# %% [markdown]
# # Modal identification of a vibrating string from grid observations
#
# We observe a 1D vibrating string at $N_x = 101$ uniformly spaced points over
# $t \in [0, 1]$ s with $T = 1001$ samples, giving $U \in \mathbb{R}^{101\times T}$.
# We learn a continuous, interpretable representation
#
# $$u(x, t) = \sum_{n=1}^{N} e^{-\gamma_n t}\bigl[a_n\cos(\omega_n t) + b_n\sin(\omega_n t)\bigr]\phi_n(x),$$
#
# jointly identifying spatial modes $\phi_n$ (a shallow $\sin$-network), modal
# frequencies $\omega_n$, damping rates $\gamma_n \ge 0$, and amplitudes
# $a_n, b_n$.

# %%
from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

print("jax", jax.__version__, "devices:", jax.devices())

# %% [markdown]
# ## Problem parameters

# %%
L = 1.0
Nx = 101
T_steps = 1001
T_end = 1.0
N = 24                  # number of modes to fit
m = 128                 # width of spatial hidden layer

x = jnp.linspace(0.0, L, Nx)
t = jnp.linspace(0.0, T_end, T_steps)
dt = float(t[1] - t[0])
dx = float(x[1] - x[0])

omega_nyquist = jnp.pi / dt   # rad/s
print(f"dt={dt:.4e}  dx={dx:.4e}  omega_nyquist={float(omega_nyquist):.1f} rad/s")

# %% [markdown]
# ## Synthetic data
#
# A linear damped string with $N_\text{true}$ Dirichlet modes, $\phi_n(x) =
# \sqrt{2/L}\sin(n\pi x/L)$, frequencies $\omega_n = n\pi c/L$, and damping
# growing mildly with $n$. We use this only to have something to identify; the
# identification code never sees `truth`.

# %%
def make_synthetic(key, c=150.0, N_true=6):
    n_arr = jnp.arange(1, N_true + 1, dtype=jnp.float32)
    omega_true = n_arr * jnp.pi * c / L
    gamma_true = 0.3 + 0.15 * n_arr
    k1, k2 = jax.random.split(key)
    a_true = jax.random.normal(k1, (N_true,))
    b_true = jax.random.normal(k2, (N_true,)) * 0.5
    phi_true = jnp.sqrt(2.0 / L) * jnp.sin(
        n_arr[None, :] * jnp.pi * x[:, None] / L
    )                                                       # (Nx, N_true)
    env = jnp.exp(-gamma_true[None, :] * t[:, None])         # (T, N_true)
    osc = (
        a_true[None, :] * jnp.cos(omega_true[None, :] * t[:, None])
        + b_true[None, :] * jnp.sin(omega_true[None, :] * t[:, None])
    )
    B_true = env * osc                                       # (T, N_true)
    U = phi_true @ B_true.T                                  # (Nx, T)
    truth = dict(
        omega=omega_true, gamma=gamma_true, a=a_true, b=b_true, phi=phi_true,
        c=c,
    )
    return U, truth


key = jax.random.key(0)
key, sub = jax.random.split(key)
U, truth = make_synthetic(sub)
print("U shape:", U.shape, "U range:", float(U.min()), float(U.max()))
print("true omegas (rad/s):", np.asarray(truth["omega"]))
print("true gammas (1/s):", np.asarray(truth["gamma"]))

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
im = axes[0].imshow(
    np.asarray(U), aspect="auto", origin="lower",
    extent=[float(t[0]), float(t[-1]), 0.0, L], cmap="RdBu_r",
    vmin=-float(jnp.abs(U).max()), vmax=float(jnp.abs(U).max()),
)
axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("x [m]")
axes[0].set_title("U(x, t) — synthetic data")
plt.colorbar(im, ax=axes[0])
for xi in (0.13, 0.37, 0.71):
    k = int(round(xi / L * (Nx - 1)))
    axes[1].plot(np.asarray(t), np.asarray(U[k]), lw=0.7, label=f"x={xi}")
axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("u")
axes[1].set_title("time series at three points"); axes[1].legend()
plt.tight_layout()

# %% [markdown]
# ## Architecture
#
# **Spatial trunk** (SIREN-style, single hidden layer):
# $h_j(x) = \sin(w_j x + b_j)$, $\phi_n(x) = \sum_j A_{nj} h_j(x)$, then
# multiplied by $x(L-x)/L^2$ so $\phi_n(0)=\phi_n(L)=0$ exactly.
#
# **Temporal**: per mode, scalars $\omega_n,\ \tilde\gamma_n,\ a_n,\ b_n$, with
# $\gamma_n = \mathrm{softplus}(\tilde\gamma_n) \ge 0$.

# %%
def init_spatial(key, m=128, N=24, L=1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    w = jax.random.uniform(k1, (m,), minval=-N * jnp.pi / L, maxval=N * jnp.pi / L)
    b = jax.random.uniform(k2, (m,), minval=-jnp.pi, maxval=jnp.pi)
    A = jax.random.normal(k3, (N, m)) / jnp.sqrt(m)
    return {"w": w, "b": b, "A": A}


def spatial_basis(theta, x, L):
    """Phi: (Nx, N) with exact Dirichlet ends."""
    h = jnp.sin(theta["w"][None, :] * x[:, None] + theta["b"][None, :])  # (Nx, m)
    phi_raw = h @ theta["A"].T                                            # (Nx, N)
    envelope = x * (L - x) / (L ** 2)
    return envelope[:, None] * phi_raw


def temporal_matrix(omega, gamma_tilde, a, b, t):
    """B: (T, N)."""
    gamma = jax.nn.softplus(gamma_tilde)
    env = jnp.exp(-gamma[None, :] * t[:, None])
    osc = (
        a[None, :] * jnp.cos(omega[None, :] * t[:, None])
        + b[None, :] * jnp.sin(omega[None, :] * t[:, None])
    )
    return env * osc


def predict(params, x, t, L):
    Phi = spatial_basis(params["theta"], x, L)
    B = temporal_matrix(
        params["omega"], params["gamma_tilde"], params["a"], params["b"], t
    )
    return Phi @ B.T

# %% [markdown]
# ## Initialisation
#
# Frequencies are seeded from FFT peaks of the data (averaged over $x$). The
# remaining $\omega_n$ slots (if fewer than $N$ peaks are usable) are filled
# with a geometric sweep.

# %%
def fft_peak_init(U, t, N, height_frac=1e-3):
    U_np = np.asarray(U)
    dt = float(t[1] - t[0])
    U_hat = np.fft.rfft(U_np, axis=1)
    freqs_hz = np.fft.rfftfreq(U_np.shape[1], dt)
    omega = 2.0 * np.pi * freqs_hz
    power = (np.abs(U_hat) ** 2).mean(axis=0)
    power[0] = 0.0                              # ignore DC
    peaks, props = find_peaks(power, height=height_frac * power.max())
    order = np.argsort(props["peak_heights"])[::-1]
    peaks = peaks[order]
    chosen = omega[peaks][:N]
    if len(chosen) < N:
        # fill the rest with a coarse geometric sweep up to Nyquist
        omega_max = float(np.pi / dt)
        extra = np.geomspace(50.0, 0.9 * omega_max, N - len(chosen))
        chosen = np.concatenate([chosen, extra])
    return jnp.asarray(np.sort(chosen)), omega, power


omega_init, fft_omega, fft_power = fft_peak_init(U, t, N)
print("FFT-init omegas (rad/s):", np.asarray(omega_init))

fig, ax = plt.subplots(figsize=(7, 3))
ax.semilogy(fft_omega, fft_power, lw=0.6)
for w in np.asarray(omega_init):
    ax.axvline(w, color="r", lw=0.4, alpha=0.6)
for w in np.asarray(truth["omega"]):
    ax.axvline(w, color="g", lw=0.6, ls="--", alpha=0.8)
ax.set_xlabel("ω [rad/s]"); ax.set_ylabel("mean |Û|²")
ax.set_title("temporal FFT (red: init picks, green dashed: truth)")
ax.set_xlim(0, 1.05 * float(omega_init.max()))
plt.tight_layout()

# %%
def init_params(key, N, m, L, omega_init):
    k_th, k_a, k_b = jax.random.split(key, 3)
    theta = init_spatial(k_th, m=m, N=N, L=L)
    gamma_tilde0 = jnp.full(
        (N,), float(np.log(np.expm1(0.01)))   # softplus^{-1}(0.01)
    )
    a = jax.random.normal(k_a, (N,)) * 0.1
    b = jax.random.normal(k_b, (N,)) * 0.1
    return {
        "theta": theta,
        "omega": omega_init.astype(jnp.float32),
        "gamma_tilde": gamma_tilde0,
        "a": a,
        "b": b,
    }


key, sub = jax.random.split(key)
params = init_params(sub, N=N, m=m, L=L, omega_init=omega_init)
Phi_init = np.asarray(spatial_basis(params["theta"], x, L))
print("param shapes:",
      {k: (v.shape if hasattr(v, "shape") else {kk: vv.shape for kk, vv in v.items()})
       for k, v in params.items()})

# %% [markdown]
# ## Loss
#
# $$\mathcal{L} = \tfrac{1}{N_x T}\|U_\text{pred} - U\|_F^2 + \lambda_\perp\mathcal{L}_\perp + \lambda_\omega\mathcal{L}_\omega,$$
#
# with trapezoidal orthogonality penalty and Gaussian frequency-separation
# penalty.

# %%
# trapezoidal weights for L^2 inner products on the grid
trap_w = jnp.ones(Nx).at[0].set(0.5).at[-1].set(0.5) * dx


def ortho_penalty(Phi, trap_w, eps=1e-8):
    # Gram matrix in L^2: G = Phi^T diag(w) Phi
    G = (Phi * trap_w[:, None]).T @ Phi              # (N, N)
    norms = jnp.sqrt(jnp.clip(jnp.diag(G), eps))
    G_norm = G / (norms[:, None] * norms[None, :] + eps)
    off = G_norm - jnp.diag(jnp.diag(G_norm))
    return jnp.sum(off ** 2)


def freq_separation_penalty(omega, sigma):
    d = omega[:, None] - omega[None, :]
    K = jnp.exp(-(d ** 2) / (sigma ** 2))
    # subtract diagonal contributions (each is 1)
    return jnp.sum(K) - omega.shape[0]


def loss_fn(params, U, x, t, L, trap_w, lam_perp, lam_omega, sigma_omega):
    Phi = spatial_basis(params["theta"], x, L)
    B = temporal_matrix(
        params["omega"], params["gamma_tilde"], params["a"], params["b"], t
    )
    U_pred = Phi @ B.T
    rec = jnp.mean((U_pred - U) ** 2)
    L_perp = ortho_penalty(Phi, trap_w)
    L_omega = freq_separation_penalty(params["omega"], sigma_omega)
    return rec + lam_perp * L_perp + lam_omega * L_omega, {
        "rec": rec, "perp": L_perp, "omega_sep": L_omega,
    }

# %% [markdown]
# ## Optimiser with per-group learning rates
#
# We use `optax.multi_transform` to apply distinct Adam learning rates to
# $(\omega,\ \tilde\gamma,\ \text{amplitudes},\ \theta)$. The spatial-only warm
# phase is built by replacing the $\omega$ and $\tilde\gamma$ transforms with
# `optax.set_to_zero()`.

# %%
def label_fn(params):
    return {
        "theta": jax.tree_util.tree_map(lambda _: "spatial", params["theta"]),
        "omega": "omega",
        "gamma_tilde": "gamma_tilde",
        "a": "amp",
        "b": "amp",
    }


def make_optimizer(total_steps, freeze_temporal=False, lr_scale=1.0):
    def sched(base):
        return optax.cosine_decay_schedule(base * lr_scale, total_steps, alpha=0.05)
    transforms = {
        "omega":       optax.set_to_zero() if freeze_temporal else optax.adam(sched(1e-4)),
        "gamma_tilde": optax.set_to_zero() if freeze_temporal else optax.adam(sched(1e-3)),
        "amp":         optax.adam(sched(1e-2)),
        "spatial":     optax.adam(sched(1e-3)),
    }
    return optax.multi_transform(transforms, label_fn)

# %% [markdown]
# ## Training step (JIT-compiled)

# %%
def make_step(opt, L_const=L):
    """Build a JIT-compiled training step closing over the optimiser."""

    @jax.jit
    def step(params, opt_state, U, x, t, trap_w,
             lam_perp, lam_omega, sigma_omega):
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, U, x, t, L_const, trap_w,
            lam_perp, lam_omega, sigma_omega,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    return step

# %% [markdown]
# ## Training schedule
#
# 1. **Spatial-only** (1k iters): freeze $\omega, \tilde\gamma$.
# 2. **Joint** (10k iters): unfreeze, decay $\lambda_\omega$ linearly to 0.
# 3. **Refinement** (2k iters): drop $\lambda_\perp$ tenfold, anneal LR.

# %%
def train(params, U, x, t,
          n_warm=1000, n_joint=8000, n_refine=2000,
          lam_perp=1e-2, lam_omega_start=1e-3, sigma_omega=10.0,
          log_every=200):
    history = {"step": [], "loss": [], "rec": [], "perp": [], "omega_sep": []}

    # Phase 1: spatial-only
    opt1 = make_optimizer(n_warm, freeze_temporal=True)
    state = opt1.init(params)
    step_fn = make_step(opt1)
    for i in range(n_warm):
        params, state, loss_val, aux = step_fn(
            params, state, U, x, t, trap_w,
            jnp.float32(lam_perp), jnp.float32(0.0), jnp.float32(sigma_omega),
        )
        if i % log_every == 0 or i == n_warm - 1:
            history["step"].append(("warm", i))
            history["loss"].append(float(loss_val))
            history["rec"].append(float(aux["rec"]))
            history["perp"].append(float(aux["perp"]))
            history["omega_sep"].append(float(aux["omega_sep"]))
            print(f"[warm  {i:5d}] loss={float(loss_val):.4e} rec={float(aux['rec']):.4e}")

    # Phase 2: joint, decaying lam_omega
    opt2 = make_optimizer(n_joint, freeze_temporal=False)
    state = opt2.init(params)
    step_fn = make_step(opt2)
    for i in range(n_joint):
        frac = 1.0 - i / max(1, n_joint - 1)
        lo = lam_omega_start * frac
        params, state, loss_val, aux = step_fn(
            params, state, U, x, t, trap_w,
            jnp.float32(lam_perp), jnp.float32(lo), jnp.float32(sigma_omega),
        )
        if i % log_every == 0 or i == n_joint - 1:
            history["step"].append(("joint", i))
            history["loss"].append(float(loss_val))
            history["rec"].append(float(aux["rec"]))
            history["perp"].append(float(aux["perp"]))
            history["omega_sep"].append(float(aux["omega_sep"]))
            print(f"[joint {i:5d}] loss={float(loss_val):.4e} rec={float(aux['rec']):.4e} "
                  f"perp={float(aux['perp']):.3e} λω={lo:.2e}")

    # Phase 3: refinement, drop lam_perp tenfold
    opt3 = make_optimizer(n_refine, freeze_temporal=False, lr_scale=0.5)
    state = opt3.init(params)
    step_fn = make_step(opt3)
    for i in range(n_refine):
        params, state, loss_val, aux = step_fn(
            params, state, U, x, t, trap_w,
            jnp.float32(lam_perp * 0.1), jnp.float32(0.0), jnp.float32(sigma_omega),
        )
        if i % log_every == 0 or i == n_refine - 1:
            history["step"].append(("refine", i))
            history["loss"].append(float(loss_val))
            history["rec"].append(float(aux["rec"]))
            history["perp"].append(float(aux["perp"]))
            history["omega_sep"].append(float(aux["omega_sep"]))
            print(f"[refine{i:5d}] loss={float(loss_val):.4e} rec={float(aux['rec']):.4e}")

    return params, history


# %% Train (full schedule). Reduce iterations on first run if just sanity-checking.
params, history = train(
    params, U, x, t,
    n_warm=1000, n_joint=8000, n_refine=2000,
    lam_perp=1e-2, lam_omega_start=1e-3, sigma_omega=20.0,
    log_every=500,
)

# %% [markdown]
# ## Diagnostics

# %%
# Loss curve
fig, ax = plt.subplots(figsize=(7, 3))
ax.semilogy(history["rec"], label="reconstruction MSE")
ax.semilogy(history["perp"], label="orthogonality")
ax.set_xlabel("log step"); ax.set_ylabel("value"); ax.legend()
ax.set_title("training diagnostics"); plt.tight_layout()

# %%
# Identified vs true frequencies
omega_id = np.asarray(params["omega"])
gamma_id = np.asarray(jax.nn.softplus(params["gamma_tilde"]))
a_id = np.asarray(params["a"])
b_id = np.asarray(params["b"])

# mode energy = 0.5 (a^2 + b^2) * ||phi||^2
Phi_id = np.asarray(spatial_basis(params["theta"], x, L))
phi_norms_sq = np.trapezoid(Phi_id ** 2, np.asarray(x), axis=0)
energy = 0.5 * (a_id ** 2 + b_id ** 2) * phi_norms_sq
order = np.argsort(-energy)

print("Top modes by energy:")
print(f"{'idx':>4} {'omega':>10} {'gamma':>8} {'energy':>10}")
for k in order[:10]:
    print(f"{k:4d} {omega_id[k]:10.2f} {gamma_id[k]:8.3f} {energy[k]:10.3e}")

# %%
# Compare top-N_true modes with truth
n_true = int(truth["omega"].shape[0])
top = order[:n_true]
top_omega = omega_id[top]
top_gamma = gamma_id[top]
# match top to truth by nearest omega
true_omega = np.asarray(truth["omega"])
true_gamma = np.asarray(truth["gamma"])
match = []
remaining = list(top)
for w_t in true_omega:
    j = int(np.argmin(np.abs(omega_id[remaining] - w_t)))
    match.append(remaining.pop(j))
match = np.array(match)

print(f"\n{'n':>2} {'omega_true':>11} {'omega_id':>11} {'Δω':>9} "
      f"{'γ_true':>8} {'γ_id':>8}")
for n, m_idx in enumerate(match, start=1):
    dw = omega_id[m_idx] - true_omega[n - 1]
    print(f"{n:2d} {true_omega[n-1]:11.3f} {omega_id[m_idx]:11.3f} {dw:9.3e} "
          f"{true_gamma[n-1]:8.3f} {gamma_id[m_idx]:8.3f}")



# %%
# Compare identified spatial modes against true sin(n π x / L)
fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
for k, ax in enumerate(axes.ravel()):
    if k >= n_true:
        ax.axis("off"); continue
    mode_idx = match[k]
    phi_id = Phi_id[:, mode_idx]
    phi_init = Phi_init[:, mode_idx]
    phi_true = np.asarray(truth["phi"][:, k])
    # match sign and scale by least squares
    s = np.dot(phi_id, phi_true) / (np.dot(phi_id, phi_id) + 1e-12)
    s_init = np.dot(phi_init, phi_true) / (np.dot(phi_init, phi_init) + 1e-12)
    ax.plot(np.asarray(x), s_init * phi_init, color="tab:orange", alpha=0.8, label="init (scaled)")
    ax.plot(np.asarray(x), s * phi_id, color="tab:blue", label="id (scaled)")
    ax.plot(np.asarray(x), phi_true, "--", color="black", label="true")
    ax.set_title(f"mode {k+1}  ω={omega_id[mode_idx]:.1f}")
    if k == 0:
        ax.legend()
plt.tight_layout()

# %%
# Reconstruction RMSE per timestep
U_pred = np.asarray(predict(params, x, t, L))
rmse_t = np.sqrt(((U_pred - np.asarray(U)) ** 2).mean(axis=0))
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(np.asarray(t), rmse_t)
ax.set_xlabel("t [s]"); ax.set_ylabel("RMSE_x(t)")
ax.set_title("reconstruction RMSE per timestep")
plt.tight_layout()

# %%
# Extrapolate beyond the training window
t_ext = jnp.linspace(0.0, 2.0 * T_end, 2 * T_steps - 1)
U_ext = np.asarray(predict(params, x, t_ext, L))
fig, ax = plt.subplots(figsize=(9, 3.4))
amax = float(np.abs(U_ext).max())
ax.imshow(U_ext, aspect="auto", origin="lower",
          extent=[float(t_ext[0]), float(t_ext[-1]), 0.0, L],
          cmap="RdBu_r", vmin=-amax, vmax=amax)
ax.axvline(T_end, color="k", lw=0.7, ls="--")
ax.set_xlabel("t [s]"); ax.set_ylabel("x [m]")
ax.set_title("extrapolation (dashed line: end of training window)")
plt.tight_layout()

# %%
# Continuous-x evaluation: off-grid sampling
x_fine = jnp.linspace(0.0, L, 401)
Phi_fine = np.asarray(spatial_basis(params["theta"], x_fine, L))
fig, ax = plt.subplots(figsize=(7, 3))
for k in match[:4]:
    s = np.sign(np.dot(Phi_fine[:, k], np.sin(np.pi * np.asarray(x_fine) / L)))
    ax.plot(np.asarray(x_fine), s * Phi_fine[:, k] / np.abs(Phi_fine[:, k]).max(),
            label=f"ω≈{omega_id[k]:.1f}")
ax.set_xlabel("x"); ax.set_ylabel("φ_n(x) (normalised)")
ax.set_title("identified spatial basis on a finer x grid"); ax.legend()
plt.tight_layout()

# %% [markdown]
# ## What we have
#
# - Identified eigenvalues $\lambda_n = -\gamma_n + i\omega_n$.
# - Continuous spatial basis $\phi_n(x;\theta)$ evaluable off the training grid.
# - Recovered initial-condition coefficients $a_n, b_n$ (so
#   $u_0 = \sum_n a_n\phi_n$ and $v_0 = \sum_n(\omega_n b_n - \gamma_n a_n)\phi_n$).
# - A model that extrapolates with error governed by spectral accuracy rather
#   than accumulated stepwise drift.

# %%
