# %% [markdown]
# # Complex modal recursion
#
# In Week 2, each linear mode became an independent damped oscillator:
#
# $$\ddot q_\mu + 2\gamma_\mu \dot q_\mu + \omega_{0,\mu}^2 q_\mu = 0.$$
#
# This activity rewrites those real second-order oscillators as first-order complex recursions. The physical parameters, modal frequencies, damping rates, readout weights and initial modal state are provided. Your task is to assemble the complex diagonal system, advance it in time, and listen to the resulting sound.

# %%
from jax._src import frozen_dict
import dataclasses

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import IPython.display as ipd

from jaxdiffmodal.excitations import create_pluck_modal
from jaxdiffmodal.ftm import (
    StringParameters,
    damping_term,
    evaluate_string_eigenfunctions,
    stiffness_term,
    string_eigenfunctions,
    string_eigenvalues,
)

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Given data
#
# We use `jaxdiffmodal` to compute the ingredients for a damped stiff string. You do not need to edit this section.

# %%
n_modes = 48
sample_rate = 44100
duration = 2.0
n_steps = int(sample_rate * duration)
dt = 1.0 / sample_rate

pluck_position = 0.18
readout_position = 0.82
initial_deflection = 0.003

params = dataclasses.replace(
    StringParameters(),
    d1=8e-4,
    d3=1.4e-5,
)

lambda_mu = string_eigenvalues(n_modes, params.length)
omega0_squared = stiffness_term(params, lambda_mu)
gamma2 = damping_term(params, lambda_mu)

gamma = 0.5 * gamma2
omega0 = jnp.sqrt(omega0_squared)
omega_d = jnp.sqrt(jnp.maximum(omega0_squared - gamma**2, 0.0))

mode_numbers = jnp.arange(1, n_modes + 1)
readout_weights = evaluate_string_eigenfunctions(
    mode_numbers,
    readout_position,
    params,
)

q0 = create_pluck_modal(
    lambda_mu,
    pluck_position=pluck_position,
    initial_deflection=initial_deflection,
    string_length=params.length,
)
q0 = jnp.asarray(q0)
v0 = jnp.zeros_like(q0)

print("omega_d shape:", omega_d.shape)
print("gamma shape:", gamma.shape)
print("q0 shape:", q0.shape)
print("readout_weights shape:", readout_weights.shape)

# %% [markdown]
# ## Inspect the initial condition

# %%
grid = jnp.linspace(0.0, params.length, 301)
wavenumbers = jnp.sqrt(lambda_mu)
K = string_eigenfunctions(wavenumbers, grid)
u0_grid = (2.0 / params.length) * (K.T @ q0)

fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot(grid, u0_grid)
ax.axvline(pluck_position * params.length, color="tab:red", linestyle=":")
ax.axvline(readout_position * params.length, color="tab:green", linestyle=":")
ax.set_xlabel("Position [m]")
ax.set_ylabel("Initial deflection [m]")
ax.set_title("Initial pluck shape")
ax.grid(True)
fig.tight_layout()

# %% [markdown]
# ## Convert real initial conditions to a complex state
#
# For one mode, use
#
# $$q_\mu(t) = \operatorname{Re}\{z_\mu(t)\}, \qquad \dot z_\mu = s_\mu z_\mu, \qquad s_\mu = -\gamma_\mu + i\omega_{d,\mu}.$$
#
# The complex initial state that matches $(q_\mu^0 = q_\mu(0))$ and $(v_\mu^0 = \dot q_\mu(0))$ is:
#
# $$z_\mu^0 = q_\mu^0 - i\frac{v_\mu^0+\gamma_\mu q_\mu^0}{\omega_{d,\mu}}.$$
#
# The small `eps` only prevents division by zero if a mode is critically damped.

# %%
eps = 1e-12
z0 = q0 - 1j * (v0 + gamma * q0) / jnp.maximum(omega_d, eps)

# %% [markdown]
# ## Task 1: assemble the complex diagonal system
#
# Fill in the three lines below. Use
#
# $$s_\mu = -\gamma_\mu + i\omega_{d,\mu}, \qquad \mathbf A = \operatorname{diag}(\mathbf s), \qquad \overline{\mathbf A} = e^{\mathbf A\Delta t} = \operatorname{diag}\left(e^{\mathbf s\Delta t}\right).$$
#
# Here $\mathbf A$ is the continuous-time diagonal matrix, while $\overline{\mathbf A}$ is the discrete-time update matrix. The diagonal entries of $\overline{\mathbf A}$ are the discrete poles.
#
# Expected shapes:
#
# - `s`: `(n_modes,)`
# - `A`: `(n_modes, n_modes)`
# - `A_bar`: `(n_modes, n_modes)`

# %%
# TODO: continuous-time poles
s = -gamma + 1j * omega_d

# TODO: complex diagonal matrix
A = jnp.diag(s)

# TODO: discrete-time diagonal update matrix, A_bar = diag(exp(s * dt))
A_bar = jnp.diag(jnp.exp(s * dt))

# Discrete poles, extracted from A_bar for the vectorised implementations below.
a = jnp.diag(A_bar)

# %% [markdown]
# ## Task 2: advance the state
#
# We sample the exact continuous-time flow at the audio rate:
#
# $$\mathbf z^n = e^{\mathbf A n\Delta t}\mathbf z^0 = \overline{\mathbf A}^{\,n}\mathbf z^0.$$
#
# So the recursion we implement is discrete time:
#
# $$\mathbf z^{n+1} = \overline{\mathbf A}\mathbf z^n.$$
#
# Choose one implementation. Both should produce an array `z` with shape `(n_steps, n_modes)`.

# %% [markdown]
# ### Option A: direct exponentiation

# %%
n = jnp.arange(n_steps)
z = z0[None, :] * a[None, :] ** n[:, None]

# %% [markdown]
# ### Option B: sequential recursion


# %%
# def step(z_prev, _):
#     z_next = a * z_prev
#     return z_next, z_next
#
# _, z_tail = jax.lax.scan(f=step, init=z0, xs=None, length=n_steps - 1)
# z = jnp.vstack([z0[None, :], z_tail])

# %% [markdown]
# ## Task 3: listen to the readout
#
# The readout is a weighted sum of modal states:
#
# $$y^n = \operatorname{Re}\left\{\sum_{\mu=1}^{N} c_\mu z_\mu^n\right\}.$$

# %%
# TODO: replace the right-hand side with the modal readout
y = (z @ readout_weights.T).real

# Normalize
y = y / (jnp.max(jnp.abs(y)) + 1e-12)

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(np.asarray(y[: sample_rate // 10]))
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title("First 100 ms")
ax.grid(True)
fig.tight_layout()

# %%
ipd.Audio(np.asarray(y), rate=sample_rate)

# %% [markdown]
# ## Checks
#
# Use these checks after filling the TODOs.

# %%
assert s.shape == (n_modes,)
assert A.shape == (n_modes, n_modes)
assert A_bar.shape == (n_modes, n_modes)
assert a.shape == (n_modes,)
assert z.shape == (n_steps, n_modes)
assert y.shape == (n_steps,)
assert jnp.all(jnp.abs(a) < 1.0)

print("All checks passed.")

# %% [markdown]
# ## Extra
#
# - Plot magnitudes `abs(a)`, what do they tell us about stability?
# - Which implementation is fastest for this small example?
# - Which implementation would you choose if `n_modes` and `n_steps` were much larger?

# %% [markdown]
# # Part 2: Recover the Linear Poles
#
# Now we turn the simulation around. A target modal trajectory is generated from known frequencies and decay rates. Your task is to recover those parameters by gradient descent.
#
# We will optimise an unconstrained parameter vector, then map it to stable poles:
#
# $$s_\mu = -\gamma_\mu + i\omega_\mu, \qquad \overline{\mathbf A} = \operatorname{diag}\left(e^{\mathbf s\Delta t}\right).$$
#
# We recover the continuous poles $s_\mu$, then use the discrete update matrix $\overline{\mathbf A}$ to generate the sampled trajectory.
#
# The constraints are:
#
# $$\gamma_\mu \ge 0 \quad\Longleftrightarrow\quad \operatorname{Re}(s_\mu) \le 0$$
#
# and therefore the magnitudes of the discrete poles are within the unit circle:
#
# $$|a_\mu| \le 1.$$

# %% [markdown]
# ## Target simulation
#
# We use the first few modes to keep the optimisation fast. This section is fully specified; do not edit it.

# %%
recovery_n_modes = 4
recovery_n_steps = 2048

omega_target = omega_d[:recovery_n_modes]
gamma_target = gamma[:recovery_n_modes]
z0_target = z0[:recovery_n_modes]

s_target = -gamma_target + 1j * omega_target
a_target = jnp.exp(s_target * dt)
n_recovery = jnp.arange(recovery_n_steps)

z_target = z0_target[None, :] * a_target[None, :] ** n_recovery[:, None]
q_target = jnp.real(z_target)

target_scale = jnp.maximum(jnp.std(q_target, axis=0), 1e-10)

print("target frequencies [Hz]:", np.asarray(omega_target / (2.0 * jnp.pi)))
print("target decays [1/s]:", np.asarray(gamma_target))

# %% [markdown]
# ## Task 4: choose initial pole guesses
#
# Choose rough starting values for the optimiser. They do not need to be perfect, but they should be plausible. Frequencies are in Hz and decays are in \(1/\mathrm{s}\).

# %%
# TODO: edit these initial frequency guesses and observe convergence
omega_init_hz = jnp.array([250.0, 490.0, 745.0, 985.0])

# TODO: edit these initial decay guesses and observe convergence
gamma_init = jnp.array([1.0, 1.7, 3.2, 5.0])

# %% [markdown]
# We optimise unconstrained values, then transform them into physically valid frequencies and decays.

# %%
omega_nyquist = jnp.pi / dt


def logit(x):
    return jnp.log(x) - jnp.log1p(-x)


def inv_softplus(x):
    return jnp.log(jnp.expm1(x))


omega_init = 2.0 * jnp.pi * omega_init_hz
omega_init_unit = jnp.clip(omega_init / omega_nyquist, 1e-4, 1.0 - 1e-4)
gamma_init = jnp.maximum(gamma_init, 1e-6)

fit_params = {
    "omega_raw": logit(omega_init_unit),
    "gamma_raw": inv_softplus(gamma_init),
}

# %% [markdown]
# ## Task 5: constrain the poles
#
# Fill in the constraints below.
#
# Hints:
#
# - use `jax.nn.sigmoid` to keep \(\omega_\mu\) between \(0\) and Nyquist
# - use `jax.nn.softplus` to keep \(\gamma_\mu \ge 0\)
# - build $s_\mu=-\gamma_\mu+i\omega_\mu$
# - build $\overline{\mathbf A}=\operatorname{diag}(e^{s_\mu\Delta t})$


# %%
def constrain_poles(params):
    # TODO: constrained angular frequencies, 0 <= omega <= omega_nyquist
    omega = omega_nyquist * jax.nn.sigmoid(params["omega_raw"])

    # TODO: constrained decays, gamma >= 0
    gamma_fit = jax.nn.softplus(params["gamma_raw"])

    # TODO: continuous-time poles, real part must be <= 0
    s_fit = -gamma_fit + 1j * omega

    # TODO: discrete-time update matrix, diagonal magnitudes must be <= 1
    A_bar_fit = jnp.diag(jnp.exp(s_fit * dt))

    return omega, gamma_fit, s_fit, A_bar_fit


# Run these checks after filling the TODOs.
omega_probe, gamma_probe, s_probe, A_bar_probe = constrain_poles(fit_params)
assert omega_probe.shape == (recovery_n_modes,)
assert gamma_probe.shape == (recovery_n_modes,)
assert s_probe.shape == (recovery_n_modes,)
assert A_bar_probe.shape == (recovery_n_modes, recovery_n_modes)
assert jnp.all(jnp.real(s_probe) <= 1e-12)
assert jnp.allclose(A_bar_probe, jnp.diag(jnp.diag(A_bar_probe)))
a_probe = jnp.diag(A_bar_probe)
assert jnp.all(jnp.abs(a_probe) <= 1.0 + 1e-12)

# %% [markdown]
# ## Differentiable simulator
#
# This is the same diagonal modal recursion as before, but now it is inside a loss function.


# %%
def predict_modal_trajectory(params):
    _, _, _, A_bar_fit = constrain_poles(params)
    a_fit = jnp.diag(A_bar_fit)
    z_pred = z0_target[None, :] * a_fit[None, :] ** n_recovery[:, None]
    return jnp.real(z_pred)


def loss_fn(params):
    q_pred = predict_modal_trajectory(params)
    error = (q_pred - q_target) / target_scale[None, :]
    return jnp.mean(error**2)


# %% [markdown]
# ## Task 6: optimise with Optax
#
# Run the optimiser after the pole constraints are implemented.

# %%
learning_rate = 1e-2
n_opt_steps = 600

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(fit_params)


@jax.jit
def train_step(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


loss_history = []
for step_idx in range(n_opt_steps):
    fit_params, opt_state, loss = train_step(fit_params, opt_state)
    loss_history.append(float(loss))
    if step_idx % 100 == 0:
        print(f"step {step_idx:04d}  loss={float(loss):.6e}")

print(f"final loss={loss_history[-1]:.6e}")

# %% [markdown]
# ## Inspect the recovered parameters

# %%
omega_fit, gamma_fit, s_fit, A_bar_fit = constrain_poles(fit_params)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].plot(loss_history)
axes[0].set_yscale("log")
axes[0].set_title("Loss")
axes[0].set_xlabel("Step")
axes[0].grid(True)

axes[1].plot(np.asarray(omega_target / (2.0 * jnp.pi)), "o-", label="target")
axes[1].plot(np.asarray(omega_fit / (2.0 * jnp.pi)), "x--", label="fit")
axes[1].set_title("Frequency [Hz]")
axes[1].set_xlabel("Mode")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(np.asarray(gamma_target), "o-", label="target")
axes[2].plot(np.asarray(gamma_fit), "x--", label="fit")
axes[2].set_title("Decay [1/s]")
axes[2].set_xlabel("Mode")
axes[2].legend()
axes[2].grid(True)

fig.tight_layout()

# %%
q_fit = predict_modal_trajectory(fit_params)

fig, axes = plt.subplots(recovery_n_modes, 1, figsize=(8, 7), sharex=True)
plot_steps = min(1000, recovery_n_steps)
for mode_idx, ax in enumerate(axes):
    ax.plot(np.asarray(q_target[:plot_steps, mode_idx]), label="target")
    ax.plot(np.asarray(q_fit[:plot_steps, mode_idx]), "--", label="fit")
    ax.set_ylabel(f"mode {mode_idx + 1}")
    ax.grid(True)
axes[0].legend()
axes[-1].set_xlabel("Sample")
fig.tight_layout()

# %% [markdown]
#
# - What happens if you initialise a frequency far from the target?
# - Why do we constrain $\gamma_\mu$ to be non-negative?
# - What would happen to the audio if $|a_\mu| > 1$?

# %%
# %% [markdown]
# ## Further reading
#
# [Methods for Synthesizing Very High Q Parametrically Well Behaved Two Pole Filters](https://ccrma.stanford.edu/~jos/smac03maxjos/) is a nice companion reference. It also treats oscillators through stable pole locations and exact pole evolution, which is the same idea behind $a_\mu = e^{s_\mu\Delta t}$ in this notebook.

# %%
