# %% [markdown]
# # Progressive guitar string
#
# This example builds up a realistic guitar string in four stages,
# starting from the simplest possible model and adding one physical
# effect at a time:
#
# 1. **Ideal string**: only tension and density (no damping, no
#    stiffness, no tension modulation).
# 2. **+ Stiffness**: add the bending stiffness term coming from
#    Young's modulus and the moment of inertia.
# 3. **+ Damping**: add the frequency-independent and
#    frequency-dependent loss terms ($d_1$ and $d_3$).
# 4. **+ Tension modulation**: add the geometric non-linearity
#    that couples the modes through the average string elongation.
#
# At every stage we pluck the string with the same initial deflection
# and listen to the readout at the same position so the differences
# between models are easy to hear.

# %%
import dataclasses

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

from jaxdiffmodal.excitations import create_pluck_modal
from jaxdiffmodal.ftm import (
    damping_term,
    evaluate_string_eigenfunctions,
    inverse_STL,
    stiffness_term,
    string_eigenfunctions,
    string_eigenvalues,
    StringParameters,
)
from jaxdiffmodal.time_integrators import (
    make_tm_nl_fn,
    solve_tf,
    string_tau_with_density,
)


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Shared setup
#
# We use the same number of modes, sample rate, pluck position and
# readout position for every stage.

# %%
n_modes = 100
sample_rate = 192000
n_steps = sample_rate * 4
dt = 1.0 / sample_rate
excitation_position = 0.1
readout_position = 0.9
initial_deflection = 0.003
n_gridpoints = 101

base_params = StringParameters(d1=8e-4)

lambda_mu = string_eigenvalues(n_modes, base_params.length)
wn = np.sqrt(lambda_mu)
grid = np.linspace(0, base_params.length, n_gridpoints)
K = string_eigenfunctions(wn, grid)

mu = np.arange(1, n_modes + 1)
readout_weights = evaluate_string_eigenfunctions(
    mu,
    readout_position,
    base_params,
)

u0_modal = create_pluck_modal(
    lambda_mu,
    pluck_position=excitation_position,
    initial_deflection=initial_deflection,
    string_length=base_params.length,
)
u0 = inverse_STL(K, u0_modal, base_params.length)

fig, ax = plt.subplots(1, 1, figsize=(6, 2))
ax.plot(grid, u0)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Deflection (m)")
ax.set_title("Initial pluck shape")
ax.grid(True)
fig.tight_layout()


# %% [markdown]
# ## Helper to integrate and read out
#
# A small helper that takes a `StringParameters` instance and an
# optional non-linear term and returns the readout signal in the
# physical domain.

# %%
def simulate_string(params, nl_fn=None, u0=None):
    gamma2_mu = damping_term(params, lambda_mu)
    omega_mu_squared = stiffness_term(params, lambda_mu)

    if nl_fn is None:
        nl_fn = lambda q: jnp.zeros_like(q)
    if u0 is None:
        u0 = u0_modal

    _, modal_sol, _ = solve_tf(
        gamma2_mu,
        omega_mu_squared,
        dt=dt,
        n_steps=n_steps,
        nl_fn=nl_fn,
        u0=u0,
        v0=jnp.zeros_like(u0),
    )
    modal_sol = modal_sol.T  # (n_modes, n_steps)
    return readout_weights @ modal_sol


def show_result(signal, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 2), dpi=150)
    ax.plot(np.asarray(signal))
    ax.set_xlabel("Sample")
    ax.set_ylabel("Deflection (m)")
    ax.set_title(title)
    ax.set_xlim(-2, sample_rate // 10)
    ax.grid(True)
    fig.tight_layout()
    display(Audio(np.asarray(signal), rate=sample_rate))


# %% [markdown]
# ## Stage 1: ideal string
#
# We start from the wave equation with no losses and no stiffness:
#
# $$\rho A\,\ddot w - T_0\,\Delta w = 0.$$
#
# In the parameters this means $E = 0$, $d_1 = 0$ and $d_3 = 0$. The
# resulting tone is perfectly harmonic and rings forever.

# %%
params_ideal = dataclasses.replace(
    base_params,
    E=0.0,
    d1=0.0,
    d3=0.0,
)
signal_ideal = simulate_string(params_ideal)
show_result(signal_ideal, "Stage 1: ideal string")

# %% [markdown]
# ## Stage 2: add stiffness
#
# Real strings have a small bending stiffness $D = E I$ that
# stretches the partials slightly above the harmonic series
# (inharmonicity). We restore the stiffness term while keeping the
# damping at zero.

# %%
params_stiff = dataclasses.replace(
    base_params,
    d1=0.0,
    d3=0.0,
)
signal_stiff = simulate_string(params_stiff)
show_result(signal_stiff, "Stage 2: + stiffness")

# %% [markdown]
# ## Stage 3: add damping
#
# Energy is lost both uniformly across modes ($d_1$) and faster for
# higher modes ($d_3 \lambda_\mu$). With these terms the tone now
# decays naturally.

# %%
params_damped = dataclasses.replace(
    base_params,
    d1=8e-4,
    d3=1.4e-5,
)
signal_damped = simulate_string(params_damped)
show_result(signal_damped, "Stage 3: + damping")

# %% [markdown]
# ## Stage 4: add tension modulation
#
# At larger amplitudes the average elongation of the string raises
# the effective tension, which in turn pulls the partials up while
# the string is still vibrating. In modal coordinates the extra
# force is
#
# $$\bar f_{\text{nl},\mu} = \lambda_\mu q_\mu \,\frac{E A}{2 L}\,
#   \sum_{\nu} \frac{\lambda_\nu q_\nu^2}{\lVert\Phi_\nu\rVert^2}.$$
#
# We pluck the string a bit harder so the effect is audible.

# %%
initial_deflection_nl = 0.01
u0_modal_nl = create_pluck_modal(
    lambda_mu,
    pluck_position=excitation_position,
    initial_deflection=initial_deflection_nl,
    string_length=base_params.length,
)

string_tau = string_tau_with_density(base_params)
string_norm = base_params.length / 2
string_tau = string_tau * lambda_mu / string_norm
nl_fn_tm = make_tm_nl_fn(lambda_mu, string_tau)

signal_tm = simulate_string(base_params, nl_fn=nl_fn_tm, u0=u0_modal_nl)
show_result(signal_tm, "Stage 4: + tension modulation")

# %% [markdown]
# ## Compare spectra
#
# Plotting the magnitude spectra side by side highlights how each
# effect changes the partials: stiffness stretches them upward,
# damping rounds the peaks, and tension modulation introduces small
# pitch glides and extra sidebands.

# %%
def spectrum(signal):
    window = np.hanning(len(signal))
    spec = np.fft.rfft(np.asarray(signal) * window)
    freqs = np.fft.rfftfreq(len(signal), dt)
    mag_db = 20.0 * np.log10(np.abs(spec) + 1e-12)
    return freqs, mag_db


fig, ax = plt.subplots(1, 1, figsize=(7, 3))
for name, sig in [
    ("ideal", signal_ideal),
    ("+ stiffness", signal_stiff),
    ("+ damping", signal_damped),
    ("+ tension mod.", signal_tm),
]:
    freqs, mag_db = spectrum(sig)
    ax.plot(freqs, mag_db, label=name, linewidth=0.8)

ax.set_xscale("log")
ax.set_xlim(20, sample_rate / 2)
ax.set_ylim(-80, ax.get_ylim()[1])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Readout spectra at each stage")
ax.legend()
ax.grid(True)
fig.tight_layout()
# %%
