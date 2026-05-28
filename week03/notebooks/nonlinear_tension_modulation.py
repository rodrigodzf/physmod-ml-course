# %% [markdown]
# # Nonlinear tension modulation
#
# Week 3 derived the modal Kirchhoff-Carrier force
#
# $$\bar f_{\mathrm{nl},\mu}(\mathbf q) \propto \lambda_\mu^2 q_\mu \sum_\nu \lambda_\nu^2 q_\nu^2.$$
#
# This exercise compares a linear modal string with the same string after adding that structured cubic term. The aim is not to build the most accurate string model, but to see what a state-dependent restoring force does to the sound and to the modal trajectories.

# %%
import dataclasses

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

from jaxdiffmodal.excitations import create_pluck_modal
from jaxdiffmodal.ftm import (
    StringParameters,
    damping_term,
    evaluate_string_eigenfunctions,
    stiffness_term,
    string_eigenvalues,
)
from jaxdiffmodal.time_integrators import solve_sv_two_step, solve_sv_one_step

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Given data
#
# You do not need to edit this section. The value `beta` below is a deliberately scaled nonlinearity strength, so that the effect is audible in a short classroom experiment.

# %%
n_modes = 32
sample_rate = 44100
duration = 1.5
n_steps = int(sample_rate * duration)
dt = 1.0 / sample_rate

pluck_position = 0.18
readout_position = 0.82
initial_deflection = 0.09

params = dataclasses.replace(
    StringParameters(),
    d1=8e-4,
    d3=1.4e-5,
)

lambda_mu = string_eigenvalues(n_modes, params.length)
omega0_squared = stiffness_term(params, lambda_mu)
gamma2 = damping_term(params, lambda_mu)
gamma = 0.5 * gamma2

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

# Try values from 0.0 to 5e5 after the TODOs are filled.
beta = 10e5

# %% [markdown]
# Note: in this notebook, `lambda_mu` stores the squared wavenumber used by `jaxdiffmodal`. It corresponds to the $\lambda_\mu^2$ factor in the slides.
#
# %% [markdown]
# ## Task 1: implement the nonlinear modal force
#
# The nonlinear force can be written as one scalar stretch factor multiplied by each modal displacement. The output-side `lambda_mu` can stay outside the contraction.
#
# $$r_\mu(\mathbf q) = \beta\,\lambda_\mu^2 q_\mu \sum_\nu \lambda_\nu^2q_\nu^2.$$
#
# Fill in the TODO below. When `beta = 0`, this function should return only zeros.


# %%
def nonlinear_force(q, beta):
    # TODO: compute stretch = sum_n lambda_n q_n^2 using jnp.einsum
    # TODO: return the structured cubic restoring force, shape (n_modes,)
    stretch = ...

    return force


# %% [markdown]
# ## Task 2: compare linear and nonlinear acceleration
#
# The modal equation is
#
# $$\ddot q_\mu = -2\gamma_\mu\dot q_\mu -\omega_\mu^2q_\mu -r_\mu(\mathbf q).$$


# %%
def acceleration(q, v, beta):
    linear = -2.0 * gamma * v - omega0_squared * q
    nonlinear = nonlinear_force(q, beta)
    return linear - nonlinear


# %% [markdown]
# ## Task 3: simulate with velocity Verlet
#
# This is the same kick-drift-kick update used by `jaxdiffmodal.time_integrators.solve_sv_one_step` in the no-excitation case. It is good enough for this listening exercise, but the nonlinear case can become unstable if `beta` or the initial deflection are made too large.


# %%
damping_factor = 1.0 + gamma2 * dt / 2.0


def step(state, _):
    q, v, beta = state

    v_half_next = v + 0.5 * dt * (
        -gamma2 * v - omega0_squared * q - nonlinear_force(q, beta)
    )
    q_next = q + dt * v_half_next
    a_next = -omega0_squared * q_next - nonlinear_force(q_next, beta)
    v_next = (v_half_next + 0.5 * dt * a_next) / damping_factor

    return (q_next, v_next, beta), q_next


def simulate(beta):
    state0 = (q0, v0, beta)
    _, q_tail = jax.lax.scan(step, state0, None, length=n_steps - 1)
    q = jnp.vstack([q0[None, :], q_tail])
    y = q @ readout_weights
    y = y / (jnp.max(jnp.abs(y)) + 1e-12)
    return y, q


# %% [markdown]
# ## Task 4: listen and inspect
#
# After the TODOs are filled:
#
# 1. Run the simulation with `beta = 0`.
# 2. Run it again with the provided non-zero `beta`.
# 3. Try halving and doubling `initial_deflection`.
# 4. Listen for pitch glide.

# %%
y_linear, q_linear = simulate(beta=0.0)
y_nonlinear, q_nonlinear = simulate(beta=beta)

fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
axes[0].plot(np.asarray(y_linear[: sample_rate // 5]))
axes[0].set_title("Linear, first 200 ms")
axes[0].grid(True)

axes[1].plot(np.asarray(y_nonlinear[: sample_rate // 5]))
axes[1].set_title("Nonlinear, first 200 ms")
axes[1].set_xlabel("Sample")
axes[1].grid(True)
fig.tight_layout()

print("Linear")
display(Audio(np.asarray(y_linear), rate=sample_rate))

print("Nonlinear")
display(Audio(np.asarray(y_nonlinear), rate=sample_rate))

# %% [markdown]
# ## Questions
#
# - What changes when `beta` is increased?
# - Does the effect become stronger or weaker as the sound decays?
# - Why does changing `initial_deflection` matter more in the nonlinear case?
# - Which line in `nonlinear_force` makes the modes depend on each other?

# %%
