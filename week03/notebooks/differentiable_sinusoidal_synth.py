# %% [markdown]
# # Differentiable sinusoidal synth
#
# Continue from `jax_basics.py`: the same synthesis, loss and gradient loop can
# recover more than one parameter. Run this file's cells from top to bottom
# in the course Python environment. It includes its own setup.
#
# First fit amplitude and damping with frequency and phase fixed. Then group
# the learnable parameters in a dictionary and use Optax for updates. Finally,
# investigate how the initial frequency affects convergence. The complex
# recursion at the end is optional.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from IPython import get_ipython
from IPython.display import Audio, display

jax.config.update("jax_enable_x64", True)

print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())


def show_audio(signal, sample_rate):
    if get_ipython() is not None:
        display(Audio(np.asarray(signal), rate=sample_rate, normalize=False))


# %% [markdown]
# ## 1. Add damping to the synthesiser
#
# The sinusoid now has an exponential envelope:
#
# $$y_n = A e^{-\gamma t_n}\sin(2\pi f t_n + \varphi).$$
#
# $A$ is the initial amplitude, $\gamma$ is the decay rate in inverse seconds,
# $f$ is the frequency in Hz, and $\varphi$ is the phase in radians. Larger
# positive $\gamma$ means faster decay. Setting it to zero removes damping.
# We use 64-bit arithmetic for the comparisons later in this tutorial.


# %%
def damped_sinusoid(t, amplitude, gamma, frequency_hz, phase=0.0):
    return (
        amplitude
        * jnp.exp(-gamma * t)
        * jnp.sin(2.0 * jnp.pi * frequency_hz * t + phase)
    )


sample_rate = 8000
duration = 0.5
t = jnp.arange(int(sample_rate * duration)) / sample_rate

target_amplitude = 0.8
target_gamma = 3.0
known_frequency_hz = 440.0
known_phase = 0.0

y_target = damped_sinusoid(
    t, target_amplitude, target_gamma, known_frequency_hz, known_phase
)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.asarray(t), np.asarray(y_target), alpha=0.6, label="target")
ax.plot(
    np.asarray(t),
    np.asarray(target_amplitude * jnp.exp(-target_gamma * t)),
    "k--",
    label="amplitude envelope",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.legend()
fig.tight_layout()
plt.show()
show_audio(y_target, sample_rate)

# %% [markdown]
# **Try it:** compare `target_gamma = 0.0`, `3.0` and `8.0`. Rerun the previous
# cell each time. Restore `3.0` before continuing.
#
# ## 2. Fit two parameters with familiar gradient updates
#
# Keep frequency and phase fixed. Our two learnable parameters are $A$ and
# $\gamma$, and the loss is still the mean squared sample error:
#
# $$\mathcal L(A,\gamma) = \frac{1}{N}\sum_n(\hat y_n(A,\gamma)-y_n)^2.$$
#
# `argnums=(0, 1)` asks for derivatives with respect to the first two arguments.


# %%
def amplitude_damping_loss(amplitude, gamma, t, target, frequency_hz, phase):
    prediction = damped_sinusoid(t, amplitude, gamma, frequency_hz, phase)
    return jnp.mean((prediction - target) ** 2)


amplitude_damping_gradient = jax.value_and_grad(amplitude_damping_loss, argnums=(0, 1))

initial_amplitude = 0.3
initial_gamma = 1.0
loss_value, (amplitude_gradient, gamma_gradient) = amplitude_damping_gradient(
    initial_amplitude, initial_gamma, t, y_target, known_frequency_hz, known_phase
)
print("Initial loss:", float(loss_value))
print("Amplitude gradient:", float(amplitude_gradient))
print("Damping gradient:", float(gamma_gradient))

# %% [markdown]
# **Predict:** would you expect both gradients to point directly towards the
# target parameters? Amplitude and damping affect the same envelope, so their
# updates interact. Check the fitted result after several updates.


# %%
@jax.jit
def manual_step(amplitude, gamma, t, target, frequency_hz, phase, learning_rate):
    loss_value, (amplitude_gradient, gamma_gradient) = amplitude_damping_gradient(
        amplitude, gamma, t, target, frequency_hz, phase
    )
    return (
        amplitude - learning_rate * amplitude_gradient,
        gamma - learning_rate * gamma_gradient,
        loss_value,
    )


learning_rate = 1.0
n_manual_updates = 3000
amplitude = jnp.asarray(initial_amplitude)
gamma = jnp.asarray(initial_gamma)
manual_history = []

for step in range(n_manual_updates):
    amplitude, gamma, loss_value = manual_step(
        amplitude, gamma, t, y_target, known_frequency_hz, known_phase, learning_rate
    )
    manual_history.append(float(loss_value))

print(f"Amplitude: target={target_amplitude:.4f}, fitted={float(amplitude):.4f}")
print(f"Damping:   target={target_gamma:.4f}, fitted={float(gamma):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].semilogy(manual_history)
axes[0].set_xlabel("Update")
axes[0].set_ylabel("Mean squared error")
axes[0].set_title("Fitting amplitude and damping")
for label, envelope, style in (
    ("target", target_amplitude * jnp.exp(-target_gamma * t), "-"),
    ("initial", initial_amplitude * jnp.exp(-initial_gamma * t), "--"),
    ("fitted", amplitude * jnp.exp(-gamma * t), ":"),
):
    axes[1].plot(np.asarray(t), np.asarray(envelope), style, label=label)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Amplitude envelope")
axes[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Try it:** change `n_manual_updates` to `100`. Does a small waveform error
# already imply accurate physical parameters? Restore `3000` afterwards.
#
# ## 3. Collect learnable parameters in a dictionary
#
# A dictionary gives each parameter a name. JAX treats it as a *pytree*, a
# nested container of values. Differentiating a scalar loss with respect to
# that dictionary returns a dictionary of gradients with the same structure.
# The frequency and phase remain separate, fixed inputs.


# %%
def parameter_loss(params, t, target, frequency_hz, phase):
    return amplitude_damping_loss(
        params["amplitude"], params["gamma"], t, target, frequency_hz, phase
    )


initial_params = {
    "amplitude": jnp.asarray(initial_amplitude),
    "gamma": jnp.asarray(initial_gamma),
}
loss_value, grads = jax.value_and_grad(parameter_loss)(
    initial_params, t, y_target, known_frequency_hz, known_phase
)
print("Parameters:", initial_params)
print("Gradients:", grads)

# %% [markdown]
# ## 4. Let Optax handle the updates
#
# Optax supplies optimisation algorithms. Here we use Adam, which adapts
# updates using running gradient statistics. Its state is passed into each
# step and returned for the next one. `apply_updates` adds the updates to the
# corresponding parameters; Optax already includes the descent direction.
#
# Start from the same guesses as the manual loop. Rerunning this cell resets
# both the parameters and optimiser state.

# %%
optimizer = optax.adam(learning_rate=0.03)


@jax.jit
def train_step(params, opt_state, t, target, frequency_hz, phase):
    loss_value, grads = jax.value_and_grad(parameter_loss)(
        params, t, target, frequency_hz, phase
    )
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params,
        value=loss_value,
        grad=grads,
        value_fn=parameter_loss,
        t=t,
        target=target,
        frequency_hz=frequency_hz,
        phase=phase,
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


params = dict(initial_params)
opt_state = optimizer.init(params)
n_adam_updates = 600
adam_history = []

for step in range(n_adam_updates):
    params, opt_state, loss_value = train_step(
        params, opt_state, t, y_target, known_frequency_hz, known_phase
    )
    adam_history.append(float(loss_value))

print("Fitted parameters:", {name: float(value) for name, value in params.items()})
print(
    "Final loss:",
    float(parameter_loss(params, t, y_target, known_frequency_hz, known_phase)),
)
y_fitted = damped_sinusoid(
    t, params["amplitude"], params["gamma"], known_frequency_hz, known_phase
)
show_audio(y_fitted, sample_rate)

fig, ax = plt.subplots(figsize=(7, 3))
ax.semilogy(manual_history, label="manual gradient descent")
ax.semilogy(adam_history, label="Adam")
ax.set_xlabel("Update")
ax.set_ylabel("Mean squared error")
ax.set_title("Two update rules from the same starting point")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Try it:** replace `optax.adam(learning_rate=0.03)` with
# `optax.sgd(learning_rate=1.0)` and rerun this cell. Compare the first 600
# updates with the manual loop. These particular learning rates suit this
# example; neither algorithm guarantees convergence for arbitrary settings.
#
# We used direct physical parameters to keep the first fit simple. These
# updates do not enforce positivity. For a constrained model, we could learn
# an unconstrained value and map it to positive damping with `jax.nn.softplus`.
#
# ## 5. What changes when we learn frequency?
#
# Now hold amplitude, damping and phase at their true values and learn only
# frequency. This isolates the effect of initialisation from errors in the
# other parameters. We will compare one starting point close to 440 Hz with
# another farther away, using the same waveform MSE and optimiser.


# %%
def frequency_loss(frequency_hz, t, target, fixed_params):
    prediction = damped_sinusoid(
        t,
        fixed_params["amplitude"],
        fixed_params["gamma"],
        frequency_hz,
        fixed_params["phase"],
    )
    return jnp.mean((prediction - target) ** 2)


fixed_params = {
    "amplitude": jnp.asarray(target_amplitude),
    "gamma": jnp.asarray(target_gamma),
    "phase": jnp.asarray(known_phase),
}
frequency_optimizer = optax.adam(learning_rate=0.1)


@jax.jit
def frequency_step(frequency_hz, opt_state, t, target, fixed_params):
    loss_value, gradient = jax.value_and_grad(frequency_loss)(
        frequency_hz, t, target, fixed_params
    )
    updates, opt_state = frequency_optimizer.update(gradient, opt_state, frequency_hz)
    frequency_hz = optax.apply_updates(frequency_hz, updates)
    return frequency_hz, opt_state, loss_value


def fit_single_frequency(
    initial_frequency_hz, t, target, fixed_params, n_updates=500, log_every=25
):
    frequency_hz = jnp.asarray(float(initial_frequency_hz))
    opt_state = frequency_optimizer.init(frequency_hz)
    history = []
    for update in range(n_updates + 1):
        if update % log_every == 0 or update == n_updates:
            loss_value = frequency_loss(frequency_hz, t, target, fixed_params)
            history.append((update, float(frequency_hz), float(loss_value)))
        if update < n_updates:
            frequency_hz, opt_state, _ = frequency_step(
                frequency_hz, opt_state, t, target, fixed_params
            )
    return np.asarray(history)


near_start_hz = known_frequency_hz - 0.5
far_start_hz = known_frequency_hz - 10.0
near_history = fit_single_frequency(near_start_hz, t, y_target, fixed_params)
far_history = fit_single_frequency(far_start_hz, t, y_target, fixed_params)

for label, history in (("near", near_history), ("far", far_history)):
    _, final_frequency_hz, final_loss = history[-1]
    print(f"{label}: frequency={final_frequency_hz:.4f} Hz, loss={final_loss:.3e}")

# %% [markdown]
# Look at the loss as a function of frequency. `vmap` evaluates the same loss
# at each candidate frequency; `None` keeps the other inputs shared. You can
# run this cell even if you skipped the optional batching example in JAX basics.

# %%
frequency_grid = jnp.linspace(known_frequency_hz - 15.0, known_frequency_hz + 5.0, 401)
frequency_losses = jax.vmap(frequency_loss, in_axes=(0, None, None, None))(
    frequency_grid, t, y_target, fixed_params
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(np.asarray(frequency_grid), np.asarray(frequency_losses))
for label, history in (("near", near_history), ("far", far_history)):
    axes[0].scatter(history[0, 1], history[0, 2], label=f"{label} start")
    axes[1].plot(history[:, 0], history[:, 1], label=f"{label} start")
axes[0].axvline(known_frequency_hz, color="k", linestyle=":", label="target")
axes[0].set_xlabel("Candidate frequency [Hz]")
axes[0].set_ylabel("Mean squared error")
axes[0].set_title("The frequency loss landscape")
axes[1].axhline(known_frequency_hz, color="k", linestyle=":", label="target")
axes[1].set_xlabel("Update")
axes[1].set_ylabel("Frequency [Hz]")
axes[1].set_title("Where the optimiser ends up")
for ax in axes:
    ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Discuss:** does the farther start reach the target? A gradient describes
# the local slope. For a finite sound, waveform MSE can have several minima
# as frequency changes, so following that slope may lead to the wrong pitch.
#
# **Try it:** change `near_start_hz` and `far_start_hz` and rerun from their cell.
# Explore the convergence range with the working scan below. The tolerance
# defines success as ending within 0.5 Hz of the known target.

# %%
initial_frequencies = np.arange(known_frequency_hz - 6.0, known_frequency_hz + 7.0)
tolerance_hz = 0.5
frequency_scan = []

for initial_frequency_hz in initial_frequencies:
    history = fit_single_frequency(initial_frequency_hz, t, y_target, fixed_params)
    _, final_frequency_hz, final_loss = history[-1]
    converged = abs(final_frequency_hz - known_frequency_hz) < tolerance_hz
    frequency_scan.append(
        (initial_frequency_hz, final_frequency_hz, final_loss, converged)
    )
    print(
        f"{initial_frequency_hz:.1f} Hz -> {final_frequency_hz:.3f} Hz; "
        f"loss={final_loss:.2e}; within tolerance={converged}"
    )

# %% [markdown]
# **Explore further:** double `duration` and rerun from Section 1. How does
# the frequency loss landscape change? Try a different damping rate too.
#
# The target values are available here so we can assess the fit. With recorded
# audio, we would also need suitable model assumptions and validation data.
#
# ## 6. Optional: express the oscillator as a complex recursion
#
# A damped sinusoid also follows a one-step update:
#
# $$z_{n+1} = a z_n,\qquad a = e^{(-\gamma+i2\pi f)\Delta t},\qquad
# z_0 = A e^{i\varphi}.$$
#
# Its **imaginary part** matches the sine convention used above:
# $y_n = \operatorname{Im}(z_n)$. The magnitude of $a$ controls decay, and its
# angle controls rotation per sample. No previous complex-recursion exercise
# is needed for the main parts of this tutorial.
#
# `lax.scan` repeatedly calls a step that returns `(next_state, output)`.
# We output the current state before advancing it, so sample zero matches $z_0$.


# %%
def complex_modal_synth(t, amplitude, gamma, frequency_hz, phase=0.0):
    dt = t[1] - t[0]
    a = jnp.exp((-gamma + 1j * 2.0 * jnp.pi * frequency_hz) * dt)
    z0 = amplitude * jnp.exp(1j * phase)

    def step(z, _):
        return a * z, z

    _, trajectory = jax.lax.scan(step, z0, xs=None, length=t.shape[0])
    return trajectory.imag


y_recursive = complex_modal_synth(
    t, target_amplitude, target_gamma, known_frequency_hz, known_phase
)
print("Maximum waveform difference:", float(jnp.max(jnp.abs(y_recursive - y_target))))
show_audio(y_recursive, sample_rate)

# %% [markdown]
# **Try it:** set a non-zero `known_phase` in Section 1 and rerun the file.
# Do the direct and recursive synthesisers still agree? What happens if you
# take `trajectory.real` instead of `trajectory.imag`?
#
# You have extended a one-parameter fit to two physical parameters, used
# dictionaries and Optax, and tested how frequency initialisation affects
# optimisation. Next, try learning phase, or replace waveform MSE with a
# spectral loss and compare its frequency landscape.
#
# Further reading:
# - [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
# - [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html)
