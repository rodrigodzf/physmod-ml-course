# %% [markdown]
# # Differentiable sinusoidal synth
#
# This notebook is the smallest possible version of the Week 4 idea:
#
# $$\theta \rightarrow \hat y \rightarrow \mathcal L(\hat y, y)
# \rightarrow \nabla_\theta \mathcal L.$$
#
# We create a target damped sinusoid, write the synthesiser as a JAX function,
# and use automatic differentiation to fit amplitude, damping, frequency and
# phase.

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

try:
    from IPython import get_ipython
    from IPython.display import Audio, display
except ImportError:  # Allows the file to run as a plain Python script.
    get_ipython = None
    Audio = None
    display = None

jax.config.update("jax_enable_x64", True)

print("jax", jax.__version__, "devices:", jax.devices())


def show_audio(y, rate):
    if Audio is not None and get_ipython is not None and get_ipython() is not None:
        display(Audio(np.asarray(y), rate=rate))

# %% [markdown]
# ## Target sound
#
# The signal is a damped sinusoid:
#
# $$y_n = A e^{-\gamma t_n}\sin(2\pi f t_n + \varphi).$$

# %%
sample_rate = 8000
duration = 0.5
n_steps = int(sample_rate * duration)
t = jnp.arange(n_steps) / sample_rate


def damped_sinusoid(t, amplitude, gamma, frequency_hz, phase):
    return amplitude * jnp.exp(-gamma * t) * jnp.sin(
        2.0 * jnp.pi * frequency_hz * t + phase
    )


target_params = {
    "amplitude": jnp.array(0.8),
    "gamma": jnp.array(3.0),
    "frequency_hz": jnp.array(440.0),
    "phase": jnp.array(0.6),
}

y_target = damped_sinusoid(t, **target_params)

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(np.asarray(t[:800]), np.asarray(y_target[:800]))
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.set_title("Target damped sinusoid, first 100 ms")
ax.grid(True)
fig.tight_layout()

show_audio(y_target, sample_rate)

# %% [markdown]
# ## Learnable synthesiser
#
# We optimise unconstrained parameters, then map them to physical values. This
# keeps amplitude, damping and frequency positive while still letting Optax make
# unconstrained gradient updates.

# %%
def inv_softplus(x):
    x = jnp.asarray(x)
    return x + jnp.log(-jnp.expm1(-x))


def init_raw_params(
    amplitude=0.3,
    gamma=1.0,
    frequency_hz=430.0,
    phase=0.0,
):
    return {
        "raw_amplitude": inv_softplus(amplitude),
        "raw_gamma": inv_softplus(gamma),
        "raw_frequency_hz": inv_softplus(frequency_hz),
        "phase": jnp.asarray(phase),
    }


def physical_params(raw_params):
    return {
        "amplitude": jax.nn.softplus(raw_params["raw_amplitude"]),
        "gamma": jax.nn.softplus(raw_params["raw_gamma"]),
        "frequency_hz": jax.nn.softplus(raw_params["raw_frequency_hz"]),
        "phase": raw_params["phase"],
    }


def synth(raw_params):
    params = physical_params(raw_params)
    return damped_sinusoid(t, **params)


raw_params = init_raw_params()
y_initial = synth(raw_params)

print("Initial physical parameters:")
for name, value in physical_params(raw_params).items():
    print(f"  {name:>12s}: {float(value):8.4f}")

# %% [markdown]
# ## Loss and gradient
#
# We start with the simplest possible audio loss:
#
# $$\mathcal L(\theta)
# =
# \frac{1}{N}
# \sum_n
# \left(
# \hat y_n(\theta) - y_n
# \right)^2.$$

# %%
def loss_fn(raw_params):
    y_hat = synth(raw_params)
    return jnp.mean((y_hat - y_target) ** 2)


loss_value, grads = jax.value_and_grad(loss_fn)(raw_params)

print("Initial loss:", float(loss_value))
print("Gradient leaves:")
for name, value in grads.items():
    print(f"  {name:>16s}: {float(value): .6e}")

# %% [markdown]
# ## Optimisation loop
#
# Automatic differentiation gives the gradient. Optax turns that gradient into
# parameter updates.

# %%
learning_rate = 3e-2
n_updates = 3000
log_every = 25

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(raw_params)


@jax.jit
def train_step(raw_params, opt_state):
    loss_value, grads = jax.value_and_grad(loss_fn)(raw_params)
    updates, opt_state = optimizer.update(grads, opt_state, raw_params)
    raw_params = optax.apply_updates(raw_params, updates)
    return raw_params, opt_state, loss_value


history = []

for step in range(n_updates + 1):
    if step % log_every == 0:
        history.append((step, float(loss_fn(raw_params))))

    if step < n_updates:
        raw_params, opt_state, loss_value = train_step(raw_params, opt_state)

fitted_params = physical_params(raw_params)
y_fitted = synth(raw_params)

print("Final physical parameters:")
for name, value in fitted_params.items():
    print(f"  {name:>12s}: {float(value):8.4f}")
print("Final loss:", float(loss_fn(raw_params)))

# %% [markdown]
# ## Inspect the result

# %%
history_steps = np.asarray([item[0] for item in history])
history_loss = np.asarray([item[1] for item in history])

fig, axes = plt.subplots(2, 1, figsize=(8, 5))

axes[0].semilogy(history_steps, history_loss)
axes[0].set_xlabel("Update")
axes[0].set_ylabel("MSE loss")
axes[0].set_title("Optimisation history")
axes[0].grid(True)

n_plot = 800
axes[1].plot(np.asarray(t[:n_plot]), np.asarray(y_target[:n_plot]), label="target")
axes[1].plot(
    np.asarray(t[:n_plot]),
    np.asarray(y_initial[:n_plot]),
    "--",
    label="initial",
    alpha=0.8,
)
axes[1].plot(
    np.asarray(t[:n_plot]),
    np.asarray(y_fitted[:n_plot]),
    ":",
    label="fitted",
    linewidth=2.0,
)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("First 100 ms")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()

print("Fitted")
show_audio(y_fitted, sample_rate)

# %% [markdown]
# ## Complex recursion version
#
# The same oscillator can be written using the complex update from Week 3:
#
# $$z^{n+1} = a z^n,$$
#
# with
#
# $$a = e^{(-\gamma + i2\pi f)\Delta t}.$$
#
# The real-valued output is:
#
# $$\hat y_n = \operatorname{Re}\{z^n\}.$$

# %%
def complex_modal_synth(frequency_hz, amplitude=0.8, gamma=3.0, phase=0.6):
    dt = 1.0 / sample_rate
    pole = -gamma + 1j * 2.0 * jnp.pi * frequency_hz
    a = jnp.exp(pole * dt)
    z0 = amplitude * jnp.exp(1j * phase)
    z = z0 * a ** jnp.arange(n_steps)
    return z.real


y_complex_target = complex_modal_synth(target_params["frequency_hz"])

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(np.asarray(t[:800]), np.asarray(y_complex_target[:800]))
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.set_title("Complex recursion target, first 100 ms")
ax.grid(True)
fig.tight_layout()

# %% [markdown]
# ## Optimise one frequency
#
# Now keep amplitude, damping and phase fixed, and optimise only the frequency.
# This makes the loss a one-dimensional function:
#
# $$\mathcal L(f)
# =
# \frac{1}{N}
# \sum_n
# \left(
# \hat y_n(f) - y_n
# \right)^2.$$

# %%
def complex_frequency_loss(frequency_hz):
    y_hat = complex_modal_synth(frequency_hz)
    return jnp.mean((y_hat - y_complex_target) ** 2)


def fit_single_frequency(
    initial_frequency_hz,
    learning_rate=0.1,
    n_updates=700,
    log_every=25,
):
    frequency_hz = jnp.asarray(float(initial_frequency_hz))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(frequency_hz)

    @jax.jit
    def step(frequency_hz, opt_state):
        loss_value, grad = jax.value_and_grad(complex_frequency_loss)(frequency_hz)
        updates, opt_state = optimizer.update(grad, opt_state, frequency_hz)
        frequency_hz = optax.apply_updates(frequency_hz, updates)
        return frequency_hz, opt_state, loss_value, grad

    history = []
    for update in range(n_updates + 1):
        if update % log_every == 0 or update == n_updates:
            loss_value, grad = jax.value_and_grad(complex_frequency_loss)(frequency_hz)
            history.append(
                (update, float(frequency_hz), float(loss_value), float(grad))
            )

        if update < n_updates:
            frequency_hz, opt_state, _, _ = step(frequency_hz, opt_state)

    return history


near_history = fit_single_frequency(438.0)
far_history = fit_single_frequency(350.0)

print("Single-frequency optimisation:")
for label, history in (("near", near_history), ("far", far_history)):
    _, frequency_hz, loss_value, grad = history[-1]
    print(
        f"  {label:>4s}: f={frequency_hz:8.3f} Hz, "
        f"loss={loss_value:.3e}, grad={grad:.3e}"
    )

fig, ax = plt.subplots(figsize=(8, 3))
for label, history in (("near start", near_history), ("far start", far_history)):
    steps = [item[0] for item in history]
    frequencies = [item[1] for item in history]
    ax.plot(steps, frequencies, label=label)

ax.axhline(float(target_params["frequency_hz"]), color="k", linestyle=":", label="target")
ax.set_xlabel("Update")
ax.set_ylabel("Frequency [Hz]")
ax.set_title("Optimising one frequency through the complex recursion")
ax.legend()
ax.grid(True)
fig.tight_layout()

# %% [markdown]
# ## Exercise: find the convergence range
#
# Try one or two initial frequencies by hand in the cell above. Then fill in the
# TODOs below to automate the experiment: run the same optimiser from several
# starting frequencies and check which ones end near the target.

# %%
initial_frequencies = np.arange(434.0, 447.0, 1.0)
target_frequency_hz = float(target_params["frequency_hz"])
tolerance_hz = 0.5

frequency_scan = []

for initial_frequency_hz in initial_frequencies:
    # TODO: run `fit_single_frequency` from this starting frequency.
    history = ...

    # TODO: read the final frequency and loss from the last history entry.
    _, final_frequency_hz, loss_value, _ = ...

    # TODO: mark whether the final frequency is within `tolerance_hz`.
    converged = ...

    frequency_scan.append(
        (initial_frequency_hz, final_frequency_hz, loss_value, converged)
    )

print(f"Target frequency: {target_frequency_hz:.1f} Hz")
for initial_frequency_hz, final_frequency_hz, loss_value, converged in frequency_scan:
    status = "works" if converged else "gets stuck"
    print(
        f"{initial_frequency_hz:6.1f} Hz -> {final_frequency_hz:8.3f} Hz  "
        f"loss={loss_value:.2e}  {status}"
    )

fig, ax = plt.subplots(figsize=(8, 3))
initials = np.asarray([item[0] for item in frequency_scan])
finals = np.asarray([item[1] for item in frequency_scan])
converged = np.asarray([item[3] for item in frequency_scan])

ax.scatter(initials[converged], finals[converged], label="works")
ax.scatter(initials[~converged], finals[~converged], label="gets stuck")
ax.axhline(target_frequency_hz, color="k", linestyle=":", label="target")
ax.set_xlabel("Initial frequency [Hz]")
ax.set_ylabel("Final frequency [Hz]")
ax.set_title("Which initial frequencies converge?")
ax.legend()
ax.grid(True)
fig.tight_layout()

# %% [markdown]
# ## Things to try
#
# - Change the frequencies in `initial_frequencies` and find the range that
#   converges to the 440 Hz target.
# - Increase the duration from `0.5` s to `1.0` s and run the scan again.
# - Remove the exponential damping term.
# - Replace waveform MSE with a spectral loss.
# - In the complex-recursion example, compare initial frequencies near and far
#   from the target.
#
# The point is not only that gradients exist. The point is that audio losses can
# become difficult even for a one-oscillator synthesiser.
