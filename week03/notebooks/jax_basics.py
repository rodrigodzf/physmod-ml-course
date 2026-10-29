# %% [markdown]
# # JAX basics
#
# Learn the basics of JAX by writing a small synthesiser and recovering the
# amplitude of a target sound. You only need basic Python and array operations.
# The main sections take about 25 minutes; the final batching exercise is optional.
#
# Before starting, select the course Python environment, which provides JAX,
# NumPy, Matplotlib and IPython.
#
# Open this file in an editor that supports `# %%` cells. Run the cells from
# top to bottom using **Run Cell** or your editor's equivalent. The
# `# %% [markdown]` cells contain explanations; the other cells contain code.
# At each **Try it** or **Predict** prompt, make a prediction, change a value,
# and rerun the relevant cells. Each example works before you make any changes.
#
# By the end, you will be able to create JAX arrays, calculate derivatives with
# `grad`, fit a parameter using gradient descent, and compile an update with
# `jit`. The optional section introduces batching with `vmap`.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())

# %% [markdown]
# ## 1. Start with familiar arrays
#
# `jax.numpy`, usually imported as `jnp`, provides familiar array operations.
# Use it for the numerical calculations that JAX will differentiate or compile.
# We use ordinary NumPy only when preparing results for plotting or playback.

# %%
x = jnp.array([0.0, 1.0, 2.0])
print("x:", x)
print("x squared:", x**2)
print("Mean:", jnp.mean(x))
print("Shape:", x.shape, "dtype:", x.dtype)

# %% [markdown]
# JAX arrays are immutable. An indexed update returns a new array:

# %%
x_updated = x.at[0].set(10.0)
print("Original:", x)
print("Updated:", x_updated)

# %% [markdown]
# **Try it:** replace the second element instead. Which index do you need?
#
# ## 2. Turn a function into its derivative
#
# For $f(x) = x^2$, we know that $f'(x) = 2x$. Start with a floating-point
# scalar, then ask JAX to build the derivative function.


# %%
def square(x):
    return x**2


square_gradient = jax.grad(square)
print("Function at 3:", square(3.0))
print("Derivative at 3:", square_gradient(3.0))
print("Derivative operations (JAXPR):")
print(jax.make_jaxpr(square_gradient)(3.0))

# %% [markdown]
# `jax.grad(square)` returns a **function**. Calling `square_gradient(3.0)`
# evaluates that derivative at 3.
# `grad` expects a scalar output, so later we will reduce all sample errors
# to one loss value. This is automatic differentiation through the operations
# in our function, rather than a finite-difference approximation.
#
# `make_jaxpr` displays the recorded operations. For this derivative, look for
# multiplication by 2. It traces the function without compiling an executable.
#
# **Try it:** change `square` to return `x**3`, rerun the cell, and predict the
# derivative at 3 before looking at the result.
#
# ## 3. Make a target sound
#
# Our synthesiser has one learnable parameter, amplitude $A$:
#
# $$\hat y_n(A) = A\sin(2\pi f t_n).$$
#
# Keep the frequency fixed for now. Pass the inputs explicitly and return the
# result. This is a **pure function**: the same inputs give the same output,
# without changing external state. Keep plotting and playback outside it.


# %%
def synthesise(amplitude, time, frequency_hz):
    return amplitude * jnp.sin(2.0 * jnp.pi * frequency_hz * time)


sample_rate = 8000
duration = 0.5
time = jnp.arange(int(sample_rate * duration)) / sample_rate
frequency_hz = 220.0
target_amplitude = 0.8
initial_amplitude = 0.2

target = synthesise(target_amplitude, time, frequency_hz)
initial = synthesise(initial_amplitude, time, frequency_hz)

print("Target")
display(Audio(np.asarray(target), rate=sample_rate, normalize=False))
print("Initial guess")
display(Audio(np.asarray(initial), rate=sample_rate, normalize=False))

# %% [markdown]
# Playback normalisation is disabled so you can hear the amplitude difference.
#
# ## 4. Recover the amplitude
#
# Compare the predicted samples with the target using mean squared error:
#
# $$\mathcal L(A) = \frac{1}{N}\sum_n(\hat y_n(A)-y_n)^2.$$
#
# With several arguments, `grad` differentiates the first one by default.
# `value_and_grad` gives us both the loss and its derivative in one call.


# %%
def loss(amplitude, time, target, frequency_hz):
    prediction = synthesise(amplitude, time, frequency_hz)
    return jnp.mean((prediction - target) ** 2)


loss_and_gradient = jax.value_and_grad(loss)
loss_value, gradient = loss_and_gradient(initial_amplitude, time, target, frequency_hz)
print("Initial loss:", float(loss_value))
print("Amplitude gradient:", float(gradient))

# %% [markdown]
# With an initial amplitude of 0.2 and a target of 0.8, the gradient is negative.
# Subtracting it increases our amplitude:
#
# $$A_{k+1} = A_k - \eta\,\frac{\mathrm d\mathcal L}{\mathrm d A}(A_k).$$
#
# **Predict:** starting above the target amplitude, what sign should the
# gradient have? Change `initial_amplitude` in Section 3 and rerun from there.

# %%
learning_rate = 0.2
n_updates = 60
amplitude = jnp.asarray(initial_amplitude)
loss_history = []

for step in range(n_updates):
    loss_value, gradient = loss_and_gradient(amplitude, time, target, frequency_hz)
    loss_history.append(float(loss_value))
    amplitude = amplitude - learning_rate * gradient

fitted = synthesise(amplitude, time, frequency_hz)
print("Target amplitude:", target_amplitude)
print("Fitted amplitude:", float(amplitude))
print("Final loss:", float(loss(amplitude, time, target, frequency_hz)))

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].semilogy(loss_history)
axes[0].set_xlabel("Update")
axes[0].set_ylabel("Mean squared error")
axes[0].set_title("Learning the amplitude")

n_plot = 80
axes[1].plot(np.asarray(time[:n_plot]), np.asarray(target[:n_plot]), label="target")
axes[1].plot(
    np.asarray(time[:n_plot]), np.asarray(initial[:n_plot]), "--", label="initial"
)
axes[1].plot(
    np.asarray(time[:n_plot]), np.asarray(fitted[:n_plot]), ":", label="fitted"
)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Try it:** use a learning rate of `0.02`, then `2.2`. Rerun the previous
# cell each time. Do the updates converge, slow down, or grow? Restore `0.2`
# before continuing.
#
# ### Visualise the loss landscape
#
# Evaluate the loss at several candidate amplitudes. Its minimum should occur
# at the target amplitude. `vmap` applies the loss across the amplitude grid,
# keeping time, target and frequency shared. Section 6 explains batching further.

# %%
amplitude_grid = jnp.linspace(0.0, 1.5, 100)
batched_loss = jax.vmap(loss, in_axes=(0, None, None, None))
loss_grid = batched_loss(amplitude_grid, time, target, frequency_hz)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(np.asarray(amplitude_grid), np.asarray(loss_grid))
ax.axvline(target_amplitude, color="black", linestyle=":", label="target amplitude")
ax.set_xlabel("Amplitude")
ax.set_ylabel("Mean squared error")
ax.set_title("Amplitude loss landscape")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Compile a working step with `jit`
#
# Put one update into a function, then compile it. The first call normally
# traces and compiles; later calls with compatible shapes and dtypes reuse
# that compilation. `jit` is a performance tool, and tiny examples may not
# benefit. New shapes or dtypes can trigger compilation again.
#
# Inside this function, keep calculations in JAX and return the new values.
# Ordinary Python printing, plotting, and conversion to `float` stay outside.


# %%
def update_amplitude(amplitude, time, target, frequency_hz, learning_rate):
    loss_value, gradient = loss_and_gradient(amplitude, time, target, frequency_hz)
    new_amplitude = amplitude - learning_rate * gradient
    return new_amplitude, loss_value


compiled_update = jax.jit(update_amplitude)
compiled_amplitude = jnp.asarray(initial_amplitude)

for step in range(n_updates):
    compiled_amplitude, loss_before_update = compiled_update(
        compiled_amplitude, time, target, frequency_hz, learning_rate
    )

print("Without jit:", float(amplitude))
print("With jit:", float(compiled_amplitude))

# %% [markdown]
# **Check:** both loops should recover the same amplitude to numerical
# precision. Compilation changes how the computation runs.
#
# ## 6. Optional: several sounds with `vmap`
#
# Our function generates one sound. `vmap` adds a batch axis without us
# rewriting it. `in_axes=(None, None, 0)` keeps amplitude and time shared,
# and takes one frequency at a time from axis 0 of `frequencies_hz`.

# %%
frequencies_hz = jnp.array([220.0, 330.0, 440.0])
batched_synthesise = jax.vmap(synthesise, in_axes=(None, None, 0))
sounds = batched_synthesise(0.5, time, frequencies_hz)
print("Batch shape (sounds, samples):", sounds.shape)

fig, ax = plt.subplots(figsize=(7, 3))
for frequency, sound in zip(frequencies_hz, sounds):
    ax.plot(
        np.asarray(time[:n_plot]),
        np.asarray(sound[:n_plot]),
        label=f"{float(frequency):.0f} Hz",
    )
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Try it:** add a fourth frequency and rerun. Predict the new output shape.
#
# You have built a complete parameter-fitting loop: synthesise a sound,
# measure its error, calculate the gradient, and update the amplitude.
# To explore further, add damping to the synthesiser or investigate what
# changes when you also try to learn frequency.
#
# Further reading:
# - [How to think in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)
# - [Pure functions and common pitfalls](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
