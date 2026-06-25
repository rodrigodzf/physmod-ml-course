# %% [markdown]
# # Tiny FNO for a nonlinear string solution operator
#
# This notebook-script is the Week 7 companion example. It follows the same
# operator-learning shape as the ISMIR 2025 tutorial FNO example, but generates a
# small dataset locally so the notebook is self-contained.
#
# The model learns a one-shot solution operator: a map from the initial string
# state to a short future trajectory. It is not autoregressive; it predicts the
# whole training window in one forward pass.
#
# $$
# \mathcal G_\theta:
# \bigl(w(x,0), \partial_t w(x,0)\bigr)
# \longmapsto
# \bigl(w(x,t_1), \partial_t w(x,t_1), \ldots, w(x,t_T), \partial_t w(x,t_T)\bigr).
# $$
#
# Targets are generated with a modal Kirchhoff-Carrier style nonlinear string:
#
# $$
# r_\mu(\mathbf q)
# =
# \beta\lambda_\mu q_\mu
# \sum_\nu \lambda_\nu q_\nu^2.
# $$

# %%
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from jaxdiffmodal.ftm import (
    StringParameters,
    damping_term,
    evaluate_string_eigenfunctions,
    stiffness_term,
    string_eigenvalues,
)
from jaxdiffmodal.time_integrators import solve_sv_one_step

jax.config.update("jax_enable_x64", True)

print("jax", jax.__version__, "devices:", jax.devices())


def show_if_interactive():
    if "ipykernel" in sys.modules and plt.get_backend().lower() != "agg":
        plt.show()


# %% [markdown]
# ## Generate nonlinear string trajectories
#
# We sample random band-limited modal initial displacements, set the initial
# velocity to zero, and roll out a short nonlinear modal string. The spatial
# field is reconstructed on a grid from the modal basis.

# %%
n_train = 128
n_test = 32
n_modal_modes = 12
n_grid = 64
n_time = 64
sample_rate = 44_100
dt = 1.0 / sample_rate

params = dataclasses.replace(
    StringParameters(),
    d1=8e-4,
    d3=1.4e-5,
)

lambda_mu = string_eigenvalues(n_modal_modes, params.length)
omega_mu_squared = stiffness_term(params, lambda_mu)
gamma2_mu = damping_term(params, lambda_mu)
mode_numbers = jnp.arange(1, n_modal_modes + 1)

nonlinearity_strength = 3e9


def nonlinear_force(q):
    stretch = jnp.sum(lambda_mu * q**2)
    return nonlinearity_strength * lambda_mu * q * stretch


def modal_basis(n_points):
    x = jnp.linspace(0.0, params.length, n_points)
    basis = jax.vmap(
        lambda position: evaluate_string_eigenfunctions(
            mode_numbers,
            position,
            params,
        )
    )(x)
    return x, basis


def sample_initial_modal_displacements(key, n_examples):
    key_amp, key_shape = jax.random.split(key)
    amplitude = jax.random.uniform(
        key_amp,
        (n_examples, 1),
        minval=8e-5,
        maxval=3e-4,
    )
    modal_envelope = jnp.exp(-0.35 * jnp.arange(n_modal_modes))
    modal_noise = jax.random.normal(key_shape, (n_examples, n_modal_modes))
    return amplitude * modal_envelope[None, :] * modal_noise


def simulate_modal(q0):
    v0 = jnp.zeros_like(q0)
    _, q, v = solve_sv_one_step(
        gamma2_mu=gamma2_mu,
        omega_mu_squared=omega_mu_squared,
        dt=dt,
        n_steps=n_time,
        nl_fn=nonlinear_force,
        u0=q0,
        v0=v0,
    )
    return q, v


def reconstruct_spatial_state(q, v, basis):
    displacement = q @ basis.T
    velocity = v @ basis.T
    return jnp.stack([displacement, velocity], axis=-1)


def make_dataset(key, n_examples, n_points):
    _, basis = modal_basis(n_points)
    q0 = sample_initial_modal_displacements(key, n_examples)
    q, v = jax.vmap(simulate_modal)(q0)
    return reconstruct_spatial_state(q, v, basis), q0


key = jax.random.PRNGKey(1707)
key_train, key_test, key_model = jax.random.split(key, 3)

train_state_raw, train_q0 = make_dataset(key_train, n_train, n_grid)
test_state_raw, test_q0 = make_dataset(key_test, n_test, n_grid)

channel_scale = jnp.max(jnp.abs(train_state_raw), axis=(0, 1, 2))
train_state = train_state_raw / channel_scale
test_state = test_state_raw / channel_scale

print("train state", train_state.shape)
print("channel scale", channel_scale)


def add_coordinate_channel(state_sequence):
    n_examples, n_steps, n_points, _ = state_sequence.shape
    coordinate = jnp.linspace(-1.0, 1.0, n_points)
    coordinate = jnp.broadcast_to(
        coordinate[None, None, :, None],
        (n_examples, n_steps, n_points, 1),
    )
    return jnp.concatenate([state_sequence, coordinate], axis=-1)


train_input = add_coordinate_channel(train_state[:, 0:1])
train_target = train_state[:, 1:]
test_input = add_coordinate_channel(test_state[:, 0:1])
test_target = test_state[:, 1:]

print("train input", train_input.shape, "train target", train_target.shape)


# %% [markdown]
# ## An ISMIR-style 1D FNO
#
# The network receives one state frame of shape `(1, width, channels)` and
# predicts the remaining trajectory. The spectral layer uses FFT padding so the
# Fourier path behaves like a linear convolution rather than a circular one.

# %%
class SpectralConv1d(eqx.Module):
    weight_real: jax.Array
    weight_imag: jax.Array
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)

    def __init__(self, in_channels, out_channels, n_modes, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes

        key_real, key_imag = jax.random.split(key)
        weight_shape = (in_channels, out_channels, n_modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = scale * jax.random.normal(key_real, weight_shape)
        self.weight_imag = scale * jax.random.normal(key_imag, weight_shape)

    @property
    def weight(self):
        return self.weight_real + 1j * self.weight_imag

    def __call__(self, x):
        # x shape: (width, in_channels)
        width = x.shape[0]
        padded_width = 2 * width - 1
        x_ft = jnp.fft.rfft(x, n=padded_width, axis=0, norm="ortho")

        n_freq = x_ft.shape[0]
        n_modes = min(self.n_modes, n_freq)
        out_ft = jnp.zeros((n_freq, self.out_channels), dtype=x_ft.dtype)
        # m: retained Fourier mode,
        # i: input hidden channel,
        # o: output hidden channel.
        out_modes = jnp.einsum(
            "mi,iom->mo",
            x_ft[:n_modes],
            self.weight[:, :, :n_modes],
        )
        out_ft = out_ft.at[:n_modes].set(out_modes)
        return jnp.fft.irfft(out_ft, n=padded_width, axis=0, norm="ortho")[:width]


class FNO1D(eqx.Module):
    spectral_convs: tuple[SpectralConv1d, ...]
    pointwise_layers: tuple[eqx.nn.Linear, ...]
    lifting: eqx.nn.Linear
    projection_hidden: eqx.nn.Linear
    projection_out: eqx.nn.Linear
    n_steps: int = eqx.field(static=True)
    output_channels: int = eqx.field(static=True)

    def __init__(
        self,
        input_channels,
        hidden_channels,
        n_modes,
        output_channels,
        n_layers,
        n_steps,
        key,
    ):
        keys = jax.random.split(key, 2 * n_layers + 3)
        self.spectral_convs = tuple(
            SpectralConv1d(
                hidden_channels,
                hidden_channels,
                n_modes,
                keys[i],
            )
            for i in range(n_layers)
        )
        self.pointwise_layers = tuple(
            eqx.nn.Linear(hidden_channels, hidden_channels, key=keys[n_layers + i])
            for i in range(n_layers)
        )
        self.lifting = eqx.nn.Linear(input_channels, hidden_channels, key=keys[-3])
        self.projection_hidden = eqx.nn.Linear(hidden_channels, 64, key=keys[-2])
        self.projection_out = eqx.nn.Linear(
            64,
            output_channels * n_steps,
            key=keys[-1],
        )
        self.n_steps = n_steps
        self.output_channels = output_channels

    def __call__(self, x):
        # x shape: (time, width, channels). Treat time as part of the channel
        # representation, matching the ISMIR tutorial model interface.
        _, width, input_channels = x.shape
        x = jnp.transpose(x, (1, 0, 2)).reshape(width, -1)

        h = jax.vmap(self.lifting)(x)
        for spectral_conv, pointwise in zip(self.spectral_convs, self.pointwise_layers):
            h_spectral = spectral_conv(h)
            h_pointwise = jax.vmap(pointwise)(h)
            h = jax.nn.gelu(h_spectral + h_pointwise)

        y = jax.vmap(self.projection_hidden)(h)
        y = jax.nn.gelu(y)
        y = jax.vmap(self.projection_out)(y)
        y = y.reshape(width, self.n_steps, self.output_channels)
        return jnp.transpose(y, (1, 0, 2))


model = FNO1D(
    input_channels=train_input.shape[-1],
    hidden_channels=32,
    n_modes=32,
    output_channels=train_target.shape[-1],
    n_layers=4,
    n_steps=train_target.shape[1],
    key=key_model,
)

initial_prediction = jax.vmap(model)(test_input[:2])
print("initial prediction", initial_prediction.shape)


# %% [markdown]
# ## Train the solution operator
#
# This is a one-shot operator: the model predicts the full short trajectory from
# the initial state. An autoregressive FNO would instead learn one time-step
# operator and roll it forward.

# %%
def loss_fn(model, inputs, targets):
    predictions = jax.vmap(model)(inputs)
    return jnp.mean((predictions - targets) ** 2)


optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(2e-3),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, inputs, targets):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


n_training_steps = 250
loss_history = []

for step in range(1, n_training_steps + 1):
    model, opt_state, loss_value = train_step(
        model,
        opt_state,
        train_input,
        train_target,
    )
    loss_history.append(float(loss_value))
    if step == 1 or step % 50 == 0:
        test_loss = loss_fn(model, test_input, test_target)
        print(
            f"step {step:4d}  train {float(loss_value):.4e}  "
            f"test {float(test_loss):.4e}"
        )


# %% [markdown]
# ## Inspect one held-out trajectory

# %%
test_index = 0
prediction = model(test_input[test_index])
target = test_target[test_index]

prediction_raw = prediction * channel_scale
target_raw = target * channel_scale
error_raw = prediction_raw - target_raw

time_ms = 1000.0 * dt * jnp.arange(1, n_time)
x_grid = jnp.linspace(0.0, params.length, n_grid)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 3.4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
mesh_kwargs = {
    "x": np.asarray(x_grid),
    "y": np.asarray(time_ms),
    "shading": "auto",
}

vmax = float(jnp.max(jnp.abs(target_raw[..., 0])))
err_vmax = float(jnp.max(jnp.abs(error_raw[..., 0])))

im0 = axes[0].pcolormesh(
    mesh_kwargs["x"],
    mesh_kwargs["y"],
    np.asarray(target_raw[..., 0]),
    vmin=-vmax,
    vmax=vmax,
    shading="auto",
)
axes[0].set_title("Target displacement")

im1 = axes[1].pcolormesh(
    mesh_kwargs["x"],
    mesh_kwargs["y"],
    np.asarray(prediction_raw[..., 0]),
    vmin=-vmax,
    vmax=vmax,
    shading="auto",
)
axes[1].set_title("FNO prediction")

im2 = axes[2].pcolormesh(
    mesh_kwargs["x"],
    mesh_kwargs["y"],
    np.asarray(error_raw[..., 0]),
    vmin=-err_vmax,
    vmax=err_vmax,
    shading="auto",
)
axes[2].set_title("Error")

for ax in axes:
    ax.set_xlabel("position x [m]")
axes[0].set_ylabel("time [ms]")

fig.colorbar(im0, ax=axes[:2], shrink=0.82, label="displacement [m]")
fig.colorbar(im2, ax=axes[2], shrink=0.82, label="error [m]")
output_path = Path(__file__).with_name("fno_wave_operator_results.png")
fig.savefig(output_path, dpi=160)
print("saved", output_path)
show_if_interactive()


# %% [markdown]
# ## Resolution-transfer probe
#
# The model was trained at 64 spatial samples. Because the spectral layer is
# expressed through Fourier modes, we can evaluate the same weights at a
# different spatial resolution. This is not a proof of representation
# equivalence, but it is a useful discretisation-invariance probe:
#
# 1. evaluate the same underlying trajectory on the training grid,
# 2. evaluate it again on a finer grid,
# 3. restrict the fine-grid prediction back to the coarse grid,
# 4. compare both predictions on the same points.

# %%
n_grid_fine = 128
_, fine_basis = modal_basis(n_grid_fine)
fine_state_raw = reconstruct_spatial_state(
    *jax.vmap(simulate_modal)(test_q0[:1]),
    fine_basis,
)
fine_state = fine_state_raw / channel_scale
fine_input = add_coordinate_channel(fine_state[:, 0:1])
fine_target = fine_state[:, 1:]
fine_prediction = jax.vmap(model)(fine_input)
fine_loss = jnp.mean((fine_prediction - fine_target) ** 2)

same_state_raw = test_state_raw[:1]
same_state = same_state_raw / channel_scale
same_input = add_coordinate_channel(same_state[:, 0:1])
same_target = same_state[:, 1:]
same_prediction = jax.vmap(model)(same_input)

fine_prediction_on_coarse = fine_prediction[:, :, ::2, :]
fine_target_on_coarse = fine_target[:, :, ::2, :]

coarse_loss = jnp.mean((same_prediction - same_target) ** 2)
restricted_fine_loss = jnp.mean(
    (fine_prediction_on_coarse - fine_target_on_coarse) ** 2
)
prediction_discrepancy = jnp.mean(
    (same_prediction - fine_prediction_on_coarse) ** 2
)

print(f"same-example coarse-grid MSE at n={n_grid}: {float(coarse_loss):.4e}")
print(f"same-example fine-grid MSE at n={n_grid_fine}: {float(fine_loss):.4e}")
print(
    "restricted fine-grid MSE on coarse points: "
    f"{float(restricted_fine_loss):.4e}"
)
print(
    "coarse vs. restricted-fine prediction discrepancy: "
    f"{float(prediction_discrepancy):.4e}"
)
