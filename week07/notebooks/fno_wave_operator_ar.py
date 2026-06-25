# %% [markdown]
# # Tiny autoregressive FNO for a nonlinear string
#
# This notebook-script is the autoregressive companion to
# `fno_wave_operator.py`. It uses the same self-contained nonlinear string data,
# but trains the FNO as a **time-step operator** rather than a one-shot solution
# operator.
#
# The learned map advances one state frame:
#
# $$
# \mathcal T_\theta:
# \bigl(w(x,t_n), \partial_t w(x,t_n)\bigr)
# \longmapsto
# \bigl(w(x,t_{n+1}), \partial_t w(x,t_{n+1})\bigr).
# $$
#
# A full trajectory is produced by applying the same operator repeatedly:
#
# $$
# \mathbf s^{n+1} = \mathcal T_\theta(\mathbf s^n),
# \qquad
# \mathbf s^n = \bigl(w(\cdot,t_n), \partial_t w(\cdot,t_n)\bigr).
# $$
#
# Training uses short random windows from many simulated string trajectories.
# This makes the notebook close in spirit to the ISMIR 2025 tutorial AR example,
# but keeps the course example local and lightweight.

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
# We use the same data generator as the one-shot FNO notebook: random
# band-limited modal initial displacements, zero initial velocity, and a short
# nonlinear modal string rollout. The spatial field is reconstructed on a grid.

# %%
n_train = 96
n_test = 24
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


key = jax.random.PRNGKey(1708)
key_train, key_test, key_model, key_windows = jax.random.split(key, 4)

train_state_raw, train_q0 = make_dataset(key_train, n_train, n_grid)
test_state_raw, test_q0 = make_dataset(key_test, n_test, n_grid)

channel_scale = jnp.max(jnp.abs(train_state_raw), axis=(0, 1, 2))
train_state = train_state_raw / channel_scale
test_state = test_state_raw / channel_scale

print("train state", train_state.shape)
print("channel scale", channel_scale)


# %% [markdown]
# ## A one-step FNO wrapped in an autoregressive rollout
#
# The single-step FNO receives one state frame plus a coordinate channel. It
# predicts a residual update, which is added to the current state to obtain the
# next displacement/velocity frame. This is easier than predicting the whole
# next state from scratch because adjacent audio-rate frames are close.

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


class FNO1DStep(eqx.Module):
    spectral_convs: tuple[SpectralConv1d, ...]
    pointwise_layers: tuple[eqx.nn.Linear, ...]
    lifting: eqx.nn.Linear
    projection_hidden: eqx.nn.Linear
    projection_out: eqx.nn.Linear
    output_channels: int = eqx.field(static=True)

    def __init__(
        self,
        input_channels,
        hidden_channels,
        n_modes,
        output_channels,
        n_layers,
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
        self.projection_out = eqx.nn.Linear(64, output_channels, key=keys[-1])
        self.output_channels = output_channels

    def __call__(self, x):
        # x shape: (time, width, channels). Here time is normally 1.
        _, width, _ = x.shape
        x = jnp.transpose(x, (1, 0, 2)).reshape(width, -1)

        h = jax.vmap(self.lifting)(x)
        for spectral_conv, pointwise in zip(self.spectral_convs, self.pointwise_layers):
            h_spectral = spectral_conv(h)
            h_pointwise = jax.vmap(pointwise)(h)
            h = jax.nn.gelu(h_spectral + h_pointwise)

        y = jax.vmap(self.projection_hidden)(h)
        y = jax.nn.gelu(y)
        return jax.vmap(self.projection_out)(y)


class FNO1DAutoregressive(eqx.Module):
    step_model: FNO1DStep
    n_steps: int

    def __call__(self, initial_state):
        # initial_state shape: (width, state_channels), without coordinate.
        width = initial_state.shape[0]
        coordinate = jnp.linspace(-1.0, 1.0, width)[:, None]

        def scan_fn(current_state, _):
            step_input = jnp.concatenate([current_state, coordinate], axis=-1)
            state_delta = self.step_model(step_input[None, ...])
            next_state = current_state + state_delta
            return next_state, next_state

        _, predictions = jax.lax.scan(
            scan_fn,
            initial_state,
            xs=None,
            length=self.n_steps,
        )
        return predictions


model = FNO1DAutoregressive(
    step_model=FNO1DStep(
        input_channels=train_state.shape[-1] + 1,
        hidden_channels=32,
        n_modes=24,
        output_channels=train_state.shape[-1],
        n_layers=3,
        key=key_model,
    ),
    n_steps=16,
)

initial_prediction = jax.vmap(model)(test_state[:2, 0])
print("initial AR prediction", initial_prediction.shape)


# %% [markdown]
# ## Train on random rollout windows
#
# A one-step model can overfit the first transition if it only ever sees
# $t_0 \to t_1$. Instead, each training step samples short windows from many
# trajectories. The model rolls out freely over each window and is penalised
# against the true future states.

# %%
n_window_steps = model.n_steps
n_windows_per_example = 1
batch_size = 16


def loss_fn_ar(model, batch, key):
    total_steps = batch.shape[1]
    max_start = total_steps - n_window_steps - 1

    def loss_one_trajectory(trajectory, trajectory_key):
        starts = jax.random.randint(
            trajectory_key,
            (n_windows_per_example,),
            0,
            max_start + 1,
        )

        def loss_one_window(start):
            window = jax.lax.dynamic_slice_in_dim(
                trajectory,
                start,
                n_window_steps + 1,
                axis=0,
            )
            prediction = model(window[0])
            target = window[1:]
            return jnp.mean((prediction - target) ** 2)

        return jnp.mean(jax.vmap(loss_one_window)(starts))

    trajectory_keys = jax.random.split(key, batch.shape[0])
    losses = jax.vmap(loss_one_trajectory)(batch, trajectory_keys)
    return jnp.mean(losses)


optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(2e-3),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, batch, key):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn_ar)(model, batch, key)
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def full_rollout_loss(model, state_sequences):
    rollout_model = eqx.tree_at(lambda m: m.n_steps, model, state_sequences.shape[1] - 1)
    predictions = jax.vmap(rollout_model)(state_sequences[:, 0])
    targets = state_sequences[:, 1:]
    return jnp.mean((predictions - targets) ** 2)


n_training_steps = 250
loss_history = []
rng = np.random.default_rng(1708)

for step in range(1, n_training_steps + 1):
    batch_indices = rng.choice(n_train, size=batch_size, replace=False)
    batch = train_state[batch_indices]
    key_windows, step_key = jax.random.split(key_windows)
    model, opt_state, loss_value = train_step(
        model,
        opt_state,
        batch,
        step_key,
    )
    loss_history.append(float(loss_value))
    if step == 1 or step % 30 == 0:
        test_loss = loss_fn_ar(model, test_state[:batch_size], step_key)
        print(
            f"step {step:4d}  train {float(loss_value):.4e}  "
            f"test-window {float(test_loss):.4e}"
        )

full_test_loss = full_rollout_loss(model, test_state)
print(f"full held-out rollout MSE: {float(full_test_loss):.4e}")


# %% [markdown]
# ## Inspect one held-out autoregressive rollout

# %%
test_index = 0
rollout_model = eqx.tree_at(lambda m: m.n_steps, model, n_time - 1)
prediction = rollout_model(test_state[test_index, 0])
target = test_state[test_index, 1:]

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
axes[1].set_title("AR FNO prediction")

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
output_path = Path(__file__).with_name("fno_wave_operator_ar_results.png")
fig.savefig(output_path, dpi=160)
print("saved", output_path)
show_if_interactive()


# %% [markdown]
# ## Resolution-transfer probe
#
# The same autoregressive model can be evaluated on a finer spatial grid because
# the spectral layer is expressed through Fourier modes. This is a stricter test
# than the one-shot case: any grid mismatch can compound through the rollout.

# %%
n_grid_fine = 128
_, fine_basis = modal_basis(n_grid_fine)
fine_state_raw = reconstruct_spatial_state(
    *jax.vmap(simulate_modal)(test_q0[:1]),
    fine_basis,
)
fine_state = fine_state_raw / channel_scale

fine_rollout_model = eqx.tree_at(lambda m: m.n_steps, model, n_time - 1)
fine_prediction = fine_rollout_model(fine_state[0, 0])
fine_target = fine_state[0, 1:]
fine_loss = jnp.mean((fine_prediction - fine_target) ** 2)

same_prediction = rollout_model(test_state[0, 0])
same_target = test_state[0, 1:]

fine_prediction_on_coarse = fine_prediction[:, ::2, :]
fine_target_on_coarse = fine_target[:, ::2, :]

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
    "coarse vs. restricted-fine AR prediction discrepancy: "
    f"{float(prediction_discrepancy):.4e}"
)
