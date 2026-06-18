# %% [markdown]
# # PINN for a damped harmonic oscillator
#
# This notebook-script adapts the harmonic oscillator PINN example used in the
# ISMIR 2025 tutorial. The goal here is to match the Week 6 lecture notation:
# a coordinate network \(q_\theta(t)\) is trained from sparse data, initial
# conditions, and an ODE residual.
#
# The physical model is
#
# $$
# \ddot q(t) + \sigma_2 \dot q(t) + \omega_0^2 q(t) = 0.
# $$
#
# We give the network only a few observations from the first third of the
# trajectory, then ask whether the residual term helps recover the full
# oscillator and the unknown physical parameters.

# %%
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

print("jax", jax.__version__, "devices:", jax.devices())


def show_if_interactive():
    if plt.get_backend().lower() != "agg":
        plt.show()


# %% [markdown]
# ## Analytical reference solution
#
# For the underdamped case, the exact solution is
#
# $$
# q(t)
# =
# e^{-\sigma t}
# \left[
# q_0\cos(\omega_d t)
# +
# \frac{v_0+\sigma q_0}{\omega_d}\sin(\omega_d t)
# \right],
# \qquad
# \sigma = \frac{\sigma_2}{2},
# \quad
# \omega_d = \sqrt{\omega_0^2-\sigma^2}.
# $$

# %%
def damped_oscillator_solution(t, sigma2, omega0, q0=1.0, v0=0.0):
    sigma = 0.5 * sigma2
    omega_d = jnp.sqrt(omega0**2 - sigma**2)
    return jnp.exp(-sigma * t) * (
        q0 * jnp.cos(omega_d * t)
        + ((v0 + sigma * q0) / omega_d) * jnp.sin(omega_d * t)
    )


sample_rate = 1000
duration = 1.0
n_steps = int(sample_rate * duration)
t = jnp.arange(n_steps) / sample_rate
t_final = (n_steps - 1) / sample_rate

true_sigma2 = 3.5
true_omega0 = 30.0
q0 = 0.75
v0 = 0.0

q_true = damped_oscillator_solution(
    t,
    sigma2=true_sigma2,
    omega0=true_omega0,
    q0=q0,
    v0=v0,
)

# Sparse measurements: every 10 ms, but only during the first third.
observed_until = 1.0 / 3.0
data_stride = 10
data_stop = int(observed_until * sample_rate)
data_indices = jnp.arange(0, data_stop, data_stride)
t_data = t[data_indices]
q_data = q_true[data_indices]

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.asarray(t), np.asarray(q_true), label="true trajectory")
ax.plot(np.asarray(t_data), np.asarray(q_data), "o", label="observed samples")
ax.set_xlabel("time [s]")
ax.set_ylabel("displacement")
ax.set_title("Training observations are sparse and early")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
show_if_interactive()


# %% [markdown]
# ## Coordinate network
#
# The model takes a time coordinate and returns a displacement estimate:
#
# $$
# t \mapsto q_\theta(t).
# $$
#
# A Fourier feature layer helps the MLP represent oscillatory functions. The
# damping and natural frequency are trainable physical parameters, stored in log
# form so they stay positive during optimisation.

# %%
def init_linear_weight(model, init_fn, key):
    def is_linear(layer):
        return isinstance(layer, eqx.nn.Linear)

    def get_weights(tree):
        return [
            layer.weight
            for layer in jax.tree_util.tree_leaves(tree, is_leaf=is_linear)
            if is_linear(layer)
        ]

    weights = get_weights(model)
    keys = jax.random.split(key, len(weights))
    new_weights = [
        init_fn(subkey, weight.shape, weight.dtype)
        for weight, subkey in zip(weights, keys)
    ]
    return eqx.tree_at(get_weights, model, new_weights)


class OscillatorPINN(eqx.Module):
    log_sigma2: jax.Array
    log_omega0: jax.Array
    fourier_matrix: jax.Array
    mlp: eqx.nn.MLP

    def __init__(
        self,
        initial_sigma2,
        initial_omega0,
        n_fourier,
        fourier_scale,
        key,
    ):
        key_features, key_mlp, key_init = jax.random.split(key, 3)
        self.log_sigma2 = jnp.log(jnp.asarray(initial_sigma2))
        self.log_omega0 = jnp.log(jnp.asarray(initial_omega0))
        self.fourier_matrix = (
            fourier_scale * jax.random.normal(key_features, (n_fourier, 1))
        )

        mlp = eqx.nn.MLP(
            in_size=2 * n_fourier,
            out_size="scalar",
            width_size=64,
            depth=3,
            activation=jax.nn.tanh,
            key=key_mlp,
        )
        self.mlp = init_linear_weight(
            mlp,
            jax.nn.initializers.glorot_uniform(),
            key_init,
        )

    @property
    def sigma2(self):
        return jnp.exp(self.log_sigma2)

    @property
    def omega0(self):
        return jnp.exp(self.log_omega0)

    def fourier_features(self, time):
        time = jnp.atleast_1d(time)
        phase = 2.0 * jnp.pi * self.fourier_matrix @ time
        return jnp.concatenate([jnp.cos(phase), jnp.sin(phase)])

    def __call__(self, time):
        return self.mlp(self.fourier_features(time))


model_key = jax.random.PRNGKey(3407)
model = OscillatorPINN(
    initial_sigma2=5.0,
    initial_omega0=20.0,
    n_fourier=32,
    fourier_scale=5.0,
    key=model_key,
)
q_initial = jax.vmap(model)(t)

print("Initial physical parameters:")
print(f"  sigma2 = {float(model.sigma2):7.3f}  target {true_sigma2:7.3f}")
print(f"  omega0 = {float(model.omega0):7.3f}  target {true_omega0:7.3f}")


# %% [markdown]
# ## PINN loss
#
# The loss combines four terms:
#
# $$
# \mathcal L
# =
# \lambda_{\rm data}\|q_\theta-q_{\rm data}\|^2
# +
# \lambda_{\rm phys}\|\ddot q_\theta
# + \sigma_2 \dot q_\theta
# + \omega_0^2 q_\theta\|^2
# +
# \lambda_{q_0}\|q_\theta(0)-q_0\|^2
# +
# \lambda_{v_0}\|\dot q_\theta(0)-v_0\|^2.
# $$
#
# The residual is evaluated at randomly sampled collocation times. No measured
# displacement is needed at those points.

# %%
n_collocation = 128
data_weight = 1.0
physics_weight = 1e-4
position_ic_weight = 1.0
velocity_ic_weight = 1.0


def oscillator_residual(model, time):
    q = model(time)
    q_t = jax.grad(model)(time)
    q_tt = jax.grad(jax.grad(model))(time)
    return q_tt + model.sigma2 * q_t + model.omega0**2 * q


def loss_fn(model, key):
    q_pred = jax.vmap(model)(t_data)
    data_loss = jnp.mean((q_pred - q_data) ** 2)

    t_residual = jax.random.uniform(
        key,
        (n_collocation,),
        minval=0.0,
        maxval=t_final,
    )
    residual = jax.vmap(lambda time: oscillator_residual(model, time))(t_residual)
    physics_loss = jnp.mean(residual**2)

    position_ic_loss = (model(0.0) - q0) ** 2
    velocity_ic_loss = (jax.grad(model)(0.0) - v0) ** 2

    total = (
        data_weight * data_loss
        + physics_weight * physics_loss
        + position_ic_weight * position_ic_loss
        + velocity_ic_weight * velocity_ic_loss
    )
    terms = (data_loss, physics_loss, position_ic_loss, velocity_ic_loss)
    return total, terms


# %% [markdown]
# ## Optimisation

# %%
n_updates = 2500
learning_rate = 2e-3
log_every = 100

schedule = optax.cosine_onecycle_schedule(
    transition_steps=n_updates,
    peak_value=learning_rate,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(schedule),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, key):
    (loss_value, terms), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model,
        key,
    )
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, terms


history = []
train_key = jax.random.PRNGKey(0)
eval_key = jax.random.PRNGKey(1)

for update in range(n_updates + 1):
    if update % log_every == 0 or update == n_updates:
        loss_value, terms = loss_fn(model, eval_key)
        data_loss, physics_loss, position_ic_loss, velocity_ic_loss = terms
        history.append(
            (
                update,
                float(loss_value),
                float(data_loss),
                float(physics_loss),
                float(position_ic_loss),
                float(velocity_ic_loss),
                float(model.sigma2),
                float(model.omega0),
            )
        )
        print(
            f"{update:5d}  loss={float(loss_value):.3e}  "
            f"sigma2={float(model.sigma2):6.3f}/{true_sigma2:6.3f}  "
            f"omega0={float(model.omega0):6.3f}/{true_omega0:6.3f}"
        )

    if update < n_updates:
        train_key, step_key = jax.random.split(train_key)
        model, opt_state, _, _ = train_step(model, opt_state, step_key)

q_pinn = jax.vmap(model)(t)

sigma2_error = abs(float(model.sigma2) - true_sigma2) / true_sigma2
omega0_error = abs(float(model.omega0) - true_omega0) / true_omega0
trajectory_error = jnp.sqrt(jnp.mean((q_pinn - q_true) ** 2)) / (
    jnp.sqrt(jnp.mean(q_true**2)) + 1e-12
)

print("Final relative errors:")
print(f"  sigma2:     {100.0 * sigma2_error:6.2f}%")
print(f"  omega0:     {100.0 * omega0_error:6.2f}%")
print(f"  trajectory: {100.0 * float(trajectory_error):6.2f}%")


# %% [markdown]
# ## Inspect the learned solution

# %%
history_array = np.asarray(history)

fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)

axes[0].plot(np.asarray(t), np.asarray(q_true), label="true")
axes[0].plot(np.asarray(t), np.asarray(q_initial), "--", label="initial network")
axes[0].plot(np.asarray(t), np.asarray(q_pinn), ":", linewidth=2.5, label="PINN")
axes[0].plot(np.asarray(t_data), np.asarray(q_data), "o", label="data")
axes[0].axvline(observed_until, color="0.5", linestyle="--", linewidth=1.0)
axes[0].set_xlabel("time [s]")
axes[0].set_ylabel("displacement")
axes[0].set_title("Sparse-data PINN reconstruction")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(history_array[:, 0], history_array[:, 1], label="total")
axes[1].semilogy(history_array[:, 0], history_array[:, 2], label="data")
axes[1].semilogy(
    history_array[:, 0],
    physics_weight * history_array[:, 3],
    label="weighted residual",
)
axes[1].set_xlabel("update")
axes[1].set_ylabel("loss")
axes[1].set_title("Training losses")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(history_array[:, 0], history_array[:, 6], label="learned sigma2")
axes[2].axhline(true_sigma2, color="tab:blue", linestyle=":", label="true sigma2")
axes[2].plot(history_array[:, 0], history_array[:, 7], label="learned omega0")
axes[2].axhline(true_omega0, color="tab:orange", linestyle=":", label="true omega0")
axes[2].set_xlabel("update")
axes[2].set_ylabel("parameter value")
axes[2].set_title("Learned physical parameters")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.tight_layout()

try:
    output_dir = Path(__file__).resolve().parent
except NameError:
    output_dir = Path.cwd()

output_path = output_dir / "pinn_harmonic_oscillator_results.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
print("Saved", output_path)
show_if_interactive()


# %% [markdown]
# ## Things to try
#
# - Set `physics_weight = 0.0` and compare extrapolation after the observed
#   region.
# - Move `observed_until` earlier or later.
# - Change `initial_sigma2` and `initial_omega0`.
# - Reduce `n_fourier` or `fourier_scale` and watch how the oscillator fit
#   changes.
# - Increase `true_omega0` to make the trajectory more audio-like and harder to
#   optimise.
