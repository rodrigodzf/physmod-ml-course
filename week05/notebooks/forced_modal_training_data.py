# %% [markdown]
# # Forced Modal Training Data
#
# This notebook-script creates one short training trajectory for Week 5.
#
# We keep the initial displacement and velocity at zero, then drive the modal
# system with a short cosine burst. The physical string is represented with the
# same `StringModel` abstraction used by the nonlinear Week 5 notebook, so the
# target generation and parameter fitting use the same model interface.

# %%
import dataclasses

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


def show_if_interactive():
    if plt.get_backend().lower() != "agg":
        plt.show()


# %% [markdown]
# ## Simulation setup
#
# The full trajectory is only 441 samples long. At 44.1 kHz this is
# 10 ms, so this is training data for inspecting a short forced response,
# not a complete audio example.

# %%
n_modes = 24
sample_rate = 44100
n_steps = 441
dt = 1.0 / sample_rate

excitation_position = 0.18
readout_position = 0.82

params = dataclasses.replace(
    StringParameters(),
    d1=8e-4,
    d3=1.4e-5,
)

lambda_mu = string_eigenvalues(n_modes, params.length)
omega_mu_squared = stiffness_term(params, lambda_mu)
gamma2_mu = damping_term(params, lambda_mu)

mode_numbers = jnp.arange(1, n_modes + 1)
input_weights = evaluate_string_eigenfunctions(
    mode_numbers,
    excitation_position,
    params,
)
readout_weights = evaluate_string_eigenfunctions(
    mode_numbers,
    readout_position,
    params,
)

q0 = jnp.zeros(n_modes)
v0 = jnp.zeros(n_modes)

omega_target = jnp.sqrt(omega_mu_squared)
gamma2_target = gamma2_mu
physical_linear_target = {
    "tension": params.Ts0,
    "bending_stiffness": params.bending_stiffness,
    "d1": params.d1,
    "d3": params.d3,
}


def init_physical_raw_params(physical_params):
    return {
        "log_tension": jnp.log(physical_params["tension"]),
        "log_bending_stiffness": jnp.log(physical_params["bending_stiffness"]),
        "log_d1": jnp.log(physical_params["d1"]),
        "log_d3": jnp.log(physical_params["d3"]),
    }


def init_random_linear_params(key):
    key_tension, key_bending, key_d1, key_d3 = jax.random.split(key, 4)

    tension_init = params.Ts0 * jax.random.uniform(
        key_tension,
        (),
        minval=0.75,
        maxval=1.25,
    )
    bending_init = params.bending_stiffness * jax.random.uniform(
        key_bending,
        (),
        minval=0.75,
        maxval=1.25,
    )
    d1_init = params.d1 * jax.random.uniform(
        key_d1,
        (),
        minval=0.75,
        maxval=1.25,
    )
    d3_init = params.d3 * jax.random.uniform(
        key_d3,
        (),
        minval=0.75,
        maxval=1.25,
    )

    return {
        "log_tension": jnp.log(tension_init),
        "log_bending_stiffness": jnp.log(bending_init),
        "log_d1": jnp.log(d1_init),
        "log_d3": jnp.log(d3_init),
    }


def constrain_modal_params(raw_params):
    tension_fit = jnp.exp(raw_params["log_tension"])
    bending_fit = jnp.exp(raw_params["log_bending_stiffness"])
    d1_fit = jnp.exp(raw_params["log_d1"])
    d3_fit = jnp.exp(raw_params["log_d3"])

    omega_mu_squared_fit = (
        bending_fit * lambda_mu**2 + tension_fit * lambda_mu
    ) / params.density
    gamma2_mu_fit = (d1_fit + d3_fit * lambda_mu) / params.density
    omega_mu = jnp.sqrt(omega_mu_squared_fit)

    return omega_mu, gamma2_mu_fit, omega_mu_squared_fit


def physical_linear_params(raw_params):
    return {
        "tension": jnp.exp(raw_params["log_tension"]),
        "bending_stiffness": jnp.exp(raw_params["log_bending_stiffness"]),
        "d1": jnp.exp(raw_params["log_d1"]),
        "d3": jnp.exp(raw_params["log_d3"]),
    }


def print_physical_param_errors(label, fitted_params):
    print(label)
    for name, target_value in physical_linear_target.items():
        rel_error = jnp.abs(fitted_params[name] - target_value) / jnp.abs(target_value)
        print(f"  {name}:", float(rel_error))


class StringModel(eqx.Module):
    raw_params: dict
    mlp: eqx.Module | None = None
    q_scale: float = 1.0
    residual_scale: float = 1.0

    def modal_params(self):
        return constrain_modal_params(self.raw_params)

    def physical_params(self):
        return physical_linear_params(self.raw_params)

    def residual_force(self, q_state):
        if self.mlp is None:
            return jnp.zeros_like(q_state)
        return self.residual_scale * self.mlp(q_state / self.q_scale)

    def __call__(
        self, modal_force_sequence, u0_state=q0, v0_state=v0, return_modal=False
    ):
        _, gamma2_current, omega_squared_current = self.modal_params()

        if self.mlp is None:
            _, q_hat, v_hat = solve_sv_one_step(
                gamma2_mu=gamma2_current,
                omega_mu_squared=omega_squared_current,
                dt=dt,
                xs=modal_force_sequence,
                u0=u0_state,
                v0=v0_state,
            )
        else:
            damping_factor_current = 1.0 + gamma2_current * dt / 2.0

            def step(state, inputs):
                q_state, v_state = state
                force_n, force_np1 = inputs

                residual_n = self.residual_force(q_state)
                v_half = v_state + 0.5 * dt * (
                    -gamma2_current * v_state
                    - omega_squared_current * q_state
                    - residual_n
                    + force_n
                )

                q_next = q_state + dt * v_half

                residual_next = self.residual_force(q_next)
                a_next = -omega_squared_current * q_next - residual_next + force_np1
                v_next = (v_half + 0.5 * dt * a_next) / damping_factor_current

                return (q_next, v_next), (q_next, v_next)

            _, (q_tail, v_tail) = jax.lax.scan(
                step,
                (u0_state, v0_state),
                (modal_force_sequence[:-1], modal_force_sequence[1:]),
            )
            q_hat = jnp.concatenate([u0_state[None], q_tail], axis=0)
            v_hat = jnp.concatenate([v0_state[None], v_tail], axis=0)

        y_hat = q_hat @ readout_weights
        if return_modal:
            return q_hat, v_hat, y_hat
        return y_hat


# %% [markdown]
# ## Short cosine excitation
#
# The scalar excitation is a Hann-windowed cosine burst centred around
# sample 50. We then project it into modal coordinates with the input
# mode-shape weights.


# %%
def cosine_burst(
    n_steps,
    center_sample=50,
    length=33,
    frequency_hz=1200.0,
    amplitude=40.0,
    sample_rate=sample_rate,
):
    n = jnp.arange(length)
    n_centered = n - 0.5 * (length - 1)

    window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (length - 1))
    carrier = jnp.cos(2.0 * jnp.pi * frequency_hz * n_centered / sample_rate)
    burst = amplitude * window * carrier

    start = center_sample - length // 2
    force = jnp.zeros(n_steps)
    return force.at[start : start + length].set(burst)


force = cosine_burst(n_steps)
modal_force = force[:, None] * input_weights[None, :]

# %% [markdown]
# ## Generate the trajectory
#
# The target is generated through `StringModel`. Internally it still uses the
# same velocity-Verlet style solver, but the notebook now has one model object
# for the physical string.

# %%
target_model = StringModel(
    raw_params=init_physical_raw_params(physical_linear_target),
)
q, v, y = target_model(modal_force, return_modal=True)
time = jnp.arange(n_steps) * dt

training_data = {
    "time": time,
    "force": force,
    "modal_force": modal_force,
    "q": q,
    "v": v,
    "y": y,
    "dt": dt,
    "sample_rate": sample_rate,
}

print(f"force: {force.shape}")
print(f"q:     {q.shape}")
print(f"v:     {v.shape}")
print(f"y:     {y.shape}")

# %% [markdown]
# ## Inspect the target

# %%
fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

axes[0].plot(np.asarray(force), color="tab:orange")
axes[0].set_ylabel("force")
axes[0].set_title("Short input force")
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.asarray(y), color="tab:blue")
axes[1].set_ylabel("readout")
axes[1].set_title("Physical readout")
axes[1].grid(True, alpha=0.3)

axes[2].plot(np.asarray(q[:, :4]))
axes[2].set_ylabel("modal q")
axes[2].set_xlabel("sample")
axes[2].set_title("First four modal displacements")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
show_if_interactive()

# %% [markdown]
# ## Pure physical case: fit string parameters from the trajectory
#
# Now treat the generated trajectory as training data. We assume the modal
# input force is known, but the physical string parameters are not.
#
# The fitted parameters are the same constrained linear-string parameters used
# in `learn_modal_nonlinearity.py`:
#
# $$
# T_0 > 0,
# \qquad
# B > 0,
# \qquad
# d_1 > 0,
# \qquad
# d_3 > 0.
# $$
#
# The modal frequencies and damping rates are then implied by those physical
# parameters rather than fitted independently per mode.

# %%
fit_key = jax.random.PRNGKey(0)
linear_model = StringModel(raw_params=init_random_linear_params(fit_key))
q_initial, v_initial, y_initial = linear_model(modal_force, return_modal=True)

omega_initial, gamma2_initial, _ = linear_model.modal_params()
physical_initial = linear_model.physical_params()

print_physical_param_errors(
    "Initial physical-parameter relative errors:",
    physical_initial,
)
print("Initial implied modal-parameter relative errors:")
print(
    "  omega:",
    float(
        jnp.linalg.norm(omega_initial - omega_target) / jnp.linalg.norm(omega_target)
    ),
)
print(
    "  gamma2:",
    float(
        jnp.linalg.norm(gamma2_initial - gamma2_target) / jnp.linalg.norm(gamma2_target)
    ),
)

# %% [markdown]
# ## Rollout loss
#
# Because we have the modal state trajectory, we can use a state loss. The
# displacement and velocity terms are normalised so that the larger velocity
# values do not dominate the optimisation.

# %%
q_scale = jnp.sqrt(jnp.mean(q**2)) + 1e-12
v_scale = jnp.sqrt(jnp.mean(v**2)) + 1e-12


def loss_fn(model):
    q_hat, v_hat, _ = model(modal_force, return_modal=True)
    q_loss = jnp.mean(((q_hat - q) / q_scale) ** 2)
    v_loss = jnp.mean(((v_hat - v) / v_scale) ** 2)
    return q_loss + v_loss


learning_rate = 3e-2
n_updates = 6000
log_every = 25

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate),
)
opt_state = optimizer.init(eqx.filter(linear_model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


history = []

for update in range(n_updates + 1):
    if update % log_every == 0 or update == n_updates:
        history.append((update, float(loss_fn(linear_model))))

    if update < n_updates:
        linear_model, opt_state, _ = train_step(linear_model, opt_state)

q_fit, v_fit, y_fit = linear_model(modal_force, return_modal=True)
omega_fit, gamma2_fit, _ = linear_model.modal_params()
physical_fit = linear_model.physical_params()

print_physical_param_errors(
    "Final physical-parameter relative errors:",
    physical_fit,
)
print("Final implied modal-parameter relative errors:")
print(
    "  omega:",
    float(jnp.linalg.norm(omega_fit - omega_target) / jnp.linalg.norm(omega_target)),
)
print(
    "  gamma2:",
    float(jnp.linalg.norm(gamma2_fit - gamma2_target) / jnp.linalg.norm(gamma2_target)),
)
print("Final loss:", float(loss_fn(linear_model)))

# %% [markdown]
# ## Inspect the pure physical fit

# %%
history_steps = np.asarray([item[0] for item in history])
history_loss = np.asarray([item[1] for item in history])

fig, axes = plt.subplots(4, 1, figsize=(8, 8))

axes[0].semilogy(history_steps, history_loss)
axes[0].set_xlabel("update")
axes[0].set_ylabel("loss")
axes[0].set_title("Rollout loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.asarray(omega_target / (2.0 * jnp.pi)), "o-", label="target")
axes[1].plot(np.asarray(omega_initial / (2.0 * jnp.pi)), "x--", label="initial")
axes[1].plot(np.asarray(omega_fit / (2.0 * jnp.pi)), ".-", label="fit")
axes[1].set_ylabel("Hz")
axes[1].set_title("Modal frequencies")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(np.asarray(gamma2_target), "o-", label="target")
axes[2].plot(np.asarray(gamma2_initial), "x--", label="initial")
axes[2].plot(np.asarray(gamma2_fit), ".-", label="fit")
axes[2].set_ylabel(r"$\gamma_{2,\mu}$")
axes[2].set_title("Modal damping rates")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(np.asarray(y), label="target")
axes[3].plot(np.asarray(y_initial), "--", label="initial", alpha=0.7)
axes[3].plot(np.asarray(y_fit), ":", label="fit", linewidth=2.0)
axes[3].set_xlabel("sample")
axes[3].set_ylabel("readout")
axes[3].set_title("Readout trajectory")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

fig.tight_layout()
show_if_interactive()

# %% [markdown]
# ## Pure neural case: learn the acceleration
#
# Now replace the physical acceleration law with a small MLP:
#
# $$
# \ddot{\mathbf q}^n
# =
# \mathbf a_\theta(\mathbf q^n,\dot{\mathbf q}^n,\mathbf x^n).
# $$
#
# No modal frequencies or damping rates are fitted in this block. The only
# learned object is the neural acceleration function.
#
# We write the velocity-Verlet update explicitly here because
# `solve_sv_one_step` assumes the structured form
#
# $$
# \ddot{\mathbf q}
# =
# -\boldsymbol\gamma_2\dot{\mathbf q}
# -
# \boldsymbol\omega^2\mathbf q
# -
# \mathbf f_{\mathrm{nl}}(\mathbf q)
# +
# \mathbf x.
# $$
#
# That is exactly what we want for a physical or hybrid model. For the pure
# neural case, however, the MLP replaces the whole acceleration and takes
# \((\mathbf q,\dot{\mathbf q},\mathbf x)\) as input.

# %%
force_scale = jnp.sqrt(jnp.mean(modal_force**2)) + 1e-12
acceleration_scale = jnp.sqrt(jnp.mean(((v[1:] - v[:-1]) / dt) ** 2)) + 1e-12


def neural_acceleration(mlp, q_state, v_state, force_state):
    features = jnp.concatenate(
        [
            q_state / q_scale,
            v_state / v_scale,
            force_state / force_scale,
        ]
    )
    return acceleration_scale * mlp(features)


def neural_rollout(mlp):
    def step(state, inputs):
        q_state, v_state = state
        force_n, force_np1 = inputs

        a_n = neural_acceleration(mlp, q_state, v_state, force_n)
        v_half = v_state + 0.5 * dt * a_n
        q_next = q_state + dt * v_half

        a_next = neural_acceleration(mlp, q_next, v_half, force_np1)
        v_next = v_half + 0.5 * dt * a_next

        return (q_next, v_next), (q_next, v_next)

    _, (q_tail, v_tail) = jax.lax.scan(
        step,
        (q0, v0),
        (modal_force[:-1], modal_force[1:]),
    )
    q_hat = jnp.concatenate([q0[None], q_tail], axis=0)
    v_hat = jnp.concatenate([v0[None], v_tail], axis=0)
    y_hat = q_hat @ readout_weights
    return q_hat, v_hat, y_hat


def neural_rollout_loss(mlp):
    q_hat, v_hat, _ = neural_rollout(mlp)
    q_loss = jnp.mean(((q_hat - q) / q_scale) ** 2)
    v_loss = jnp.mean(((v_hat - v) / v_scale) ** 2)
    return q_loss + v_loss


mlp_key = jax.random.PRNGKey(43)
mlp = eqx.nn.MLP(
    in_size=3 * n_modes,
    out_size=n_modes,
    width_size=64,
    depth=2,
    activation=jnp.tanh,
    key=mlp_key,
)
mlp = eqx.tree_at(
    lambda model: (model.layers[-1].weight, model.layers[-1].bias),
    mlp,
    (
        1e-3 * mlp.layers[-1].weight,
        jnp.zeros_like(mlp.layers[-1].bias),
    ),
)
_, _, y_neural_initial = neural_rollout(mlp)

nn_learning_rate = 1e-3
nn_updates = 4000
nn_log_every = 25

nn_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(nn_learning_rate),
)
nn_opt_state = nn_optimizer.init(eqx.filter(mlp, eqx.is_array))


@eqx.filter_jit
def neural_train_step(mlp, nn_opt_state):
    loss_value, grads = eqx.filter_value_and_grad(neural_rollout_loss)(mlp)
    updates, nn_opt_state = nn_optimizer.update(grads, nn_opt_state, mlp)
    mlp = eqx.apply_updates(mlp, updates)
    return mlp, nn_opt_state, loss_value


nn_history = []

for update in range(nn_updates + 1):
    if update % nn_log_every == 0 or update == nn_updates:
        nn_history.append((update, float(neural_rollout_loss(mlp))))

    if update < nn_updates:
        mlp, nn_opt_state, _ = neural_train_step(mlp, nn_opt_state)

q_neural, v_neural, y_neural = neural_rollout(mlp)
neural_readout_error = jnp.mean((y_neural - y) ** 2) / (jnp.mean(y**2) + 1e-12)

print("Pure neural rollout:")
print("  final loss:", float(neural_rollout_loss(mlp)))
print("  normalized readout MSE:", float(neural_readout_error))

# %% [markdown]
# ## Inspect the pure neural fit

# %%
nn_history_steps = np.asarray([item[0] for item in nn_history])
nn_history_loss = np.asarray([item[1] for item in nn_history])

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

axes[0].semilogy(nn_history_steps, nn_history_loss)
axes[0].set_xlabel("update")
axes[0].set_ylabel("loss")
axes[0].set_title("Pure neural rollout loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.asarray(y), label="target")
axes[1].plot(np.asarray(y_neural_initial), "--", label="initial", alpha=0.7)
axes[1].plot(np.asarray(y_neural), ":", label="neural", linewidth=2.0)
axes[1].set_xlabel("sample")
axes[1].set_ylabel("readout")
axes[1].set_title("Readout trajectory")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(np.asarray(q[:, :4]), alpha=0.7)
axes[2].plot(np.asarray(q_neural[:, :4]), ":", linewidth=2.0)
axes[2].set_xlabel("sample")
axes[2].set_ylabel("modal q")
axes[2].set_title("First four modal displacements: target vs. neural")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
show_if_interactive()

# %% [markdown]
# ## Optional: save the arrays
#
# Uncomment this cell if you want a small `.npz` file for a later training
# notebook.

# %%
# np.savez(
#     "forced_modal_training_data.npz",
#     time=np.asarray(time),
#     force=np.asarray(force),
#     modal_force=np.asarray(modal_force),
#     q=np.asarray(q),
#     v=np.asarray(v),
#     y=np.asarray(y),
#     dt=dt,
#     sample_rate=sample_rate,
# )
