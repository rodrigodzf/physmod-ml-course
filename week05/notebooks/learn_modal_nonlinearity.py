# %% [markdown]
# # Learn a Modal Nonlinearity
#
# This notebook-script is the staged linear-plus-residual version of the Week 5
# forced modal example.
#
# We first learn a linear modal approximation to the nonlinear trajectory:
#
# $$
# \ddot{\mathbf q}
# =
# -\boldsymbol\gamma_2\dot{\mathbf q}
# -
# \boldsymbol\omega^2\mathbf q
# +
# \mathbf x.
# $$
#
# Then we train the linear modal parameters and a neural restoring force
# together:
#
# $$
# \ddot{\mathbf q}
# =
# -\boldsymbol\gamma_2^{\star}\dot{\mathbf q}
# -
# (\boldsymbol\omega^{\star})^2\mathbf q
# -
# \mathbf r_\theta(\mathbf q)
# +
# \mathbf x.
# $$
#
# This follows the Neural ODE window-training pattern in `NODE.py`, but the MLP
# is not allowed to replace the whole acceleration. It can only estimate a
# missing restoring force term.

# %%
import dataclasses
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from IPython.display import Audio, display
from scipy.io import wavfile

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


def init_linear_weight(model, init_fn, key):
    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)

    def get_weights(m):
        return [
            x.weight
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
            if is_linear(x)
        ]

    weights = get_weights(model)
    if not weights:
        return model

    keys = jax.random.split(key, len(weights))
    new_weights = [
        init_fn(subkey, weight.shape, weight.dtype)
        for weight, subkey in zip(weights, keys)
    ]

    return eqx.tree_at(get_weights, model, new_weights)


# %% [markdown]
# ## Simulation setup
#
# The full one-second trajectory is used for listening and as the source for
# random training windows. We use a stronger burst than the purely linear
# training-data notebook so the cubic residual changes the trajectory by a
# visible amount.

# %%
n_modes = 15
sample_rate = 44100
full_n_steps = sample_rate
train_n_steps = int(0.010 * sample_rate)
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


# %% [markdown]
# ## Short excitation burst
#
# The same modal input is used for the linear reference trajectory and the
# nonlinear target trajectory. The difference is whether the hidden nonlinear
# restoring force is active.


# %%
def cosine_burst(
    n_steps,
    center_sample=70,
    length=49,
    frequency_hz=600.0,
    amplitude=2000.0,
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


force_full = cosine_burst(full_n_steps)
modal_force_full = force_full[:, None] * input_weights[None, :]
time_full = jnp.arange(full_n_steps) * dt

force = force_full[:train_n_steps]
modal_force = modal_force_full[:train_n_steps]
time = time_full[:train_n_steps]


# %% [markdown]
# ## Hidden nonlinear force
#
# The target residual is the modal Kirchhoff-Carrier style cubic force used in
# the Week 3 nonlinear string exercise:
#
# $$
# r_\mu(\mathbf q)
# =
# \beta\lambda_\mu q_\mu
# \sum_\nu \lambda_\nu q_\nu^2.
# $$
#
# `solve_sv_one_step` subtracts `nl_fn(q)` from the acceleration, so this
# function returns the positive restoring-force term.

# %%
beta = 3e9


def target_nonlinear_force(q):
    stretch = jnp.sum(lambda_mu * q**2)
    return beta * lambda_mu * q * stretch


# %% [markdown]
# ## Generate reference and target trajectories
#
# Full one-second trajectories are generated for listening and training. Each
# optimizer update samples several `train_n_steps` windows from the full
# trajectory, where `train_n_steps` is 10 ms at 44.1 kHz.

# %%
_, q_linear_full, v_linear_full = solve_sv_one_step(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_force_full,
    u0=q0,
    v0=v0,
)

_, q_target_full, v_target_full = solve_sv_one_step(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_force_full,
    u0=q0,
    v0=v0,
    nl_fn=target_nonlinear_force,
)

q_linear = q_linear_full[:train_n_steps]
v_linear = v_linear_full[:train_n_steps]
q_target = q_target_full[:train_n_steps]
v_target = v_target_full[:train_n_steps]

y_linear_full = q_linear_full @ readout_weights
y_target_full = q_target_full @ readout_weights
y_linear = q_linear @ readout_weights
y_target = q_target @ readout_weights

target_residual = jax.vmap(target_nonlinear_force)(q_target)
target_residual_full = jax.vmap(target_nonlinear_force)(q_target_full)

print(f"force_full:           {force_full.shape}")
print(f"force first 10 ms:    {force.shape}")
print(f"q_linear_full:        {q_linear_full.shape}")
print(f"q_target_full:        {q_target_full.shape}")
print(f"q_linear first 10 ms: {q_linear.shape}")
print(f"q_target first 10 ms: {q_target.shape}")
print(f"target_residual:      {target_residual.shape}")
print(
    "relative nonlinear displacement change:",
    float(jnp.linalg.norm(q_target - q_linear) / (jnp.linalg.norm(q_linear) + 1e-12)),
)


# %% [markdown]
# ## Inspect the target

# %%
fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)

axes[0].plot(np.asarray(force), color="tab:orange")
axes[0].set_ylabel("force")
axes[0].set_title("Input force")
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.asarray(y_linear), "--", label="purely linear reference")
axes[1].plot(np.asarray(y_target), label="nonlinear target")
axes[1].set_ylabel("readout")
axes[1].set_title("Readout trajectory")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(np.asarray(q_target[:, :4]))
axes[2].set_ylabel("modal q")
axes[2].set_title("First four target modal displacements")
axes[2].grid(True, alpha=0.3)

axes[3].plot(np.asarray(target_residual[:, :4]))
axes[3].set_ylabel("residual")
axes[3].set_xlabel("sample")
axes[3].set_title("Hidden nonlinear restoring force")
axes[3].grid(True, alpha=0.3)

fig.tight_layout()
show_if_interactive()


# %% [markdown]
# ## Inspect the full one-second target

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

axes[0].plot(np.asarray(time_full), np.asarray(force_full), color="tab:orange")
axes[0].set_ylabel("force")
axes[0].set_title("Full one-second input force")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    np.asarray(time_full),
    np.asarray(y_linear_full),
    "--",
    label="physical linear reference",
)
axes[1].plot(np.asarray(time_full), np.asarray(y_target_full), label="nonlinear target")
axes[1].axvspan(
    0.0,
    train_n_steps / sample_rate,
    color="tab:green",
    alpha=0.15,
    label="training slice",
)
axes[1].set_xlabel("time [s]")
axes[1].set_ylabel("readout")
axes[1].set_title("Full one-second readout")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
show_if_interactive()

display(Audio(y_target_full, rate=sample_rate))
display(Audio(y_linear_full, rate=sample_rate))


# %% [markdown]
# # Learn the linear approximation
#
# Here the linear model is deliberately fit to the nonlinear target trajectory.
# It is constrained to remain a physical linear string: the fit can move only
# the tension, bending stiffness, and two damping coefficients. It cannot choose
# an independent frequency and damping rate for every mode, because that can
# overfit one short nonlinear trajectory too well.

# %%
omega_target = jnp.sqrt(omega_mu_squared)
gamma2_target = gamma2_mu
physical_linear_target = {
    "tension": params.Ts0,
    "bending_stiffness": params.bending_stiffness,
    "d1": params.d1,
    "d3": params.d3,
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


linear_fit_key = jax.random.PRNGKey(0)
linear_model = StringModel(raw_params=init_random_linear_params(linear_fit_key))
q_initial, v_initial, y_initial = linear_model(modal_force, return_modal=True)

omega_initial, gamma2_initial, _ = linear_model.modal_params()
physical_initial = linear_model.physical_params()

print_physical_param_errors(
    "Initial physical-parameter relative errors:", physical_initial
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
# ## Linear random-window state loss

# %%
window_size = train_n_steps
n_training_windows = 8
window_eval_key = jax.random.PRNGKey(123)

q_target_scale = jnp.sqrt(jnp.mean(q_target_full**2)) + 1e-12
v_target_scale = jnp.sqrt(jnp.mean(v_target_full**2)) + 1e-12


def window_state_loss(model, key):
    total_steps = q_target_full.shape[0]
    max_start = total_steps - window_size
    random_starts = jax.random.randint(
        key,
        (n_training_windows,),
        0,
        max_start + 1,
    )

    def compute_window_loss(start_idx):
        force_window = jax.lax.dynamic_slice_in_dim(
            modal_force_full,
            start_idx,
            window_size,
            axis=0,
        )
        q_window = jax.lax.dynamic_slice_in_dim(
            q_target_full,
            start_idx,
            window_size,
            axis=0,
        )
        v_window = jax.lax.dynamic_slice_in_dim(
            v_target_full,
            start_idx,
            window_size,
            axis=0,
        )

        q_hat, v_hat, _ = model(
            force_window,
            u0_state=q_window[0],
            v0_state=v_window[0],
            return_modal=True,
        )
        q_loss = jnp.mean(((q_hat - q_window) / q_target_scale) ** 2)
        v_loss = jnp.mean(((v_hat - v_window) / v_target_scale) ** 2)
        return q_loss + v_loss

    return jnp.mean(jax.vmap(compute_window_loss)(random_starts))


def linear_loss_fn(model, key):
    return window_state_loss(model, key)


linear_learning_rate = 3e-2
linear_updates = 500
linear_log_every = 25

linear_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(linear_learning_rate),
)
linear_opt_state = linear_optimizer.init(eqx.filter(linear_model, eqx.is_array))


@eqx.filter_jit
def linear_train_step(model, opt_state, key):
    loss_value, grads = eqx.filter_value_and_grad(linear_loss_fn)(model, key)
    updates, opt_state = linear_optimizer.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


linear_history = []
linear_train_key = jax.random.PRNGKey(1)

for update in range(linear_updates + 1):
    if update % linear_log_every == 0 or update == linear_updates:
        linear_history.append(
            (update, float(linear_loss_fn(linear_model, window_eval_key)))
        )

    if update < linear_updates:
        linear_train_key, update_key = jax.random.split(linear_train_key)
        linear_model, linear_opt_state, _ = linear_train_step(
            linear_model,
            linear_opt_state,
            update_key,
        )

q_linear_approx, v_linear_approx, y_linear_approx = linear_model(
    modal_force,
    return_modal=True,
)
omega_fit, gamma2_fit, omega_mu_squared_fit = linear_model.modal_params()
physical_linear_approx = linear_model.physical_params()
linear_model_approx = linear_model
omega_linear_approx = omega_fit
gamma2_linear_approx = gamma2_fit

pure_linear_readout_error = jnp.mean((y_linear - y_target) ** 2) / (
    jnp.mean(y_target**2) + 1e-12
)
linear_readout_error = jnp.mean((y_linear_approx - y_target) ** 2) / (
    jnp.mean(y_target**2) + 1e-12
)
linear_window_state_loss = window_state_loss(linear_model_approx, window_eval_key)

print_physical_param_errors(
    "Final physical-parameter relative errors after linear approximation:",
    physical_linear_approx,
)
print("Final implied modal-parameter relative errors after linear approximation:")
print(
    "  omega:",
    float(jnp.linalg.norm(omega_fit - omega_target) / jnp.linalg.norm(omega_target)),
)
print(
    "  gamma2:",
    float(jnp.linalg.norm(gamma2_fit - gamma2_target) / jnp.linalg.norm(gamma2_target)),
)
print(
    "Final linear objective:",
    float(linear_loss_fn(linear_model_approx, window_eval_key)),
)
print("Final linear window state loss:", float(linear_window_state_loss))
print(
    "Pure physical-linear reference normalized readout MSE:",
    float(pure_linear_readout_error),
)
print(
    "Fitted linear-approximation normalized readout MSE:",
    float(linear_readout_error),
)


# %% [markdown]
# ## Inspect the learned linear approximation

# %%
linear_history_steps = np.asarray([item[0] for item in linear_history])
linear_history_loss = np.asarray([item[1] for item in linear_history])

fig, axes = plt.subplots(4, 1, figsize=(8, 8))

axes[0].semilogy(linear_history_steps, linear_history_loss)
axes[0].set_xlabel("update")
axes[0].set_ylabel("loss")
axes[0].set_title("Linear approximation random-window state loss")
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

axes[3].plot(np.asarray(y_linear), "--", label="physical linear reference")
axes[3].plot(np.asarray(y_target), label="nonlinear target")
axes[3].plot(np.asarray(y_initial), "--", label="initial", alpha=0.7)
axes[3].plot(
    np.asarray(y_linear_approx),
    ":",
    label="fitted linear approximation",
    linewidth=2.0,
)
axes[3].set_xlabel("sample")
axes[3].set_ylabel("readout")
axes[3].set_title("Nonlinear-target readout")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

fig.tight_layout()
show_if_interactive()

_, _, y_linear_approx_full = linear_model_approx(modal_force_full, return_modal=True)

display(Audio(y_linear_approx_full, rate=sample_rate))
display(Audio(y_target_full, rate=sample_rate))


# %% [markdown]
# # Learn the linear and nonlinear parts together
#
# The linear approximation is used only as an initialization. In this stage,
# both the physical linear parameters and the neural residual are trainable, and
# the objective is the same random-window state loss used above.

# %%
residual_scale = jnp.sqrt(jnp.mean(target_residual_full**2)) + 1e-12
mlp_key = jax.random.PRNGKey(43)
mlp = eqx.nn.MLP(
    in_size=n_modes,
    out_size=n_modes,
    width_size=128,
    depth=4,
    activation=jax.nn.selu,
    key=mlp_key,
)
mlp = init_linear_weight(
    mlp,
    jax.nn.initializers.lecun_normal(),
    mlp_key,
)


def joint_loss_fn(model, key):
    return window_state_loss(model, key)


joint_model = StringModel(
    raw_params=linear_model_approx.raw_params,
    mlp=mlp,
    q_scale=float(q_target_scale),
    residual_scale=float(residual_scale),
)
joint_initial_loss = float(joint_loss_fn(joint_model, window_eval_key))

joint_learning_rate = 5e-3
joint_updates = 750
joint_log_every = 25

joint_schedule = optax.cosine_onecycle_schedule(
    transition_steps=joint_updates,
    peak_value=joint_learning_rate,
)
joint_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(joint_schedule),
)
joint_opt_state = joint_optimizer.init(eqx.filter(joint_model, eqx.is_array))


@eqx.filter_jit
def joint_train_step(model, opt_state, key):
    loss_value, grads = eqx.filter_value_and_grad(joint_loss_fn)(model, key)
    updates, opt_state = joint_optimizer.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


joint_history = []
joint_train_key = jax.random.PRNGKey(99)

for update in range(joint_updates + 1):
    if update % joint_log_every == 0 or update == joint_updates:
        joint_history.append(
            (update, float(joint_loss_fn(joint_model, window_eval_key)))
        )

    if update < joint_updates:
        joint_train_key, update_key = jax.random.split(joint_train_key)
        joint_model, joint_opt_state, _ = joint_train_step(
            joint_model,
            joint_opt_state,
            update_key,
        )

q_joint, v_joint, y_joint = joint_model(modal_force, return_modal=True)
omega_joint, gamma2_joint, omega_joint_squared = joint_model.modal_params()
physical_joint = joint_model.physical_params()
joint_residual_hat = jax.vmap(joint_model.residual_force)(q_target_full)
joint_residual_force_error = jnp.linalg.norm(
    joint_residual_hat - target_residual_full
) / (jnp.linalg.norm(target_residual_full) + 1e-12)
joint_readout_error = jnp.mean((y_joint - y_target) ** 2) / (
    jnp.mean(y_target**2) + 1e-12
)

print("Joint fine-tuned rollout:")
print("  initial random-window state loss:", joint_initial_loss)
print(
    "  final random-window state loss:",
    float(joint_loss_fn(joint_model, window_eval_key)),
)
print("  normalized readout MSE:", float(joint_readout_error))
print("  hidden-force reference relative error:", float(joint_residual_force_error))
print(
    "  omega change from linear approximation:",
    float(
        jnp.linalg.norm(omega_joint - omega_linear_approx)
        / (jnp.linalg.norm(omega_linear_approx) + 1e-12)
    ),
)
print(
    "  gamma2 change from linear approximation:",
    float(
        jnp.linalg.norm(gamma2_joint - gamma2_linear_approx)
        / (jnp.linalg.norm(gamma2_linear_approx) + 1e-12)
    ),
)
print_physical_param_errors(
    "  final physical-parameter relative errors:",
    physical_joint,
)


# %% [markdown]
# ## Full one-second rollouts and audio export
#
# The models were trained on random 10 ms windows sampled from the full
# one-second target. These full one-second rollouts are for listening and
# checking the open-loop trajectory.

# %%
_, _, y_linear_approx_full = linear_model_approx(modal_force_full, return_modal=True)
q_joint_full, v_joint_full, y_joint_full = joint_model(
    modal_force_full,
    return_modal=True,
)

full_linear_readout_error = jnp.mean((y_linear_approx_full - y_target_full) ** 2) / (
    jnp.mean(y_target_full**2) + 1e-12
)
full_joint_readout_error = jnp.mean((y_joint_full - y_target_full) ** 2) / (
    jnp.mean(y_target_full**2) + 1e-12
)

print("Full one-second normalized readout MSE:")
print("  fitted linear approximation:", float(full_linear_readout_error))
print("  joint fine-tuned model:", float(full_joint_readout_error))


def write_normalized_wav(path, signal):
    signal_np = np.asarray(signal)
    signal_np = signal_np / (np.max(np.abs(signal_np)) + 1e-12)
    wavfile.write(path, sample_rate, np.asarray(0.95 * signal_np, dtype=np.float32))


audio_dir = Path(__file__).resolve().parent / "audio"
audio_dir.mkdir(exist_ok=True)

write_normalized_wav(
    audio_dir / "modal_nonlinearity_physical_linear.wav", y_linear_full
)
write_normalized_wav(audio_dir / "modal_nonlinearity_target.wav", y_target_full)
write_normalized_wav(
    audio_dir / "modal_nonlinearity_linear_approx.wav",
    y_linear_approx_full,
)
write_normalized_wav(
    audio_dir / "modal_nonlinearity_joint_model.wav",
    y_joint_full,
)

print("Wrote full-second audio files to:", audio_dir)


# %% [markdown]
# ## Inspect the staged fit

# %%
joint_history_steps = np.asarray([item[0] for item in joint_history])
joint_history_loss = np.asarray([item[1] for item in joint_history])

linear_readout_residual = y_linear_approx - y_target
joint_readout_residual = y_joint - y_target

print(
    "Final plot linear-approximation normalized readout MSE:",
    float(linear_readout_error),
)
print(
    "Final plot physical-linear reference normalized readout MSE:",
    float(pure_linear_readout_error),
)

fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=False)

axes[0].semilogy(joint_history_steps, joint_history_loss, label="joint fine-tune")
axes[0].set_xlabel("update")
axes[0].set_ylabel("loss")
axes[0].set_title("Random-window state loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    np.asarray(y_target),
    color="black",
    label="nonlinear target",
    alpha=0.75,
    linewidth=1.2,
    zorder=3,
)
axes[1].plot(
    np.asarray(y_linear_approx),
    "--",
    color="tab:orange",
    label="linear approximation",
    linewidth=2.0,
    zorder=5,
)
axes[1].plot(
    np.asarray(y_joint),
    ":",
    color="tab:green",
    label="joint fine-tuned model",
    linewidth=2.0,
    zorder=5,
)
axes[1].set_xlabel("sample")
axes[1].set_ylabel("readout")
axes[1].set_title("Readout trajectory")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(
    np.asarray(linear_readout_residual),
    "--",
    color="tab:orange",
    label="linear approximation",
)
axes[2].plot(
    np.asarray(joint_readout_residual),
    ":",
    color="tab:green",
    label="joint fine-tuned model",
    linewidth=2.0,
)
axes[2].set_xlabel("sample")
axes[2].set_ylabel("readout error")
axes[2].set_title("Readout error relative to nonlinear target")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(np.asarray(q_target[:, :4]), alpha=0.7)
axes[3].plot(np.asarray(q_joint[:, :4]), ":", linewidth=2.0)
axes[3].set_xlabel("sample")
axes[3].set_ylabel("modal q")
axes[3].set_title("First four modal displacements: target vs. joint model")
axes[3].grid(True, alpha=0.3)

residual_hat = jax.vmap(joint_model.residual_force)(q_target)
axes[4].plot(np.asarray(target_residual[:, :3]), alpha=0.7)
axes[4].plot(np.asarray(residual_hat[:, :3]), ":", linewidth=2.0)
axes[4].set_xlabel("sample")
axes[4].set_ylabel("residual force")
axes[4].set_title("Learned residual vs. hidden-force reference")
axes[4].grid(True, alpha=0.3)

fig.tight_layout()
show_if_interactive()

display(Audio(y_joint_full, rate=sample_rate))
display(Audio(y_target_full, rate=sample_rate))

# %%
