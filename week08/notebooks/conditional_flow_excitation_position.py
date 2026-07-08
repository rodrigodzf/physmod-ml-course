# %% [markdown]
# # Conditional Normalizing Flow for Modal Parameter Estimation
#
# This notebook-script is a Week 8 companion example for distributional
# parameter estimation.
#
# We generate synthetic modal observations from a plucked string and train a
# **conditional discrete normalizing flow** to estimate a distribution over the
# excitation parameters:
#
# $$
# q_\phi(\theta \mid y),
# \qquad
# \theta =
# \begin{bmatrix}
# x_\mathrm{pluck} \\
# A_\mathrm{pluck}
# \end{bmatrix}.
# $$
#
# The flow is conditional because it receives features extracted from the
# observed sound. It is discrete because it is a finite stack of affine coupling
# layers, not an ODE-defined continuous normalizing flow.
#
# The example is intentionally modal-only. The synthetic observations are modal
# magnitudes from the same plucked-string initial-condition convention used in
# earlier notebooks.

# %%
from __future__ import annotations

import dataclasses
import math
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from jaxdiffmodal.excitations import create_pluck_modal
from jaxdiffmodal.ftm import (
    StringParameters,
    damping_term,
    evaluate_string_eigenfunctions,
    stiffness_term,
    string_eigenvalues,
)

jax.config.update("jax_enable_x64", True)

print("jax", jax.__version__, "devices:", jax.devices())


def show_if_interactive():
    if "ipykernel" in sys.modules and plt.get_backend().lower() != "agg":
        plt.show()


# %% [markdown]
# ## Modal string setup
#
# We use `jaxdiffmodal` for the physical string ingredients. The inference
# target is only the excitation position and amplitude; the string and pickup
# parameters are assumed known.
#
# The observation is a noisy log-magnitude vector of the first modal readout
# coefficients:
#
# $$
# y_\mu
# =
# \log\left(
# |c_\mu q_\mu^0| + \epsilon
# \right)
# +
# \eta_\mu.
# $$
#
# This is a compact proxy for the resonant spectrum of the recorded sound.

# %%
n_modes = 32

params = dataclasses.replace(
    StringParameters(),
    d1=8e-4,
    d3=1.4e-5,
)

length = float(params.length)
readout_position = 0.82 * length
print(f"string length L = {length:.3f} m")

lambda_mu = jnp.asarray(string_eigenvalues(n_modes, params.length))
mode_numbers = jnp.arange(1, n_modes + 1)
readout_weights = jnp.asarray(
    evaluate_string_eigenfunctions(mode_numbers, readout_position, params)
)

# The true parameters are restricted to a physically plausible training range.
theta_low = jnp.asarray([0.08 * length, 7e-4])
theta_high = jnp.asarray([0.92 * length, 5e-3])
theta_names = ("pluck position [m]", "initial deflection [m]")


def modal_pluck_coefficients(pluck_position, initial_deflection):
    """JAX version of jaxdiffmodal.excitations.create_pluck_modal."""
    lambdas_sqrt = jnp.sqrt(lambda_mu)
    deflection_scaling = initial_deflection * (
        length / (length - pluck_position)
    )
    coefficients = (
        deflection_scaling
        * jnp.sin(lambdas_sqrt * pluck_position)
        / (lambdas_sqrt * pluck_position)
    )
    return coefficients / lambdas_sqrt


# Check that the differentiable JAX implementation matches the helper used in
# the earlier modal notebooks.
check_position = 0.23 * length
check_amplitude = 2.5e-3
q0_numpy = create_pluck_modal(
    np.asarray(lambda_mu),
    pluck_position=check_position,
    initial_deflection=check_amplitude,
    string_length=length,
)
q0_jax = modal_pluck_coefficients(check_position, check_amplitude)
print("pluck helper max error:", float(jnp.max(jnp.abs(q0_jax - q0_numpy))))


def clean_features(theta):
    pluck_position, initial_deflection = theta
    q0 = modal_pluck_coefficients(pluck_position, initial_deflection)
    modal_readout = q0 * readout_weights
    return jnp.log(jnp.abs(modal_readout) + 1e-10)


def noisy_features(theta, key, noise_std=0.06):
    noise = noise_std * jax.random.normal(key, (n_modes,))
    return clean_features(theta) + noise


# %% [markdown]
# ## Synthetic training data
#
# We sample excitation parameters from a broad prior range, generate modal
# observations, and then standardise those observations. Since the observation
# uses magnitudes, a pluck at $x$ and a pluck at $L-x$ can become difficult to
# distinguish. This is exactly the kind of ambiguity where a distribution is
# more honest than a single point estimate.

# %%
n_train = 4096
n_test = 512
feature_noise_std = 0.06


def sample_theta(key, n_examples):
    key_pos, key_amp = jax.random.split(key)
    position = jax.random.uniform(
        key_pos,
        (n_examples,),
        minval=theta_low[0],
        maxval=theta_high[0],
    )
    # Log-uniform amplitude prior.
    log_amp = jax.random.uniform(
        key_amp,
        (n_examples,),
        minval=jnp.log(theta_low[1]),
        maxval=jnp.log(theta_high[1]),
    )
    amplitude = jnp.exp(log_amp)
    return jnp.stack([position, amplitude], axis=-1)


def make_dataset(key, n_examples):
    key_theta, key_noise = jax.random.split(key)
    theta = sample_theta(key_theta, n_examples)
    noise_keys = jax.random.split(key_noise, n_examples)
    features = jax.vmap(noisy_features, in_axes=(0, 0, None))(
        theta,
        noise_keys,
        feature_noise_std,
    )
    return features, theta


key = jax.random.PRNGKey(808)
key_train, key_test, key_flow, key_target = jax.random.split(key, 4)

train_features_raw, train_theta = make_dataset(key_train, n_train)
test_features_raw, test_theta = make_dataset(key_test, n_test)

feature_mean = jnp.mean(train_features_raw, axis=0)
feature_std = jnp.std(train_features_raw, axis=0) + 1e-6

train_features = (train_features_raw - feature_mean) / feature_std
test_features = (test_features_raw - feature_mean) / feature_std

print("train features", train_features.shape)
print("train theta", train_theta.shape)
print(
    "position range / L",
    float(train_theta[:, 0].min() / length),
    float(train_theta[:, 0].max() / length),
)
print("amplitude range", float(train_theta[:, 1].min()), float(train_theta[:, 1].max()))

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

for x_pluck, color in [(0.18, "tab:blue"), (0.50, "tab:orange"), (0.82, "tab:green")]:
    theta = jnp.asarray([x_pluck * length, 2.5e-3])
    axes[0].plot(
        np.arange(1, n_modes + 1),
        np.asarray(clean_features(theta)),
        marker="o",
        markersize=3,
        label=f"x={x_pluck:.2f}L",
        color=color,
    )

axes[0].set_xlabel("Mode number")
axes[0].set_ylabel("log modal magnitude")
axes[0].set_title("Clean modal observations")
axes[0].grid(True)
axes[0].legend()

axes[1].scatter(
    np.asarray(train_theta[:600, 0] / length),
    np.asarray(train_theta[:600, 1] * 1e3),
    s=8,
    alpha=0.5,
)
axes[1].set_xlabel("pluck position / L")
axes[1].set_ylabel("initial deflection [mm]")
axes[1].set_title("Training parameters")
axes[1].grid(True)

fig.tight_layout()
show_if_interactive()


# %% [markdown]
# ## Parameter transforms
#
# The flow operates on an unconstrained vector $u \in \mathbb R^2$. We map
# between physical parameters and unconstrained coordinates with a logit
# transform over the training range.

# %%
theta_dim = 2


def logit(p):
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def sigmoid(x):
    return jax.nn.sigmoid(x)


def unconstrain_theta(theta):
    unit = (theta - theta_low) / (theta_high - theta_low)
    return logit(unit)


def constrain_u(u):
    unit = sigmoid(u)
    return theta_low + unit * (theta_high - theta_low)


train_u = unconstrain_theta(train_theta)
test_u = unconstrain_theta(test_theta)


# %% [markdown]
# ## Conditional RealNVP flow
#
# The model has two pieces:
#
# 1. an encoder that maps modal features to a conditioning vector;
# 2. a stack of affine coupling layers conditioned on that vector.
#
# For likelihood training we evaluate:
#
# $$
# \log q_\phi(u \mid y)
# =
# \log p(z)
# +
# \log \left|\det \frac{\partial z}{\partial u}\right|,
# \qquad
# z = f_\phi^{-1}(u; y).
# $$

# %%
context_dim = 32
hidden_width = 64
n_coupling_layers = 8
log_scale_limit = 2.0


def initialise_weights(mlp, key, final_scale):
    def get_weights(model):
        return [layer.weight for layer in model.layers]

    def get_biases(model):
        return [layer.bias for layer in model.layers]

    keys = jax.random.split(key, len(mlp.layers))
    new_weights = []
    for i, (subkey, layer) in enumerate(zip(keys, mlp.layers, strict=True)):
        in_dim = layer.weight.shape[1]
        scale = final_scale if i == len(mlp.layers) - 1 else math.sqrt(2.0 / in_dim)
        new_weights.append(
            scale * jax.random.normal(subkey, layer.weight.shape, layer.weight.dtype)
        )
    new_biases = [jnp.zeros_like(layer.bias) for layer in mlp.layers]

    mlp = eqx.tree_at(get_weights, mlp, new_weights)
    return eqx.tree_at(
        get_biases,
        mlp,
        new_biases,
    )


def init_mlp(key, in_size, out_size, final_scale):
    mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=hidden_width,
        depth=2,
        activation=jax.nn.swish,
        key=key,
    )
    return initialise_weights(mlp, key, final_scale)


def init_flow(key):
    keys = jax.random.split(key, n_coupling_layers + 1)
    encoder = init_mlp(
        keys[0],
        n_modes,
        context_dim,
        final_scale=0.05,
    )
    coupling_mlps = tuple(
        init_mlp(
            keys[i + 1],
            theta_dim + context_dim,
            2 * theta_dim,
            final_scale=1e-2,
        )
        for i in range(n_coupling_layers)
    )
    return {"encoder": encoder, "couplings": coupling_mlps}


masks = tuple(
    jnp.asarray([1.0, 0.0]) if i % 2 == 0 else jnp.asarray([0.0, 1.0])
    for i in range(n_coupling_layers)
)


def encode_context(params, features):
    return jax.vmap(params["encoder"])(features)


def coupling_forward(coupling_params, mask, x, context):
    x_masked = x * mask
    net_input = jnp.concatenate([x_masked, context], axis=-1)
    shift_log_scale = jax.vmap(coupling_params)(net_input)
    shift, log_scale = jnp.split(shift_log_scale, 2, axis=-1)
    log_scale = log_scale_limit * jnp.tanh(log_scale) * (1.0 - mask)
    shift = shift * (1.0 - mask)
    y = x_masked + (1.0 - mask) * (x * jnp.exp(log_scale) + shift)
    log_det = jnp.sum(log_scale, axis=-1)
    return y, log_det


def coupling_inverse(coupling_params, mask, y, context):
    y_masked = y * mask
    net_input = jnp.concatenate([y_masked, context], axis=-1)
    shift_log_scale = jax.vmap(coupling_params)(net_input)
    shift, log_scale = jnp.split(shift_log_scale, 2, axis=-1)
    log_scale = log_scale_limit * jnp.tanh(log_scale) * (1.0 - mask)
    shift = shift * (1.0 - mask)
    x = y_masked + (1.0 - mask) * ((y - shift) * jnp.exp(-log_scale))
    log_det = -jnp.sum(log_scale, axis=-1)
    return x, log_det


def flow_forward(params, z, features):
    context = encode_context(params, features)
    x = z
    total_log_det = jnp.zeros((z.shape[0],))
    for coupling_params, mask in zip(params["couplings"], masks, strict=True):
        x, log_det = coupling_forward(coupling_params, mask, x, context)
        total_log_det = total_log_det + log_det
    return x, total_log_det


def flow_inverse(params, u, features):
    context = encode_context(params, features)
    x = u
    total_log_det = jnp.zeros((u.shape[0],))
    for coupling_params, mask in reversed(
        tuple(zip(params["couplings"], masks, strict=True))
    ):
        x, log_det = coupling_inverse(coupling_params, mask, x, context)
        total_log_det = total_log_det + log_det
    return x, total_log_det


def standard_normal_log_prob(z):
    """Log density of the standard Gaussian base distribution for each sample."""
    return -0.5 * jnp.sum(z**2 + jnp.log(2.0 * jnp.pi), axis=-1)


def flow_log_prob(params, u, features):
    """Conditional log density log q_phi(u | features)."""
    z, inverse_log_det = flow_inverse(params, u, features)
    return standard_normal_log_prob(z) + inverse_log_det


def nll(params, features, u):
    """Mean negative conditional log likelihood used to train the flow."""
    return -jnp.mean(flow_log_prob(params, u, features))


# %% [markdown]
# ## Train by supervised conditional likelihood
#
# Because the data are synthetic, we know the true excitation parameters. This
# lets us train the conditional flow directly:
#
# $$
# \min_\phi
# -\log q_\phi(u_\mathrm{true} \mid y).
# $$
#
# This is the cleanest way to teach the posterior estimator. A later
# analysis-by-synthesis stage could refine the distribution on real audio.

# %%
batch_size = 256
n_updates = 1600
learning_rate = 1e-3
log_every = 100

flow_params = init_flow(key_flow)
optimizer = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adamw(learning_rate, weight_decay=1e-5),
)
opt_state = optimizer.init(eqx.filter(flow_params, eqx.is_array))


@eqx.filter_jit
def train_step(flow_params, opt_state, key):
    """Take one stochastic gradient step on a minibatch of the NLL objective."""
    idx = jax.random.randint(key, (batch_size,), minval=0, maxval=n_train)
    batch_features = train_features[idx]
    batch_u = train_u[idx]
    loss_value, grads = eqx.filter_value_and_grad(nll)(
        flow_params,
        batch_features,
        batch_u,
    )
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        eqx.filter(flow_params, eqx.is_array),
    )
    flow_params = eqx.apply_updates(flow_params, updates)
    return flow_params, opt_state, loss_value


@eqx.filter_jit
def eval_losses(flow_params):
    """Evaluate full-dataset train and test NLL values for monitoring."""
    return nll(flow_params, train_features, train_u), nll(
        flow_params,
        test_features,
        test_u,
    )


history = []
loop_key = key_flow

for step in range(n_updates + 1):
    if step % log_every == 0:
        train_loss, test_loss = eval_losses(flow_params)
        history.append((step, float(train_loss), float(test_loss)))
        print(
            f"step {step:4d} | train nll {float(train_loss):8.4f} "
            f"| test nll {float(test_loss):8.4f}"
        )

    if step < n_updates:
        loop_key, subkey = jax.random.split(loop_key)
        flow_params, opt_state, loss_value = train_step(
            flow_params,
            opt_state,
            subkey,
        )

# %%
history_np = np.asarray(history)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(history_np[:, 0], history_np[:, 1], label="train")
ax.plot(history_np[:, 0], history_np[:, 2], label="test")
ax.set_xlabel("Update")
ax.set_ylabel("negative log likelihood")
ax.set_title("Conditional flow training")
ax.grid(True)
ax.legend()
fig.tight_layout()
show_if_interactive()


# %% [markdown]
# ## Sample a posterior for one target
#
# We now choose one synthetic target observation and sample many parameter
# vectors:
#
# $$
# z_k \sim \mathcal N(0, I),
# \qquad
# u_k = f_\phi(z_k; y),
# \qquad
# \theta_k = \mathrm{constrain}(u_k).
# $$
#
# The posterior over pluck position can be broad or multi-peaked if the modal
# magnitudes do not uniquely identify the excitation.

# %%
def standardise_feature(feature_raw):
    return (feature_raw - feature_mean) / feature_std


def sample_flow(params, key, feature, n_samples):
    z = jax.random.normal(key, (n_samples, theta_dim))
    features = jnp.broadcast_to(feature[None, :], (n_samples, n_modes))
    u, _ = flow_forward(params, z, features)
    return constrain_u(u)


target_theta = jnp.asarray([0.22 * length, 2.4e-3])
target_feature_raw = noisy_features(target_theta, key_target, feature_noise_std)
target_feature = standardise_feature(target_feature_raw)

key_samples = jax.random.PRNGKey(9001)
posterior_samples = sample_flow(flow_params, key_samples, target_feature, 8192)

posterior_mean = jnp.mean(posterior_samples, axis=0)
posterior_std = jnp.std(posterior_samples, axis=0)
posterior_quantiles = jnp.quantile(
    posterior_samples,
    jnp.asarray([0.05, 0.5, 0.95]),
    axis=0,
)

print("target theta:")
for name, value in zip(theta_names, target_theta, strict=True):
    print(f"  {name:24s}: {float(value): .6f}")
print(f"  {'pluck position / L':24s}: {float(target_theta[0] / length): .6f}")

print("\nposterior mean +/- std:")
for name, mean, std in zip(theta_names, posterior_mean, posterior_std, strict=True):
    print(f"  {name:24s}: {float(mean): .6f} +/- {float(std): .6f}")
print(
    f"  {'pluck position / L':24s}: "
    f"{float(posterior_mean[0] / length): .6f} +/- "
    f"{float(posterior_std[0] / length): .6f}"
)

print("\nposterior 5/50/95 percentiles:")
for i, name in enumerate(theta_names):
    q05, q50, q95 = posterior_quantiles[:, i]
    print(f"  {name:24s}: {float(q05): .6f}, {float(q50): .6f}, {float(q95): .6f}")

# %%
samples_np = np.asarray(posterior_samples)
target_np = np.asarray(target_theta)
mirror_position = length - target_np[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

axes[0].hist(samples_np[:, 0] / length, bins=80, density=True, alpha=0.8)
axes[0].axvline(target_np[0] / length, color="tab:red", linewidth=2, label="true")
axes[0].axvline(
    mirror_position / length,
    color="tab:purple",
    linestyle=":",
    linewidth=2,
    label="mirror",
)
axes[0].set_xlabel("pluck position / L")
axes[0].set_ylabel("posterior density")
axes[0].set_title("Posterior over excitation position")
axes[0].grid(True)
axes[0].legend()

axes[1].scatter(
    samples_np[:2500, 0] / length,
    samples_np[:2500, 1] * 1e3,
    s=6,
    alpha=0.25,
)
axes[1].axvline(target_np[0] / length, color="tab:red", linewidth=2)
axes[1].axhline(target_np[1] * 1e3, color="tab:red", linewidth=2)
axes[1].set_xlabel("pluck position / L")
axes[1].set_ylabel("initial deflection [mm]")
axes[1].set_title("Joint posterior samples")
axes[1].grid(True)

fig.tight_layout()
show_if_interactive()


# %% [markdown]
# ## Posterior predictive check
#
# A posterior is useful only if sampled parameters reproduce the observation.
# Here we compare the target modal features against several samples from the
# learned posterior.

# %%
sample_indices = np.linspace(0, len(posterior_samples) - 1, 8, dtype=int)
predictive_theta = posterior_samples[sample_indices]
predictive_features = jax.vmap(clean_features)(predictive_theta)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(
    np.arange(1, n_modes + 1),
    np.asarray(target_feature_raw),
    color="black",
    linewidth=2.0,
    label="target noisy features",
)
for i, feature in enumerate(np.asarray(predictive_features)):
    ax.plot(
        np.arange(1, n_modes + 1),
        feature,
        alpha=0.45,
        linewidth=1.0,
        label="posterior samples" if i == 0 else None,
    )
ax.set_xlabel("Mode number")
ax.set_ylabel("log modal magnitude")
ax.set_title("Posterior predictive modal features")
ax.grid(True)
ax.legend()
fig.tight_layout()
show_if_interactive()


# %% [markdown]
# ## Optional: synthesize audio from posterior samples
#
# The flow was trained on modal features, not waveforms. Still, each sampled
# parameter vector defines a modal initial condition and therefore an audio
# signal through the known modal string model.

# %%
sample_rate = 16_000
duration = 0.35
n_steps = int(sample_rate * duration)
time = jnp.arange(n_steps) / sample_rate

omega0_squared = stiffness_term(params, lambda_mu)
gamma2 = damping_term(params, lambda_mu)
gamma = 0.5 * gamma2
omega_d = jnp.sqrt(jnp.maximum(omega0_squared - gamma**2, 0.0))


def synthesize_audio(theta):
    q0 = modal_pluck_coefficients(theta[0], theta[1])
    z0 = q0 - 1j * (gamma * q0) / jnp.maximum(omega_d, 1e-12)
    poles = -gamma + 1j * omega_d
    z = z0[None, :] * jnp.exp(poles[None, :] * time[:, None])
    y = (z @ readout_weights).real
    return y / (jnp.max(jnp.abs(y)) + 1e-12)


y_target = synthesize_audio(target_theta)
y_samples = jax.vmap(synthesize_audio)(predictive_theta[:4])

fig, ax = plt.subplots(figsize=(8, 3.2))
plot_samples = int(0.08 * sample_rate)
ax.plot(
    np.asarray(time[:plot_samples]),
    np.asarray(y_target[:plot_samples]),
    color="black",
    linewidth=2,
    label="target",
)
for i, y_sample in enumerate(np.asarray(y_samples)):
    ax.plot(
        np.asarray(time[:plot_samples]),
        y_sample[:plot_samples],
        alpha=0.6,
        linewidth=1.0,
        label="posterior synths" if i == 0 else None,
    )
ax.set_xlabel("Time [s]")
ax.set_ylabel("normalised amplitude")
ax.set_title("Audio implied by posterior samples")
ax.grid(True)
ax.legend()
fig.tight_layout()
show_if_interactive()


# %% [markdown]
# ## Takeaway
#
# A point estimator would return one excitation position. The conditional flow
# returns a **set of plausible explanations**:
#
# $$
# q_\phi(x_\mathrm{pluck}, A_\mathrm{pluck} \mid y).
# $$
#
# That distinction matters when the observation is partial. In this example,
# modal magnitudes can make symmetric pluck positions hard to distinguish, and
# amplitude trades against the overall modal level. The flow exposes that
# ambiguity instead of hiding it behind one best-fit value.
