# %% [markdown]
# # Audio loss landscapes
#
# Continue from `differentiable_sinusoidal_synth.py`. Keep its damped sinusoid,
# target and frequency sweep, but change how we compare the waveforms.
# This short notebook is self-contained; it repeats the same synthesiser so
# importing the earlier notebook does not run all its fitting exercises.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


def damped_sinusoid(t, amplitude, gamma, frequency_hz, phase=0.0):
    return (
        amplitude
        * jnp.exp(-gamma * t)
        * jnp.sin(2.0 * jnp.pi * frequency_hz * t + phase)
    )


sample_rate = 8000
duration = 0.5
t = jnp.arange(int(sample_rate * duration)) / sample_rate
target_amplitude, target_gamma, target_frequency_hz = 0.8, 3.0, 440.0
y_target = damped_sinusoid(t, target_amplitude, target_gamma, target_frequency_hz)

# %% [markdown]
# ## 1. Three ways to compare the same sound
#
# **Waveform MSE:** compare samples, including phase.
#
# **MRSTFT:** at each resolution, add mean absolute magnitude error and mean
# absolute log-magnitude error; then average over resolutions. These settings
# define this particular MRSTFT loss, not a universal one.
#
# **Spectral OT:** use one STFT resolution and normalise each power spectrum
# to unit mass. On a shared frequency grid, exact Wasserstein-1 is
# $W_1(p,q)=\sum_k |\operatorname{CDF}_p[k]-\operatorname{CDF}_q[k]|\Delta f_k$.
# Average over frames. The units are Hz; global amplitude is discarded.
# This is W1, not squared W2, and needs no transport solver or regularisation.

# %%
window_sizes = (128, 512, 2048)
sot_window_size = 2048
log_epsilon = 1e-5


def magnitude_stft(signal, window_size):
    hop_size = window_size // 4
    frame_starts = jnp.arange(0, signal.shape[0] - window_size + 1, hop_size)
    frames = signal[frame_starts[:, None] + jnp.arange(window_size)]
    window = jnp.hanning(window_size)
    return jnp.abs(jnp.fft.rfft(frames * window, axis=-1)) / window.sum()


def waveform_mse_loss(prediction, target):
    return jnp.mean((prediction - target) ** 2)


def multiresolution_stft_loss(prediction_spectra, target_spectra):
    losses = [
        jnp.mean(jnp.abs(prediction - target))
        + jnp.mean(
            jnp.abs(jnp.log(prediction + log_epsilon) - jnp.log(target + log_epsilon))
        )
        for prediction, target in zip(prediction_spectra, target_spectra)
    ]
    return jnp.mean(jnp.stack(losses))


def spectral_ot_loss(prediction_magnitude, target_magnitude, frequency_bins_hz):
    def spectral_mass(magnitude):
        power = magnitude**2
        return power / jnp.maximum(power.sum(axis=-1, keepdims=True), 1e-30)

    cdf_difference = jnp.cumsum(
        spectral_mass(prediction_magnitude) - spectral_mass(target_magnitude), axis=-1
    )
    frame_distances = jnp.sum(
        jnp.abs(cdf_difference[..., :-1]) * jnp.diff(frequency_bins_hz), axis=-1
    )
    return jnp.mean(frame_distances)


# %% [markdown]
# ## 2. Sweep frequency and plot the landscapes
#
# Only frequency varies; amplitude and damping stay at their target values.
# For a live phase experiment, change `candidate_phase` and rerun from here.
# The target stays fixed. After changing STFT settings above, rerun from
# their cell onwards to rebuild the target spectra and compiled sweep.
# STFTs use complete Hann-windowed frames, 75% overlap and no padding.

# %%
candidate_phase = 0.0
frequency_grid = jnp.linspace(target_frequency_hz - 30, target_frequency_hz + 30, 601)
analysis_sizes = tuple(sorted(set(window_sizes + (sot_window_size,))))
target_spectra = {size: magnitude_stft(y_target, size) for size in analysis_sizes}
sot_frequency_bins_hz = jnp.fft.rfftfreq(sot_window_size, d=1 / sample_rate)


def losses_at_frequency(frequency_hz):
    prediction = damped_sinusoid(
        t, target_amplitude, target_gamma, frequency_hz, candidate_phase
    )
    spectra = {size: magnitude_stft(prediction, size) for size in analysis_sizes}
    return jnp.stack(
        [
            waveform_mse_loss(prediction, y_target),
            multiresolution_stft_loss(
                [spectra[size] for size in window_sizes],
                [target_spectra[size] for size in window_sizes],
            ),
            spectral_ot_loss(
                spectra[sot_window_size],
                target_spectra[sot_window_size],
                sot_frequency_bins_hz,
            ),
        ]
    )


landscapes = np.asarray(jax.jit(jax.vmap(losses_at_frequency))(frequency_grid))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharex=True)
for column, (ax, title, units) in enumerate(
    zip(axes, ("Waveform MSE", "MRSTFT", "Spectral OT (W1)"), ("MSE", "Loss", "Hz"))
):
    ax.plot(np.asarray(frequency_grid), landscapes[:, column])
    ax.axvline(target_frequency_hz, color="k", linestyle=":", label="target")
    ax.set(title=title, xlabel="Candidate frequency [Hz]", ylabel=units)
    ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# Compare the slopes and local minima, not the numerical heights: the losses
# have different units and scales. Try changing `window_sizes` or
# `log_epsilon`. SOT can also have finite-window and frequency-bin artefacts;
# a smooth-looking curve is not a proof of convexity.
#
# ## 3. Extra: spectral OT and string length
#
# For an ideal string with fixed tension $T$ and linear density $\mu$,
# $f(L)=\frac{1}{2L}\sqrt{T/\mu}=c/L$.
# Compare the SAME computed spectral W1 loss with frequency or string length
# as the candidate parameter. Only the fundamental frequency
# changes with length; amplitude and damping stay fixed in this example.

# %%
target_length = 0.65
linear_density = 2e-4
tension = linear_density * (2 * target_length * target_frequency_hz) ** 2


def frequency_from_length(length):
    return jnp.sqrt(tension / linear_density) / (2 * length)


def spectral_ot_at_frequency(frequency_hz):
    prediction = damped_sinusoid(
        t, target_amplitude, target_gamma, frequency_hz, candidate_phase
    )
    return spectral_ot_loss(
        magnitude_stft(prediction, sot_window_size),
        target_spectra[sot_window_size],
        sot_frequency_bins_hz,
    )


def spectral_ot_at_length(length):
    return spectral_ot_at_frequency(frequency_from_length(length))


length_grid = jnp.linspace(0.5 * target_length, 3 * target_length, 401)
extra_frequency_grid = jnp.linspace(target_frequency_hz / 3, 2 * target_frequency_hz, 401)
frequency_ot_landscape = jax.jit(jax.vmap(spectral_ot_at_frequency))(extra_frequency_grid)
length_ot_landscape = jax.jit(jax.vmap(spectral_ot_at_length))(length_grid)
fig_extra, axes_extra = plt.subplots(1, 2, figsize=(10, 3.3))
axes_extra[0].plot(extra_frequency_grid, frequency_ot_landscape)
axes_extra[0].axvline(target_frequency_hz, color="k", linestyle=":", label="target")
axes_extra[0].set(
    title="Computed spectral W1 vs frequency", xlabel="Frequency [Hz]", ylabel="W1 [Hz]"
)
axes_extra[1].plot(length_grid, length_ot_landscape)
axes_extra[1].axvline(target_length, color="k", linestyle=":", label="target")
axes_extra[1].set(
    title="Computed spectral W1 vs length", xlabel="String length [m]", ylabel="W1 [Hz]"
)
for ax in axes_extra:
    ax.legend()
fig_extra.tight_layout()
plt.show()

# %% [markdown]
# For an ideal moving spectral peak, W1 gives $|c/L-f_\star|$, which is
# concave for $L>L_\star$. The computed spectra approximate this behaviour.
# Changing to OT therefore does not guarantee convexity in string length.
# Non-convexity does not necessarily introduce additional minima.
#
# ## 4. Extra: noise and several sinusoids
#
# Keep each candidate clean, but add fixed white noise to the target.
# Compare one partial with a harmonic stack whose amplitudes decay as 1/h.
# SNR is measured against the clean waveform RMS: 20 dB is mild noise,
# 0 dB means equal signal and noise power. Edit the SNRs or partial counts.
# This is a controlled stress test, not a claim that many partials always
# make OT fail. Here the harmonic structure and amplitudes are known.

# %%
partial_counts = (1, 8)
noise_snrs_db = (20.0, 0.0)
noise = jnp.asarray(np.random.default_rng(0).normal(size=t.shape))
noise = noise / jnp.sqrt(jnp.mean(noise**2))


def harmonic_synth(fundamental_hz, n_partials):
    harmonics = jnp.arange(1, n_partials + 1)
    return damped_sinusoid(
        t[:, None], target_amplitude / harmonics, target_gamma,
        fundamental_hz * harmonics,
    ).sum(axis=-1)


fig_noise, axes_noise = plt.subplots(1, len(partial_counts), figsize=(10, 3.3), squeeze=False)
noise_landscapes = {}
for ax, n_partials in zip(axes_noise[0], partial_counts):
    assert float(frequency_grid[-1]) * n_partials < sample_rate / 2
    clean_target = harmonic_synth(target_frequency_hz, n_partials)
    signal_rms = jnp.sqrt(jnp.mean(clean_target**2))
    target_variants = [("Clean", clean_target)] + [
        (f"SNR {snr_db:g} dB", clean_target + signal_rms * 10 ** (-snr_db / 20) * noise)
        for snr_db in noise_snrs_db
    ]

    def noisy_target_losses(frequency_hz, reference_spectra):
        spectrum = magnitude_stft(harmonic_synth(frequency_hz, n_partials), sot_window_size)
        return jax.vmap(lambda reference: spectral_ot_loss(
            spectrum, reference, sot_frequency_bins_hz
        ))(reference_spectra)

    reference_spectra = jnp.stack([
        magnitude_stft(signal, sot_window_size) for _, signal in target_variants
    ])
    values = np.asarray(jax.jit(jax.vmap(
        noisy_target_losses, in_axes=(0, None)
    ))(frequency_grid, reference_spectra))
    noise_landscapes[n_partials] = values
    for column, (label, _) in enumerate(target_variants):
        best = np.argmin(values[:, column])
        minimum = values[best, column]
        line, = ax.plot(frequency_grid, values[:, column] - minimum, label=label)
        ax.scatter(frequency_grid[best], 0, color=line.get_color(), marker="x")
        print(
            f"{n_partials} partial(s), {label}: "
            f"best on grid = {float(frequency_grid[best]):.1f} Hz; W1 = {minimum:.2f} Hz"
        )
    ax.axvline(target_frequency_hz, color="k", linestyle=":", label="true frequency")
    ax.set(
        title=f"{n_partials} partial" + ("s" if n_partials != 1 else ""),
        xlabel="Candidate fundamental [Hz]", ylabel="W1 minus grid minimum [Hz]",
    )
    ax.legend()
fig_noise.tight_layout()
plt.show()

# %% [markdown]
# Crosses mark the best candidates ON THIS GRID; an edge minimum suggests
# widening the sweep. Subtracting each curve's minimum makes their shapes
# easier to compare without changing slopes or minima; raw minima are printed.
# We reuse the same noise at every candidate frequency.
# Noise adds spectral mass that the clean model cannot reproduce. OT still
# tries to transport that mass, which can raise the loss floor or bias pitch.
# Harmonic amplitudes also determine how much mass each partial contributes.
# Neither a higher loss floor nor more partials alone proves a worse estimate:
# inspect where the minimum moves. Try a different noise seed as well.
#
# References: [MRSTFT configuration](https://doi.org/10.1109/LSP.2023.3333205),
# [spectral OT](https://arxiv.org/abs/2312.14507),
# [W1 as a CDF integral](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).
