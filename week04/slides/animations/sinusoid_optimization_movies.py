from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm

jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    ROOT / "week04" / "slides" / "animations" / "media" / "videos" / "sinusoid_optimization"
)

SAMPLE_RATE = 8000
DURATION = 0.5
N_STEPS = int(SAMPLE_RATE * DURATION)
T = jnp.arange(N_STEPS) / SAMPLE_RATE

AMPLITUDE = 0.8
GAMMA = 3.0
TARGET_FREQUENCY_HZ = 440.0
TARGET_PHASE = 0.6

FREQUENCY_RANGE = (340.0, 452.0)
PHASE_RANGE = (-np.pi, np.pi)

CASES = {
    "near_start": {
        "title": "Near initialisation",
        "frequency_hz": 435.0,
        "phase": 0.0,
        "n_updates": 700,
        "learning_rate": 0.1,
        "filename": "near_start.mp4",
    },
    "far_start": {
        "title": "Far initialisation",
        "frequency_hz": 350.0,
        "phase": 0.0,
        "n_updates": 700,
        "learning_rate": 0.1,
        "filename": "far_start.mp4",
    },
}


def damped_sinusoid(t, frequency_hz, phase):
    return AMPLITUDE * jnp.exp(-GAMMA * t) * jnp.sin(
        2.0 * jnp.pi * frequency_hz * t + phase
    )


Y_TARGET = damped_sinusoid(T, TARGET_FREQUENCY_HZ, TARGET_PHASE)


def wrap_phase(phase):
    return (phase + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def loss_fn(params):
    y_hat = damped_sinusoid(T, params["frequency_hz"], params["phase"])
    return jnp.mean((y_hat - Y_TARGET) ** 2)


def trace_optimization(case):
    params = {
        "frequency_hz": jnp.asarray(case["frequency_hz"]),
        "phase": jnp.asarray(case["phase"]),
    }
    optimizer = optax.adam(case["learning_rate"])
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params["phase"] = wrap_phase(params["phase"])
        return params, opt_state, loss_value, grads

    rows = []
    for step in range(case["n_updates"] + 1):
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        rows.append(
            {
                "step": step,
                "frequency_hz": float(params["frequency_hz"]),
                "phase": float(params["phase"]),
                "loss": float(loss_value),
                "grad_frequency_hz": float(grads["frequency_hz"]),
                "grad_phase": float(grads["phase"]),
            }
        )

        if step < case["n_updates"]:
            params, opt_state, _, _ = train_step(params, opt_state)

    return rows


def loss_surface(n_frequency=210, n_phase=180):
    frequency = np.linspace(*FREQUENCY_RANGE, n_frequency)
    phase = np.linspace(*PHASE_RANGE, n_phase)
    t_np = np.asarray(T)
    target_np = np.asarray(Y_TARGET)
    surface = np.empty((n_phase, n_frequency))

    for col, f in enumerate(frequency):
        y_grid = AMPLITUDE * np.exp(-GAMMA * t_np)[None, :] * np.sin(
            2.0 * np.pi * f * t_np[None, :] + phase[:, None]
        )
        surface[:, col] = np.mean((y_grid - target_np[None, :]) ** 2, axis=1)

    return frequency, phase, np.maximum(surface, 1e-10)


def synth_np(frequency_hz, phase):
    t_np = np.asarray(T)
    return AMPLITUDE * np.exp(-GAMMA * t_np) * np.sin(
        2.0 * np.pi * frequency_hz * t_np + phase
    )


def render_movie(case_name, output_dir, fps=30, n_frames=180, dpi=140):
    case = CASES[case_name]
    output_path = output_dir / case["filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = trace_optimization(case)
    frequency_grid, phase_grid, surface = loss_surface()
    frame_indices = np.unique(
        np.linspace(0, len(rows) - 1, min(n_frames, len(rows)), dtype=int)
    )

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#101114")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.35, wspace=0.25)
    ax_surface = fig.add_subplot(gs[:, 0])
    ax_wave = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, 1])

    for ax in (ax_surface, ax_wave, ax_loss):
        ax.set_facecolor("#15171c")
        ax.tick_params(colors="#d7dae0", labelsize=9, which="both")
        for spine in ax.spines.values():
            spine.set_color("#555a66")

    levels = np.geomspace(surface.min(), surface.max(), 32)
    ax_surface.contourf(
        frequency_grid,
        phase_grid,
        surface,
        levels=levels,
        norm=LogNorm(vmin=surface.min(), vmax=surface.max()),
        cmap="magma",
    )
    ax_surface.contour(
        frequency_grid,
        phase_grid,
        surface,
        levels=np.geomspace(1e-5, surface.max(), 13),
        colors="#ffffff",
        alpha=0.18,
        linewidths=0.5,
    )
    ax_surface.scatter(
        [TARGET_FREQUENCY_HZ],
        [TARGET_PHASE],
        marker="*",
        s=160,
        color="#87f7b5",
        edgecolor="#101114",
        linewidth=0.8,
        label="target",
        zorder=5,
    )
    ax_surface.set_xlim(FREQUENCY_RANGE)
    ax_surface.set_ylim(PHASE_RANGE)
    ax_surface.set_xlabel("frequency [Hz]", color="#f2f3f5")
    ax_surface.set_ylabel("phase [rad]", color="#f2f3f5")
    ax_surface.set_title("Waveform MSE surface", color="#f2f3f5", pad=12)

    (path_line,) = ax_surface.plot([], [], color="#6ee7ff", linewidth=2.2)
    current_point = ax_surface.scatter([], [], s=64, color="#6ee7ff", edgecolor="#101114", zorder=6)
    gradient_arrow = ax_surface.quiver(
        [],
        [],
        [],
        [],
        color="#ffffff",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        zorder=7,
    )

    t_np = np.asarray(T)
    y_target_np = np.asarray(Y_TARGET)
    time_limit = 0.1
    wave_mask = t_np <= time_limit
    ax_wave.plot(t_np[wave_mask], y_target_np[wave_mask], color="#87f7b5", linewidth=2.0, label="target")
    (wave_line,) = ax_wave.plot([], [], color="#6ee7ff", linewidth=1.7, label="current")
    ax_wave.set_xlim(0.0, time_limit)
    ax_wave.set_ylim(-0.95, 0.95)
    ax_wave.set_xlabel("time [s]", color="#f2f3f5")
    ax_wave.set_ylabel("amplitude", color="#f2f3f5")
    ax_wave.set_title("Current waveform", color="#f2f3f5", pad=10)
    ax_wave.grid(color="#ffffff", alpha=0.08)
    ax_wave.legend(facecolor="#15171c", edgecolor="#555a66", labelcolor="#f2f3f5", loc="upper right")

    steps = np.asarray([row["step"] for row in rows])
    losses = np.asarray([row["loss"] for row in rows])
    ax_loss.semilogy(steps, losses, color="#555a66", linewidth=1.0, alpha=0.7)
    (loss_line,) = ax_loss.semilogy([], [], color="#6ee7ff", linewidth=2.0)
    loss_point = ax_loss.scatter([], [], s=42, color="#6ee7ff", edgecolor="#101114", zorder=5)
    ax_loss.set_xlim(0, rows[-1]["step"])
    ax_loss.set_ylim(max(losses.min() * 0.5, 1e-10), losses.max() * 1.35)
    ax_loss.set_xlabel("update", color="#f2f3f5")
    ax_loss.set_ylabel("MSE loss", color="#f2f3f5")
    ax_loss.set_title("Optimisation history", color="#f2f3f5", pad=10)
    ax_loss.grid(color="#ffffff", alpha=0.08)
    ax_loss.tick_params(colors="#d7dae0", labelsize=9, which="both")
    ax_loss.yaxis.get_offset_text().set_color("#d7dae0")

    title = fig.suptitle("", color="#f2f3f5", fontsize=16, y=0.98)
    readout = ax_surface.text(
        0.03,
        0.03,
        "",
        transform=ax_surface.transAxes,
        color="#f2f3f5",
        fontsize=10,
        bbox={"facecolor": "#101114", "edgecolor": "#555a66", "alpha": 0.86, "pad": 6},
    )

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": f"Sinusoid optimisation - {case['title']}"},
        codec="libx264",
        bitrate=2600,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    with writer.saving(fig, output_path, dpi=dpi):
        for idx in frame_indices:
            row = rows[int(idx)]
            upto = rows[: int(idx) + 1]
            freq_path = [item["frequency_hz"] for item in upto]
            phase_path = [item["phase"] for item in upto]
            path_line.set_data(freq_path, phase_path)
            current_point.set_offsets([[row["frequency_hz"], row["phase"]]])

            grad = np.asarray([row["grad_frequency_hz"], row["grad_phase"]])
            direction = -grad
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * np.asarray([7.0, 0.45])
            gradient_arrow.set_offsets([[row["frequency_hz"], row["phase"]]])
            gradient_arrow.set_UVC([direction[0]], [direction[1]])

            y_current = synth_np(row["frequency_hz"], row["phase"])
            wave_line.set_data(t_np[wave_mask], y_current[wave_mask])

            loss_line.set_data(steps[: int(idx) + 1], losses[: int(idx) + 1])
            loss_point.set_offsets([[row["step"], row["loss"]]])

            title.set_text(
                f"{case['title']}: gradient descent on a differentiable sinusoid"
            )
            readout.set_text(
                "step: {step:4d}\n"
                "loss: {loss:.3e}\n"
                "f:    {frequency_hz:7.2f} Hz\n"
                "phi:  {phase:7.3f} rad\n"
                "|grad|: {grad_norm:.3e}".format(
                    grad_norm=np.linalg.norm(grad),
                    **row,
                )
            )
            writer.grab_frame()

    plt.close(fig)
    final = rows[-1]
    print(
        f"{case_name}: wrote {output_path} | "
        f"final loss={final['loss']:.3e}, "
        f"f={final['frequency_hz']:.3f} Hz, phase={final['phase']:.3f}"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Render optimisation movies for the Week 4 sinusoid demo."
    )
    parser.add_argument(
        "--case",
        choices=[*CASES.keys(), "all"],
        default="all",
        help="Which initialization to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for MP4 outputs.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    case_names = CASES.keys() if args.case == "all" else [args.case]
    for case_name in case_names:
        render_movie(
            case_name,
            output_dir=args.output_dir,
            fps=args.fps,
            n_frames=args.frames,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
