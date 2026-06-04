from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.animation import FFMpegWriter

jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "week04"
    / "slides"
    / "animations"
    / "media"
    / "videos"
    / "complex_frequency_optimization"
)

SAMPLE_RATE = 8000
DURATION = 0.5
N_STEPS = int(SAMPLE_RATE * DURATION)
DT = 1.0 / SAMPLE_RATE
N = jnp.arange(N_STEPS)
T = jnp.arange(N_STEPS) * DT

AMPLITUDE = 0.8
GAMMA = 3.0
PHASE = 0.6
TARGET_FREQUENCY_HZ = 440.0
LEARNING_RATE = 0.1
N_UPDATES = 700

FREQUENCY_RANGE = (345.0, 445.0)

CASES = {
    "near": {
        "label": "near start",
        "frequency_hz": 438.0,
        "color": "#6ee7ff",
    },
    "far": {
        "label": "far start",
        "frequency_hz": 350.0,
        "color": "#ffb86b",
    },
}


def complex_modal_synth(frequency_hz):
    pole = -GAMMA + 1j * 2.0 * jnp.pi * frequency_hz
    a = jnp.exp(pole * DT)
    z0 = AMPLITUDE * jnp.exp(1j * PHASE)
    z = z0 * a**N
    return z.real


Y_TARGET = complex_modal_synth(TARGET_FREQUENCY_HZ)


def loss_fn(frequency_hz):
    y_hat = complex_modal_synth(frequency_hz)
    return jnp.mean((y_hat - Y_TARGET) ** 2)


def trace_frequency(initial_frequency_hz):
    frequency_hz = jnp.asarray(initial_frequency_hz)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(frequency_hz)

    @jax.jit
    def train_step(frequency_hz, opt_state):
        loss_value, grad = jax.value_and_grad(loss_fn)(frequency_hz)
        updates, opt_state = optimizer.update(grad, opt_state, frequency_hz)
        frequency_hz = optax.apply_updates(frequency_hz, updates)
        return frequency_hz, opt_state, loss_value, grad

    rows = []
    for update in range(N_UPDATES + 1):
        loss_value, grad = jax.value_and_grad(loss_fn)(frequency_hz)
        rows.append(
            {
                "step": update,
                "frequency_hz": float(frequency_hz),
                "loss": float(loss_value),
                "grad_frequency_hz": float(grad),
            }
        )

        if update < N_UPDATES:
            frequency_hz, opt_state, _, _ = train_step(frequency_hz, opt_state)

    return rows


def compute_loss_curve(n_points=1400):
    frequency = np.linspace(*FREQUENCY_RANGE, n_points)
    loss_and_grad = jax.vmap(jax.value_and_grad(loss_fn))(jnp.asarray(frequency))
    loss = np.asarray(loss_and_grad[0])
    grad = np.asarray(loss_and_grad[1])
    return frequency, loss, grad


def synth_np(frequency_hz):
    return np.asarray(complex_modal_synth(frequency_hz))


def set_dark_axis(ax):
    ax.set_facecolor("#15171c")
    ax.tick_params(colors="#d7dae0", labelsize=9, which="both")
    for spine in ax.spines.values():
        spine.set_color("#555a66")


def render_movie(output_dir, fps=30, n_frames=360, dpi=140):
    output_path = output_dir / "single_frequency_comparison.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traces = {
        case_name: trace_frequency(case["frequency_hz"])
        for case_name, case in CASES.items()
    }
    frequency_grid, loss_grid, _ = compute_loss_curve()
    frame_indices = np.unique(
        np.linspace(0, N_UPDATES, min(n_frames, N_UPDATES + 1), dtype=int)
    )

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#101114")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.0], hspace=0.34, wspace=0.24)
    ax_loss_curve = fig.add_subplot(gs[0, 0])
    ax_gradient = fig.add_subplot(gs[0, 1])
    ax_near_wave = fig.add_subplot(gs[1, 0])
    ax_far_wave = fig.add_subplot(gs[1, 1])

    for ax in (ax_loss_curve, ax_gradient, ax_near_wave, ax_far_wave):
        set_dark_axis(ax)

    ax_loss_curve.plot(frequency_grid, loss_grid, color="#d7dae0", linewidth=1.15)
    ax_loss_curve.axvline(
        TARGET_FREQUENCY_HZ,
        color="#87f7b5",
        linestyle=":",
        linewidth=1.5,
        label="target",
    )
    ax_loss_curve.set_xlim(FREQUENCY_RANGE)
    ax_loss_curve.set_ylim(-0.005, float(loss_grid.max()) * 1.08)
    ax_loss_curve.set_xlabel("frequency [Hz]", color="#f2f3f5")
    ax_loss_curve.set_ylabel("MSE loss", color="#f2f3f5")
    ax_loss_curve.set_title("Loss for one learnable frequency", color="#f2f3f5", pad=10)
    ax_loss_curve.grid(color="#ffffff", alpha=0.08)

    loss_markers = {}
    tangent_lines = {}
    readouts = {}
    for offset, (case_name, case) in enumerate(CASES.items()):
        color = case["color"]
        (marker,) = ax_loss_curve.plot(
            [],
            [],
            marker="o",
            markersize=7,
            markeredgecolor="#101114",
            color=color,
            label=case["label"],
            linestyle="",
            zorder=6,
        )
        (tangent,) = ax_loss_curve.plot([], [], color=color, linewidth=2.0, alpha=0.8)
        text = ax_loss_curve.text(
            0.03,
            0.48 - 0.24 * offset,
            "",
            transform=ax_loss_curve.transAxes,
            color="#f2f3f5",
            fontsize=8.5,
            va="top",
            bbox={
                "facecolor": "#101114",
                "edgecolor": color,
                "alpha": 0.86,
                "pad": 5,
            },
        )
        loss_markers[case_name] = marker
        tangent_lines[case_name] = tangent
        readouts[case_name] = text

    ax_loss_curve.text(
        TARGET_FREQUENCY_HZ - 1.2,
        float(loss_grid.max()) * 0.94,
        "target",
        color="#87f7b5",
        fontsize=8.5,
        ha="right",
        va="top",
    )

    steps = np.arange(N_UPDATES + 1)
    for case_name, case in CASES.items():
        grad = np.asarray([row["grad_frequency_hz"] for row in traces[case_name]])
        ax_gradient.plot(
            steps,
            grad,
            color=case["color"],
            linewidth=1.0,
            alpha=0.35,
        )
    gradient_lines = {}
    gradient_points = {}
    for case_name, case in CASES.items():
        color = case["color"]
        (line,) = ax_gradient.plot([], [], color=color, linewidth=2.0, label=case["label"])
        point = ax_gradient.scatter([], [], s=44, color=color, edgecolor="#101114", zorder=5)
        gradient_lines[case_name] = line
        gradient_points[case_name] = point

    ax_gradient.axhline(0.0, color="#d7dae0", linewidth=0.9, alpha=0.45)
    ax_gradient.set_xlim(0, N_UPDATES)
    ax_gradient.set_yscale("symlog", linthresh=1e-5)
    ax_gradient.set_ylim(-0.18, 0.18)
    ax_gradient.set_xlabel("update", color="#f2f3f5")
    ax_gradient.set_ylabel(r"$dJ/df$", color="#f2f3f5")
    ax_gradient.set_title("Gradient through the complex recursion", color="#f2f3f5", pad=10)
    ax_gradient.grid(color="#ffffff", alpha=0.08)
    ax_gradient.legend(
        facecolor="#15171c",
        edgecolor="#555a66",
        labelcolor="#f2f3f5",
        loc="upper right",
    )

    t_np = np.asarray(T)
    target_np = np.asarray(Y_TARGET)
    wave_mask = t_np <= 0.1
    wave_axes = {"near": ax_near_wave, "far": ax_far_wave}
    wave_lines = {}
    for case_name, ax in wave_axes.items():
        case = CASES[case_name]
        ax.plot(
            t_np[wave_mask],
            target_np[wave_mask],
            color="#87f7b5",
            linewidth=2.0,
            label="target",
        )
        (line,) = ax.plot(
            [],
            [],
            color=case["color"],
            linewidth=1.8,
            label="current",
        )
        wave_lines[case_name] = line
        ax.set_xlim(0.0, 0.1)
        ax.set_ylim(-0.95, 0.95)
        ax.set_xlabel("time [s]", color="#f2f3f5")
        ax.set_ylabel("amplitude", color="#f2f3f5")
        ax.set_title(f"{case['label']}: current result", color="#f2f3f5", pad=10)
        ax.grid(color="#ffffff", alpha=0.08)
        ax.legend(
            facecolor="#15171c",
            edgecolor="#555a66",
            labelcolor="#f2f3f5",
            loc="upper right",
        )

    fig.suptitle(
        "Optimising one frequency in a complex modal recursion",
        color="#f2f3f5",
        fontsize=16,
        y=0.98,
    )

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": "Complex frequency optimisation comparison"},
        codec="libx264",
        bitrate=2800,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    with writer.saving(fig, output_path, dpi=dpi):
        for idx in frame_indices:
            for case_name, case in CASES.items():
                rows = traces[case_name]
                row = rows[int(idx)]
                color = case["color"]

                f = row["frequency_hz"]
                loss = row["loss"]
                grad = row["grad_frequency_hz"]

                loss_markers[case_name].set_data([f], [loss])
                f_tangent = np.linspace(f - 2.0, f + 2.0, 24)
                tangent = loss + grad * (f_tangent - f)
                tangent_lines[case_name].set_data(f_tangent, tangent)
                readouts[case_name].set_text(
                    "{label}  step {step:4d}\n"
                    "f={frequency_hz:7.3f} Hz   J={loss:.2e}   dJ/df={grad_frequency_hz:.2e}".format(
                        label=case["label"],
                        **row,
                    )
                )

                gradients = np.asarray(
                    [item["grad_frequency_hz"] for item in rows[: int(idx) + 1]]
                )
                gradient_lines[case_name].set_data(steps[: int(idx) + 1], gradients)
                gradient_points[case_name].set_offsets([[row["step"], grad]])

                y_current = synth_np(f)
                wave_lines[case_name].set_data(t_np[wave_mask], y_current[wave_mask])

            writer.grab_frame()

    plt.close(fig)

    print(f"wrote {output_path}")
    for case_name, rows in traces.items():
        final = rows[-1]
        print(
            f"{case_name}: final loss={final['loss']:.3e}, "
            f"f={final['frequency_hz']:.3f} Hz, "
            f"grad={final['grad_frequency_hz']:.3e}"
        )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Render the Week 4 complex-recursion frequency optimisation movie."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    render_movie(
        output_dir=args.output_dir,
        fps=args.fps,
        n_frames=args.frames,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
