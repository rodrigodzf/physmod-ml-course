from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "images" / "string_boundary_conditions.mp4"
)
STRING_LENGTH = 1.0
INITIAL_CENTRE = 0.22
PULSE_HALF_WIDTH = 0.14
BACKGROUND = "#101114"
FOREGROUND = "#f2f3f5"
INCIDENT_COLOUR = "#69aafc"
REFLECTED_COLOUR = "#fb907f"


def raised_cosine(offset):
    normalised_offset = np.asarray(offset) / PULSE_HALF_WIDTH
    return np.where(
        np.abs(normalised_offset) < 1.0,
        0.5 * (1.0 + np.cos(np.pi * normalised_offset)),
        0.0,
    )


def pulse_components(position, centre, reflection_coefficient):
    incident = raised_cosine(position - centre)
    reflected = reflection_coefficient * raised_cosine(
        2.0 * STRING_LENGTH - position - centre
    )
    return incident, reflected


def make_figure(position):
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.8), facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.13, top=0.80, hspace=0.58)
    fig.suptitle(
        "Pulse reflection at a string boundary",
        color=FOREGROUND,
        fontsize=24,
        y=0.98,
    )
    fig.text(
        0.5,
        0.89,
        "The same raised-cosine pulse approaches both ends from the left",
        ha="center",
        color="#c5c9d1",
        fontsize=14,
    )
    stage = fig.text(0.5, 0.85, "", ha="center", color=FOREGROUND, fontsize=15)
    panels = []
    cases = [
        (
            -1,
            "Dirichlet: fixed end",
            r"$u(L,t)=0$",
            "Inverted reflection: phase shift of 180°",
            "#f4c968",
        ),
        (
            1,
            "Neumann: free end",
            r"$\partial u/\partial x\,(L,t)=0$",
            "Upright reflection: no phase shift",
            "#72d6c6",
        ),
    ]

    for ax, (coefficient, title, condition, explanation, colour) in zip(axes, cases):
        ax.set_facecolor(BACKGROUND)
        ax.set_xlim(-0.025, 1.055)
        ax.set_ylim(-1.35, 2.35)
        ax.set_axis_off()
        ax.axhline(0.0, xmin=0.023, xmax=0.949, color="#515761", linewidth=1.0)
        ax.text(
            0.0,
            1.13,
            title,
            transform=ax.transAxes,
            color=colour,
            fontsize=18,
            weight="bold",
        )
        ax.text(
            0.0, 1.00, explanation, transform=ax.transAxes, color="#c5c9d1", fontsize=12
        )
        ax.text(
            0.96,
            1.09,
            condition,
            transform=ax.transAxes,
            ha="right",
            color=FOREGROUND,
            fontsize=18,
        )

        (incident_line,) = ax.plot(
            position,
            np.zeros_like(position),
            color=INCIDENT_COLOUR,
            linewidth=2.2,
            linestyle="--",
            alpha=0.9,
        )
        (reflected_line,) = ax.plot(
            position,
            np.zeros_like(position),
            color=REFLECTED_COLOUR,
            linewidth=2.2,
            linestyle=":",
            alpha=0.9,
        )
        (string_line,) = ax.plot(
            position, np.zeros_like(position), color=FOREGROUND, linewidth=3.0
        )

        if coefficient == -1:
            ax.add_patch(
                Rectangle(
                    (1.006, -1.25),
                    0.025,
                    3.45,
                    facecolor="#30353d",
                    edgecolor="#737a86",
                    hatch="////",
                    linewidth=0.7,
                )
            )
        else:
            ax.plot(
                [STRING_LENGTH, STRING_LENGTH],
                [-1.25, 2.2],
                color="#737a86",
                linewidth=2.0,
            )

        (endpoint,) = ax.plot(
            [STRING_LENGTH],
            [0.0],
            marker="o",
            markersize=7,
            markerfacecolor=BACKGROUND,
            markeredgecolor=colour,
            markeredgewidth=2.2,
        )
        ax.text(
            STRING_LENGTH, -1.47, r"$x=L$", ha="center", color="#c5c9d1", fontsize=12
        )
        panels.append(
            (coefficient, incident_line, reflected_line, string_line, endpoint)
        )

    fig.legend(
        handles=[
            Line2D(
                [], [], color=FOREGROUND, linewidth=3.0, label="String displacement"
            ),
            Line2D(
                [],
                [],
                color=INCIDENT_COLOUR,
                linewidth=2.2,
                linestyle="--",
                label="Incident pulse",
            ),
            Line2D(
                [],
                [],
                color=REFLECTED_COLOUR,
                linewidth=2.2,
                linestyle=":",
                label="Reflected pulse",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        labelcolor=FOREGROUND,
        fontsize=12,
    )
    return fig, panels, stage


def render(output, duration, fps):
    output.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    position = np.linspace(0.0, STRING_LENGTH, 1001)
    fig, panels, stage = make_figure(position)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={
            "title": "Raised-cosine pulse reflection: Dirichlet and Neumann boundaries"
        },
        extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart"],
    )
    frame_count = round(duration * fps)
    hold_duration = min(0.75, duration / 8.0)

    with writer.saving(fig, str(output), dpi=100):
        for elapsed in np.linspace(0.0, duration, frame_count):
            progress = np.clip(
                (elapsed - hold_duration) / (duration - 2.0 * hold_duration), 0.0, 1.0
            )
            centre = INITIAL_CENTRE + 2.0 * (STRING_LENGTH - INITIAL_CENTRE) * progress

            if centre + PULSE_HALF_WIDTH < STRING_LENGTH:
                stage.set_text("Approaching the boundary")
            elif centre - PULSE_HALF_WIDTH <= STRING_LENGTH:
                stage.set_text("Reflection at the boundary")
            else:
                stage.set_text("Travelling back along the string")

            for (
                coefficient,
                incident_line,
                reflected_line,
                string_line,
                endpoint,
            ) in panels:
                incident, reflected = pulse_components(position, centre, coefficient)
                displacement = incident + reflected
                incident_line.set_ydata(incident)
                reflected_line.set_ydata(reflected)
                string_line.set_ydata(displacement)
                endpoint.set_ydata([displacement[-1]])

            writer.grab_frame()

    plt.close(fig)
    print(f"Created {output} ({frame_count} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Render a raised-cosine pulse reflecting at fixed and free string ends."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if (
        not np.isfinite(args.duration)
        or args.duration <= 0.0
        or args.fps <= 0
        or args.duration * args.fps < 2
    ):
        parser.error(
            "Choose a positive duration and frame rate producing at least two frames."
        )
    render(args.output, args.duration, args.fps)


if __name__ == "__main__":
    main()
