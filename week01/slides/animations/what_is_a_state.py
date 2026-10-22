from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from manim import (
    BLUE,
    GOLD,
    GREEN,
    LEFT,
    UP,
    Arrow,
    Axes,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    Line,
    MarkupText,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
    tempconfig,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "images" / "what_is_a_state.mp4"
BACKGROUND = "#191919"
MUTED = "#bfc4cc"
ANGULAR_FREQUENCY = 2.0
INITIAL_DISPLACEMENT = 0.45
INITIAL_VELOCITY = 1.0
SAMPLE_POSITIONS = np.linspace(0.125, 0.875, 7)
MASS_PER_GRID_POINT = 0.7


def displacement(position, time, direction=1):
    amplitude = INITIAL_DISPLACEMENT * np.cos(ANGULAR_FREQUENCY * time)
    amplitude += (
        direction
        * INITIAL_VELOCITY
        / ANGULAR_FREQUENCY
        * np.sin(ANGULAR_FREQUENCY * time)
    )
    return amplitude * np.sin(np.pi * position)


def velocity(position, time, direction=1):
    amplitude = (
        -ANGULAR_FREQUENCY * INITIAL_DISPLACEMENT * np.sin(ANGULAR_FREQUENCY * time)
    )
    amplitude += direction * INITIAL_VELOCITY * np.cos(ANGULAR_FREQUENCY * time)
    return amplitude * np.sin(np.pi * position)


def caption(text, vertical_position, font_size=28, colour="#ffffff"):
    return Text(text, font_size=font_size, color=colour).move_to(
        [0, vertical_position, 0]
    )


def string_axes(centre, width=5.6, height=2.2):
    return Axes(
        x_range=[0, 1, 0.25],
        y_range=[-1.1, 1.1, 0.5],
        x_length=width,
        y_length=height,
        axis_config={"include_tip": False, "stroke_width": 1, "color": MUTED},
        x_axis_config={"include_ticks": False},
        y_axis_config={"include_ticks": False, "stroke_opacity": 0},
    ).move_to(centre)


def field_curve(axes, field, colour):
    return axes.plot(field, x_range=[0, 1, 0.01], color=colour, stroke_width=4)


def state_vector(
    variables=("w(t)", "v(t)"), state_label="h(t) =", colours=(GOLD, BLUE)
):
    components = VGroup(
        *[
            MarkupText(variable, font_size=40, color=colour, weight="BOLD")
            for variable, colour in zip(variables, colours)
        ]
    ).arrange(-UP, buff=0.35)
    brackets = VGroup()
    half_height = components.height / 2 + 0.15
    for direction in (-1, 1):
        horizontal = direction * (components.width / 2 + 0.18)
        inward = horizontal - direction * 0.16
        brackets.add(
            Line([inward, half_height, 0], [horizontal, half_height, 0]),
            Line([horizontal, half_height, 0], [horizontal, -half_height, 0]),
            Line([horizontal, -half_height, 0], [inward, -half_height, 0]),
        )
    vector = VGroup(components, brackets)
    label = MarkupText(state_label, font_size=40).next_to(vector, LEFT, buff=0.25)
    return VGroup(label, vector)


def two_step_fields():
    positions = np.linspace(0, 1, len(SAMPLE_POSITIONS) + 2)
    spacing = positions[1] - positions[0]
    time_step = 0.15
    current_time = 1.2
    previous = displacement(positions, current_time - time_step)
    current = displacement(positions, current_time)
    wave_speed = ANGULAR_FREQUENCY / np.pi
    courant_number = wave_speed * time_step / spacing
    following = np.zeros_like(current)
    following[1:-1] = (
        2 * current[1:-1]
        - previous[1:-1]
        + courant_number**2 * (current[:-2] - 2 * current[1:-1] + current[2:])
    )
    return positions, previous, current, following


class WhatIsAState(Scene):
    def construct(self):
        self.same_shape_different_motion()
        self.collect_the_state()
        self.two_displacement_steps()
        self.displacement_and_momentum()
        self.state_choices()

    def same_shape_different_motion(self):
        title = caption("What is a state?", 3.35, font_size=42)
        question = caption("Pause the simulation. What must we save to continue?", 2.55)
        time = ValueTracker(0)
        axes = [string_axes([-3.25, 0.1, 0]), string_axes([3.25, 0.1, 0])]
        directions = (1, -1)
        curves = VGroup(
            *[
                always_redraw(
                    lambda axis=axis, direction=direction: field_curve(
                        axis,
                        lambda x: displacement(x, time.get_value(), direction),
                        GOLD,
                    )
                )
                for axis, direction in zip(axes, directions)
            ]
        )
        endpoints = VGroup(
            *[Dot(axis.c2p(x, 0), radius=0.055) for axis in axes for x in (0, 1)]
        )
        labels = VGroup(
            Text("String A", font_size=28).move_to([-3.25, 1.65, 0]),
            Text("String B", font_size=28).move_to([3.25, 1.65, 0]),
        )
        observation = caption(
            "Exactly the same displacement at the paused instant", -1.7
        )
        self.play(FadeIn(title), FadeIn(question))
        self.play(
            FadeIn(VGroup(*axes)), FadeIn(curves), FadeIn(endpoints), FadeIn(labels)
        )
        self.play(FadeIn(observation))
        self.wait(2)

        arrows = VGroup(
            *[
                Arrow(
                    axis.c2p(x, displacement(x, 0)),
                    axis.c2p(x, displacement(x, 0) + 0.55 * velocity(x, 0, direction)),
                    color=BLUE,
                    buff=0,
                    stroke_width=3,
                    max_tip_length_to_length_ratio=0.28,
                )
                for axis, direction in zip(axes, directions)
                for x in np.linspace(0.15, 0.85, 5)
            ]
        )
        motion_labels = VGroup(
            Text("Moving upwards", font_size=26, color=BLUE).move_to([-3.25, -2.3, 0]),
            Text("Moving downwards", font_size=26, color=BLUE).move_to([3.25, -2.3, 0]),
        )
        self.play(FadeIn(arrows), FadeIn(motion_labels))
        self.wait(3)
        self.play(
            FadeOut(arrows),
            FadeOut(question),
            FadeOut(observation),
            FadeOut(motion_labels),
        )
        resumed = MarkupText(
            "Resume: different velocities <i>v</i> give different futures", font_size=28
        ).move_to([0, 2.55, 0])
        self.play(FadeIn(resumed))
        self.play(time.animate.set_value(0.6), run_time=3, rate_func=linear)
        self.wait(1)
        self.play(time.animate.set_value(2.1), run_time=4, rate_func=linear)
        self.wait(1)
        curves.clear_updaters(recursive=True)
        self.play(FadeOut(Group(*self.mobjects)))

    def collect_the_state(self):
        title = caption("Save displacement and velocity", 3.35, font_size=42)
        subtitle = caption("Both fields across the whole string", 2.65, colour=MUTED)
        position_axes = string_axes([-2.6, 1.0, 0], width=6.0, height=1.7)
        velocity_axes = string_axes([-2.6, -1.0, 0], width=6.0, height=1.7)
        position_curve = field_curve(position_axes, lambda x: displacement(x, 0), GOLD)
        velocity_curve = field_curve(velocity_axes, lambda x: velocity(x, 0), BLUE)
        field_labels = VGroup(
            Text("Displacement w(x,t)", font_size=25, color=GOLD).move_to(
                [-2.6, 2.05, 0]
            ),
            Text("Velocity v(x,t)", font_size=25, color=BLUE).move_to([-2.6, 0.1, 0]),
        )
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(
            FadeIn(VGroup(position_axes, velocity_axes)),
            FadeIn(VGroup(position_curve, velocity_curve)),
            FadeIn(field_labels),
        )
        self.wait(2)
        samples = VGroup(
            *[
                Dot(axis.c2p(x, field(x, 0)), radius=0.065, color=colour)
                for axis, field, colour in (
                    (position_axes, displacement, GOLD),
                    (velocity_axes, velocity, BLUE),
                )
                for x in SAMPLE_POSITIONS
            ]
        )
        sample_caption = caption(
            "Sample both fields at N interior positions", -2.25, font_size=27
        )
        self.play(FadeIn(samples), FadeIn(sample_caption))
        self.wait(2)
        vector = state_vector().move_to([3.35, 0.5, 0])
        vector_label = Text("One state vector", font_size=27).move_to([3.35, 1.95, 0])
        vector_size = Text("2N values", font_size=26, color=MUTED).move_to(
            [3.35, -0.65, 0]
        )
        self.play(FadeIn(vector), FadeIn(vector_label), FadeIn(vector_size))
        self.wait(3)
        definition = caption(
            "The state is the information needed to continue the model",
            -2.9,
            font_size=28,
        )
        self.play(FadeIn(definition))
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))

    def two_displacement_steps(self):
        title = caption("Or save two displacement snapshots", 3.35, font_size=40)
        subtitle = caption(
            "Discrete time: a scheme using two consecutive steps", 2.6, colour=MUTED
        )
        positions, previous, current, following = two_step_fields()
        previous_axes = string_axes([-3.25, 0.75, 0], height=1.8)
        current_axes = string_axes([3.25, 0.75, 0], height=1.8)
        previous_curve = field_curve(
            previous_axes, lambda x: np.interp(x, positions, previous), MUTED
        )
        current_curve = field_curve(
            current_axes, lambda x: np.interp(x, positions, current), GOLD
        )
        labels = VGroup(
            MarkupText("Previous: w<sub>n-1</sub>", font_size=28, color=MUTED).move_to(
                [-3.25, 1.9, 0]
            ),
            MarkupText("Current: w<sub>n</sub>", font_size=28, color=GOLD).move_to(
                [3.25, 1.9, 0]
            ),
        )
        vector = state_vector(
            ("w<sub>n</sub>", "w<sub>n-1</sub>"), "h<sub>n</sub> =", (GOLD, MUTED)
        ).move_to([0, -1.1, 0])
        information = caption(
            "Their difference carries information about motion", -2.65
        )
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(
            FadeIn(VGroup(previous_axes, current_axes)),
            FadeIn(VGroup(previous_curve, current_curve)),
            FadeIn(labels),
        )
        self.play(FadeIn(vector), FadeIn(information))
        self.wait(3)
        next_previous = field_curve(
            previous_axes, lambda x: np.interp(x, positions, current), MUTED
        )
        next_current = field_curve(
            current_axes, lambda x: np.interp(x, positions, following), GOLD
        )
        next_labels = VGroup(
            MarkupText("Previous: w<sub>n</sub>", font_size=28, color=MUTED).move_to(
                [-3.25, 1.9, 0]
            ),
            MarkupText("Current: w<sub>n+1</sub>", font_size=28, color=GOLD).move_to(
                [3.25, 1.9, 0]
            ),
        )
        next_vector = state_vector(
            ("w<sub>n+1</sub>", "w<sub>n</sub>"), "h<sub>n+1</sub> =", (GOLD, MUTED)
        ).move_to(vector.get_center())
        update = caption(
            "Compute the next displacement and keep the current one", -2.65
        )
        self.play(
            previous_curve.animate.become(next_previous),
            current_curve.animate.become(next_current),
            labels.animate.become(next_labels),
            vector.animate.become(next_vector),
            information.animate.become(update),
            run_time=2,
        )
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))

    def displacement_and_momentum(self):
        title = caption("Or save displacement and momentum", 3.35, font_size=40)
        subtitle = caption("For the spatially discretised string", 2.65, colour=MUTED)
        position_axes = string_axes([-2.6, 1.0, 0], width=6.0, height=1.7)
        momentum_axes = string_axes([-2.6, -1.0, 0], width=6.0, height=1.7)
        position_curve = field_curve(position_axes, lambda x: displacement(x, 0), GOLD)
        momentum_curve = field_curve(
            momentum_axes, lambda x: MASS_PER_GRID_POINT * velocity(x, 0), GREEN
        )
        field_labels = VGroup(
            Text("Displacement w(t)", font_size=25, color=GOLD).move_to(
                [-2.6, 2.05, 0]
            ),
            Text("Momentum p(t)", font_size=25, color=GREEN).move_to([-2.6, 0.1, 0]),
        )
        samples = VGroup(
            *[
                Dot(axis.c2p(x, field(x)), radius=0.065, color=colour)
                for axis, field, colour in (
                    (position_axes, lambda x: displacement(x, 0), GOLD),
                    (
                        momentum_axes,
                        lambda x: MASS_PER_GRID_POINT * velocity(x, 0),
                        GREEN,
                    ),
                )
                for x in SAMPLE_POSITIONS
            ]
        )
        vector = state_vector(("w(t)", "p(t)"), colours=(GOLD, GREEN)).move_to(
            [3.35, 0.5, 0]
        )
        vector_label = Text("A Hamiltonian state", font_size=27).move_to(
            [3.35, 1.95, 0]
        )
        relation = Text("p(t) = M v(t)", font_size=30).move_to([3.35, -1.1, 0])
        mass_label = Text("M: mass matrix", font_size=24, color=MUTED).move_to(
            [3.35, -1.8, 0]
        )
        explanation = caption(
            "Known masses let us recover velocity from momentum", -2.9
        )
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(
            FadeIn(VGroup(position_axes, momentum_axes)),
            FadeIn(VGroup(position_curve, momentum_curve)),
            FadeIn(field_labels),
            FadeIn(samples),
        )
        self.play(FadeIn(vector), FadeIn(vector_label))
        self.wait(2)
        self.play(FadeIn(relation), FadeIn(mass_label), FadeIn(explanation))
        self.wait(4)
        self.play(FadeOut(Group(*self.mobjects)))

    def state_choices(self):
        title = caption("We can choose different state variables", 3.25, font_size=40)
        labels = VGroup(
            Text("Displacement + velocity", font_size=24).move_to([-4.5, 2.0, 0]),
            Text("Two displacement steps", font_size=24).move_to([0, 2.0, 0]),
            Text("Displacement + momentum", font_size=24).move_to([4.5, 2.0, 0]),
        )
        vectors = VGroup(
            state_vector().scale(0.8).move_to([-4.5, 0.6, 0]),
            state_vector(
                ("w<sub>n</sub>", "w<sub>n-1</sub>"), "h<sub>n</sub> =", (GOLD, MUTED)
            )
            .scale(0.8)
            .move_to([0, 0.6, 0]),
            state_vector(("w(t)", "p(t)"), colours=(GOLD, GREEN))
            .scale(0.8)
            .move_to([4.5, 0.6, 0]),
        )
        continuous = Text("Continuous time", font_size=24, color=MUTED)
        time_labels = VGroup(
            continuous.copy().move_to([-4.5, -0.8, 0]),
            Text("Discrete time", font_size=24, color=MUTED).move_to([0, -0.8, 0]),
            continuous.copy().move_to([4.5, -0.8, 0]),
        )
        definition = caption(
            "The state stores what we need to continue the model", -1.9
        )
        space_definition = caption(
            "State space: the possible values of the chosen variables",
            -2.7,
            font_size=27,
        )
        self.play(FadeIn(title), FadeIn(labels), FadeIn(vectors), FadeIn(time_labels))
        self.play(FadeIn(definition))
        self.wait(2)
        self.play(FadeIn(space_definition))
        self.wait(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quality", choices=("low_quality", "high_quality"), default="high_quality"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempconfig(
        {
            "quality": args.quality,
            "frame_rate": 30,
            "background_color": BACKGROUND,
            "media_dir": str(Path(__file__).resolve().parent / "media"),
            "output_file": str(output),
            "preview": False,
            "disable_caching": True,
        }
    ):
        WhatIsAState().render()

    with av.open(str(output)) as video:
        for frame in video.decode(video=0):
            if frame.time is not None and frame.time >= 7:
                frame.to_image().save(output.with_suffix(".jpg"), quality=90)
                break


if __name__ == "__main__":
    main()
