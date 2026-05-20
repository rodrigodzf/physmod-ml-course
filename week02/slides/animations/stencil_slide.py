import numpy as np
from manim import (
    Axes,
    BLUE,
    Create,
    DOWN,
    Dot,
    FadeIn,
    FadeOut,
    GOLD,
    GREY,
    LEFT,
    Line,
    MathTex,
    RED,
    RIGHT,
    Rectangle,
    Scene,
    UP,
    VGroup,
    ValueTracker,
    WHITE,
    Write,
    always_redraw,
)


FIELD_COLOR = GOLD
STENCIL_COLOR = RED
RESULT_COLOR = BLUE


class StencilSlide(Scene):
    """Show the [1, -2, 1] stencil sliding along a discrete field w
    and producing the second-difference output one point at a time."""

    def construct(self):
        L = 1.0
        n_interior = 13
        n_total = n_interior + 2

        dx = L / (n_total - 1)
        x_grid = np.linspace(0.0, L, n_total)

        def w_continuous(x):
            x = np.asarray(x)
            return 0.9 * np.sin(np.pi * x / L) + 0.35 * np.sin(3.0 * np.pi * x / L)

        def d2w_continuous(x):
            x = np.asarray(x)
            return -((np.pi / L) ** 2) * 0.9 * np.sin(np.pi * x / L) \
                   - ((3.0 * np.pi / L) ** 2) * 0.35 * np.sin(3.0 * np.pi * x / L)

        w_vals = w_continuous(x_grid)
        d2_exact = d2w_continuous(x_grid)
        d2_fd = np.zeros_like(w_vals)
        for i in range(1, n_total - 1):
            d2_fd[i] = (w_vals[i - 1] - 2.0 * w_vals[i] + w_vals[i + 1]) / dx**2

        eq = MathTex(
            r"(\mathbf{D}_{xx}\,\mathbf{w})_i \;=\; ",
            r"\frac{w_{i-1} - 2\,w_i + w_{i+1}}{\Delta x^2}",
            font_size=40,
        ).to_edge(UP, buff=0.3)
        eq[1].set_color(STENCIL_COLOR)

        top_axes = Axes(
            x_range=[0.0, L, 0.2],
            y_range=[-1.6, 1.6, 0.5],
            x_length=11.0,
            y_length=2.2,
            axis_config={"include_tip": False, "stroke_width": 1.0},
        ).move_to([0.0, 1.4, 0.0])
        top_label = MathTex(r"w_i", font_size=30, color=FIELD_COLOR).next_to(
            top_axes, LEFT, buff=0.15
        )

        smooth_curve = top_axes.plot(
            lambda x: w_continuous(x),
            x_range=[0.0, L, 0.005],
            color=GREY,
            stroke_width=1.2,
        )

        top_dots = VGroup(
            *[
                Dot(top_axes.coords_to_point(x_grid[i], w_vals[i]),
                    radius=0.06, color=FIELD_COLOR)
                for i in range(n_total)
            ]
        )
        top_stems = VGroup(
            *[
                Line(
                    top_axes.coords_to_point(x_grid[i], 0.0),
                    top_axes.coords_to_point(x_grid[i], w_vals[i]),
                    color=FIELD_COLOR,
                    stroke_width=1.0,
                )
                for i in range(n_total)
            ]
        )

        d2_exact_fine = d2w_continuous(np.linspace(0.0, L, 400))
        d2_max = float(
            max(np.max(np.abs(d2_fd)), np.max(np.abs(d2_exact_fine)))
        )
        if d2_max == 0.0:
            d2_max = 1.0
        bot_axes = Axes(
            x_range=[0.0, L, 0.2],
            y_range=[-1.15 * d2_max, 1.15 * d2_max, d2_max / 2.0],
            x_length=11.0,
            y_length=2.2,
            axis_config={"include_tip": False, "stroke_width": 1.0},
        ).move_to([0.0, -1.8, 0.0])
        bot_label = MathTex(
            r"(\mathbf{D}_{xx}\mathbf{w})_i", font_size=30, color=RESULT_COLOR
        ).next_to(bot_axes, LEFT, buff=0.15)

        bot_boundary_dots = VGroup(
            Dot(bot_axes.coords_to_point(x_grid[0], 0.0),
                radius=0.05, color=GREY),
            Dot(bot_axes.coords_to_point(x_grid[-1], 0.0),
                radius=0.05, color=GREY),
        )

        active_idx = ValueTracker(1.0)

        def stencil_box():
            i = int(round(active_idx.get_value()))
            i = max(1, min(n_total - 2, i))
            left = top_axes.coords_to_point(x_grid[i - 1], 0.0)[0] - 0.12
            right = top_axes.coords_to_point(x_grid[i + 1], 0.0)[0] + 0.12
            top = top_axes.coords_to_point(0.0, 1.55)[1]
            bot = top_axes.coords_to_point(0.0, -1.55)[1]
            width = right - left
            height = top - bot
            rect = Rectangle(
                width=width,
                height=height,
                stroke_color=STENCIL_COLOR,
                stroke_width=2.0,
                fill_opacity=0.08,
                fill_color=STENCIL_COLOR,
            )
            rect.move_to([(left + right) / 2.0, (top + bot) / 2.0, 0.0])
            return rect

        stencil_rect = always_redraw(stencil_box)

        def stencil_weights():
            i = int(round(active_idx.get_value()))
            i = max(1, min(n_total - 2, i))
            group = VGroup()
            for offset, weight_str in [(-1, "1"), (0, "-2"), (1, "1")]:
                point = top_axes.coords_to_point(x_grid[i + offset], 0.0)
                label = MathTex(weight_str, font_size=26, color=STENCIL_COLOR)
                label.move_to([point[0], top_axes.coords_to_point(0.0, -1.35)[1], 0.0])
                group.add(label)
            return group

        weight_labels = always_redraw(stencil_weights)

        def highlight_dots():
            i = int(round(active_idx.get_value()))
            i = max(1, min(n_total - 2, i))
            group = VGroup()
            for offset in (-1, 0, 1):
                j = i + offset
                group.add(
                    Dot(
                        top_axes.coords_to_point(x_grid[j], w_vals[j]),
                        radius=0.09,
                        color=STENCIL_COLOR,
                    )
                )
            return group

        hl = always_redraw(highlight_dots)

        result_dots = VGroup()
        result_stems = VGroup()

        running_eq = always_redraw(
            lambda: self._running_eq(active_idx, w_vals, dx, n_total).move_to(
                [0.0, -3.4, 0.0]
            )
        )

        self.play(Write(eq))
        self.play(
            Create(top_axes),
            Write(top_label),
            Create(bot_axes),
            Write(bot_label),
            run_time=1.2,
        )
        self.play(
            Create(smooth_curve),
            Create(top_stems),
            Create(top_dots),
            Create(bot_boundary_dots),
            run_time=1.5,
        )
        self.add(stencil_rect, weight_labels, hl, running_eq)
        self.wait(0.4)

        for i in range(1, n_total - 1):
            self.play(active_idx.animate.set_value(float(i)), run_time=0.35)
            dot = Dot(
                bot_axes.coords_to_point(x_grid[i], d2_fd[i]),
                radius=0.06,
                color=RESULT_COLOR,
            )
            stem = Line(
                bot_axes.coords_to_point(x_grid[i], 0.0),
                bot_axes.coords_to_point(x_grid[i], d2_fd[i]),
                color=RESULT_COLOR,
                stroke_width=1.0,
            )
            result_dots.add(dot)
            result_stems.add(stem)
            self.play(FadeIn(stem), FadeIn(dot), run_time=0.25)

        self.remove(stencil_rect, weight_labels, hl, running_eq)
        self.wait(0.4)

        exact_curve = bot_axes.plot(
            lambda x: d2w_continuous(x),
            x_range=[0.0, L, 0.005],
            color=GREY,
            stroke_width=1.4,
        )
        exact_label = MathTex(
            r"\partial_x^2 w", font_size=26, color=GREY
        ).move_to(bot_axes.coords_to_point(0.5, 0.95 * d2_max))
        self.play(Create(exact_curve), Write(exact_label), run_time=1.2)
        self.wait(1.5)

    @staticmethod
    def _running_eq(active_idx, w_vals, dx, n_total):
        i = int(round(active_idx.get_value()))
        i = max(1, min(n_total - 2, i))
        wm, w0, wp = w_vals[i - 1], w_vals[i], w_vals[i + 1]
        result = (wm - 2.0 * w0 + wp) / dx**2
        return MathTex(
            rf"i = {i}: \quad",
            rf"\frac{{({wm:+.3f}) - 2({w0:+.3f}) + ({wp:+.3f})}}{{\Delta x^2}}",
            rf"\;=\; {result:+.2f}",
            font_size=28,
            color=STENCIL_COLOR,
        )
