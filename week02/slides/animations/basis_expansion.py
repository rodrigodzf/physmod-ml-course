import numpy as np
from manim import (
    Axes,
    BLUE,
    Create,
    GOLD,
    Line,
    MathTex,
    Scene,
    UP,
    LEFT,
    VGroup,
    ValueTracker,
    WHITE,
    Write,
    always_redraw,
    config,
    linear,
)
from scipy.integrate import quad


PHI_COLOR = GOLD
Q_COLOR = BLUE


class BasisExpansion(Scene):
    """Three-panel modal decomposition of a Gaussian pulse."""

    def construct(self):
        # Physical parameters
        L = 1.0
        wave_speed = 1.0
        n_modes = 5
        x0 = 0.30
        sigma = 0.08
        duration = 4.0

        # Modal coefficients for w(x, 0) = exp(-(x-x0)^2 / (2 sigma^2)), w_dot(x,0) = 0
        def w0(x):
            return np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))

        a_mu = np.zeros(n_modes)
        for mu in range(1, n_modes + 1):
            integrand = lambda x, mu=mu: w0(x) * np.sin(mu * np.pi * x / L)
            integral, _ = quad(integrand, 0.0, L, limit=200)
            a_mu[mu - 1] = (2.0 / L) * integral

        omega = np.pi * np.arange(1, n_modes + 1) * wave_speed / L

        def q_mu_t(mu_idx, t):
            return a_mu[mu_idx] * np.cos(omega[mu_idx] * t)

        def phi_mu_x(mu_idx, x):
            return np.sin((mu_idx + 1) * np.pi * x / L)

        def w_total(x, t):
            total = 0.0
            for mu in range(n_modes):
                total += phi_mu_x(mu, x) * q_mu_t(mu, t)
            return total

        time = ValueTracker(0.0)

        # Title equation with colored parts
        eq = MathTex(
            r"w(x,t) \;=\; \sum_\mu \;",
            r"\phi_\mu(x)",
            r"\;",
            r"q_\mu(t)",
            font_size=48,
        )
        eq[1].set_color(PHI_COLOR)
        eq[3].set_color(Q_COLOR)
        eq.to_edge(UP, buff=0.4)

        # Top panel: w(x, t)
        top_axes = Axes(
            x_range=[0.0, L, 0.25],
            y_range=[-1.2, 1.2, 0.5],
            x_length=12.0,
            y_length=2.0,
            axis_config={"include_tip": False, "stroke_width": 1.5},
        ).move_to(np.array([0.0, 0.8, 0.0]))
        top_label = MathTex(r"w(x,t)", font_size=28).next_to(top_axes, UP, buff=0.05)

        top_curve = always_redraw(
            lambda: top_axes.plot(
                lambda x: w_total(x, time.get_value()),
                x_range=[0.0, L, 0.005],
                color=WHITE,
                stroke_width=2.5,
            )
        )

        # Bottom-left panel: individual modes phi_mu(x) * q_mu(t)
        bl_axes = Axes(
            x_range=[0.0, L, 0.5],
            y_range=[-3.0, 3.0, 1.0],
            x_length=5.5,
            y_length=3.6,
            axis_config={"include_tip": False, "stroke_width": 0.5},
        ).move_to(np.array([-3.6, -2.0, 0.0]))
        bl_label = MathTex(
            r"\phi_\mu(x)\,q_\mu(t)", font_size=26, color=PHI_COLOR
        ).next_to(bl_axes, UP, buff=0.05)

        mode_offsets = np.linspace(2.4, -2.4, n_modes)  # mu=1 at top
        per_mode_scale = 0.7

        mode_curves = VGroup()
        mode_labels = VGroup()
        for mu in range(n_modes):
            offset = mode_offsets[mu]

            curve = always_redraw(
                lambda offset=offset, mu=mu: bl_axes.plot(
                    lambda x, offset=offset, mu=mu: offset
                    + per_mode_scale * phi_mu_x(mu, x) * q_mu_t(mu, time.get_value()),
                    x_range=[0.0, L, 0.005],
                    color=PHI_COLOR,
                    stroke_width=1.8,
                )
            )
            mode_curves.add(curve)

            mu_label = MathTex(rf"\mu={mu + 1}", font_size=20, color=PHI_COLOR).move_to(
                bl_axes.coords_to_point(-0.05, offset) + np.array([-0.15, 0.0, 0.0])
            )
            mode_labels.add(mu_label)

        # Bottom-right panel: bar chart of q_mu(t)
        br_axes = Axes(
            x_range=[0.4, n_modes + 0.6, 1.0],
            y_range=[-1.0, 1.0, 0.5],
            x_length=5.5,
            y_length=3.6,
            axis_config={"include_tip": False, "stroke_width": 1.0},
        ).move_to(np.array([3.6, -2.0, 0.0]))
        br_label = MathTex(r"q_\mu(t)", font_size=26, color=Q_COLOR).next_to(
            br_axes, UP, buff=0.05
        )

        x_tick_labels = VGroup()
        for mu in range(1, n_modes + 1):
            tl = MathTex(str(mu), font_size=20).move_to(
                br_axes.coords_to_point(mu, 0.0) + np.array([0.0, -0.3, 0.0])
            )
            x_tick_labels.add(tl)

        bars = always_redraw(
            lambda: VGroup(
                *[
                    Line(
                        start=br_axes.coords_to_point(mu + 1, 0.0),
                        end=br_axes.coords_to_point(mu + 1, q_mu_t(mu, time.get_value())),
                        stroke_color=Q_COLOR,
                        stroke_width=18,
                    )
                    for mu in range(n_modes)
                ]
            )
        )

        # Choreography
        self.play(Write(eq))
        self.wait(0.8)
        self.play(
            Create(top_axes),
            Write(top_label),
            Create(bl_axes),
            Write(bl_label),
            Write(mode_labels),
            Create(br_axes),
            Write(br_label),
            Write(x_tick_labels),
            run_time=2.0,
        )
        self.add(top_curve, mode_curves, bars)
        self.wait(0.5)
        self.play(
            time.animate.set_value(duration),
            run_time=duration,
            rate_func=linear,
        )
        self.wait(1.0)
