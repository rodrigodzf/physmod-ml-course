import numpy as np
from scipy.integrate import quad

from manim import (
    Axes,
    BLUE,
    Create,
    DOWN,
    Dot,
    FadeIn,
    FadeOut,
    GOLD,
    LEFT,
    Line,
    MathTex,
    RED,
    RIGHT,
    Scene,
    SurroundingRectangle,
    Tex,
    Transform,
    UP,
    VGroup,
    ValueTracker,
    WHITE,
    Write,
    always_redraw,
)


SPATIAL = GOLD
TEMPORAL = BLUE


def _pluck_static(x: float, x_p: float, L: float = 1.0, peak: float = 1.0) -> float:
    """Triangular pluck shape on [0, L] with peak `peak` at x = x_p."""
    if x <= x_p:
        return peak * x / x_p
    return peak * (L - x) / (L - x_p)


def _velocity_static(
    x: float, x_v: float, sigma: float = 0.08, peak: float = 1.0
) -> float:
    """Gaussian velocity bump centred at x_v."""
    return peak * np.exp(-((x - x_v) ** 2) / (2.0 * sigma**2))


def colour_tex(tex: MathTex, spatial_idx, temporal_idx):
    """Apply the spatial / temporal colour scheme to a MathTex."""
    for i in spatial_idx:
        tex[i].set_color(SPATIAL)
    for i in temporal_idx:
        tex[i].set_color(TEMPORAL)
    return tex


class SeparationOfVariables(Scene):
    """Step-by-step derivation w = F(x) G(t) -> w = sum phi_mu(x) q_mu(t)."""

    def construct(self):
        legend = self._build_legend().to_edge(UP, buff=0.4)

        self.play(FadeIn(legend))
        self.wait(0.3)

        # --- Step 1: ansatz ----------------------------------------------
        ansatz = MathTex(
            r"w(x, t) \;=\;",
            r"F(x)",
            r"\,",
            r"G(t)",
            font_size=56,
        )
        ansatz[1].set_color(SPATIAL)
        ansatz[3].set_color(TEMPORAL)
        ansatz.move_to([0.0, 1.5, 0.0])

        caption1 = Tex(
            r"Ansatz: a product of a spatial shape and a temporal amplitude.",
            font_size=30,
        ).next_to(ansatz, DOWN, buff=0.6)

        self.play(Write(ansatz))
        self.play(FadeIn(caption1, shift=0.2 * UP))
        self.wait(1.2)

        # --- Step 2: the wave equation -----------------------------------
        pde = MathTex(
            r"\partial_t^2",
            r"w",
            r"\;=\;\hat T^2\,",
            r"\partial_x^2",
            r"w",
            font_size=52,
        )
        pde[0].set_color(TEMPORAL)
        pde[3].set_color(SPATIAL)
        pde.next_to(ansatz, DOWN, buff=1.0)

        self.play(FadeOut(caption1))
        self.play(Write(pde))
        self.wait(1.0)

        # --- Step 3: substitute the ansatz -------------------------------
        substituted = MathTex(
            r"F(x)\,",
            r"\partial_t^2 G(t)",
            r"\;=\;\hat T^2\,",
            r"G(t)\,",
            r"\partial_x^2 F(x)",
            font_size=52,
        )
        substituted[0].set_color(SPATIAL)
        substituted[1].set_color(TEMPORAL)
        substituted[3].set_color(TEMPORAL)
        substituted[4].set_color(SPATIAL)

        caption2 = Tex(
            r"Insert $w = F(x)\,G(t)$ into the wave equation.",
            font_size=30,
        ).next_to(ansatz, UP, buff=0.4)

        substituted.move_to(pde.get_center())
        self.play(FadeIn(caption2, shift=0.2 * DOWN))
        self.play(Transform(pde, substituted))
        self.wait(1.4)

        # --- Step 4: divide by hat T^2 F G -------------------------------
        divided = MathTex(
            r"\frac{1}{\hat T^2}",
            r"\frac{\partial_t^2 G}{G}",
            r"\;=\;",
            r"\frac{\partial_x^2 F}{F}",
            font_size=56,
        )
        divided[1].set_color(TEMPORAL)
        divided[3].set_color(SPATIAL)
        divided.move_to(pde.get_center())

        caption3 = Tex(
            r"Divide both sides by $\hat T^2\,F(x)\,G(t)$.",
            font_size=30,
        ).next_to(caption2, DOWN, buff=0.0)

        self.play(FadeOut(caption2), FadeOut(ansatz))
        self.play(FadeIn(caption3, shift=0.2 * DOWN))
        self.play(Transform(pde, divided))
        self.wait(1.4)

        # --- Step 5: each side depends only on its variable --------------
        observation = Tex(
            r"Left side depends only on $t$; right side only on $x$.",
            font_size=30,
        ).next_to(caption3, DOWN, buff=0.05)
        observation_2 = Tex(
            r"Both must equal a common constant, call it $-\lambda^2$.",
            font_size=30,
        ).next_to(observation, DOWN, buff=0.1)

        self.play(FadeOut(caption3))
        self.play(FadeIn(observation, shift=0.2 * DOWN))
        self.wait(0.8)
        self.play(FadeIn(observation_2, shift=0.2 * DOWN))
        self.wait(0.6)

        equal_const = MathTex(
            r"\frac{1}{\hat T^2}",
            r"\frac{\partial_t^2 G}{G}",
            r"\;=\;",
            r"\frac{\partial_x^2 F}{F}",
            r"\;=\;-\lambda^2",
            font_size=56,
        )
        equal_const[1].set_color(TEMPORAL)
        equal_const[3].set_color(SPATIAL)
        equal_const.move_to(pde.get_center())

        left_part = VGroup(*equal_const[:4])
        lambda_part = equal_const[4]

        self.play(
            pde.animate.move_to(left_part.get_center()),
            FadeIn(lambda_part),
        )
        self.wait(1.4)

        self.remove(pde, lambda_part)
        self.add(equal_const)

        # --- Step 6: the two ODEs ----------------------------------------
        ode_x = MathTex(
            r"\partial_x^2 F",
            r"\;=\;-\lambda^2\,",
            r"F",
            font_size=50,
        )
        ode_x[0].set_color(SPATIAL)
        ode_x[2].set_color(SPATIAL)

        ode_t = MathTex(
            r"\partial_t^2 G",
            r"\;=\;-\hat T^2 \lambda^2\,",
            r"G",
            font_size=50,
        )
        ode_t[0].set_color(TEMPORAL)
        ode_t[2].set_color(TEMPORAL)

        odes = VGroup(ode_x, ode_t).arrange(RIGHT, buff=1.4).move_to(
            equal_const.get_center()
        )

        self.play(
            FadeOut(observation),
            FadeOut(observation_2),
        )
        self.play(Transform(equal_const, odes))
        self.wait(0.4)

        caption4 = Tex(
            r"Two ODEs: one in space, one in time, coupled only through $\lambda$.",
            font_size=30,
        ).next_to(ansatz.get_center(), UP, buff=0.0).move_to(
            [0.0, ansatz.get_center()[1], 0.0]
        )
        self.play(FadeIn(caption4, shift=0.2 * DOWN))
        self.wait(1.4)

        # --- Step 6b: general sinusoidal solutions of the ODEs -----------
        sol_x = MathTex(
            r"F_n(x)",
            r"\;=\;A_n\cos(\lambda_n x) + B_n\sin(\lambda_n x)",
            font_size=42,
        )
        sol_x[0].set_color(SPATIAL)
        sol_x[1].set_color(SPATIAL)

        sol_t = MathTex(
            r"G_n(t)",
            r"\;=\;C_n\cos(\hat T \lambda_n t) + D_n\sin(\hat T \lambda_n t)",
            font_size=42,
        )
        sol_t[0].set_color(TEMPORAL)
        sol_t[1].set_color(TEMPORAL)

        sols = VGroup(sol_x, sol_t).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(
            equal_const.get_center()
        )

        sol_caption = Tex(
            r"Each ODE has the same form as a harmonic oscillator: "
            r"its general solution is a sinusoid.",
            font_size=30,
        ).move_to(caption4.get_center())

        self.play(Transform(caption4, sol_caption))
        self.play(Transform(equal_const, sols))
        self.wait(1.6)

        # --- Step 7: boundary conditions kill the cosine -----------------
        # Visualise F(x) = A cos(lambda x) + B sin(lambda x) and animate A -> 0.
        self.play(FadeOut(equal_const))

        bc_intro = Tex(
            r"Dirichlet boundary conditions: $F(0) = F(L) = 0$.",
            font_size=30,
        ).move_to([0.0, 2.5, 0.0])
        self.play(Transform(caption4, bc_intro))

        L_val = 1.0
        lambda_val = np.pi / L_val  # mode n = 1

        bc_axes = Axes(
            x_range=[0.0, L_val, 0.25],
            y_range=[-1.6, 1.6, 1.0],
            x_length=8.0,
            y_length=2.6,
            axis_config={"include_tip": False, "stroke_width": 1.5},
        ).move_to([0.0, -1.4, 0.0])

        bc_x0_label = MathTex("0", font_size=24).next_to(
            bc_axes.coords_to_point(0.0, 0.0), DOWN, buff=0.18
        )
        bc_xL_label = MathTex("L", font_size=24).next_to(
            bc_axes.coords_to_point(L_val, 0.0), DOWN, buff=0.18
        )

        A_tracker = ValueTracker(1.0)
        B_val = 1.0

        def F_eval(x):
            return (
                A_tracker.get_value() * np.cos(lambda_val * x)
                + B_val * np.sin(lambda_val * x)
            )

        bc_curve = always_redraw(
            lambda: bc_axes.plot(
                F_eval,
                x_range=[0.0, L_val, 0.005],
                color=SPATIAL,
                stroke_width=3.5,
            )
        )

        bc_dot_0 = always_redraw(
            lambda: Dot(
                bc_axes.coords_to_point(0.0, F_eval(0.0)),
                color=RED,
                radius=0.09,
            )
        )
        bc_dot_L = always_redraw(
            lambda: Dot(
                bc_axes.coords_to_point(L_val, F_eval(L_val)),
                color=RED,
                radius=0.09,
            )
        )

        bc_formula = always_redraw(
            lambda: MathTex(
                rf"F(x) \;=\; {A_tracker.get_value():.2f}\,\cos(\lambda x)"
                rf"\;+\; {B_val:.2f}\,\sin(\lambda x)",
                font_size=36,
                color=SPATIAL,
            ).move_to([0.0, 1.3, 0.0])
        )

        bc_status = Tex(
            r"With $A = 1$, $F(0) = A \ne 0$ -- the boundary condition fails.",
            font_size=28,
        ).move_to([0.0, -3.2, 0.0])

        self.play(Create(bc_axes), FadeIn(bc_x0_label), FadeIn(bc_xL_label))
        self.add(bc_curve, bc_dot_0, bc_dot_L, bc_formula)
        self.play(FadeIn(bc_status, shift=0.2 * UP))
        self.wait(1.5)

        bc_status_2 = Tex(
            r"Force $A = 0$: only the sine survives, vanishing at $x = 0$.",
            font_size=28,
        ).move_to(bc_status.get_center())

        self.play(Transform(bc_status, bc_status_2))
        self.play(A_tracker.animate.set_value(0.0), run_time=1.8)
        self.wait(0.8)

        bc_status_3 = Tex(
            r"At $x = L$: $B\sin(\lambda L) = 0$ quantises "
            r"$\lambda_n = n\pi/L$, giving $F_\mu(x) = \sin(\lambda_\mu x)$.",
            font_size=28,
        ).move_to(bc_status.get_center())

        self.play(Transform(bc_status, bc_status_3))
        self.wait(2.0)

        # --- Step 7b: initial conditions set the modal amplitudes --------
        self.play(
            FadeOut(bc_curve),
            FadeOut(bc_dot_0),
            FadeOut(bc_dot_L),
            FadeOut(bc_axes),
            FadeOut(bc_x0_label),
            FadeOut(bc_xL_label),
            FadeOut(bc_formula),
            FadeOut(bc_status),
            FadeOut(caption4),
        )

        ic_caption = Tex(
            r"Initial conditions $w(x, 0)$ and $\partial_t w(x, 0)$ "
            r"fix the modal amplitudes $C_\mu, D_\mu$.",
            font_size=30,
        ).move_to([0.0, 2.5, 0.0])

        x_p_tracker = ValueTracker(0.3)
        x_v_tracker = ValueTracker(0.5)
        v_sigma = 0.08
        n_modes_demo = 6

        def pluck_eval(x):
            return _pluck_static(x, x_p_tracker.get_value(), L_val)

        def velocity_eval(x):
            return _velocity_static(x, x_v_tracker.get_value(), v_sigma)

        def C_n_compute(n, x_p_val):
            integrand = lambda x: _pluck_static(x, x_p_val, L_val) * np.sin(
                n * np.pi * x / L_val
            )
            integral, _ = quad(integrand, 0.0, L_val, limit=200)
            return (2.0 / L_val) * integral

        def D_n_compute(n, x_v_val):
            integrand = lambda x: _velocity_static(x, x_v_val, v_sigma) * np.sin(
                n * np.pi * x / L_val
            )
            integral, _ = quad(integrand, 0.0, L_val, limit=200)
            lambda_n = n * np.pi / L_val
            return (2.0 / (lambda_n * L_val)) * integral

        # Common axis dimensions
        panel_x_length = 4.6
        panel_y_length = 1.5
        row1_y = 1.0
        row2_y = -1.1

        # --- Row 1: pluck shape w(x, 0) and |C_mu| -----------------------
        pluck_axes = Axes(
            x_range=[0.0, L_val, 0.25],
            y_range=[0.0, 1.2, 0.5],
            x_length=panel_x_length,
            y_length=panel_y_length,
            axis_config={"include_tip": False, "stroke_width": 1.5},
        ).move_to([-3.4, row1_y, 0.0])

        pluck_label = MathTex(r"w(x, 0)", font_size=24, color=SPATIAL).next_to(
            pluck_axes, UP, buff=0.08
        )

        pluck_curve = always_redraw(
            lambda: pluck_axes.plot(
                pluck_eval,
                x_range=[0.0, L_val, 0.002],
                color=SPATIAL,
                stroke_width=3,
            )
        )

        pluck_marker = always_redraw(
            lambda: Dot(
                pluck_axes.coords_to_point(x_p_tracker.get_value(), 1.0),
                color=RED,
                radius=0.07,
            )
        )

        c_axes = Axes(
            x_range=[0.4, n_modes_demo + 0.6, 1.0],
            y_range=[0.0, 1.0, 0.25],
            x_length=panel_x_length,
            y_length=panel_y_length,
            axis_config={"include_tip": False, "stroke_width": 1.0},
        ).move_to([3.4, row1_y, 0.0])

        c_label = MathTex(r"|C_\mu|", font_size=24, color=TEMPORAL).next_to(
            c_axes, UP, buff=0.08
        )

        c_x_labels = VGroup()
        for n in range(1, n_modes_demo + 1):
            tl = MathTex(rf"\mu={n}", font_size=16).move_to(
                c_axes.coords_to_point(n, 0.0) + np.array([0.0, -0.22, 0.0])
            )
            c_x_labels.add(tl)

        def c_bar_height(n):
            return abs(C_n_compute(n, x_p_tracker.get_value()))

        c_bars = always_redraw(
            lambda: VGroup(
                *[
                    Line(
                        start=c_axes.coords_to_point(n, 0.0),
                        end=c_axes.coords_to_point(n, min(c_bar_height(n), 1.0)),
                        stroke_color=TEMPORAL,
                        stroke_width=16,
                    )
                    for n in range(1, n_modes_demo + 1)
                ]
            )
        )

        # --- Row 2: initial velocity dot w(x, 0) and |D_mu| --------------
        vel_axes = Axes(
            x_range=[0.0, L_val, 0.25],
            y_range=[0.0, 1.2, 0.5],
            x_length=panel_x_length,
            y_length=panel_y_length,
            axis_config={"include_tip": False, "stroke_width": 1.5},
        ).move_to([-3.4, row2_y, 0.0])

        vel_label = MathTex(
            r"\partial_t w(x, 0)", font_size=24, color=SPATIAL
        ).next_to(vel_axes, UP, buff=0.08)

        vel_curve = always_redraw(
            lambda: vel_axes.plot(
                velocity_eval,
                x_range=[0.0, L_val, 0.002],
                color=SPATIAL,
                stroke_width=3,
            )
        )

        vel_marker = always_redraw(
            lambda: Dot(
                vel_axes.coords_to_point(x_v_tracker.get_value(), 1.0),
                color=RED,
                radius=0.07,
            )
        )

        d_axes = Axes(
            x_range=[0.4, n_modes_demo + 0.6, 1.0],
            y_range=[0.0, 1.0, 0.25],
            x_length=panel_x_length,
            y_length=panel_y_length,
            axis_config={"include_tip": False, "stroke_width": 1.0},
        ).move_to([3.4, row2_y, 0.0])

        d_label = MathTex(r"|D_\mu|", font_size=24, color=TEMPORAL).next_to(
            d_axes, UP, buff=0.08
        )

        d_x_labels = VGroup()
        for n in range(1, n_modes_demo + 1):
            tl = MathTex(rf"\mu={n}", font_size=16).move_to(
                d_axes.coords_to_point(n, 0.0) + np.array([0.0, -0.22, 0.0])
            )
            d_x_labels.add(tl)

        # Normalisation so D bars use the same vertical scale.
        d_norm = max(
            abs(D_n_compute(n, 0.5)) for n in range(1, n_modes_demo + 1)
        )

        def d_bar_height(n):
            return abs(D_n_compute(n, x_v_tracker.get_value())) / d_norm

        d_bars = always_redraw(
            lambda: VGroup(
                *[
                    Line(
                        start=d_axes.coords_to_point(n, 0.0),
                        end=d_axes.coords_to_point(n, min(d_bar_height(n), 1.0)),
                        stroke_color=TEMPORAL,
                        stroke_width=16,
                    )
                    for n in range(1, n_modes_demo + 1)
                ]
            )
        )

        ic_formula = MathTex(
            r"C_\mu",
            r"\;=\; \frac{2}{L}\int_0^L ",
            r"\sin(\lambda_\mu x)\, w(x, 0)",
            r"\, dx, \qquad ",
            r"D_\mu",
            r"\;=\; \frac{2}{\hat T \lambda_\mu L}\int_0^L ",
            r"\sin(\lambda_\mu x)\, \partial_t w(x, 0)",
            r"\, dx",
            font_size=24,
        ).move_to([0.0, -2.9, 0.0])
        ic_formula[0].set_color(TEMPORAL)
        ic_formula[2].set_color(SPATIAL)
        ic_formula[4].set_color(TEMPORAL)
        ic_formula[6].set_color(SPATIAL)

        self.play(Write(ic_caption))
        self.play(
            Create(pluck_axes),
            Write(pluck_label),
            Create(c_axes),
            Write(c_label),
            Write(c_x_labels),
            Create(vel_axes),
            Write(vel_label),
            Create(d_axes),
            Write(d_label),
            Write(d_x_labels),
        )
        self.add(pluck_curve, pluck_marker, c_bars, vel_curve, vel_marker, d_bars)
        self.play(Write(ic_formula))
        self.wait(1.0)

        ic_status = Tex(
            r"Move the pluck point: each $C_\mu$ rescales accordingly.",
            font_size=26,
        ).move_to([0.0, -3.5, 0.0])
        self.play(FadeIn(ic_status, shift=0.2 * UP))
        self.play(x_p_tracker.animate.set_value(0.7), run_time=2.0)
        self.wait(0.4)
        self.play(x_p_tracker.animate.set_value(0.5), run_time=1.4)
        self.wait(0.6)

        ic_status_2 = Tex(
            r"Now move the strike point: each $D_\mu$ rescales accordingly.",
            font_size=26,
        ).move_to(ic_status.get_center())
        self.play(Transform(ic_status, ic_status_2))
        self.play(x_v_tracker.animate.set_value(0.25), run_time=2.0)
        self.wait(0.4)
        self.play(x_v_tracker.animate.set_value(0.5), run_time=1.4)
        self.wait(1.0)

        self.play(
            FadeOut(ic_caption),
            FadeOut(pluck_curve),
            FadeOut(pluck_marker),
            FadeOut(pluck_axes),
            FadeOut(pluck_label),
            FadeOut(c_bars),
            FadeOut(c_axes),
            FadeOut(c_label),
            FadeOut(c_x_labels),
            FadeOut(vel_curve),
            FadeOut(vel_marker),
            FadeOut(vel_axes),
            FadeOut(vel_label),
            FadeOut(d_bars),
            FadeOut(d_axes),
            FadeOut(d_label),
            FadeOut(d_x_labels),
            FadeOut(ic_formula),
            FadeOut(ic_status),
        )

        # --- Step 8: per-mode product solution ---------------------------
        per_mode = MathTex(
            r"w_\mu(x, t) \;=\;",
            r"F_\mu(x)",
            r"\,",
            r"G_\mu(t)",
            font_size=54,
        )
        per_mode[1].set_color(SPATIAL)
        per_mode[3].set_color(TEMPORAL)
        per_mode.move_to([0.0, 0.0, 0.0])

        caption5 = Tex(
            r"Each $\mu$ gives a standing-wave solution.",
            font_size=30,
        ).next_to(per_mode, DOWN, buff=0.4)

        self.play(Write(per_mode))
        self.play(FadeIn(caption5, shift=0.2 * UP))
        self.wait(1.2)

        # --- Step 9: superposition ---------------------------------------
        superposition = MathTex(
            r"w(x, t) \;=\;\sum_\mu\;",
            r"F_\mu(x)",
            r"\,",
            r"G_\mu(t)",
            font_size=54,
        )
        superposition[1].set_color(SPATIAL)
        superposition[3].set_color(TEMPORAL)
        superposition.move_to(per_mode.get_center())

        caption6 = Tex(
            r"Linearity: any sum of solutions is again a solution.",
            font_size=30,
        ).move_to(caption5.get_center())

        self.play(Transform(per_mode, superposition), Transform(caption5, caption6))
        self.wait(1.4)

        # --- Step 10: morph superposition -> explicit BC form -> boxed --
        explicit = MathTex(
            r"w(x, t) \;=\; \sum_\mu \;",
            r"\sin(\lambda_\mu x)",
            r"\,\bigl[",
            r"C_\mu\cos(\hat T \lambda_\mu t) + D_\mu\sin(\hat T \lambda_\mu t)",
            r"\bigr]",
            font_size=38,
        )
        explicit[1].set_color(SPATIAL)
        explicit[3].set_color(TEMPORAL)
        explicit.move_to(per_mode.get_center())

        explicit_caption = Tex(
            r"Boundary conditions fix $F_\mu(x) = \sin(\lambda_\mu x)$; "
            r"initial conditions fix $C_\mu, D_\mu$.",
            font_size=30,
        ).move_to(caption5.get_center())

        self.play(
            Transform(per_mode, explicit),
            Transform(caption5, explicit_caption),
        )
        self.wait(2.0)

        final = MathTex(
            r"w(x, t) \;=\; \sum_\mu \;",
            r"\phi_\mu(x)",
            r"\,",
            r"q_\mu(t)",
            font_size=60,
        )
        final[1].set_color(SPATIAL)
        final[3].set_color(TEMPORAL)
        final.move_to(per_mode.get_center())

        self.play(
            Transform(per_mode, final),
            FadeOut(caption5),
        )

        box = SurroundingRectangle(final, color=WHITE, buff=0.4)
        self.play(Write(box))
        self.wait(2.0)

    # ---------------------------------------------------------------
    def _build_legend(self) -> VGroup:
        spatial_label = Tex(r"spatial", font_size=32, color=SPATIAL)
        temporal_label = Tex(r"temporal", font_size=32, color=TEMPORAL)
        return VGroup(spatial_label, temporal_label).arrange(RIGHT, buff=1.4)
