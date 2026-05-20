import numpy as np
from manim import (
    Axes,
    BLUE,
    Create,
    Dot,
    GOLD,
    GREY,
    Line,
    MathTex,
    Scene,
    Text,
    UP,
    VGroup,
    ValueTracker,
    WHITE,
    Write,
    always_redraw,
    linear,
)
from scipy.integrate import quad


MODAL_COLOR = GOLD
GRID_COLOR = BLUE
SUM_COLOR = WHITE


class BasisComparison(Scene):
    """One displacement field, two bases: global modal sines vs local nodal hats."""

    def construct(self):
        # Physical parameters
        L = 1.0
        wave_speed = 1.0
        n_modes = 6          # left panel: number of modal sines
        n_grid = 25          # right panel: number of interior grid nodes
        x0 = 0.30
        sigma = 0.12
        duration = 4.0

        # Display amplitudes
        modal_row_amp = 0.4  # per-mode rows, left panel
        hat_band_amp = 0.5   # weighted-hat band, right panel
        sum_amp = 0.45       # reconstructed sum, both panels

        # Initial Gaussian -> modal coefficients
        def w0(x):
            return np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))

        a_mu = np.zeros(n_modes)
        for mu in range(1, n_modes + 1):
            integrand = lambda x, mu=mu: w0(x) * np.sin(mu * np.pi * x / L)
            integral, _ = quad(integrand, 0.0, L, limit=200)
            a_mu[mu - 1] = (2.0 / L) * integral

        omega = np.pi * np.arange(1, n_modes + 1) * wave_speed / L

        # --- Modal basis (global sines) -------------------------------------
        def phi_modal(mu_idx, x):
            return np.sin((mu_idx + 1) * np.pi * np.asarray(x) / L)

        def q_modal(mu_idx, t):
            return a_mu[mu_idx] * np.cos(omega[mu_idx] * t)

        def w_modal_total(x, t):
            total = 0.0
            for mu in range(n_modes):
                total = total + phi_modal(mu, x) * q_modal(mu, t)
            return total

        # --- Nodal basis (local hats), Dirichlet BCs at x = 0 and x = L -----
        dx = L / (n_grid + 1)
        x_grid = np.array([(i + 1) * dx for i in range(n_grid)])

        def hat(i_idx, x):
            # Continuous piecewise-linear tent: 1 at node i, 0 at all others
            return np.maximum(0.0, 1.0 - np.abs(np.asarray(x) - x_grid[i_idx]) / dx)

        def w_node(i_idx, t):
            # Nodal coefficient = field value at that grid point
            return w_modal_total(x_grid[i_idx], t)

        def w_grid_total(x, t):
            x_arr = np.asarray(x, dtype=float)
            total = np.zeros_like(x_arr)
            for i in range(n_grid):
                total = total + hat(i, x_arr) * w_node(i, t)
            return total

        time = ValueTracker(0.0)

        # --- Title: the two expansions of the same field --------------------
        title = MathTex(
            r"w(x,t) = \sum_\mu \phi_\mu(x)\,q_\mu(t)"
            r" = \sum_i \Lambda_i(x)\,w_i(t)",
            font_size=34,
        ).to_edge(UP, buff=0.3)

        # --- Panels ---------------------------------------------------------
        panel_y_range = [-3.0, 3.0, 1.0]
        panel_y_length = 6.0
        panel_x_length = 5.5

        left_axes = Axes(
            x_range=[0.0, L, 0.25],
            y_range=panel_y_range,
            x_length=panel_x_length,
            y_length=panel_y_length,
            axis_config={"include_tip": False, "stroke_width": 0.4},
        ).move_to([-3.5, -0.4, 0.0])
        left_subtitle = Text(
            "Modal basis: 6 global modes", font_size=22, color=MODAL_COLOR
        ).next_to(left_axes, UP, buff=0.12)

        right_axes = Axes(
            x_range=[0.0, L, 0.25],
            y_range=panel_y_range,
            x_length=panel_x_length,
            y_length=panel_y_length,
            axis_config={"include_tip": False, "stroke_width": 0.4},
        ).move_to([3.5, -0.4, 0.0])
        right_subtitle = Text(
            "Nodal (hat) basis: 25 local tents", font_size=22, color=GRID_COLOR
        ).next_to(right_axes, UP, buff=0.12)

        # Layout offsets (in axes-y data units)
        modal_offsets = np.linspace(2.2, -1.2, n_modes)
        hat_band_offset = 1.0
        sum_offset = -2.4
        sep_y = -1.75

        # Separators between basis display and the reconstructed sum
        left_sep = Line(
            left_axes.coords_to_point(0.0, sep_y),
            left_axes.coords_to_point(L, sep_y),
            color=GREY,
            stroke_width=0.5,
        )
        right_sep = Line(
            right_axes.coords_to_point(0.0, sep_y),
            right_axes.coords_to_point(L, sep_y),
            color=GREY,
            stroke_width=0.5,
        )

        # Grid-node dots: visual reminder of where the x_i live
        grid_dots = VGroup(
            *[
                Dot(
                    right_axes.coords_to_point(x_grid[i], -3.0),
                    radius=0.04,
                    color=GRID_COLOR,
                )
                for i in range(n_grid)
            ]
        )

        # --- Left panel: stacked weighted modes phi_mu(x) q_mu(t) -----------
        left_curves = VGroup()
        left_labels = VGroup()
        for mu in range(n_modes):
            offset = modal_offsets[mu]
            curve = always_redraw(
                lambda offset=offset, mu=mu: left_axes.plot(
                    lambda x, offset=offset, mu=mu: offset
                    + modal_row_amp
                    * phi_modal(mu, x)
                    * q_modal(mu, time.get_value()),
                    x_range=[0.0, L, 0.005],
                    color=MODAL_COLOR,
                    stroke_width=1.6,
                )
            )
            left_curves.add(curve)
            label = MathTex(
                rf"\mu={mu + 1}", font_size=20, color=MODAL_COLOR
            ).move_to(
                left_axes.coords_to_point(-0.04, offset) + np.array([-0.2, 0.0, 0.0])
            )
            left_labels.add(label)

        left_sum = always_redraw(
            lambda: left_axes.plot(
                lambda x: sum_offset + sum_amp * w_modal_total(x, time.get_value()),
                x_range=[0.0, L, 0.005],
                color=SUM_COLOR,
                stroke_width=2.2,
            )
        )
        left_sum_label = MathTex(
            r"\sum_\mu \phi_\mu(x)\,q_\mu(t)", font_size=22, color=SUM_COLOR
        ).move_to(
            left_axes.coords_to_point(0.5, sum_offset) + np.array([0.0, -1.0, 0.0])
        )

        # --- Right panel: band of all weighted hats Lambda_i(x) w_i(t) ------
        # Each tent is drawn separately, so locality is visible: the peaks
        # trace the field, one nodal value per tent.
        right_curves = VGroup()
        for i in range(n_grid):
            curve = always_redraw(
                lambda i=i: right_axes.plot(
                    lambda x, i=i: hat_band_offset
                    + hat_band_amp * hat(i, x) * w_node(i, time.get_value()),
                    x_range=[0.0, L, 0.005],
                    color=GRID_COLOR,
                    stroke_width=1.4,
                )
            )
            right_curves.add(curve)

        right_band_label = MathTex(
            r"\Lambda_i(x)\,w_i(t)", font_size=22, color=GRID_COLOR
        ).move_to(right_axes.coords_to_point(0.5, hat_band_offset + 1.0))

        right_sum = always_redraw(
            lambda: right_axes.plot(
                lambda x: sum_offset + sum_amp * w_grid_total(x, time.get_value()),
                x_range=[0.0, L, 0.005],
                color=SUM_COLOR,
                stroke_width=2.2,
            )
        )
        right_sum_label = MathTex(
            r"\sum_i \Lambda_i(x)\,w_i(t)", font_size=22, color=SUM_COLOR
        ).move_to(
            right_axes.coords_to_point(0.5, sum_offset) + np.array([0.0, -1.0, 0.0])
        )

        # --- Choreography ---------------------------------------------------
        self.play(Write(title))
        self.wait(0.5)
        self.play(
            Create(left_axes),
            Write(left_subtitle),
            Create(right_axes),
            Write(right_subtitle),
            run_time=1.5,
        )
        self.play(
            Create(left_sep),
            Create(right_sep),
            Write(left_labels),
            Write(right_band_label),
            Create(grid_dots),
            run_time=1.0,
        )
        self.add(left_curves, right_curves)
        self.wait(0.3)
        self.add(left_sum, right_sum)
        self.play(Write(left_sum_label), Write(right_sum_label), run_time=1.0)
        self.wait(0.5)
        self.play(
            time.animate.set_value(duration),
            run_time=duration,
            rate_func=linear,
        )
        self.wait(1.0)