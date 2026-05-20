import numpy as np
from manim import (
    Axes,
    BLACK,
    BLUE,
    Create,
    DOWN,
    GREEN,
    MathTex,
    RED,
    Scene,
    Text,
    UP,
    VGroup,
    ValueTracker,
    Write,
    always_redraw,
    linear,
)


STABLE_COLOR = BLUE
MAGIC_COLOR = GREEN
BLOWUP_COLOR = RED


class CFLBlowup(Scene):
    """Three FDTD strings side by side at nu = 0.9, 1.0, 1.01.
    Shows clean propagation, the exact magic case, and catastrophic
    blow-up of the highest grid mode when nu > 1."""

    def construct(self):
        L = 1.0
        N = 81
        x = np.linspace(0.0, L, N)
        n_steps = 200

        sigma = 0.05
        x0 = 0.30
        w_init = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
        w_init[0] = 0.0
        w_init[-1] = 0.0

        def simulate(nu):
            w_prev = w_init.copy()
            w_curr = w_init.copy()
            traj = np.zeros((n_steps, N))
            traj[0] = w_curr
            for n in range(1, n_steps):
                w_next = np.zeros(N)
                w_next[1:-1] = (
                    2.0 * w_curr[1:-1]
                    - w_prev[1:-1]
                    + nu**2
                    * (w_curr[2:] - 2.0 * w_curr[1:-1] + w_curr[:-2])
                )
                w_prev, w_curr = w_curr, w_next
                traj[n] = w_curr
            return traj

        configs = [
            (0.9, STABLE_COLOR, r"\nu = 0.9", "stable"),
            (1.0, MAGIC_COLOR, r"\nu = 1.0", "magic (exact)"),
            (1.01, BLOWUP_COLOR, r"\nu = 1.01", "blow-up"),
        ]
        trajs = [simulate(nu) for nu, _, _, _ in configs]

        title = MathTex(
            r"\text{CFL:}\quad \nu \;=\; \hat T\, \frac{\Delta t}{\Delta x}",
            font_size=40,
        ).to_edge(UP, buff=0.3)

        time_idx = ValueTracker(0.0)
        y_max = 1.4

        panel_x = [-4.5, 0.0, 4.5]
        panels = VGroup()
        curves = []
        amp_texts = []

        for col, (nu, color, label_tex, status) in enumerate(configs):
            axes = Axes(
                x_range=[0.0, L, 0.25],
                y_range=[-y_max, y_max, 0.5],
                x_length=3.8,
                y_length=3.0,
                axis_config={"include_tip": False, "stroke_width": 0.8},
            ).move_to([panel_x[col], -0.3, 0.0])

            panel_label = MathTex(label_tex, font_size=32, color=color).next_to(
                axes, UP, buff=0.15
            )
            status_label = Text(
                status, font_size=18, color=color
            ).next_to(panel_label, UP, buff=0.05)

            traj = trajs[col]

            def make_curve(axes=axes, traj=traj, color=color):
                def f(xq, traj=traj):
                    n = int(time_idx.get_value())
                    n = max(0, min(n_steps - 1, n))
                    y = np.interp(xq, x, traj[n])
                    return float(np.clip(y, -y_max, y_max))

                return axes.plot(
                    f,
                    x_range=[0.0, L, 0.003],
                    color=color,
                    stroke_width=2.2,
                )

            curve = always_redraw(make_curve)

            def make_amp_text(axes=axes, traj=traj, color=color):
                n = int(time_idx.get_value())
                n = max(0, min(n_steps - 1, n))
                m = float(np.max(np.abs(traj[n])))
                if m < 1e3:
                    txt = rf"\max|w| = {m:.2f}"
                elif m > 1e6:
                    txt = r"\max|w| > 10^6"
                else:
                    exp = int(np.floor(np.log10(m)))
                    mant = m / 10**exp
                    txt = rf"\max|w| = {mant:.2f}\times 10^{{{exp}}}"
                amp_label = MathTex(txt, font_size=22, color=color).next_to(
                    axes, DOWN, buff=0.35
                )
                amp_label.add_background_rectangle(
                    color=BLACK, opacity=0.75, buff=0.05
                )
                amp_label.set_z_index(10)
                return amp_label

            amp_text = always_redraw(make_amp_text)

            panels.add(axes, panel_label, status_label)
            curves.append(curve)
            amp_texts.append(amp_text)

        time_text = always_redraw(
            lambda: MathTex(
                rf"n = {int(time_idx.get_value())} / {n_steps - 1}",
                font_size=26,
            ).to_edge(DOWN, buff=0.3)
        )

        self.play(Write(title))
        self.play(Create(panels), run_time=1.5)
        for c, a in zip(curves, amp_texts):
            self.add(c, a)
        self.add(time_text)
        self.wait(0.6)
        self.play(
            time_idx.animate.set_value(float(n_steps - 1)),
            run_time=9.0,
            rate_func=linear,
        )
        self.wait(1.0)
