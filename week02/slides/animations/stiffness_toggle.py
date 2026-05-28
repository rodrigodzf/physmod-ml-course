import numpy as np
from manim import *

class StiffnessToggle(Scene):
    def construct(self):
        D_hat = ValueTracker(0.0)
        T_hat = ValueTracker(1.0)
        
        axes = Axes(
            x_range=[0, 20, 2],
            y_range=[0, 150, 30],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "include_tip": False}
        )
        axes.move_to(DOWN*0.5)
        axes_labels = axes.get_axis_labels(
            x_label=MathTex(r"\mu"), y_label=MathTex(r"\omega_\mu")
        )
        
        title = MathTex(r"\omega_\mu = \sqrt{\hat T^2 \lambda_\mu^2 + \hat D^2 \lambda_\mu^4}", font_size=40)
        title.to_edge(UP, buff=0.5)
        
        def get_curve():
            D = D_hat.get_value()
            T = T_hat.get_value()
            # use a dense sampling for a smooth curve
            return axes.plot(
                lambda mu: np.sqrt(T**2 * (mu*np.pi)**2 + D**2 * (mu*np.pi)**4),
                color=YELLOW
            )
            
        curve = always_redraw(get_curve)
        
        label_group = always_redraw(lambda: VGroup(
            MathTex(rf"\hat T = {T_hat.get_value():.2f}", font_size=36),
            MathTex(rf"\hat D = {D_hat.get_value():.4f}", font_size=36)
        ).arrange(RIGHT, buff=1.0).next_to(title, DOWN, buff=0.3))
        
        self.play(Create(axes), Write(axes_labels), Write(title))
        self.add(curve, label_group)
        
        self.wait(1.0)
        self.play(T_hat.animate.set_value(2.0), run_time=3.0)
        self.wait(1.0)
        self.play(T_hat.animate.set_value(1.0), run_time=2.0)
        self.wait(1.0)
        self.play(D_hat.animate.set_value(0.030), run_time=3.0)
        self.wait(1.0)
        self.play(D_hat.animate.set_value(0.0), run_time=2.0)
        self.wait(1.0)
