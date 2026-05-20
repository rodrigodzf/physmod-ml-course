import numpy as np
import jax.numpy as jnp
from jaxdiffmodal.ftm import (
    circ_laplacian_eigenvalues,
    circ_laplacian_wavenumbers,
    plate_eigenfunctions,
    plate_eigenvalues,
    plate_wavenumbers,
)
from manim import (
    BLUE_D,
    BLUE_E,
    DOWN,
    GOLD_D,
    GOLD_E,
    GREEN_D,
    GREEN_E,
    GREY_B,
    Line3D,
    MathTex,
    Surface,
    TAU,
    Text,
    ThreeDScene,
    UP,
    VGroup,
    ValueTracker,
    always_redraw,
    linear,
)
from scipy.special import jv


def bilinear_sample(values: np.ndarray, u: float, v: float) -> float:
    """Sample a precomputed mode shape on [0, 1] x [0, 1]."""
    nx, ny = values.shape
    sx = np.clip(u, 0.0, 1.0) * (nx - 1)
    sy = np.clip(v, 0.0, 1.0) * (ny - 1)
    i0 = int(np.floor(sx))
    j0 = int(np.floor(sy))
    i1 = min(i0 + 1, nx - 1)
    j1 = min(j0 + 1, ny - 1)
    tx = sx - i0
    ty = sy - j0

    return float(
        (1.0 - tx) * (1.0 - ty) * values[i0, j0]
        + tx * (1.0 - ty) * values[i1, j0]
        + (1.0 - tx) * ty * values[i0, j1]
        + tx * ty * values[i1, j1]
    )


def polar_sample(values: np.ndarray, radius: float, theta: float) -> float:
    """Sample a circular mode shape on a polar grid."""
    theta_unit = (theta % TAU) / TAU
    return bilinear_sample(values, radius, theta_unit)


def circular_mode_shape(
    order: int,
    wavenumber: float,
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate J_n(k r) cos(n theta) on a polar grid."""
    r_mesh, theta_mesh = np.meshgrid(r_grid, theta_grid, indexing="ij")
    radial = jv(order, wavenumber * r_mesh)
    if order == 0:
        return radial
    return radial * np.cos(order * theta_mesh)


class MembraneModes2D(ThreeDScene):
    """Rectangular then circular membrane eigenmodes computed with jaxdiffmodal."""

    def construct(self):
        self.set_camera_orientation(phi=63 * np.pi / 180, theta=-58 * np.pi / 180)
        self.camera.set_zoom(0.82)
        self.begin_ambient_camera_rotation(rate=0.045)

        rectangular_group = self._build_rectangular_modes()
        circular_group = self._build_circular_modes()

        self.add_fixed_in_frame_mobjects(
            rectangular_group["title"],
            rectangular_group["subtitle"],
            rectangular_group["labels"],
        )
        self.add(
            rectangular_group["title"],
            rectangular_group["subtitle"],
            rectangular_group["frames"],
            rectangular_group["surfaces"],
            rectangular_group["labels"],
        )
        self.play(
            rectangular_group["phase"].animate.set_value(5.0 * TAU),
            run_time=10.0,
            rate_func=linear,
        )
        self.remove_fixed_in_frame_mobjects(
            rectangular_group["title"],
            rectangular_group["subtitle"],
            rectangular_group["labels"],
        )
        self.remove(
            rectangular_group["title"],
            rectangular_group["subtitle"],
            rectangular_group["frames"],
            rectangular_group["surfaces"],
            rectangular_group["labels"],
        )

        self.add_fixed_in_frame_mobjects(
            circular_group["title"],
            circular_group["subtitle"],
            circular_group["labels"],
        )
        self.add(
            circular_group["title"],
            circular_group["subtitle"],
            circular_group["frames"],
            circular_group["surfaces"],
            circular_group["labels"],
        )
        self.play(
            circular_group["phase"].animate.set_value(5.0 * TAU),
            run_time=10.0,
            rate_func=linear,
        )
        self.stop_ambient_camera_rotation()
        self.wait(0.5)

    def _build_rectangular_modes(self):
        length_x = 1.0
        length_y = 1.0
        n_grid = 61
        x_grid = jnp.linspace(0.0, length_x, n_grid)
        y_grid = jnp.linspace(0.0, length_y, n_grid)

        wavenumbers_x, wavenumbers_y = plate_wavenumbers(3, 3, length_x, length_y)
        lambda_mu_squared = np.asarray(
            plate_eigenvalues(wavenumbers_x, wavenumbers_y)
        )
        modes = np.asarray(
            plate_eigenfunctions(wavenumbers_x, wavenumbers_y, x_grid, y_grid)
        )

        mode_specs = [
            {"mn": (1, 1), "center": -4.0, "colors": (BLUE_D, BLUE_E)},
            {"mn": (2, 1), "center": 0.0, "colors": (GOLD_D, GOLD_E)},
            {"mn": (2, 2), "center": 4.0, "colors": (GREEN_D, GREEN_E)},
        ]

        omega_11 = np.sqrt(lambda_mu_squared[0, 0])
        panel_size = 2.45
        amplitude = 0.58
        phase = ValueTracker(0.0)

        title = MathTex(
            r"\phi_{m,n}(x,y)=\sin\!\left(\frac{m\pi x}{L}\right)"
            r"\sin\!\left(\frac{n\pi y}{L}\right)",
            font_size=38,
        ).to_edge(UP, buff=0.25)

        subtitle = Text(
            "Square Dirichlet membrane modes",
            font_size=24,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.12)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.add(title, subtitle)

        surfaces = VGroup()
        frames = VGroup()
        labels = VGroup()

        for spec in mode_specs:
            m, n = spec["mn"]
            center_x = spec["center"]
            mode_shape = modes[m - 1, n - 1]
            omega_ratio = np.sqrt(lambda_mu_squared[m - 1, n - 1]) / omega_11
            colors = spec["colors"]

            surface = always_redraw(
                lambda mode_shape=mode_shape,
                center_x=center_x,
                omega_ratio=omega_ratio,
                colors=colors: Surface(
                    lambda u, v: np.array(
                        [
                            center_x + panel_size * (u - 0.5),
                            panel_size * (v - 0.5),
                            amplitude
                            * np.cos(omega_ratio * phase.get_value())
                            * bilinear_sample(mode_shape, u, v),
                        ]
                    ),
                    u_range=[0.0, 1.0],
                    v_range=[0.0, 1.0],
                    resolution=(28, 28),
                    checkerboard_colors=colors,
                    fill_opacity=0.92,
                    stroke_color=GREY_B,
                    stroke_width=0.28,
                )
            )
            surfaces.add(surface)

            half = panel_size / 2.0
            z0 = 0.0
            frame = VGroup(
                Line3D([center_x - half, -half, z0], [center_x + half, -half, z0]),
                Line3D([center_x + half, -half, z0], [center_x + half, half, z0]),
                Line3D([center_x + half, half, z0], [center_x - half, half, z0]),
                Line3D([center_x - half, half, z0], [center_x - half, -half, z0]),
            )
            frames.add(frame)

            label = MathTex(
                rf"(m,n)=({m},{n})",
                rf"\quad \omega/\omega_{{1,1}}={omega_ratio:.2f}",
                font_size=26,
            )
            label.move_to([center_x, -2.85, 0.0])
            labels.add(label)

        return {
            "title": title,
            "subtitle": subtitle,
            "frames": frames,
            "surfaces": surfaces,
            "labels": labels,
            "phase": phase,
        }

    def _build_circular_modes(self):
        radius = 1.0
        r_grid = np.linspace(0.0, radius, 61)
        theta_grid = np.linspace(0.0, TAU, 91)

        wavenumbers = circ_laplacian_wavenumbers(3, 1, radius=radius)
        lambda_mu_squared = np.asarray(circ_laplacian_eigenvalues(wavenumbers))
        wavenumbers = np.asarray(wavenumbers)

        mode_specs = [
            {"nm": (0, 1), "center": -4.0, "colors": (BLUE_D, BLUE_E)},
            {"nm": (1, 1), "center": 0.0, "colors": (GOLD_D, GOLD_E)},
            {"nm": (2, 1), "center": 4.0, "colors": (GREEN_D, GREEN_E)},
        ]

        omega_01 = np.sqrt(lambda_mu_squared[0, 0])
        panel_radius = 1.2
        amplitude = 0.55
        phase = ValueTracker(0.0)

        title = MathTex(
            r"\phi_{n,m}(r,\theta)=J_n(\lambda_{n,m}r)\cos(n\theta)",
            font_size=38,
        ).to_edge(UP, buff=0.25)

        subtitle = Text(
            "Circular Dirichlet membrane modes",
            font_size=24,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.12)

        surfaces = VGroup()
        frames = VGroup()
        labels = VGroup()

        for spec in mode_specs:
            n, m = spec["nm"]
            center_x = spec["center"]
            mode_shape = circular_mode_shape(
                n, wavenumbers[n, m - 1], r_grid, theta_grid
            )
            omega_ratio = np.sqrt(lambda_mu_squared[n, m - 1]) / omega_01
            colors = spec["colors"]

            surface = always_redraw(
                lambda mode_shape=mode_shape,
                center_x=center_x,
                omega_ratio=omega_ratio,
                colors=colors: Surface(
                    lambda r, theta: np.array(
                        [
                            center_x + panel_radius * r * np.cos(theta),
                            panel_radius * r * np.sin(theta),
                            amplitude
                            * np.cos(omega_ratio * phase.get_value())
                            * polar_sample(mode_shape, r, theta),
                        ]
                    ),
                    u_range=[0.0, 1.0],
                    v_range=[0.0, TAU],
                    resolution=(24, 48),
                    checkerboard_colors=colors,
                    fill_opacity=0.92,
                    stroke_color=GREY_B,
                    stroke_width=0.25,
                )
            )
            surfaces.add(surface)

            frame = VGroup(
                *[
                    Line3D(
                        [
                            center_x + panel_radius * np.cos(theta),
                            panel_radius * np.sin(theta),
                            0.0,
                        ],
                        [
                            center_x + panel_radius * np.cos(theta + TAU / 96),
                            panel_radius * np.sin(theta + TAU / 96),
                            0.0,
                        ],
                    )
                    for theta in np.linspace(0.0, TAU, 96, endpoint=False)
                ]
            )
            frames.add(frame)

            label = MathTex(
                rf"(n,m)=({n},{m})",
                rf"\quad \omega/\omega_{{0,1}}={omega_ratio:.2f}",
                font_size=26,
            )
            label.move_to([center_x, -2.85, 0.0])
            labels.add(label)

        return {
            "title": title,
            "subtitle": subtitle,
            "frames": frames,
            "surfaces": surfaces,
            "labels": labels,
            "phase": phase,
        }
