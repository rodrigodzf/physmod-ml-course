# Physical Modelling and Machine Learning

[![Discord](https://img.shields.io/badge/Discord-Join%20the%20course-5865F2?logo=discord&logoColor=white)](https://discord.gg/cYFe46hbE4)

Course materials for the **WiSe 2026/2027** seminar on Differentiable Physical Modelling and Machine Learning for Sound Synthesis.

## Course Information

- **Format:** Seminar (SE)
- **Time:** Thursdays, 14:15–16:00, from **15 October 2026 to 11 February 2027** (local Berlin time)
- **Room:** [FH 313, Fraunhoferstraße 33–36, 10587 Berlin](https://www.tu.berlin/en/studienberatung/advising-information/studying-with-impairments/accessibility/fh-fraunhoferstrasse-gebaeude)
- **Christmas break:** No sessions on **24 or 31 December 2026**. Sessions resume on **7 January 2027**.
- **Credits:** 3 ECTS
- **Instructor:** Rodrigo Diaz, [diaz.fernandez@campus.tu-berlin.de](mailto:diaz.fernandez@campus.tu-berlin.de)

See the [introductory slides](week01/slides/index.qmd) for provisional project deadlines.

## Course Schedule

All sessions run on Thursdays, 14:15–16:00, in local Berlin time.

| Session | Date | Topic and slides |
|---------|------------|-------|
| 1 | 15.10.2026 | [Introduction to Physical Modelling](week01/slides/index.qmd) |
| 2 | 22.10.2026 | [Modal and Finite-Difference Methods](week02/slides/index.qmd) |
| 3 | 29.10.2026 | [JAX Tutorial: Differentiable Programming for Audio](week03/slides/index.qmd) |
| 4 | 05.11.2026 | [Nonlinear Systems and Coupled Models](week04/slides/index.qmd) |
| 5 | 12.11.2026 | [Neural ODEs](week05/slides/index.qmd) |
| 6 | 19.11.2026 | [Physics-Informed Neural Networks](week06/slides/index.qmd) |
| 7 | 26.11.2026 | [Neural Operators (FNO)](week07/slides/index.qmd) |
| 8 | 03.12.2026 | [Deep State-Space Models and Koopman Theory](week08/slides/index.qmd) |
| 9 | 10.12.2026 | [Differentiable Physical Modelling for Parameter Estimation](week09/slides/index.qmd) |
| 10 | 17.12.2026 | Project Pitches and Feedback |
| 11 | 07.01.2027 | Project Workshop: Model Implementation |
| 12 | 14.01.2027 | Project Workshop: Parameter Estimation and Evaluation |
| 13 | 21.01.2027 | Project Progress and Feedback |
| 14 | 28.01.2027 | Project Workshop: Results and Paper Preparation |
| 15 | 04.02.2027 | Questions and Office Hours |
| 16 | 11.02.2027 | Project Presentations |

## Setup

```sh
uv sync
```

## Week 3: JAX Tutorial

**29 October 2026, 14:15–16:00.** Use the
[Differentiable Programming for Audio slides](week03/slides/index.qmd) alongside
the following two Python files, in order.

Session 3 starts with [JAX basics](week03/notebooks/jax_basics.py), the opening
0 to 25 minute guided introduction using runnable `# %%` cells, before the
longer differentiable-synthesis exercise. Open the file in an editor with
Python cell support, select the course Python environment, and run the cells
from top to bottom. It covers arrays, gradients, fitting a sinusoid's amplitude,
and compilation, followed by an optional batching example.

Continue with the [differentiable sinusoidal synth](week03/notebooks/differentiable_sinusoidal_synth.py).
It builds on the introduction by fitting amplitude and damping, introducing
parameter dictionaries and Optax, and exploring frequency initialisation.
The complex-recursion section is optional.

### Session Sequence

| Time | Slides and coding |
|-------|-------|
| 14:15–14:40 | Opening slides and `jax_basics.py`: arrays, gradients, amplitude fitting and compilation. |
| 14:40–15:00 | Damped-sinusoid slides and the main notebook: fit amplitude and damping with frequency fixed. |
| 15:00–15:15 | The JAX Pattern slide, parameter dictionaries and Optax. |
| 15:15–15:40 | Frequency-estimation slides and experiments: initial guesses, loss landscapes and convergence. |
| 15:40–16:00 | Discussion and wrap-up; optional complex recursion with `lax.scan`. |

## Week 4: Nonlinear Systems and Coupled Models

**5 November 2026, 14:15–16:00.** Use the
[Nonlinear Systems and Coupled Models slides](week04/slides/index.qmd) with the
[complex modal recursion](week04/notebooks/complex_modal_recursion.py) and
[nonlinear tension modulation](week04/notebooks/nonlinear_tension_modulation.py)
activities.

## Render Slides

Slides are rendered with [Quarto](https://quarto.org), a standalone CLI that is not a Python package and must be installed separately (e.g. `brew install quarto` on macOS, or via the installer at quarto.org).

Render the Week 3 and Week 4 decks with:

```sh
quarto render week03/slides/index.qmd
quarto render week04/slides/index.qmd
```

## Regenerate the Week 1 Video

The boundary-conditions video compares a raised-cosine pulse reflecting at a fixed (Dirichlet) end and a free (Neumann) end. The fixed end inverts the reflected pulse; the free end preserves its polarity.

```sh
uv sync --group animations
source .venv/bin/activate
python week01/slides/animations/string_boundary_conditions.py
```

The generator saves a browser-compatible H.264 MP4 to `week01/slides/images/string_boundary_conditions.mp4`, the path used by the introductory slides.

## Regenerate the State Animation

The Manim animation shows why the same string shape can lead to different motion, then compares states using displacement and velocity, two consecutive displacement steps, and displacement and momentum.

```sh
uv sync
source .venv/bin/activate
python week01/slides/animations/what_is_a_state.py
```

The generator saves `week01/slides/images/what_is_a_state.mp4` and a matching `.jpg` poster. Use `--quality low_quality` for a quick preview.

## Interactive Energy Example

Week 1 embeds `week01/slides/widgets/energy-stability.html`. Its editable fragment is `energy-stability.fragment.html` in the same directory. After editing the fragment, re-export it with the Visualize standalone renderer. The example compares Forward Euler, centred differences (Störmer-Verlet) and classical RK4. The slider and numeric input cover time steps from 0.001 s to 3 s, including unstable regimes for Verlet and RK4.
