# Physical Modelling and Machine Learning

[![Discord](https://img.shields.io/badge/Discord-Join%20the%20course-5865F2?logo=discord&logoColor=white)](https://discord.gg/rtrq3PQCHJ)

Course materials for the **WiSe 2026/2027** seminar on Differentiable Physical Modelling and Machine Learning for Sound Synthesis.

## Course Information

- **Format:** Seminar (SE)
- **Time:** Thursdays, 14:00–16:00, from **15 October 2026 to 11 February 2027** (local Berlin time)
- **Room:** [FH 313, Fraunhoferstraße 33–36, 10587 Berlin](https://www.tu.berlin/en/studienberatung/advising-information/studying-with-impairments/accessibility/fh-fraunhoferstrasse-gebaeude)
- **Christmas break:** No sessions on **24 or 31 December 2026**. Sessions resume on **7 January 2027**.
- **Credits:** 3 ECTS
- **Instructor:** Rodrigo Diaz, [diaz.fernandez@campus.tu-berlin.de](mailto:diaz.fernandez@campus.tu-berlin.de)

See the [introductory slides](week01/slides/index.qmd) for the course schedule and provisional project deadlines.

## Setup

```sh
uv sync
```

## Render Slides

Slides are rendered with [Quarto](https://quarto.org), a standalone CLI that is not a Python package and must be installed separately (e.g. `brew install quarto` on macOS, or via the installer at quarto.org).

```sh
quarto render week01/slides/index.qmd
```
