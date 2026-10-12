# Multivariate Degradation Signal Analysis

Bachelor thesis project at IIT Guwahati under the guidance of Prof. Ayon Ganguly. This repository collects degradation modelling experiments, datasets, and their generated results in one place.

## Start here

The current runnable model is a **single-sensor degradation analysis** of C-MAPSS FD001 sensor 11. From this repository's root, run:

```bash
python models/univariate/explore_univariate.py
```

This opens separate interactive plots for the selected engine's signal fit and residuals, plus a model-comparison heatmap. Sliders change the engine, number of switch points, and polynomial degree. The heatmap compares BIC scores; it is not an RUL accuracy plot.

To save fitted parameters and CSV results:

```bash
python models/univariate/offline_univariate.py --sensor sensor_11 --switch-points 1 --degree 1
```

Install dependencies with `python -m pip install -r requirements.txt`. See [the univariate model guide](models/univariate/README.md) for model details and limitations.

## Repository map

| Section | Contents |
| --- | --- |
| `models/univariate/` | One-sensor offline degradation fit, interactive viewer, and guide |
| `models/bivariate/` | Copula exploration notebook |
| `models/multivariate/` | Multivariate degradation analysis notebook |
| `models/baselines/` | Earlier PCA and Wiener-process RUL baseline |
| `datasets/CMAPSS/` | NASA C-MAPSS train, test, RUL files, and dataset documentation |
| `results/univariate/csv/` | Per-engine phase fits, fitted sensor curves, and BIC comparison |
| `results/univariate/parameters/` | Fitted population parameters and selected model metadata |
| `results/univariate/images/` | Example sensor fit figures |
| `results/images/` | Other project figures |
| `references/` | Citations and source links for research papers |

The C-MAPSS text files are included so the experiments can run locally. Research article PDFs and extracted article text are not redistributed; citation details are in `references/`.
