# Offline univariate degradation analysis

For a graph window with interactive controls, run:

```bash
python models/univariate/explore_univariate.py
```

The viewer opens separate windows for the sensor fit, residuals, and model comparison. Sliders in the sensor-fit window change the engine, switch-point count, and polynomial degree. The curve plot compares the selected fit (orange) with the lowest-BIC fit (purple dashed). The heatmap shows each candidate's **BIC difference from the best model**; its legend labels the red outline as the slider selection and the star as the lowest-BIC pair. Edit `SENSOR` near the top of `explore_univariate.py` to inspect a different single sensor. Close the graph windows when finished.

Negative raw BIC values are valid. BIC is a relative score: smaller is better, and a zero difference on the heatmap marks the best candidate. The fixed equal-lifetime boundaries need not line up with the sensor's visible bend, so a smooth cubic can score better than a split model.

To write parameter tables and plots to files, run the batch script from the repository root:

```bash
python models/univariate/offline_univariate.py --sensor sensor_11 --switch-points 1 --degree 1
```

`--sensor` accepts one raw C-MAPSS sensor (`sensor_1` through `sensor_21`, or its number). `--degree` accepts 1, 2, or 3. `--switch-points K` makes `K + 1` phases. `--max-switch-points` sets the upper end of the BIC comparison grid (default 4). Use `--output-dir` to choose where results are written. The selected fit is always the pair supplied on the command line; the grid only recommends a pair.

The input is the complete FD001 training trajectories. Each engine is split into phases containing approximately equal numbers of cycles across its observed lifetime. Within each phase, an independent polynomial is fitted by least squares using powers of `cycle / 100`. Residual variance is estimated by Gaussian maximum likelihood (`RSS / number of observations`). The phase fits may have a jump at a boundary because they are estimated independently.

Across engines, each phase's polynomial coefficients are summarized by their maximum-likelihood mean and covariance. Its residual variances are fitted to an inverse-gamma distribution, and its phase durations to a normal distribution. BIC is summed across engines: each engine contributes `-2 log likelihood + p log(n)`, with `p = (degree + 2) × (K + 1)` for polynomial coefficients and phase variances. Switch-point locations have no fitted parameters because the equal-spacing rule fixes them from the completed trajectory.

The default output directory is `results/univariate/`, organized as:

- `csv/unit_phase_fits.csv`: phase estimates for every engine.
- `csv/fitted_curves.csv`: observed and fitted values at every cycle.
- `csv/bic_comparison.csv`: scores for candidate switch-point counts and degrees.
- `parameters/population_parameters.json`: selected settings and population distributions.
- `images/example_fits.png`: four example engines with phase boundaries.

The result is a retrospective, one-sensor degradation fit. Because boundaries use each engine's completed lifetime, the fitted curves include future information when viewed from an earlier cycle. A later Cox survival fit must form time-varying degradation covariates without that future information. The remaining offline work is to fit the Cox degradation effect with its risk sets and then estimate the baseline hazard. FD001 has no natural engine-level fixed covariate like the battery type in the paper.
