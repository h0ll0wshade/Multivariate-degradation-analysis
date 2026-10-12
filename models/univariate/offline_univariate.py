"""Offline, single-sensor degradation fitting for C-MAPSS FD001.

Each complete engine trajectory is divided into equally sized phases. A
separate Gaussian polynomial regression is fitted in each phase. This is the
degradation part of the paper's two-step offline procedure; it does not fit a
survival model or predict RUL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "btp_mplconfig"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import invgamma


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "CMAPSS" / "train_FD001.txt"
COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{number}" for number in range(1, 22)]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sensor", default="sensor_11", help="One raw sensor, e.g. sensor_11 or 11")
    parser.add_argument("--switch-points", type=int, default=1)
    parser.add_argument("--degree", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument(
        "--max-switch-points", type=int, default=4,
        help="Compare candidate counts from 0 through this value",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "univariate")
    args = parser.parse_args()
    if args.sensor.isdigit():
        args.sensor = f"sensor_{args.sensor}"
    if args.sensor not in COLUMNS[5:]:
        parser.error("--sensor must name one of sensor_1 through sensor_21")
    if args.switch_points < 0 or args.max_switch_points < 0:
        parser.error("switch-point counts must be nonnegative")
    return args


def load_data(path: Path, sensor: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if data.empty or data.shape[1] != len(COLUMNS):
        raise ValueError("Expected a nonempty C-MAPSS file with 26 columns")
    data = data[["unit", "cycle", sensor]].sort_values(["unit", "cycle"])
    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Unit, cycle, and sensor values must all be finite")
    if data.duplicated(["unit", "cycle"]).any():
        raise ValueError("Each engine must have at most one reading per cycle")
    if (data["cycle"] <= 0).any():
        raise ValueError("Cycle numbers must be positive")
    return data


def fit_unit(
    group: pd.DataFrame, sensor: str, switch_points: int, degree: int,
    collect_curves: bool = False,
) -> tuple[list[dict], float, pd.DataFrame | None]:
    """Return phase estimates, unit BIC, and optional observation-level fits."""
    n_total = len(group)
    phases = np.array_split(np.arange(n_total), switch_points + 1)
    if any(len(indices) <= degree + 1 for indices in phases):
        raise ValueError(
            f"Engine {group['unit'].iloc[0]} has too few readings for "
            f"{switch_points} switch points and degree {degree}"
        )

    cycle = group["cycle"].to_numpy(dtype=float)
    observed = group[sensor].to_numpy(dtype=float)
    fitted = np.empty(n_total, dtype=float) if collect_curves else None
    rows: list[dict] = []
    total_log_likelihood = 0.0
    for phase_number, indices in enumerate(phases, start=1):
        # One common time scale keeps coefficients comparable between engines.
        time = cycle[indices] / 100.0
        design = np.polynomial.polynomial.polyvander(time, degree)
        coefficients, _, rank, _ = np.linalg.lstsq(design, observed[indices], rcond=None)
        if rank != degree + 1:
            raise ValueError(f"Rank-deficient polynomial fit for engine {group['unit'].iloc[0]}")
        prediction = design @ coefficients
        residuals = observed[indices] - prediction
        rss = float(residuals @ residuals)
        count = len(indices)
        if rss <= 0 or not math.isfinite(rss):
            raise ValueError(f"Nonpositive residual variance for engine {group['unit'].iloc[0]}")
        variance = rss / count  # Gaussian maximum-likelihood estimate
        total_log_likelihood += -0.5 * count * (math.log(2 * math.pi * variance) + 1)
        if fitted is not None:
            fitted[indices] = prediction
        row = {
            "unit": int(group["unit"].iloc[0]),
            "phase": phase_number,
            "start_cycle": int(cycle[indices[0]]),
            "end_cycle": int(cycle[indices[-1]]),
            "duration_cycles": int(count),
            "n_observations": count,
            "residual_variance": variance,
            "rss": rss,
            "rmse": math.sqrt(variance),
        }
        row.update({f"coefficient_{order}": float(value) for order, value in enumerate(coefficients)})
        rows.append(row)

    # Boundaries are fixed by the complete trajectory length, not estimated.
    parameter_count = (degree + 2) * (switch_points + 1)
    bic = -2 * total_log_likelihood + parameter_count * math.log(n_total)
    curves = None
    if fitted is not None:
        curves = group.copy()
        curves["fitted"] = fitted
        curves["phase"] = np.concatenate(
            [np.full(len(indices), number, dtype=int) for number, indices in enumerate(phases, 1)]
        )
    return rows, bic, curves


def fit_fleet(
    data: pd.DataFrame, sensor: str, switch_points: int, degree: int,
    collect_curves: bool = False,
) -> tuple[pd.DataFrame, float, pd.DataFrame | None]:
    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    total_bic = 0.0
    for _, group in data.groupby("unit", sort=True):
        unit_rows, unit_bic, unit_curves = fit_unit(
            group, sensor, switch_points, degree, collect_curves
        )
        rows.extend(unit_rows)
        total_bic += unit_bic
        if unit_curves is not None:
            curves.append(unit_curves)
    return pd.DataFrame(rows), total_bic, pd.concat(curves, ignore_index=True) if curves else None


def population_parameters(fits: pd.DataFrame, degree: int) -> list[dict]:
    population = []
    columns = [f"coefficient_{order}" for order in range(degree + 1)]
    for phase, group in fits.groupby("phase", sort=True):
        coefficients = group[columns].to_numpy(dtype=float)
        variances = group["residual_variance"].to_numpy(dtype=float)
        durations = group["duration_cycles"].to_numpy(dtype=float)
        if len(group) < 2:
            raise ValueError("At least two engines are needed to estimate population variation")
        shape, _, scale = invgamma.fit(variances, floc=0)
        population.append({
            "phase": int(phase),
            "engines": len(group),
            "coefficient_mean": coefficients.mean(axis=0).tolist(),
            "coefficient_covariance_mle": np.cov(coefficients, rowvar=False, bias=True).tolist(),
            "residual_variance_inverse_gamma": {
                "shape": float(shape), "scale": float(scale)
            },
            "duration_normal": {
                "mean_cycles": float(durations.mean()),
                "variance_cycles_squared_mle": float(durations.var()),
            },
        })
    return population


def bic_comparison(data: pd.DataFrame, sensor: str, max_switch_points: int) -> pd.DataFrame:
    results = []
    for switch_points in range(max_switch_points + 1):
        for degree in (1, 2, 3):
            try:
                _, bic, _ = fit_fleet(data, sensor, switch_points, degree)
                results.append({"switch_points": switch_points, "degree": degree, "bic": bic})
            except ValueError as exc:
                results.append({
                    "switch_points": switch_points, "degree": degree,
                    "bic": np.nan, "error": str(exc),
                })
    return pd.DataFrame(results).sort_values("bic", na_position="last").reset_index(drop=True)


def plot_examples(curves: pd.DataFrame, fits: pd.DataFrame, sensor: str, output: Path) -> None:
    unit_ids = curves["unit"].drop_duplicates().head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, unit_id in zip(axes.flat, unit_ids):
        unit = curves[curves["unit"] == unit_id]
        ax.plot(unit["cycle"], unit[sensor], color="steelblue", alpha=0.65, label="Observed")
        ax.plot(unit["cycle"], unit["fitted"], color="darkorange", label="Phase fit")
        unit_phases = fits[fits["unit"] == unit_id]
        for boundary in unit_phases["end_cycle"].iloc[:-1]:
            ax.axvline(boundary + 0.5, color="gray", linestyle="--", alpha=0.7)
        ax.set(title=f"Engine {unit_id}", xlabel="Cycle", ylabel=sensor)
        ax.legend()
    for ax in axes.flat[len(unit_ids):]:
        ax.set_visible(False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_data(args.data, args.sensor)
    fits, selected_bic, curves = fit_fleet(
        data, args.sensor, args.switch_points, args.degree, collect_curves=True
    )
    assert curves is not None
    comparison = bic_comparison(data, args.sensor, max(args.max_switch_points, args.switch_points))
    finite = comparison.dropna(subset=["bic"])
    if finite.empty:
        raise ValueError("No candidate model could be fitted")
    best = finite.iloc[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output_dir / "csv"
    parameter_dir = args.output_dir / "parameters"
    image_dir = args.output_dir / "images"
    csv_dir.mkdir(parents=True, exist_ok=True)
    parameter_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    fits.to_csv(csv_dir / "unit_phase_fits.csv", index=False)
    curves.to_csv(csv_dir / "fitted_curves.csv", index=False)
    comparison.to_csv(csv_dir / "bic_comparison.csv", index=False)
    plot_examples(curves, fits, args.sensor, image_dir / "example_fits.png")
    report = {
        "dataset": str(args.data.resolve()),
        "sensor": args.sensor,
        "engines": int(data["unit"].nunique()),
        "observations": len(data),
        "time_basis": "(cycle / 100) ** polynomial_order",
        "boundary_rule": "equal numbers of observed cycles across each completed engine lifetime",
        "selected_model": {
            "switch_points": args.switch_points, "degree": args.degree, "bic": selected_bic
        },
        "lowest_bic_candidate": {
            "switch_points": int(best["switch_points"]),
            "degree": int(best["degree"]),
            "bic": float(best["bic"]),
        },
        "population_parameters": population_parameters(fits, args.degree),
        "interpretation": (
            "Phase locations use completed lifetimes. Duration estimates are descriptive "
            "and these retrospective fits cannot be used as earlier-time Cox covariates "
            "without removing future information."
        ),
    }
    with (parameter_dir / "population_parameters.json").open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"Fitted {report['engines']} engines ({report['observations']} observations) using {args.sensor}.")
    print(f"Selected: {args.switch_points} switch points, degree {args.degree}, BIC {selected_bic:.2f}")
    print(
        f"Lowest BIC: {int(best['switch_points'])} switch points, "
        f"degree {int(best['degree'])}, BIC {best['bic']:.2f}"
    )
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
