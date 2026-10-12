"""Open interactive graphs for the single-sensor C-MAPSS degradation fit.

Run `python models/univariate/explore_univariate.py` from the repository root.
Edit the constants below to change the default sensor or BIC comparison range.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "btp_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from offline_univariate import DEFAULT_DATA, bic_comparison, fit_unit, load_data


SENSOR = "sensor_11"
MAX_SWITCH_POINTS = 4
START_SWITCH_POINTS = 1
START_DEGREE = 1
START_ENGINE = 1


def draw_phase_fit(ax, curves, label, color, linestyle="-"):
    for phase, group in curves.groupby("phase", sort=True):
        ax.plot(
            group["cycle"], group["fitted"], color=color,
            linestyle=linestyle, linewidth=2,
            label=label if phase == 1 else None,
        )


def main() -> None:
    if plt.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo"}:
        raise RuntimeError(
            "Matplotlib has no interactive backend. On macOS, run this script "
            "from a desktop Python session without MPLBACKEND=Agg."
        )
    data = load_data(DEFAULT_DATA, SENSOR)
    groups = {int(unit): group for unit, group in data.groupby("unit", sort=True)}
    unit_ids = sorted(groups)
    comparison = bic_comparison(data, SENSOR, MAX_SWITCH_POINTS)
    finite = comparison.dropna(subset=["bic"])
    best = finite.iloc[0]
    best_k, best_degree = int(best["switch_points"]), int(best["degree"])
    bic_map = np.full((MAX_SWITCH_POINTS + 1, 3), np.nan)
    for row in finite.itertuples():
        bic_map[row.switch_points, row.degree - 1] = row.bic
    delta_bic = bic_map - float(best["bic"])

    curve_fig, curve_ax = plt.subplots(figsize=(10, 6))
    curve_fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.25)
    curve_fig.suptitle(f"C-MAPSS FD001 • {SENSOR} • selected engine", fontsize=15)

    residual_fig, residual_ax = plt.subplots(figsize=(10, 5))
    residual_fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.16)
    residual_fig.suptitle("Residuals for the selected fit", fontsize=15)

    comparison_fig, heat_ax = plt.subplots(figsize=(9, 6))
    comparison_fig.subplots_adjust(left=0.12, right=0.72, top=0.82, bottom=0.22)
    comparison_fig.suptitle("Model comparison across all 100 engines (BIC)", fontsize=15)

    image = heat_ax.imshow(delta_bic, cmap="viridis_r", aspect="auto", origin="lower")
    comparison_fig.colorbar(image, ax=heat_ax, label="BIC above best (lower is better)")
    heat_ax.set(
        title="ΔBIC relative to the best candidate (lower is better)",
        xlabel="Polynomial degree", ylabel="Number of switch points",
        xticks=np.arange(3), yticks=np.arange(MAX_SWITCH_POINTS + 1),
        xticklabels=[1, 2, 3], yticklabels=list(range(MAX_SWITCH_POINTS + 1)),
    )
    for k in range(MAX_SWITCH_POINTS + 1):
        for degree in range(1, 4):
            value = delta_bic[k, degree - 1]
            if np.isfinite(value):
                heat_ax.text(degree - 1, k, f"{value:,.0f}", ha="center", va="center", fontsize=9)
    heat_ax.scatter(best_degree - 1, best_k, marker="*", s=210, color="#20e060", edgecolors="black", linewidths=0.8, zorder=4)
    selection = Rectangle((-0.5, START_SWITCH_POINTS - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5)
    heat_ax.add_patch(selection)
    comparison_fig.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2.5, label="Slider-selected model"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#20e060", markeredgecolor="black", markersize=14, label="Lowest BIC"),
        ],
        loc="upper left", bbox_to_anchor=(0.76, 0.78), borderaxespad=0,
    )
    switch_ax = curve_fig.add_axes((0.18, 0.15, 0.68, 0.025))
    degree_ax = curve_fig.add_axes((0.18, 0.10, 0.68, 0.025))
    engine_ax = curve_fig.add_axes((0.18, 0.05, 0.68, 0.025))
    switch_slider = Slider(switch_ax, "Switch points", 0, MAX_SWITCH_POINTS, valinit=START_SWITCH_POINTS, valstep=1)
    degree_slider = Slider(degree_ax, "Degree", 1, 3, valinit=START_DEGREE, valstep=1)
    engine_slider = Slider(engine_ax, "Engine", min(unit_ids), max(unit_ids), valinit=START_ENGINE, valstep=unit_ids)

    def redraw(_=None):
        k = int(switch_slider.val)
        degree = int(degree_slider.val)
        engine_index = unit_ids.index(int(engine_slider.val))
        selected_unit = unit_ids[engine_index]
        selected_curves = None

        group = groups[selected_unit]
        chosen_rows, _, chosen_curves = fit_unit(group, SENSOR, k, degree, collect_curves=True)
        _, _, best_curves = fit_unit(group, SENSOR, best_k, best_degree, collect_curves=True)
        assert chosen_curves is not None and best_curves is not None
        selected_curves = chosen_curves
        curve_ax.clear()
        curve_ax.plot(group["cycle"], group[SENSOR], color="#6687aa", alpha=0.65, linewidth=1.2, label="Observed sensor")
        draw_phase_fit(curve_ax, chosen_curves, "Slider-selected fit", "#e77817")
        if (k, degree) != (best_k, best_degree):
            draw_phase_fit(curve_ax, best_curves, "Lowest-BIC fit", "#8a47a6", "--")
        for row in chosen_rows[:-1]:
            curve_ax.axvline(row["end_cycle"] + 0.5, color="gray", linestyle=":", linewidth=1)
        curve_ax.set(
            title=f"Engine {selected_unit} • {k} switch points • degree {degree}",
            xlabel="Cycle", ylabel=SENSOR,
        )
        curve_ax.legend(loc="upper left")

        residual_ax.clear()
        residuals = selected_curves[SENSOR] - selected_curves["fitted"]
        residual_ax.scatter(selected_curves["cycle"], residuals, s=9, alpha=0.55, color="#e77817")
        residual_ax.axhline(0, color="black", linewidth=1)
        residual_ax.set(
            title=f"Engine {selected_unit} • RMSE {np.sqrt(np.mean(residuals**2)):.3f}",
            xlabel="Cycle", ylabel="Observed − selected fit",
        )
        selection.set_xy((degree - 1 - 0.5, k - 0.5))
        curve_fig.canvas.draw_idle()
        residual_fig.canvas.draw_idle()
        comparison_fig.canvas.draw_idle()

    switch_slider.on_changed(redraw)
    degree_slider.on_changed(redraw)
    engine_slider.on_changed(redraw)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
