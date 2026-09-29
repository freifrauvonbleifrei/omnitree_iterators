"""Directionally adaptive advection versus spatially adaptive octrees.

Both methods regrid from the computed solution every 0.1 time units. The exact
solution supplies initialization, inflow data, and error evaluation only.
"""

import argparse
import csv
import json
from math import log
from pathlib import Path
from time import perf_counter

from omnitree_iterators.validation.adaptive import adaptive_experiment


def plots(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, case in zip(axes, ("stripe3d", "ellipsoid3d")):
        for strategy in ("directional", "isotropic"):
            selected = [
                r for r in rows if r["case"] == case and r["strategy"] == strategy
            ]
            ax.loglog(
                [r["cells"] for r in selected],
                [r["physical_l2"] for r in selected],
                "o-",
                label=strategy,
            )
        ax.set(title=case, xlabel="Number of rectangles N", ylabel="Physical L² error")
        budgets = sorted({r["cells"] for r in rows if r["case"] == case})
        ax.set_xticks(budgets, [str(n) for n in budgets])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"Solution-driven adaptation during upwind advection, t={rows[0]['final_time']:g}"
    )
    fig.savefig(output / "convergence.pdf")
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/adaptive-advection")
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 120, 232, 456])
    parser.add_argument("--final-time", type=float, default=0.3)
    parser.add_argument("--no-lookahead", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in ("stripe3d", "ellipsoid3d"):
        for strategy in ("directional", "isotropic"):
            previous = None
            for budget in args.budgets:
                start = perf_counter()
                row = adaptive_experiment(
                    case,
                    strategy,
                    budget,
                    final_time=args.final_time,
                    lookahead=not args.no_lookahead,
                )
                row["wall_seconds"] = perf_counter() - start
                row["N_rate"] = (
                    None
                    if previous is None or previous["cells"] == row["cells"]
                    else (
                        log(previous["physical_l2"] / row["physical_l2"])
                        / log(row["cells"] / previous["cells"])
                    )
                )
                rows.append(row)
                previous = row
                # Save each finished experiment, including parameters, before the next run.
                (args.output / "measurements.json").write_text(
                    json.dumps(
                        {
                            "indicator": "hierarchical-haar-v2-partial: subtree detail energy and limited child prediction",
                            "initializer": "four-slab directional variance with optional translated initial-data probes",
                            "ranking": "subtree energy / added leaves; directional predicted split on strongest axis",
                            "max_level_per_axis": 9,
                            "absolute_detail_floor_squared": 1e-28,
                            "seed_level": 1,
                            "cfl": 0.8,
                            "transport_lookahead": not args.no_lookahead,
                            "initial_probe_times": [0, 0.05, 0.1]
                            if not args.no_lookahead
                            else [0],
                            "center": [0.3, 0.4, 0.65],
                            "stripe_widths": [0.065, None, None],
                            "stripe_velocity": [0.35, 0, 0],
                            "ellipsoid_widths": [0.065, 0.18, 0.3],
                            "ellipsoid_velocity": [0.3, 0.15, -0.1],
                            "error": "exact continuous L2 of piecewise-constant numerical field",
                            "measurements": rows,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    f"{case:12} {strategy:11} N={row['cells']:4} "
                    f"L2={row['physical_l2']:.6g} rate={row['N_rate']} "
                    f"aspect={row['max_aspect_ratio']:g} "
                    f"seconds={row['wall_seconds']:.1f}",
                    flush=True,
                )
    keys = [k for k in rows[0] if k not in ("cell_history", "volume_weighted_levels")]
    with (args.output / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plots(rows, args.output)


if __name__ == "__main__":
    main()
