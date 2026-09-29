"""Animate actual solver steps from the adaptive 3D benchmark (no resampling in time).

Display frames hold the most recent state of each run. Numerical step times
therefore differ between panels; each panel reports its own state time.
"""

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from omnitree_iterators.validation.adaptive import Profile, adaptive_experiment


def analytical_slice(profile, time, x, y, z=0.65):
    """Pointwise translating Gaussian on a fixed z plane (not cell averages)."""
    values = np.ones_like(x)
    for coordinate, center, width, velocity in zip(
        (x, y, z), profile.center, profile.widths, profile.velocity
    ):
        if width is not None:
            values *= np.exp(
                -0.5 * ((coordinate - center - velocity * time) / width) ** 2
            )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/advection.gif"))
    parser.add_argument("--budget", type=int, default=456)
    parser.add_argument("--final-time", type=float, default=1.5)
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs, results = [], []
    for case in ("stripe3d", "ellipsoid3d"):
        for strategy in ("directional", "isotropic"):
            snapshots = []

            def capture(time, mesh, values):
                polygons, colors = [], []
                for cell, value in zip(mesh, values):
                    lo, hi = cell.bounds.lower, cell.bounds.upper
                    if lo[2] <= 0.65 < hi[2]:
                        x0, y0, x1, y1 = map(float, (lo[0], lo[1], hi[0], hi[1]))
                        polygons.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                        colors.append(value)
                snapshots.append((time, polygons, colors, len(mesh)))

            result = adaptive_experiment(
                case,
                strategy,
                args.budget,
                final_time=args.final_time,
                balancing=not args.no_balance,
                observer=capture,
            )
            runs.append((case, strategy, snapshots))
            results.append(result)
            print(f"Recorded {case} / {strategy}: {len(snapshots)} states", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    numerical_axes = axes[:, :2].flat
    for ax in axes.flat:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
    collections = []
    for ax in numerical_axes:
        collection = PolyCollection(
            [],
            cmap="viridis",
            norm=Normalize(0, 1),
            edgecolors=(1, 1, 1, 0.5),
            linewidths=0.35,
        )
        ax.add_collection(collection)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
        collections.append(collection)
    fig.colorbar(
        collections[0],
        ax=list(axes.flat),
        label="u (numerical averages / analytical point values)",
        shrink=0.8,
    )
    fig.suptitle(
        "Advection on adaptive 3D omnitrees\nz = 0.65 slice • "
        + ("face 2:1 balance • " if not args.no_balance else "unbalanced • ")
        + f"target N = {args.budget} • regrid every 0.1"
    )
    coordinates = (np.arange(240) + 0.5) / 240
    x, y = np.meshgrid(coordinates, coordinates)
    profiles = [
        Profile((0.3, 0.4, 0.65), (0.065, None, None), (0.35, 0, 0)),
        Profile((0.3, 0.4, 0.65), (0.065, 0.18, 0.3), (0.3, 0.15, -0.1)),
    ]
    exact_images = [
        ax.imshow(
            analytical_slice(profile, 0, x, y),
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        for ax, profile in zip(axes[:, 2], profiles)
    ]
    times = np.concatenate(
        (np.linspace(0, args.final_time, 151), [args.final_time] * 15)
    )

    def draw(frame):
        for ax, collection, (case, strategy, snapshots) in zip(
            axes[:, :2].flat, collections, runs
        ):
            i = bisect_right([s[0] for s in snapshots], float(times[frame]) + 1e-12) - 1
            t, polygons, colors, cells = snapshots[max(0, i)]
            collection.set_verts(polygons)
            collection.set_array(np.array(colors))
            ax.set_title(
                f"{case.removesuffix('3d').capitalize()} · {strategy}\nN = {cells} · state t = {t:.3f}",
                fontsize=10,
            )
        for ax, picture, profile, name in zip(
            axes[:, 2], exact_images, profiles, ("Stripe", "Ellipsoid")
        ):
            picture.set_data(analytical_slice(profile, times[frame], x, y))
            ax.set_title(f"{name} · analytical\nt = {times[frame]:.3f}", fontsize=10)
        return collections + exact_images

    animation = FuncAnimation(fig, draw, frames=len(times), interval=100, blit=False)
    animation.save(args.output, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    from compress_advection_gif import compress

    encoded_frames, duration = compress(args.output, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "slice_z": 0.65,
                "source_frames": len(times),
                "frames": encoded_frames,
                "fps": 5,
                "compression": {
                    "temporal_stride": 2,
                    "palette_colors": 128,
                    "duration_ms": duration,
                },
                "display": "Latest numerical state, held until the next actual solver step; no extra time steps.",
                "analytical": "Pointwise exact Gaussian at the display time on the fixed z slice; numerical panels hold their labelled solver times.",
                "measurements": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
