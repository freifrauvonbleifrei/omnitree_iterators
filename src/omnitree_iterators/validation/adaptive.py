"""Budgeted directional Haar adaptation driven by the numerical solution.

Analytic initialization uses slab indicators. Numerical regridding uses a stack
Haar transform and bounded conservative child prediction.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import count
from math import erf, exp, pi, sqrt

import numpy as np
from bitarray import bitarray
from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization
from dyada.refinement import normalize_discretization

from omnitree_iterators import Bounds, Omnitree
from omnitree_iterators.validation.advection import UpwindTransport
from omnitree_iterators.validation.balancing import balance_mesh
from omnitree_iterators.validation.haar import regrid


@dataclass(frozen=True)
class Profile:
    center: tuple[float, ...]
    widths: tuple[float | None, ...]
    velocity: tuple[float, ...]

    def average(self, bounds: Bounds, time: float, *, squared: bool = False) -> float:
        result = 1.0
        for lo, hi, center, width, speed in zip(
            bounds.lower, bounds.upper, self.center, self.widths, self.velocity
        ):
            if width is None:
                continue
            sigma = width / sqrt(2) if squared else width
            lo, hi = float(lo), float(hi)
            center += speed * time
            if lo == hi:
                result *= exp(-0.5 * ((lo - center) / sigma) ** 2)
            else:
                result *= (
                    sigma
                    * sqrt(pi / 2)
                    * (
                        erf((hi - center) / (sqrt(2) * sigma))
                        - erf((lo - center) / (sqrt(2) * sigma))
                    )
                    / (hi - lo)
                )
        return result


def slabs(bounds: Bounds, axis: int, number: int = 4) -> list[Bounds]:
    result = []
    width = (bounds.upper[axis] - bounds.lower[axis]) / number
    for i in range(number):
        lo, hi = list(bounds.lower), list(bounds.upper)
        lo[axis] = bounds.lower[axis] + i * width
        hi[axis] = lo[axis] + width
        result.append(Bounds(tuple(lo), tuple(hi)))
    return result


def directional_energy(
    bounds: Bounds, average: Callable[[Bounds], float]
) -> np.ndarray:
    """Volume times variance of four equal slabs (two Haar detail levels).

    For q0,...,q3: variance = H0^2 + (Hleft^2 + Hright^2)/2,
    H0=((q2+q3)-(q0+q1))/4, Hleft=(q1-q0)/2, Hright=(q3-q2)/2.
    This is a directional refinement indicator, not a PDE error bound.
    """
    energies = []
    for axis in range(len(bounds.lower)):
        q = np.array([average(b) for b in slabs(bounds, axis)])
        energies.append(float(bounds.measure) * float(np.mean((q - q.mean()) ** 2)))
    return np.array(energies)


class ReconstructedField:
    """Conservative, bounded linear reconstruction of numerical cell averages.

    Weighted least squares uses actual neighbor-center displacements, including
    tangential offsets at hanging/crossed faces. A corner bound limits the slope
    to the extrema of the cell and its face neighbors. Every reconstructed cell
    retains exactly its original average in exact arithmetic.
    """

    def __init__(self, mesh: Omnitree, values: np.ndarray):
        self.mesh = mesh
        self.values = np.asarray(values)
        self.centers = np.array(
            [
                [float((lo + hi) / 2) for lo, hi in zip(c.bounds.lower, c.bounds.upper)]
                for c in mesh
            ]
        )
        widths = np.array(
            [
                [float(hi - lo) for lo, hi in zip(c.bounds.lower, c.bounds.upper)]
                for c in mesh
            ]
        )
        neighbors = [[] for _ in mesh]
        for face in mesh.interfaces():
            a, b = face.minus.box_index, face.plus.box_index
            neighbors[a].append((b, float(face.measure)))
            neighbors[b].append((a, float(face.measure)))
        self.gradients = np.zeros_like(self.centers)
        for i, adjacent in enumerate(neighbors):
            if not adjacent:
                continue
            indices = [j for j, _ in adjacent]
            weights = np.sqrt([area for _, area in adjacent])
            offsets = (self.centers[indices] - self.centers[i]) / widths[i]
            scaled, *_ = np.linalg.lstsq(
                offsets * weights[:, None],
                (self.values[indices] - self.values[i]) * weights,
                rcond=None,
            )
            deviation = np.sum(np.abs(scaled)) / 2
            lo = min(self.values[i], min(self.values[indices]))
            hi = max(self.values[i], max(self.values[indices]))
            factor = (
                min(
                    1.0,
                    (hi - self.values[i]) / deviation,
                    (self.values[i] - lo) / deviation,
                )
                if deviation > 0
                else 0.0
            )
            self.gradients[i] = factor * scaled / widths[i]
        self._cache: dict[Bounds, float] = {}

    def average(self, bounds: Bounds) -> float:
        if bounds not in self._cache:
            integral = 0.0
            for hit in self.mesh.intersect(bounds.lower, bounds.upper):
                i = hit.cell.box_index
                center = np.array(
                    [
                        float((lo + hi) / 2)
                        for lo, hi in zip(hit.bounds.lower, hit.bounds.upper)
                    ]
                )
                integral += float(hit.bounds.measure) * (
                    self.values[i] + self.gradients[i] @ (center - self.centers[i])
                )
            self._cache[bounds] = integral / float(bounds.measure)
        return self._cache[bounds]


@dataclass
class _Node:
    bounds: Bounds
    level: tuple[int, ...]
    value: float
    mask: str = ""
    children: list[_Node] = field(default_factory=list)


def adapt(
    average: Callable[[Bounds], float],
    dimension: int,
    budget: int,
    strategy: str,
    *,
    max_level: int = 9,
    displacement: tuple[float, ...] | None = None,
) -> tuple[Omnitree, np.ndarray]:
    """Greedy remeshing from one uniform seed level, followed by normalization.

    Directional mode bisects the strongest eligible axis (cost one new leaf).
    Isotropic mode splits all axes (cost 2^d-1), ranked by summed directional
    energies divided by this cost. Both modes use identical source information.
    Optional displacement adds half/full-step characteristic look-ahead to
    the indicator only. Translated windows are clamped into the unit domain;
    transfer still integrates the unshifted, current numerical reconstruction.
    """
    if strategy not in ("directional", "isotropic"):
        raise ValueError("Strategy must be directional or isotropic")
    if dimension < 1 or budget < 2**dimension or max_level < 1:
        raise ValueError("Require positive dimension/level and budget >= 2^dimension")
    if displacement is not None and len(displacement) != dimension:
        raise ValueError("Displacement has the wrong dimension")
    root_bounds = Bounds((Fraction(0),) * dimension, (Fraction(1),) * dimension)
    root = _Node(root_bounds, (0,) * dimension, average(root_bounds))
    heap = []
    serial = count()

    def children(node, axes):
        node.mask = "".join("1" if j in axes else "0" for j in range(dimension))
        for ordinal in range(1 << len(axes)):
            lo, hi = list(node.bounds.lower), list(node.bounds.upper)
            level = list(node.level)
            for bit, axis in enumerate(axes):
                mid = (lo[axis] + hi[axis]) / 2
                if ordinal & (1 << bit):
                    lo[axis] = mid
                else:
                    hi[axis] = mid
                level[axis] += 1
            bounds = Bounds(tuple(lo), tuple(hi))
            node.children.append(_Node(bounds, tuple(level), average(bounds)))
        return node.children

    def propose(node):
        eligible = [j for j in range(dimension) if node.level[j] < max_level]
        if not eligible or (strategy == "isotropic" and len(eligible) < dimension):
            return
        energy = directional_energy(node.bounds, average)
        if displacement is not None:
            for fraction in (0.5, 1.0):
                widths = tuple(
                    hi - lo for lo, hi in zip(node.bounds.lower, node.bounds.upper)
                )
                lower = tuple(
                    max(
                        Fraction(0),
                        min(1 - width, lo - Fraction(str(fraction * shift))),
                    )
                    for lo, width, shift in zip(node.bounds.lower, widths, displacement)
                )
                predicted = Bounds(
                    lower, tuple(lo + width for lo, width in zip(lower, widths))
                )
                energy = np.maximum(energy, directional_energy(predicted, average))
        if strategy == "directional":
            axis = max(eligible, key=lambda j: energy[j])
            axes, score = [axis], energy[axis]
        else:
            axes, score = eligible, float(sum(energy)) / ((1 << dimension) - 1)
        # Ignore roundoff-only details, particularly in invariant directions.
        if score > 1e-28 * float(node.bounds.measure):
            heapq.heappush(heap, (-score, next(serial), node, axes))

    for child in children(root, list(range(dimension))):
        propose(child)
    leaves = 1 << dimension
    while heap:
        _, _, node, axes = heapq.heappop(heap)
        cost = (1 << len(axes)) - 1
        if leaves + cost > budget:
            break
        leaves += cost
        for child in children(node, axes):
            propose(child)
    words, values_by_bounds = [], {}
    pending = [root]
    while pending:
        node = pending.pop()
        words.append(node.mask if node.children else "0" * dimension)
        if node.children:
            pending.extend(reversed(node.children))
        else:
            values_by_bounds[node.bounds] = node.value
    descriptor = RefinementDescriptor.from_binary(dimension, bitarray("".join(words)))
    disc = Discretization(MortonOrderLinearization(), descriptor)
    disc, _, _ = normalize_discretization(disc, track_mapping="boxes")
    mesh = Omnitree.from_discretization(disc)
    return mesh, np.array([values_by_bounds[c.bounds] for c in mesh])


def adaptive_experiment(
    case: str,
    strategy: str,
    budget: int,
    *,
    final_time: float = 0.3,
    regrid_interval: float = 0.1,
    lookahead: bool = True,
    balancing: bool = True,
    observer: Callable[[float, Omnitree, np.ndarray], None] | None = None,
) -> dict:
    if case == "stripe3d":
        profile = Profile((0.3, 0.4, 0.65), (0.065, None, None), (0.35, 0, 0))
    elif case == "ellipsoid3d":
        profile = Profile((0.3, 0.4, 0.65), (0.065, 0.18, 0.3), (0.3, 0.15, -0.1))
    else:
        raise ValueError(f"Unknown profile: {case}")
    if final_time <= 0 or regrid_interval <= 0:
        raise ValueError("Time and regrid interval must be positive")
    mesh, values = adapt(
        lambda b: profile.average(b, 0),
        3,
        budget,
        strategy,
        displacement=tuple(
            v * min(regrid_interval, final_time) for v in profile.velocity
        )
        if lookahead
        else None,
    )
    balance_history = []
    if balancing:
        mesh, values, stats = balance_mesh(mesh, values)
        # Initial data are known analytically; initialize the final geometry.
        values = np.array([profile.average(c.bounds, 0) for c in mesh])
        balance_history.append(
            {
                "time": 0.0,
                "input_cells": stats.input_cells,
                "output_cells": stats.output_cells,
                "added_cells": stats.added_cells,
                "rounds": stats.rounds,
            }
        )
    mass0 = sum(float(c.volume) * u for c, u in zip(mesh, values))
    initial_keys = {c.key for c in mesh}
    time = budget_flux = balance = remap_error = 0.0
    steps = regrids = 0
    minimum = float(values.min())
    history = []
    while time < final_time - 1e-14:
        transport = UpwindTransport(mesh, profile.velocity)
        segment = min(regrid_interval, final_time - time)
        before_defect = abs(float(transport.volumes @ values) - mass0 + budget_flux)
        result = transport.evolve(
            values,
            segment,
            lambda f, t: profile.average(f.bounds, time + t),
            observer=(lambda t, u: observer(time + t, mesh, u))
            if observer is not None
            else None,
        )
        values = result.values
        budget_flux += result.boundary_outflow_integral
        balance = max(balance, before_defect + result.maximum_mass_balance_error)
        minimum = min(minimum, result.minimum)
        steps += result.steps
        time += segment
        history.append({"time": time, "cells": len(mesh)})
        if time < final_time - 1e-14:
            field = ReconstructedField(mesh, values)
            before_mass = float(transport.volumes @ values)
            mesh, values = regrid(
                field,
                budget,
                strategy,
                displacement=tuple(
                    v * min(regrid_interval, final_time - time)
                    for v in profile.velocity
                )
                if lookahead
                else None,
            )
            if balancing:
                mesh, values, stats = balance_mesh(mesh, values)
                balance_history.append(
                    {
                        "time": time,
                        "input_cells": stats.input_cells,
                        "output_cells": stats.output_cells,
                        "added_cells": stats.added_cells,
                        "rounds": stats.rounds,
                    }
                )
            after_mass = sum(float(c.volume) * u for c, u in zip(mesh, values))
            remap_error = max(remap_error, abs(after_mass - before_mass))
            balance = max(balance, abs(after_mass - mass0 + budget_flux))
            minimum = min(minimum, float(values.min()))
            regrids += 1
    volumes = np.array([float(c.volume) for c in mesh])
    exact = np.array([profile.average(c.bounds, final_time) for c in mesh])
    exact_squared = np.array(
        [profile.average(c.bounds, final_time, squared=True) for c in mesh]
    )
    error = values - exact
    # Physical L2 includes unresolved within-cell variation of the exact field.
    full_l2_squared = volumes @ (error**2 + np.maximum(0, exact_squared - exact**2))
    levels = np.array([c.level for c in mesh])
    aspects = np.array([2.0 ** (max(c.level) - min(c.level)) for c in mesh])
    return {
        "adaptation": "hierarchical-haar-v3-balanced"
        if balancing
        else "hierarchical-haar-v2-partial",
        "balancing": balancing,
        "balance_history": balance_history,
        "maximum_cells": max(h["cells"] for h in history),
        "case": case,
        "strategy": strategy,
        "budget": budget,
        "cells": len(mesh),
        "final_time": final_time,
        "regrid_interval": regrid_interval,
        "lookahead": lookahead,
        "steps": steps,
        "regrids": regrids,
        "cell_history": history,
        "physical_l2": float(sqrt(full_l2_squared)),
        "cell_average_l1": float(volumes @ np.abs(error)),
        "cell_average_l2": float(sqrt(volumes @ (error**2))),
        "minimum": minimum,
        "mass_balance_error": balance,
        "remap_mass_error": remap_error,
        "volume_weighted_levels": list(volumes @ levels),
        "max_aspect_ratio": float(max(aspects)),
        "changed_leaf_keys": len(
            initial_keys.symmetric_difference(c.key for c in mesh)
        ),
    }
