"""First-order conservative upwind transport on a fixed omnitree mesh.

The velocity is constant. Boundary inflow is supplied as face averages;
interior/outflow states are cell averages. Time integration is forward Euler.
This module is a validation experiment, not part of the stable geometry API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import erf, exp, isfinite, pi, sqrt

import numpy as np
from bitarray import bitarray
from dyada.descriptor import RefinementDescriptor

from omnitree_iterators import Bounds, Interface, Omnitree


@dataclass(frozen=True)
class Gaussian:
    """Unit-height translating Gaussian with exact rectangular averages."""

    center: tuple[float, ...]
    sigma: float
    velocity: tuple[float, ...]

    def __post_init__(self):
        if not self.center or len(self.center) != len(self.velocity):
            raise ValueError("Center and velocity must have matching dimensions")
        if not isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Sigma must be finite and positive")
        if not all(isfinite(x) for x in (*self.center, *self.velocity)):
            raise ValueError("Center and velocity must be finite")

    def average(self, bounds: Bounds, time: float) -> float:
        """Volume average, or tangential average when one extent is zero."""
        if len(bounds.lower) != len(self.center):
            raise ValueError("Bounds have the wrong dimension")
        result = 1.0
        scale = sqrt(2) * self.sigma
        for lo, hi, center, speed in zip(
            bounds.lower, bounds.upper, self.center, self.velocity
        ):
            lo, hi = float(lo), float(hi)
            center += speed * time
            if lo == hi:
                result *= exp(-0.5 * ((lo - center) / self.sigma) ** 2)
            else:
                result *= (
                    self.sigma
                    * sqrt(pi / 2)
                    * (erf((hi - center) / scale) - erf((lo - center) / scale))
                    / (hi - lo)
                )
        return result

    def cell_averages(self, mesh: Omnitree, time: float) -> np.ndarray:
        return np.array([self.average(c.bounds, time) for c in mesh])


@dataclass(frozen=True)
class TransportResult:
    values: np.ndarray
    steps: int
    minimum: float
    maximum_mass_balance_error: float
    boundary_outflow_integral: float


class UpwindTransport:
    """Compile the face iterator once, then reuse its connectivity at each step.

    Boundary flux is positive outward. Each interior patch contributes once
    with equal and opposite signs. The time-step bound is computed from the
    sum of outgoing face rates for each cell, including exterior outflow.
    """

    def __init__(self, mesh: Omnitree, velocity: Sequence[float]):
        self.mesh = mesh
        self.velocity = np.asarray(velocity, dtype=float)
        if (
            self.velocity.shape != (mesh.dimension,)
            or not np.isfinite(self.velocity).all()
        ):
            raise ValueError("Velocity must be a finite vector of mesh dimension")
        self.volumes = np.array([float(c.volume) for c in mesh])
        if not np.all(self.volumes > 0):
            raise ValueError("Cell volumes must be representable as positive floats")
        self.faces = tuple(mesh.interfaces(include_boundary=True))
        self.minus = np.array(
            [-1 if f.minus is None else f.minus.box_index for f in self.faces],
            dtype=int,
        )
        self.plus = np.array(
            [-1 if f.plus is None else f.plus.box_index for f in self.faces], dtype=int
        )
        self.rates = np.array(
            [self.velocity[f.axis] * float(f.measure) for f in self.faces]
        )
        self.donors = np.where(self.rates >= 0, self.minus, self.plus)
        self.inflow = np.flatnonzero((self.donors < 0) & (self.rates != 0))
        interior_donors = self.donors >= 0
        outgoing = np.bincount(
            self.donors[interior_donors],
            weights=np.abs(self.rates[interior_donors]),
            minlength=len(mesh),
        )
        moving = outgoing > 0
        self.dt_limit = (
            float(np.min(self.volumes[moving] / outgoing[moving]))
            if np.any(moving)
            else float("inf")
        )
        if self.dt_limit <= 0:
            raise ValueError("The floating-point CFL time-step bound underflows")

    def residual(
        self,
        values: np.ndarray,
        time: float,
        inflow: Callable[[Interface, float], float],
    ) -> tuple[np.ndarray, float]:
        """Return cell-average time derivatives and total outward boundary flux."""
        fluxes = self.rates * values[np.maximum(self.donors, 0)]
        for i in self.inflow:
            fluxes[i] = self.rates[i] * inflow(self.faces[i], time)
        has_minus, has_plus = self.minus >= 0, self.plus >= 0
        mass_rate = np.bincount(
            self.plus[has_plus],
            weights=fluxes[has_plus],
            minlength=len(self.mesh),
        ) - np.bincount(
            self.minus[has_minus],
            weights=fluxes[has_minus],
            minlength=len(self.mesh),
        )
        boundary_flux = float(np.sum(fluxes[~has_plus]) - np.sum(fluxes[~has_minus]))
        return mass_rate / self.volumes, boundary_flux

    def evolve(
        self,
        initial: Sequence[float],
        final_time: float,
        inflow: Callable[[Interface, float], float],
        *,
        cfl: float = 0.8,
        observer: Callable[[float, np.ndarray], None] | None = None,
    ) -> TransportResult:
        """Evolve from time zero, clipping the last step to final_time.

        The reported balance error is the maximum over all steps of
        |mass(t) - mass(0) + integrated numerical boundary outflow|. It checks
        the numerical budget, not equality to an exact continuous mass budget.
        """
        if not isfinite(final_time) or final_time < 0:
            raise ValueError("Final time must be finite and nonnegative")
        if not isfinite(cfl) or not 0 < cfl <= 1:
            raise ValueError("CFL must be in (0, 1]")
        values = np.array(initial, dtype=float, copy=True)
        if values.shape != (len(self.mesh),) or not np.isfinite(values).all():
            raise ValueError("Initial data must be a finite vector of leaf averages")
        mass0 = float(self.volumes @ values)
        time, budget, max_balance = 0.0, 0.0, 0.0
        minimum = float(values.min())
        steps = 0
        if observer is not None:
            observer(time, values.copy())
        while time < final_time:
            dt = min(cfl * self.dt_limit, final_time - time)
            if time + dt == time:
                raise ValueError(
                    "Time step is too small to advance floating-point time"
                )
            derivative, outward = self.residual(values, time, inflow)
            values += dt * derivative
            budget += dt * outward
            time += dt
            steps += 1
            if observer is not None:
                observer(time, values.copy())
            minimum = min(minimum, float(values.min()))
            max_balance = max(
                max_balance, abs(float(self.volumes @ values) - mass0 + budget)
            )
        return TransportResult(values, steps, minimum, max_balance, budget)


def validation_mesh(case: str, refinement: int) -> Omnitree:
    """Nested families: uniform 2D, hanging 2D, or crossed-interface 3D.

    One increment halves every leaf edge while preserving the coarse/fine
    pattern. Descriptors are normalized by construction. In the crossed case,
    the x=1/2 face meets y-refined cells on its left and z-refined cells on its
    right; no isotropic replacement of the crossed interface is performed.
    """
    if not isinstance(refinement, int) or refinement < 0:
        raise ValueError("Refinement must be a nonnegative integer")
    if case == "uniform2d":
        return Omnitree(RefinementDescriptor(2, refinement))
    if case not in ("hanging2d", "crossed3d"):
        raise ValueError(f"Unknown mesh family: {case}")
    d = 2 if case == "hanging2d" else 3
    if refinement == 0:
        words = "10 00 01 00 00" if d == 2 else "100 010 000 000 001 000 000"
        return Omnitree(
            RefinementDescriptor.from_binary(d, bitarray(words.replace(" ", "")))
        )
    words = []
    pending = [(0, 0)]
    while pending:
        depth, half = pending.pop()
        if depth == refinement:
            # x needs one extra level everywhere; the tangential extra level
            # depends on which half of the domain contains this subtree.
            mask = (
                ("10" if half == 0 else "11")
                if d == 2
                else ("110" if half == 0 else "101")
            )
            words.append(mask)
            words.extend(["0" * d] * (1 << mask.count("1")))
        else:
            words.append("1" * d)
            pending.extend(
                (depth + 1, child & 1 if depth == 0 else half)
                for child in reversed(range(1 << d))
            )
    return Omnitree(RefinementDescriptor.from_binary(d, bitarray("".join(words))))


def gaussian_experiment(case: str, refinement: int, *, final_time: float = 0.5) -> dict:
    """Run one benchmark and return JSON-serializable diagnostics."""
    mesh = validation_mesh(case, refinement)
    center = (0.3, 0.35) if mesh.dimension == 2 else (0.3, 0.35, 0.65)
    velocity = (0.4, 0.25) if mesh.dimension == 2 else (0.4, 0.25, -0.2)
    profile = Gaussian(center, 0.1, velocity)
    transport = UpwindTransport(mesh, velocity)
    initial = profile.cell_averages(mesh, 0)
    result = transport.evolve(
        initial,
        final_time,
        lambda face, t: profile.average(face.bounds, t),
    )
    exact = profile.cell_averages(mesh, final_time)
    difference = result.values - exact
    return {
        "case": case,
        "refinement": refinement,
        "cells": len(mesh),
        "interfaces": len(transport.faces),
        "h_max": max(
            float(hi - lo)
            for c in mesh
            for lo, hi in zip(c.bounds.lower, c.bounds.upper)
        ),
        "final_time": final_time,
        "steps": result.steps,
        "l1": float(transport.volumes @ np.abs(difference)),
        "l2": float(sqrt(transport.volumes @ (difference**2))),
        "minimum": result.minimum,
        "mass_balance_error": result.maximum_mass_balance_error,
        "boundary_outflow_integral": result.boundary_outflow_integral,
    }
