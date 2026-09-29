"""Numerical validation of face-based first-order upwind transport."""

from math import prod

import numpy as np
import pytest
from dyada.descriptor import RefinementDescriptor

from omnitree_iterators import Bounds, Omnitree
from omnitree_iterators.validation.advection import (
    Gaussian,
    UpwindTransport,
    gaussian_experiment,
    validation_mesh,
)


@pytest.mark.parametrize("case", ["uniform2d", "hanging2d", "crossed3d"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_constant_preservation(case, direction):
    mesh = validation_mesh(case, 2)
    velocity = [direction * (j + 1) / 5 for j in range(mesh.dimension)]
    transport = UpwindTransport(mesh, velocity)
    result = transport.evolve(np.full(len(mesh), 1.75), 0.31, lambda face, t: 1.75)
    np.testing.assert_allclose(result.values, 1.75, atol=2e-14, rtol=0)
    assert result.maximum_mass_balance_error < 2e-14


@pytest.mark.parametrize("case", ["uniform2d", "hanging2d", "crossed3d"])
def test_positive_solution_and_boundary_mass_budget(case):
    mesh = validation_mesh(case, 2)
    velocity = [(-1) ** j * (j + 1) / 5 for j in range(mesh.dimension)]
    transport = UpwindTransport(mesh, velocity)
    initial = np.random.default_rng(17).uniform(0, 0.4, len(mesh))
    original = initial.copy()
    result = transport.evolve(initial, 0.37, lambda face, t: 1.25, cfl=1)
    assert result.minimum >= -1e-14
    assert result.values.max() <= 1.25 + 1e-14
    assert result.maximum_mass_balance_error < 2e-14
    assert result.boundary_outflow_integral < -0.01  # meaningful net inflow
    np.testing.assert_array_equal(initial, original)
    assert (
        abs(
            transport.volumes @ (result.values - initial)
            + result.boundary_outflow_integral
        )
        < 2e-14
    )


def test_exact_one_cell_translation_at_courant_one():
    mesh = Omnitree(RefinementDescriptor(1, 3))
    transport = UpwindTransport(mesh, [1])
    assert transport.dt_limit == 1 / 8
    result = transport.evolve(np.arange(1, 9), 1 / 8, lambda face, t: 0, cfl=1)
    np.testing.assert_array_equal(result.values, np.arange(8))
    assert result.steps == 1
    assert result.boundary_outflow_integral == 1
    assert result.maximum_mass_balance_error == 0


@pytest.mark.parametrize("velocity", [(0.4, -0.2, 0.3), (-0.4, 0.2, -0.3)])
def test_per_cell_neighbor_assembly_matches_global_faces(velocity):
    mesh = validation_mesh("crossed3d", 1)
    transport = UpwindTransport(mesh, velocity)
    values = np.random.default_rng(42).uniform(0, 1, len(mesh))

    def boundary(face, t):
        return 0.2 + t + sum(float(x) for x in face.bounds.lower)

    actual, _ = transport.residual(values, 0.1, boundary)
    # Independently assemble each cell's outward normal flux via neighbor queries.
    local = np.zeros(len(mesh))
    for cell in mesh:
        for axis, speed in enumerate(velocity):
            for side in (-1, 1):
                for face in mesh.face_neighbors(cell.box_index, axis, side):
                    neighbor = face.minus if side == -1 else face.plus
                    state = (
                        values[cell.box_index]
                        if side * speed >= 0
                        else values[neighbor.box_index]
                    )
                    local[cell.box_index] -= side * speed * float(face.measure) * state
    for face in mesh.boundary_faces():
        cell, side = (face.plus, -1) if face.minus is None else (face.minus, 1)
        outward = side * velocity[face.axis]
        state = values[cell.box_index] if outward >= 0 else boundary(face, 0.1)
        local[cell.box_index] -= outward * float(face.measure) * state
    local /= np.array([float(c.volume) for c in mesh])
    np.testing.assert_allclose(actual, local, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize(
    "case, levels",
    [
        ("uniform2d", (3, 4, 5)),
        ("hanging2d", (2, 3, 4)),
        ("crossed3d", (2, 3, 4)),
    ],
)
def test_gaussian_error_decreases_under_refinement(case, levels):
    # Coarser grids smear the reference itself through cell averaging and need
    # not have monotone discrete L2 errors. The report retains those samples.
    rows = [gaussian_experiment(case, r) for r in levels]
    for before, after in zip(rows, rows[1:]):
        assert after["h_max"] == before["h_max"] / 2
        assert after["l1"] < before["l1"]
        assert after["l2"] < before["l2"]
    for row in rows:
        assert row["minimum"] >= -1e-14
        assert row["mass_balance_error"] < 2e-14


def test_gaussian_averages_against_independent_quadrature():
    profile = Gaussian((0.3, 0.65), 0.1, (0.4, -0.2))
    x, w = np.polynomial.legendre.leggauss(80)
    for bounds in (Bounds((0, 0), (1, 1)), Bounds((0.5, 0), (0.5, 1))):
        factors = []
        for lo, hi, center, speed in zip(
            bounds.lower, bounds.upper, profile.center, profile.velocity
        ):
            lo, hi = float(lo), float(hi)
            points = lo + (x + 1) * (hi - lo) / 2
            factors.append(
                w
                @ np.exp(-0.5 * ((points - center - 0.3 * speed) / profile.sigma) ** 2)
                / 2
            )
        assert profile.average(bounds, 0.3) == pytest.approx(prod(factors), abs=1e-14)


def test_zero_velocity_and_zero_duration():
    mesh = validation_mesh("hanging2d", 0)
    values = np.arange(len(mesh), dtype=float)
    transport = UpwindTransport(mesh, [0, 0])

    def no_inflow(face, t):
        pytest.fail("Zero velocity must never request inflow data")

    stationary = transport.evolve(values, 0.7, no_inflow)
    np.testing.assert_array_equal(stationary.values, values)
    assert stationary.maximum_mass_balance_error == 0
    assert transport.evolve(values, 0, no_inflow).steps == 0


@pytest.mark.parametrize(
    "case, base_cells, d", [("hanging2d", 3, 2), ("crossed3d", 4, 3)]
)
def test_irregular_family_preserves_pattern(case, base_cells, d):
    for r in range(3):
        mesh = validation_mesh(case, r)
        assert len(mesh) == base_cells * 2 ** (d * r)
        # All x lengths equal 2^(-r-1); the transverse refinement varies by side.
        for cell in mesh:
            assert cell.level[0] == r + 1
            left = cell.bounds.upper[0] <= 0.5
            if d == 2:
                assert cell.level[1] == r + (not left)
            else:
                assert cell.level[1:] == ((r + 1, r) if left else (r, r + 1))


@pytest.mark.parametrize("velocity", [(1,), (float("nan"), 0)])
def test_invalid_velocity(velocity):
    with pytest.raises(ValueError):
        UpwindTransport(validation_mesh("uniform2d", 1), velocity)


@pytest.mark.parametrize(
    "time, cfl", [(-1, 0.8), (float("inf"), 0.8), (1, 0), (1, 1.1)]
)
def test_invalid_evolution_parameters(time, cfl):
    mesh = validation_mesh("uniform2d", 1)
    with pytest.raises(ValueError):
        UpwindTransport(mesh, (1, 1)).evolve(
            [0] * len(mesh), time, lambda face, t: 0, cfl=cfl
        )


@pytest.mark.parametrize("values", [[0], [0, 0, 0, float("nan")]])
def test_invalid_initial_data(values):
    with pytest.raises(ValueError):
        UpwindTransport(validation_mesh("uniform2d", 1), (1, 1)).evolve(
            values, 1, lambda face, t: 0
        )


@pytest.mark.parametrize(
    "case, level", [("unknown", 1), ("uniform2d", -1), ("uniform2d", 1.5)]
)
def test_invalid_mesh_parameters(case, level):
    with pytest.raises(ValueError):
        validation_mesh(case, level)


@pytest.mark.parametrize(
    "center, sigma, velocity",
    [
        ((), 1, ()),
        ((0,), 0, (1,)),
        ((0,), 1, (1, 2)),
        ((float("nan"),), 1, (1,)),
    ],
)
def test_invalid_gaussian(center, sigma, velocity):
    with pytest.raises(ValueError):
        Gaussian(center, sigma, velocity)


def test_gaussian_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        Gaussian((0,), 1, (1,)).average(Bounds((0, 0), (1, 1)), 0)


def test_observer_does_not_change_steps_or_solution():
    transport = UpwindTransport(Omnitree(RefinementDescriptor(1, 3)), [1])
    initial = np.arange(8, dtype=float)
    baseline = transport.evolve(initial, 0.3, lambda face, t: 1)
    times = []

    def observe(time, values):
        times.append(time)
        values[:] = -100  # Observers receive a copy, not solver-owned state.

    observed = transport.evolve(initial, 0.3, lambda face, t: 1, observer=observe)
    np.testing.assert_array_equal(observed.values, baseline.values)
    assert observed.steps == baseline.steps
    assert len(times) == observed.steps + 1
    assert times[0] == 0
    assert times[-1] == 0.3
