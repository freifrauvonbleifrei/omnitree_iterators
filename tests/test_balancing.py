"""Componentwise face balance, anisotropy, conservative transfer and closure."""

import numpy as np
import pytest
from bitarray import bitarray
from dyada.descriptor import RefinementDescriptor

from omnitree_iterators import Omnitree
from omnitree_iterators.validation.balancing import balance_marks, balance_mesh


def uniform_words(d, axis, depth):
    if depth == 0:
        return "0" * d
    mask = "".join("1" if j == axis else "0" for j in range(d))
    return mask + uniform_words(d, axis, depth - 1) * 2


def assert_balanced_geometrically(mesh):
    # Independent all-pairs oracle, including every positive-area face contact.
    for i, a in enumerate(mesh.cells):
        for b in mesh.cells[i + 1 :]:
            touching = any(
                (
                    a.bounds.upper[n] == b.bounds.lower[n]
                    or b.bounds.upper[n] == a.bounds.lower[n]
                )
                and all(
                    min(a.bounds.upper[j], b.bounds.upper[j])
                    > max(a.bounds.lower[j], b.bounds.lower[j])
                    for j in range(mesh.dimension)
                    if j != n
                )
                for n in range(mesh.dimension)
            )
            if touching:
                assert all(abs(x - y) <= 1 for x, y in zip(a.level, b.level))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_closure_conservation_and_bounds(d):
    # Strong normal jump, plus crossed anisotropy in d > 1.
    right_axis = 0 if d == 1 else d - 1
    words = (
        "1"
        + "0" * (d - 1)
        + uniform_words(d, 0, 3)
        + uniform_words(d, right_axis, 0 if d == 1 else 3)
    )
    mesh = Omnitree(RefinementDescriptor.from_binary(d, bitarray(words)))
    values = np.random.default_rng(14).uniform(0.1, 0.9, len(mesh))
    assert balance_marks(mesh)
    result, transferred, stats = balance_mesh(mesh, values)
    assert_balanced_geometrically(result)
    assert not balance_marks(result)
    assert stats.added_cells == len(result) - len(mesh) > 0
    assert stats.rounds >= 2
    assert transferred.min() >= values.min() - 1e-14
    assert transferred.max() <= values.max() + 1e-14
    # Each old cell's mass is conserved, not just the domain total.
    for old, u in zip(mesh, values):
        descendants = [
            (c, v)
            for c, v in zip(result, transferred)
            if all(
                a <= b and e <= f
                for a, b, e, f in zip(
                    old.bounds.lower, c.bounds.lower, c.bounds.upper, old.bounds.upper
                )
            )
        ]
        assert sum(float(c.volume) * v for c, v in descendants) == pytest.approx(
            float(old.volume) * u, abs=1e-14
        )
    maximum = np.max([c.level for c in mesh], axis=0)
    assert all(all(ell <= cap for ell, cap in zip(c.level, maximum)) for c in result)
    again, same, stats = balance_mesh(result, transferred)
    assert again is result
    np.testing.assert_array_equal(same, transferred)
    assert stats.rounds == stats.added_cells == 0


def test_balancing_does_not_require_isotropic_cells():
    mesh = Omnitree(RefinementDescriptor(3, [5, 1, 1]))
    result, values, stats = balance_mesh(mesh, np.ones(len(mesh)))
    assert result is mesh
    assert stats.rounds == 0
    assert all(c.level == (5, 1, 1) for c in result)
    np.testing.assert_array_equal(values, 1)


def test_crossed_faces_are_balanced_in_all_tangential_axes():
    mesh = Omnitree(
        RefinementDescriptor.from_binary(
            3, bitarray("100" + uniform_words(3, 1, 3) + uniform_words(3, 2, 3))
        )
    )
    result, values, _ = balance_mesh(mesh, np.full(len(mesh), 2.0))
    assert_balanced_geometrically(result)
    np.testing.assert_array_equal(values, 2.0)


def test_advection_observes_only_balanced_snapshots():
    from omnitree_iterators.validation.adaptive import adaptive_experiment

    observed = []

    def observe(time, mesh, values):
        assert not balance_marks(mesh)
        observed.append(time)

    result = adaptive_experiment("ellipsoid3d", "directional", 64, observer=observe)
    assert observed[0] == 0
    assert observed[-1] == pytest.approx(0.3)
    assert len(result["balance_history"]) == 3
    assert result["remap_mass_error"] < 1e-13
    assert result["minimum"] >= 0
