"""Indicators, conservative transfer, and actual directional transport."""

from fractions import Fraction

import numpy as np
import pytest
from dyada.descriptor import RefinementDescriptor

from omnitree_iterators import Bounds, Omnitree
from omnitree_iterators.validation.adaptive import (
    Profile,
    ReconstructedField,
    adapt,
    adaptive_experiment,
    directional_energy,
    slabs,
)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_directional_energy_detects_symmetric_peak(axis):
    box = Bounds((0, 0, 0), (1, 1, 1))
    widths = tuple(0.1 if j == axis else None for j in range(3))
    profile = Profile((0.5, 0.5, 0.5), widths, (0, 0, 0))

    def average(b):
        return profile.average(b, 0)

    halves = [average(b) for b in slabs(box, axis, 2)]
    assert halves[0] == pytest.approx(halves[1])  # single-level Haar is blind
    energies = directional_energy(box, average)
    assert energies[axis] > 0.01
    assert all(energies[j] == 0 for j in range(3) if j != axis)
    mesh, _ = adapt(average, 3, 64, "directional")
    assert all(c.level[j] == 1 for c in mesh for j in range(3) if j != axis)
    assert max(c.level[axis] for c in mesh) > 1


def test_indicator_for_linear_field_has_expected_scaling():
    box = Bounds((0, 0), (1, Fraction(1, 2)))

    # Four equally spaced slab centers have variance 5 h^2 / 64.
    def average(b):
        return float((b.lower[0] + b.upper[0]) / 2 + 2 * (b.lower[1] + b.upper[1]))

    np.testing.assert_allclose(directional_energy(box, average), [5 / 128, 20 / 128])


@pytest.mark.parametrize("strategy", ["directional", "isotropic"])
def test_constant_field_is_not_refined(strategy):
    mesh, values = adapt(lambda b: 2.0, 3, 64, strategy)
    assert len(mesh) == 8
    np.testing.assert_array_equal(values, 2)
    field = ReconstructedField(mesh, values)
    np.testing.assert_array_equal(field.gradients, 0)
    assert field.average(Bounds((0, 0, 0), (1, 1, 1))) == 2


def test_reconstruction_and_remapping_conserve_mass_and_bounds():
    mesh = Omnitree(RefinementDescriptor(3, [2, 1, 1]))
    values = np.random.default_rng(17).uniform(0.1, 0.9, len(mesh))
    source = ReconstructedField(mesh, values)
    # Every old cell retains its mean, including cells with nonzero slopes.
    np.testing.assert_allclose(
        [source.average(c.bounds) for c in mesh], values, atol=2e-15
    )
    assert np.any(source.gradients != 0)
    mass = sum(float(c.volume) * u for c, u in zip(mesh, values))
    for strategy in ("directional", "isotropic"):
        new_mesh, new_values = adapt(source.average, 3, 64, strategy)
        assert len(new_mesh) == 64
        new_mass = sum(float(c.volume) * u for c, u in zip(new_mesh, new_values))
        assert new_mass == pytest.approx(mass, abs=2e-15)
        assert new_values.min() >= values.min() - 1e-14
        assert new_values.max() <= values.max() + 1e-14


@pytest.mark.parametrize("case", ["stripe3d", "ellipsoid3d"])
def test_dynamic_advection_at_matched_budget(case):
    directional = adaptive_experiment(case, "directional", 64, balancing=False)
    isotropic = adaptive_experiment(case, "isotropic", 64, balancing=False)
    assert 8 <= directional["cells"] <= 64
    assert 8 <= isotropic["cells"] <= 64
    for result in (directional, isotropic):
        assert result["regrids"] == 2
        assert result["minimum"] >= -1e-14
        assert result["mass_balance_error"] < 1e-13
        assert result["remap_mass_error"] < 1e-13
        assert result["physical_l2"] >= result["cell_average_l2"]
    assert isotropic["max_aspect_ratio"] == 1
    assert directional["changed_leaf_keys"] > 0
    assert directional["max_aspect_ratio"] > 1
    if case == "stripe3d":
        assert directional["volume_weighted_levels"][1:] == pytest.approx([1, 1])


def test_max_level_is_respected():
    mesh, _ = adapt(lambda b: float(sum(b.lower)), 3, 100, "directional", max_level=1)
    assert len(mesh) == 8


@pytest.mark.parametrize("strategy, budget", [("invalid", 64), ("directional", 2)])
def test_invalid_adaptation_parameters(strategy, budget):
    with pytest.raises(ValueError):
        adapt(lambda b: 1.0, 3, budget, strategy)


def test_profile_face_average_and_square_average():
    p = Profile((0.5,), (0.1,), (0.0,))
    b = Bounds((Fraction(1, 2),), (Fraction(1, 2),))
    assert p.average(b, 0) == p.average(b, 0, squared=True) == 1


def test_transport_lookahead_preserves_constant_at_domain_boundaries():
    mesh, values = adapt(
        lambda b: 0.75, 3, 64, "directional", displacement=(0.1, -0.1, 0.2)
    )
    assert len(mesh) == 8
    np.testing.assert_array_equal(values, 0.75)


def test_invalid_lookahead_dimension():
    with pytest.raises(ValueError, match="dimension"):
        adapt(lambda b: 1.0, 3, 64, "directional", displacement=(0.1,))
