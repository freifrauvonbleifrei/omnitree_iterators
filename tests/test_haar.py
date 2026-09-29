import numpy as np
import pytest
from bitarray import bitarray
from dyada.descriptor import RefinementDescriptor

from omnitree_iterators import Omnitree
from omnitree_iterators.validation.adaptive import ReconstructedField
from omnitree_iterators.validation.haar import analyze, haar_block, regrid


def test_mixed_detail_and_parseval():
    mesh = Omnitree(RefinementDescriptor(2, [1, 1]))
    values = np.array([1.0, -1.0, -1.0, 1.0])
    result = analyze(mesh, values)
    np.testing.assert_allclose(result.coefficients[0], [0, 0, 0, 1])
    np.testing.assert_allclose(result.directional_energy[0], [1, 1])
    assert result.subtree_energy[0] == 1


def test_unequal_depth_scaling_and_energy():
    mesh = Omnitree(RefinementDescriptor.from_binary(1, bitarray("10100")))
    values = np.array([2.0, 4.0, 8.0])
    result = analyze(mesh, values)
    mean = sum(float(c.volume) * u for c, u in zip(mesh, values))
    assert result.coefficients[0][0] == mean
    assert result.subtree_energy[0] == pytest.approx(
        sum(float(c.volume) * (u - mean) ** 2 for c, u in zip(mesh, values))
    )


@pytest.mark.parametrize("strategy", ["directional", "isotropic"])
def test_transfer_without_average_queries(strategy, monkeypatch):
    mesh = Omnitree(RefinementDescriptor(2, [2, 2]))
    values = np.random.default_rng(7).uniform(0.1, 0.9, len(mesh))
    field = ReconstructedField(mesh, values)

    def forbidden(*args):
        raise AssertionError("Regridding must not request rectangle averages")

    monkeypatch.setattr(field, "average", forbidden)
    mass = sum(float(c.volume) * u for c, u in zip(mesh, values))
    for budget in (4, 16, 64):
        result, transferred = regrid(field, budget, strategy, displacement=(0.1, -0.2))
        assert len(result) <= budget
        assert sum(
            float(c.volume) * u for c, u in zip(result, transferred)
        ) == pytest.approx(mass, abs=1e-14)
        assert transferred.min() >= values.min() - 1e-14
        assert transferred.max() <= values.max() + 1e-14


def test_constant_coarsens():
    mesh = Omnitree(RefinementDescriptor(2, [3, 3]))
    result, values = regrid(
        ReconstructedField(mesh, np.full(len(mesh), 2.0)), 64, "directional"
    )
    assert len(result) == 4
    np.testing.assert_array_equal(values, 2.0)


def test_prediction_refines_only_active_axis():
    mesh = Omnitree(RefinementDescriptor(2, [2, 1]))
    values = np.array(
        [float((c.bounds.lower[0] + c.bounds.upper[0]) / 2) for c in mesh]
    )
    result, _ = regrid(ReconstructedField(mesh, values), 32, "directional")
    assert max(c.level[0] for c in result) == 3
    assert all(c.level[1] == 1 for c in result)


@pytest.mark.parametrize("values", [[], [1, 2, 3]])
def test_invalid_block(values):
    with pytest.raises(ValueError):
        haar_block(values)


def test_partial_axis_coarsening_preserves_exact_stripe():
    # Pure x variation: four quadrants can become two x halves without error.
    from types import SimpleNamespace

    mesh = Omnitree(RefinementDescriptor(2, [1, 1]))
    values = np.array([1.0, 3.0, 1.0, 3.0])
    field = SimpleNamespace(mesh=mesh, values=values, gradients=np.zeros((4, 2)))
    result, transferred = regrid(field, 4, "directional", max_level=1)
    assert len(result) == 2
    assert all(c.level == (1, 0) for c in result)
    np.testing.assert_array_equal(transferred, [1, 3])
    isotropic, _ = regrid(field, 4, "isotropic", max_level=1)
    assert len(isotropic) == 4


def test_mixed_parent_exposes_only_one_mergeable_pair():
    from types import SimpleNamespace

    # Bottom-right quadrant has x children; the two left leaves are equal.
    mesh = Omnitree(RefinementDescriptor.from_binary(2, bitarray("11001000000000")))
    values = np.array([1.0, 8.0, 8.0, 1.0, 4.0])
    field = SimpleNamespace(mesh=mesh, values=values, gradients=np.zeros((5, 2)))
    # No prediction: all leaf gradients vanish. Existing detail marks could
    # propose splits, so cap at existing maximum and provide centers as usual.
    field.centers = np.array(
        [
            [(float(a) + float(b)) / 2 for a, b in zip(c.bounds.lower, c.bounds.upper)]
            for c in mesh
        ]
    )
    result, transferred = regrid(field, 4, "directional", max_level=2)
    assert len(result) <= 4
    assert sum(
        float(c.volume) * u for c, u in zip(result, transferred)
    ) == pytest.approx(3.5)
    # Check eligibility independently of prediction-driven budget priorities.
    from omnitree_iterators.validation.haar import _Candidate, _downsplit_leaf_pairs

    analysis = analyze(mesh, values)
    candidates = [
        _Candidate(
            n.bounds,
            tuple(
                (hi - lo).denominator.bit_length() - 1
                for lo, hi in zip(n.bounds.lower, n.bounds.upper)
            ),
            analysis.coefficients[i][0],
            [j for j, b in enumerate(n.mask) if b],
        )
        for i, n in enumerate(mesh._nodes)
    ]
    for i, n in enumerate(mesh._nodes):
        candidates[i].children = [candidates[j] for j in n.children]
    _downsplit_leaf_pairs(candidates[0])
    root = candidates[0]
    assert root.axes == [0]
    assert root.children[0].axes == [1]
    assert all(not c.children for c in root.children[0].children)
    assert root.children[0].value == 1
    assert any(c.children for c in root.children[1].children)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_downsplit_is_lossless_for_every_morton_axis(axis):
    from fractions import Fraction

    from omnitree_iterators import Bounds
    from omnitree_iterators.validation.haar import _Candidate, _downsplit_leaf_pairs

    mesh = Omnitree(RefinementDescriptor(3, [1, 1, 1]))
    # Axis under test has zero pair differences; other axes have distinct jumps.
    values = [sum((j + 1) * c.index[j] for j in range(3) if j != axis) for c in mesh]
    leaves = [_Candidate(c.bounds, c.level, v) for c, v in zip(mesh, values)]
    root = _Candidate(
        Bounds((Fraction(0),) * 3, (Fraction(1),) * 3),
        (0, 0, 0),
        float(np.mean(values)),
        [0, 1, 2],
        leaves.copy(),
    )
    _downsplit_leaf_pairs(root)
    assert root.axes == [j for j in range(3) if j != axis]
    assert all(c.axes == [axis] for c in root.children)
    exposed = [leaf for c in root.children for leaf in c.children]
    assert {id(c) for c in exposed} == {id(c) for c in leaves}
    for c in root.children:
        assert c.value == sum(child.value for child in c.children) / 2
        assert c.children[0].value == c.children[1].value
        assert c.bounds.measure == sum(child.bounds.measure for child in c.children)


def test_normalization_mapping_preserves_values_after_nontrivial_permutation():
    from dyada.discretization import Discretization
    from dyada.linearization import MortonOrderLinearization

    from omnitree_iterators.validation.haar import _normalize_leaf_values

    # Split x, then z, then y. Old leaf bits are y,z,x, normalized bits x,y,z:
    # a non-self-inverse permutation, detecting gather/scatter confusion.
    descriptor = RefinementDescriptor.from_binary(
        3, bitarray("100" + ("001" + ("010" + "000" * 2) * 2) * 2)
    )
    disc = Discretization(MortonOrderLinearization(), descriptor)
    before = np.arange(8, dtype=float) + 0.25
    mesh, after = _normalize_leaf_values(disc, before)
    for cell, value in zip(mesh, after):
        x, y, z = cell.index
        assert value == before[y + 2 * z + 4 * x]
    # Already normalized: DyAda's empty mapping denotes identity.
    normalized = Discretization(
        MortonOrderLinearization(), RefinementDescriptor(3, [1, 1, 1])
    )
    _, unchanged = _normalize_leaf_values(normalized, after)
    np.testing.assert_array_equal(unchanged, after)
