"""Geometry-only validation against independent leaf-pair calculations."""

import random
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations
from math import prod

import bitarray as ba
import pytest
from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization
from dyada.refinement import apply_single_refinement, normalize_discretization

from omnitree_iterators import Bounds, Omnitree


def descriptor(d, words):
    return RefinementDescriptor.from_binary(d, ba.bitarray("".join(words.split())))


def signature(face):
    return (
        None if face.minus is None else face.minus.box_index,
        None if face.plus is None else face.plus.box_index,
        face.axis,
        face.bounds.lower,
        face.bounds.upper,
    )


def oracle(mesh):
    """Independent exhaustive leaf comparison, without the traversal helpers."""
    result = set()
    for a, b in combinations(mesh.cells, 2):
        for axis in range(mesh.dimension):
            if a.bounds.upper[axis] == b.bounds.lower[axis]:
                minus, plus = a, b
            elif b.bounds.upper[axis] == a.bounds.lower[axis]:
                minus, plus = b, a
            else:
                continue
            lower = tuple(max(x, y) for x, y in zip(a.bounds.lower, b.bounds.lower))
            upper = tuple(min(x, y) for x, y in zip(a.bounds.upper, b.bounds.upper))
            if all(lower[j] < upper[j] for j in range(mesh.dimension) if j != axis):
                result.add((minus.box_index, plus.box_index, axis, lower, upper))
    return result


def check_partition(mesh):
    faces = list(mesh.interfaces())
    signatures = [signature(f) for f in faces]
    assert len(signatures) == len(set(signatures)), "Duplicate interface emission"
    assert set(signatures) == oracle(mesh)
    assert sum(cell.volume for cell in mesh) == 1
    areas = defaultdict(F)
    expected_neighbors = defaultdict(set)
    for face in mesh.interfaces(include_boundary=True):
        assert face.measure > 0
        assert face.bounds.lower[face.axis] == face.bounds.upper[face.axis]
        for cell, side in ((face.minus, 1), (face.plus, -1)):
            if cell is not None:
                areas[cell.box_index, face.axis, side] += face.measure
                if not face.is_boundary:
                    expected_neighbors[cell.box_index, face.axis, side].add(
                        signature(face)
                    )
    for cell in mesh:
        assert (
            mesh.locate(
                tuple(
                    (lo + hi) / 2
                    for lo, hi in zip(cell.bounds.lower, cell.bounds.upper)
                )
            )
            == cell
        )
        for axis in range(mesh.dimension):
            area = prod(
                cell.bounds.upper[j] - cell.bounds.lower[j]
                for j in range(mesh.dimension)
                if j != axis
            )
            for side in (-1, 1):
                key = cell.box_index, axis, side
                assert areas[key] == area
                actual = [signature(f) for f in mesh.face_neighbors(*key)]
                assert len(actual) == len(set(actual))
                assert set(actual) == expected_neighbors[key]
    # Domain boundary measure is two per coordinate in any dimension.
    boundary = list(mesh.boundary_faces())
    assert sum(f.measure for f in boundary) == 2 * mesh.dimension
    assert len(boundary) == len(set(map(signature, boundary)))


@pytest.mark.parametrize(
    "levels", [(0,), (3,), (0, 0), (3, 1), (1, 2, 1), (1, 1, 1, 1)]
)
def test_regular_and_single_leaf(levels):
    mesh = Omnitree(RefinementDescriptor(len(levels), levels))
    check_partition(mesh)
    counts = [1 << ell for ell in levels]
    assert len(list(mesh.interfaces())) == sum(
        (n - 1) * prod(counts[j] for j in range(len(counts)) if j != axis)
        for axis, n in enumerate(counts)
    )


def test_crossed_3d_faces():
    mesh = Omnitree(descriptor(3, "100 010 000 000 001 000 000"))
    middle = [f for f in mesh.interfaces() if f.axis == 0]
    assert {(f.minus.box_index, f.plus.box_index) for f in middle} == {
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    }
    assert all(f.measure == F(1, 4) for f in middle)
    assert all(f.bounds.lower[0] == f.bounds.upper[0] == F(1, 2) for f in middle)
    check_partition(mesh)


@pytest.mark.parametrize("depth", [1, 3, 6])
def test_unbounded_number_of_face_neighbors(depth):
    right = RefinementDescriptor(2, [0, depth]).get_data()
    desc = RefinementDescriptor.from_binary(2, ba.bitarray("1000") + right)
    mesh = Omnitree(desc)
    neighbors = list(mesh.face_neighbors(0, 0, +1))
    assert len(neighbors) == 1 << depth
    assert sum(f.measure for f in neighbors) == 1
    check_partition(mesh)


def random_normalized_words(rng, d, depth):
    if depth == 0 or rng.random() < 0.4:
        return ["0" * d]
    mask = rng.randrange(1, 1 << d)
    word = "".join(str((mask >> j) & 1) for j in range(d))
    children = [
        random_normalized_words(rng, d, depth - 1) for _ in range(1 << mask.bit_count())
    ]
    # Ensure property 3 locally; descendants already satisfy it by construction.
    if any(word[j] == "0" and all(c[0][j] == "1" for c in children) for j in range(d)):
        children[0] = ["0" * d]
    return [word] + [w for child in children for w in child]


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_random_normalized_partitions(d, seed):
    words = random_normalized_words(random.Random(seed), d, 3)
    mesh = Omnitree(descriptor(d, " ".join(words)))
    check_partition(mesh)
    # Compare region coverage to independently clipped leaf bounds.
    lower = (F(1, 5),) * d
    upper = (F(4, 5),) * d
    hits = list(mesh.intersect(lower, upper))
    expected = {}
    for cell in mesh:
        lo = tuple(max(a, b) for a, b in zip(lower, cell.bounds.lower))
        hi = tuple(min(a, b) for a, b in zip(upper, cell.bounds.upper))
        if all(a < b for a, b in zip(lo, hi)):
            expected[cell.box_index] = (lo, hi)
    assert {
        h.cell.box_index: (h.bounds.lower, h.bounds.upper) for h in hits
    } == expected
    assert len(hits) == len(expected)
    assert sum(h.bounds.measure for h in hits) == F(3, 5) ** d


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_dyada_adaptation_and_leaf_order(d):
    rng = random.Random(d)
    disc = Discretization(MortonOrderLinearization(), RefinementDescriptor(d))
    for _ in range(12):
        mask = rng.randrange(1, 1 << d)
        disc, _ = apply_single_refinement(
            disc,
            rng.randrange(len(disc)),
            ba.bitarray([(mask >> j) & 1 for j in range(d)]),
        )
    disc, _, _ = normalize_discretization(disc, track_mapping="boxes")
    mesh = Omnitree.from_discretization(disc)
    for cell in mesh:
        level_index = disc.get_level_index(cell.box_index)
        assert cell.level == tuple(level_index.d_level)
        assert cell.index == tuple(level_index.d_index)
    check_partition(mesh)


def test_non_normalized_rejected_then_dyada_normalized():
    desc = descriptor(2, "10 01 00 00 01 00 00")
    with pytest.raises(ValueError, match="node 0, axis 1.*property 3"):
        Omnitree(desc)
    disc, _, _ = normalize_discretization(
        Discretization(MortonOrderLinearization(), desc)
    )
    mesh = Omnitree.from_discretization(disc)
    check_partition(mesh)
    assert {c.key for c in mesh} == {((1, 1), (i, j)) for i in (0, 1) for j in (0, 1)}


def test_normalization_checked_below_root():
    desc = descriptor(2, "10 10 01 00 00 01 00 00 00")
    with pytest.raises(ValueError, match="node 1, axis 1"):
        Omnitree(desc)


def test_deep_tree_uses_exact_integers_and_explicit_stacks():
    depth = 1050
    # Repeatedly refine the lower child in 1D. Every such tree is normalized.
    desc = RefinementDescriptor.from_binary(
        1, ba.bitarray("1" * depth + "0" * (depth + 1))
    )
    mesh = Omnitree(desc)
    assert len(mesh) == depth + 1
    assert mesh.cells[0].bounds.upper == (F(1, 1 << depth),)
    assert mesh.locate((F(1, 1 << (depth + 1)),)) == mesh.cells[0]
    assert len(list(mesh.interfaces())) == depth
    assert len(list(mesh.face_neighbors(0, 0, 1))) == 1
    assert list(mesh.face_neighbors(0, 0, -1)) == []
    assert len(list(mesh.intersect((0,), (1,)))) == depth + 1


def test_queries_and_boundary_conventions():
    mesh = Omnitree(RefinementDescriptor(2, [1, 1]))
    assert mesh.locate((0, 0)).box_index == 0
    assert mesh.locate((F(1, 2), F(1, 2))).box_index == 3
    assert mesh.locate((0.75, 0.25)).box_index == 1
    assert len(list(mesh.intersect((-1, -1), (2, 2)))) == 4
    assert list(mesh.intersect((1, 0), (2, 1))) == []
    hits = list(mesh.intersect((0, 0), (F(1, 2), F(1, 2))))
    assert len(hits) == 1 and hits[0].cell.box_index == 0
    assert hits[0].bounds.measure == F(1, 4)
    for face in mesh.boundary_faces():
        if face.minus is None:
            assert face.bounds.lower[face.axis] == 0
        else:
            assert face.plus is None
            assert face.bounds.lower[face.axis] == 1


def test_snapshot_independent_of_mutated_descriptor():
    desc = RefinementDescriptor(2, [1, 1])
    mesh = Omnitree(desc)
    expected = list(mesh.interfaces())
    desc.get_data().clear()
    assert list(mesh.interfaces()) == expected
    assert len(mesh) == 4


@pytest.mark.parametrize("bits", ["", "1", "00", "10"])
def test_malformed_descriptor_rejected(bits):
    # Mutation bypasses DyAda's constructor validation, exercising our boundary.
    desc = RefinementDescriptor(1)
    desc.get_data().clear()
    desc.get_data().extend(bits)
    with pytest.raises(ValueError, match="Descriptor"):
        Omnitree(desc)


def test_incomplete_mask_rejected():
    desc = RefinementDescriptor(2)
    desc.get_data().append(0)
    with pytest.raises(ValueError, match="incomplete mask"):
        Omnitree(desc)


def test_unsupported_linearization_rejected():
    disc = Discretization(object(), RefinementDescriptor(2))
    with pytest.raises(ValueError, match="Morton"):
        Omnitree.from_discretization(disc)


@pytest.mark.parametrize("point", [(1, 0), (-0.1, 0), (0,), (float("nan"), 0)])
def test_invalid_point(point):
    with pytest.raises(ValueError):
        Omnitree(RefinementDescriptor(2)).locate(point)


@pytest.mark.parametrize(
    "args, error",
    [
        ((-1, 0, 1), IndexError),
        ((1, 0, 1), IndexError),
        ((0, -1, 1), ValueError),
        ((0, 2, 1), ValueError),
        ((0, 0, 0), ValueError),
        ((0, 0, 1.5), TypeError),
    ],
)
def test_invalid_neighbor_query(args, error):
    with pytest.raises(error):
        list(Omnitree(RefinementDescriptor(2)).face_neighbors(*args))


@pytest.mark.parametrize(
    "lo, hi",
    [((0,), (1,)), ((0, 0), (1,)), ((0, 0), (0, 1)), ((1, 0), (0, 1)), ((), ())],
)
def test_invalid_region(lo, hi):
    with pytest.raises(ValueError):
        list(Omnitree(RefinementDescriptor(2)).intersect(lo, hi))


def test_bounds_and_geometric_keys_are_immutable():
    from dataclasses import FrozenInstanceError

    mesh = Omnitree(RefinementDescriptor(1))
    assert mesh.cells[0].key == ((0,), (0,))
    assert Bounds((0,), (1,)).measure == 1
    with pytest.raises(FrozenInstanceError):
        mesh.cells[0].box_index = 2
