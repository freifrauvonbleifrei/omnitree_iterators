"""Exact, dimension-independent traversal without an all-pairs leaf search.

All traversal stacks are explicit: deep anisotropic trees do not consume the
Python call stack. Geometry is rational; cell keys use arbitrary-size integers.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import prod
from operator import index as integer_index

from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization


@dataclass(frozen=True)
class Bounds:
    """Exact axis-aligned bounds. Degenerate axes are allowed for faces."""

    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", tuple(Fraction(x) for x in self.lower))
        object.__setattr__(self, "upper", tuple(Fraction(x) for x in self.upper))
        if not self.lower or len(self.lower) != len(self.upper):
            raise ValueError("Bounds must have matching, positive dimensions")
        if any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Lower bounds must not exceed upper bounds")

    @property
    def measure(self) -> Fraction:
        """Full-dimensional volume (zero for a face)."""
        return prod((b - a for a, b in zip(self.lower, self.upper)), start=Fraction(1))


@dataclass(frozen=True)
class Cell:
    """Leaf of the snapshot; box_index indexes DyAda's leaf data ordering."""

    box_index: int
    descriptor_index: int
    level: tuple[int, ...]
    index: tuple[int, ...]
    bounds: Bounds

    @property
    def key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Geometric identity, independent of descriptor/leaf ordering."""
        return self.level, self.index

    @property
    def volume(self) -> Fraction:
        return self.bounds.measure


@dataclass(frozen=True)
class Interface:
    """A shared face patch, oriented from minus to plus along ``axis``.

    minus/plus are the cells on the lower/upper coordinate side, respectively.
    A domain boundary has exactly one missing cell. Bounds have zero extent
    along axis. In 1D the face measure is one (the empty product).
    """

    minus: Cell | None
    plus: Cell | None
    axis: int
    bounds: Bounds

    @property
    def is_boundary(self) -> bool:
        return self.minus is None or self.plus is None

    @property
    def measure(self) -> Fraction:
        return prod(
            (
                hi - lo
                for j, (lo, hi) in enumerate(zip(self.bounds.lower, self.bounds.upper))
                if j != self.axis
            ),
            start=Fraction(1),
        )


@dataclass(frozen=True)
class Intersection:
    """A leaf and its positive-volume intersection with a query region."""

    cell: Cell
    bounds: Bounds


@dataclass
class _Node:
    mask: tuple[bool, ...]
    bounds: Bounds
    parent: int | None
    ordinal: int
    children: list[int] = field(default_factory=list)
    cell: Cell | None = None


def _overlap(a: Bounds, b: Bounds, skip: int | None = None) -> bool:
    return all(
        max(a.lower[j], b.lower[j]) < min(a.upper[j], b.upper[j])
        for j in range(len(a.lower))
        if j != skip
    )


class Omnitree:
    """Indexed snapshot of a normalized DyAda descriptor on [0, 1)^d.

    The descriptor is interpreted in Morton order, first dimension fastest.
    Construction validates tree structure and normalization (property 3).
    Later mutation/refinement of the source does not alter this snapshot:
    construct a new Omnitree after adaptation.
    """

    def __init__(self, descriptor: RefinementDescriptor):
        d = integer_index(descriptor.get_num_dimensions())
        if d < 1:
            raise ValueError("Dimension must be positive")
        if len(descriptor.get_data()) % d:
            raise ValueError("Descriptor contains an incomplete mask")
        self._dimension = d
        self._nodes: list[_Node] = []
        leaves: list[Cell] = []
        # Pending nodes carry their exact dyadic geometry, parent and ordinal.
        pending = [(None, 0, (0,) * d, (0,) * d)]
        masks = iter(descriptor)
        while pending:
            parent, ordinal, level, index = pending.pop()
            try:
                mask = tuple(bool(x) for x in next(masks))
            except StopIteration as exc:
                raise ValueError(
                    "Descriptor ends before all children are present"
                ) from exc
            if len(mask) != d:
                raise ValueError("Descriptor mask has the wrong dimension")
            bounds = Bounds(
                tuple(Fraction(i, 1 << ell) for i, ell in zip(index, level)),
                tuple(Fraction(i + 1, 1 << ell) for i, ell in zip(index, level)),
            )
            node_id = len(self._nodes)
            node = _Node(mask, bounds, parent, ordinal)
            self._nodes.append(node)
            if parent is not None:
                self._nodes[parent].children.append(node_id)
            axes = [j for j, split in enumerate(mask) if split]
            if not axes:
                node.cell = Cell(len(leaves), node_id, level, index, bounds)
                leaves.append(node.cell)
                continue
            child_level = tuple(ell + split for ell, split in zip(level, mask))
            for child in reversed(range(1 << len(axes))):
                child_index = list(index)
                for bit, axis in enumerate(axes):
                    child_index[axis] = 2 * index[axis] + ((child >> bit) & 1)
                pending.append((node_id, child, child_level, tuple(child_index)))
        if next(masks, None) is not None:
            raise ValueError("Descriptor contains nodes after the root subtree")
        for node_id, node in enumerate(self._nodes):
            if node.children:
                for axis, split in enumerate(node.mask):
                    if not split and all(
                        self._nodes[c].mask[axis] for c in node.children
                    ):
                        raise ValueError(
                            f"Descriptor is not normalized: node {node_id}, axis {axis} "
                            "violates property 3"
                        )
        self._cells = tuple(leaves)

    @classmethod
    def from_discretization(cls, discretization: Discretization) -> Omnitree:
        """Snapshot a DyAda discretization, rejecting unsupported orderings."""
        if type(discretization.linearization) is not MortonOrderLinearization:
            raise ValueError("Only MortonOrderLinearization is supported")
        return cls(discretization.descriptor)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def cells(self) -> tuple[Cell, ...]:
        return self._cells

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[Cell]:
        return iter(self._cells)

    def _match_faces(self, minus: int, plus: int, axis: int) -> Iterator[Interface]:
        """Traverse two touching subtrees, pruning nonoverlapping face pieces."""
        pending = [(minus, plus)]
        while pending:
            a_id, b_id = pending.pop()
            a, b = self._nodes[a_id], self._nodes[b_id]
            if a.bounds.upper[axis] != b.bounds.lower[axis]:
                continue
            if not _overlap(a.bounds, b.bounds, skip=axis):
                continue
            if a.cell is not None and b.cell is not None:
                yield Interface(
                    a.cell,
                    b.cell,
                    axis,
                    Bounds(
                        tuple(
                            max(x, y) for x, y in zip(a.bounds.lower, b.bounds.lower)
                        ),
                        tuple(
                            min(x, y) for x, y in zip(a.bounds.upper, b.bounds.upper)
                        ),
                    ),
                )
                continue
            # Descend the side with the larger tangential extent first. This
            # avoids expanding a fine side while its opposite remains coarse.
            a_area = prod(
                a.bounds.upper[j] - a.bounds.lower[j]
                for j in range(self.dimension)
                if j != axis
            )
            b_area = prod(
                b.bounds.upper[j] - b.bounds.lower[j]
                for j in range(self.dimension)
                if j != axis
            )
            if a.children and (not b.children or a_area >= b_area):
                pending.extend(
                    (c, b_id)
                    for c in reversed(a.children)
                    if self._nodes[c].bounds.upper[axis] == a.bounds.upper[axis]
                )
            else:
                pending.extend(
                    (a_id, c)
                    for c in reversed(b.children)
                    if self._nodes[c].bounds.lower[axis] == b.bounds.lower[axis]
                )

    def interfaces(self, *, include_boundary: bool = False) -> Iterator[Interface]:
        """Yield every interior face patch exactly once, optionally boundaries.

        Internal interfaces belong to their unique lowest common ancestor.
        Sibling pairs differ in exactly one Morton child bit. No adjacency
        graph or all-pairs leaf comparison is constructed.
        """
        for node in self._nodes:
            axes = [j for j, split in enumerate(node.mask) if split]
            for bit, axis in enumerate(axes):
                for ordinal, child in enumerate(node.children):
                    if not (ordinal & (1 << bit)):
                        yield from self._match_faces(
                            child, node.children[ordinal ^ (1 << bit)], axis
                        )
        if include_boundary:
            yield from self.boundary_faces()

    def boundary_faces(self) -> Iterator[Interface]:
        """Yield domain boundary faces; edges and corners are not extra faces."""
        for cell in self._cells:
            for axis in range(self.dimension):
                for side, coordinate in (
                    (-1, cell.bounds.lower[axis]),
                    (1, cell.bounds.upper[axis]),
                ):
                    if coordinate != (0 if side == -1 else 1):
                        continue
                    lower, upper = list(cell.bounds.lower), list(cell.bounds.upper)
                    lower[axis] = upper[axis] = coordinate
                    yield Interface(
                        cell if side == 1 else None,
                        cell if side == -1 else None,
                        axis,
                        Bounds(tuple(lower), tuple(upper)),
                    )

    def face_neighbors(
        self, box_index: int, axis: int, side: int
    ) -> Iterator[Interface]:
        """Yield patches touching a leaf's requested face (side -1 or +1).

        Returned patches retain global minus-to-plus orientation. At a domain
        boundary this yields nothing; use boundary_faces for exterior faces.
        The search ascends to a crossing sibling then descends its face subtree.
        """
        box_index, axis, side = map(integer_index, (box_index, axis, side))
        if not 0 <= box_index < len(self):
            raise IndexError("Leaf index out of range")
        if not 0 <= axis < self.dimension:
            raise ValueError("Axis out of range")
        if side not in (-1, 1):
            raise ValueError("Side must be -1 or +1")
        leaf = self._cells[box_index].descriptor_index
        current = leaf
        while self._nodes[current].parent is not None:
            node = self._nodes[current]
            parent_id = node.parent
            assert parent_id is not None
            parent = self._nodes[parent_id]
            if parent.mask[axis]:
                bit = sum(parent.mask[:axis])
                upper_child = bool(node.ordinal & (1 << bit))
                if upper_child == (side == -1):
                    sibling = parent.children[node.ordinal ^ (1 << bit)]
                    yield from self._match_faces(
                        leaf if side == 1 else sibling,
                        sibling if side == 1 else leaf,
                        axis,
                    )
                    return
            current = parent_id

    def locate(self, point: Sequence) -> Cell:
        """Find the leaf containing a point, using half-open cells [lower, upper).

        Internal split points belong to the upper child. Points on the unit
        domain's upper boundary are outside. Floats retain their exact binary
        value; pass Fraction for exact rational input.
        """
        point = tuple(Fraction(x) for x in point)
        if len(point) != self.dimension:
            raise ValueError("Point has the wrong dimension")
        if any(x < 0 or x >= 1 for x in point):
            raise ValueError("Point is outside [0, 1)^d")
        node = self._nodes[0]
        while node.cell is None:
            ordinal, bit = 0, 0
            for axis, split in enumerate(node.mask):
                if split:
                    midpoint = (node.bounds.lower[axis] + node.bounds.upper[axis]) / 2
                    ordinal |= int(point[axis] >= midpoint) << bit
                    bit += 1
            node = self._nodes[node.children[ordinal]]
        return node.cell

    def intersect(self, lower: Sequence, upper: Sequence) -> Iterator[Intersection]:
        """Yield positive-volume leaf intersections with a rectangular region.

        Region bounds may extend outside the unit domain. Mere face/edge/point
        contacts are excluded; every requested extent must be strictly positive.
        """
        region = Bounds(tuple(lower), tuple(upper))
        if len(region.lower) != self.dimension:
            raise ValueError("Region has the wrong dimension")
        if any(lo == hi for lo, hi in zip(region.lower, region.upper)):
            raise ValueError("Region must have positive extent in every dimension")
        pending = [0]
        while pending:
            node = self._nodes[pending.pop()]
            if not _overlap(node.bounds, region):
                continue
            if node.cell is None:
                pending.extend(reversed(node.children))
            else:
                yield Intersection(
                    node.cell,
                    Bounds(
                        tuple(
                            max(x, y) for x, y in zip(node.bounds.lower, region.lower)
                        ),
                        tuple(
                            min(x, y) for x, y in zip(node.bounds.upper, region.upper)
                        ),
                    ),
                )
