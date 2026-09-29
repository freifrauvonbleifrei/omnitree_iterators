"""Refinement-only componentwise 2:1 closure across positive-area face contacts."""

from dataclasses import dataclass

import numpy as np
from bitarray import bitarray
from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization

from omnitree_iterators import Omnitree
from omnitree_iterators.validation.haar import _normalize_leaf_values


@dataclass(frozen=True)
class BalanceStats:
    input_cells: int
    output_cells: int
    rounds: int

    @property
    def added_cells(self):
        return self.output_cells - self.input_cells


def balance_marks(mesh: Omnitree) -> dict[int, set[int]]:
    """Union the offending axes for each directionally coarser leaf."""
    marks: dict[int, set[int]] = {}
    for face in mesh.interfaces():
        left, right = face.minus, face.plus
        for axis, (a, b) in enumerate(zip(left.level, right.level)):
            if abs(a - b) > 1:
                coarse = left if a < b else right
                marks.setdefault(coarse.box_index, set()).add(axis)
    return marks


def balance_mesh(mesh: Omnitree, values: np.ndarray):
    """Refine until every face-neighbor directional level jump is at most one.

    Balanced meshes are returned unchanged. Every pass bisects marked axes,
    conservatively predicts child means with bounded linear reconstruction,
    and normalizes using DyAda's box mapping. No axis exceeds its initial
    global maximum level, so refinement-only closure terminates. Cell budget
    is deliberately not enforced here. Corners and edges are not balanced.
    """
    from omnitree_iterators.validation.adaptive import ReconstructedField

    values = np.asarray(values, dtype=float)
    if values.shape != (len(mesh),) or not np.all(np.isfinite(values)):
        raise ValueError("Require one finite value per leaf")
    initial_cells, rounds = len(mesh), 0
    d = mesh.dimension
    while marks := balance_marks(mesh):
        field = ReconstructedField(mesh, values)
        words, averages = [], []
        for node in mesh._nodes:
            if node.cell is None:
                words.append("".join("1" if b else "0" for b in node.mask))
                continue
            cell = node.cell
            axes = sorted(marks.get(cell.box_index, ()))
            if not axes:
                words.append("0" * d)
                averages.append(values[cell.box_index])
                continue
            words.append("".join("1" if j in axes else "0" for j in range(d)))
            for ordinal in range(1 << len(axes)):
                offset = np.zeros(d)
                for bit, axis in enumerate(axes):
                    width = float(cell.bounds.upper[axis] - cell.bounds.lower[axis])
                    offset[axis] = (1 if ordinal & (1 << bit) else -1) * width / 4
                words.append("0" * d)
                averages.append(
                    values[cell.box_index] + field.gradients[cell.box_index] @ offset
                )
        descriptor = RefinementDescriptor.from_binary(d, bitarray("".join(words)))
        disc = Discretization(MortonOrderLinearization(), descriptor)
        mesh, values = _normalize_leaf_values(disc, averages)
        rounds += 1
    return mesh, values, BalanceStats(initial_cells, len(mesh), rounds)
