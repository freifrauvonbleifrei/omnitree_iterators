"""Stack Haar analysis and conservative, hierarchy-based numerical regridding.

The reverse traversal follows the algorithm in transform_to_all_wavelet_coefficients
in freifrauvonbleifrei/wavelets_with_omnitrees. This implementation uses butterfly
updates rather than dense matrices. Axis zero is the fastest Morton bit.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import count

import numpy as np
from bitarray import bitarray
from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization
from dyada.refinement import normalize_discretization

from omnitree_iterators import Bounds, Omnitree


def haar_block(values):
    """Mean-normalized tensor Haar transform; detail energy is volume*c**2."""
    result = np.array(values, dtype=float, copy=True)
    if result.ndim != 1 or not len(result) or len(result) & (len(result) - 1):
        raise ValueError("Haar block must have power-of-two length")
    stride = 1
    while stride < len(result):
        blocks = result.reshape(-1, 2 * stride)
        left = blocks[:, :stride].copy()
        right = blocks[:, stride:].copy()
        blocks[:, :stride] = (left + right) / 2
        blocks[:, stride:] = (left - right) / 2
        stride *= 2
    return result


@dataclass
class HaarAnalysis:
    coefficients: list[np.ndarray]
    directional_energy: np.ndarray
    subtree_energy: np.ndarray


def analyze(mesh: Omnitree, values: np.ndarray) -> HaarAnalysis:
    """One reverse stack pass. Includes mixed details in every participating axis.

    Subtree energy counts each detail once (directional energies can overlap).
    Scaling coefficients are volume averages, even with unequal leaf depths.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (len(mesh),) or not np.all(np.isfinite(values)):
        raise ValueError("Require one finite value per leaf")
    coefficients = [None] * len(mesh._nodes)
    directional = np.zeros((len(coefficients), mesh.dimension))
    energy = np.zeros(len(coefficients))
    stack = []
    for i in reversed(range(len(coefficients))):
        node = mesh._nodes[i]
        if node.cell is not None:
            block = np.array([values[node.cell.box_index]])
        else:
            block = haar_block([stack.pop() for _ in node.children])
            volume = float(node.bounds.measure)
            axes = [j for j, active in enumerate(node.mask) if active]
            for bit, axis in enumerate(axes):
                directional[i, axis] = volume * sum(
                    c * c for k, c in enumerate(block) if k & (1 << bit)
                )
            energy[i] = volume * float(block[1:] @ block[1:]) + sum(
                energy[j] for j in node.children
            )
        coefficients[i] = block
        stack.append(block[0])
    return HaarAnalysis(coefficients, directional, energy)


@dataclass
class _Candidate:
    bounds: Bounds
    level: tuple[int, ...]
    value: float
    axes: list[int] = field(default_factory=list)
    children: list = field(default_factory=list)
    score: float = 0.0
    retained: bool = False


def _downsplit_leaf_pairs(root):
    """Expose adjacent leaf siblings by pushing their common axis down.

    Every eligible multi-axis parent is visited, including parents with mixed
    leaf/internal children. Choose the axis with the smallest mean discarded
    L2 energy per eligible pair. Restructuring itself is exact: new scaling
    values are pair means; the subsequent budget selection decides which pairs
    to merge. Original children (and all descendant coefficients) are retained.
    """
    pending = [root]
    while pending:
        node = pending.pop()
        if len(node.axes) < 2:
            pending.extend(node.children)
            continue
        choices = []
        for bit, axis in enumerate(node.axes):
            energies = []
            for ordinal, left in enumerate(node.children):
                if ordinal & (1 << bit):
                    continue
                right = node.children[ordinal | (1 << bit)]
                if not left.children and not right.children:
                    volume = float(left.bounds.measure + right.bounds.measure)
                    energies.append(volume * ((left.value - right.value) / 2) ** 2)
            if energies:
                choices.append((sum(energies) / len(energies), axis, bit))
        if choices:
            _, axis, bit = min(choices)
            remaining = [j for j in node.axes if j != axis]
            children = []
            for ordinal in range(1 << len(remaining)):
                # Insert a zero at the pushed Morton bit.
                low = ordinal & ((1 << bit) - 1)
                index = low | ((ordinal >> bit) << (bit + 1))
                left, right = node.children[index], node.children[index | (1 << bit)]
                lower, upper = list(left.bounds.lower), list(left.bounds.upper)
                upper[axis] = right.bounds.upper[axis]
                level = list(left.level)
                level[axis] -= 1
                children.append(
                    _Candidate(
                        Bounds(tuple(lower), tuple(upper)),
                        tuple(level),
                        (left.value + right.value) / 2,
                        [axis],
                        [left, right],
                    )
                )
            node.axes = remaining
            node.children = children
        pending.extend(node.children)


def _preorder(root):
    nodes, pending = [], [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(reversed(node.children))
    return nodes


def _normalize_leaf_values(discretization, averages):
    """Scatter old leaf averages through DyAda's old-to-new box mapping."""
    disc, mapping, _ = normalize_discretization(discretization, track_mapping="boxes")
    values = np.asarray(averages, dtype=float)
    # DyAda returns an empty mapping when no normalization was necessary.
    if mapping:
        reordered = np.empty_like(values)
        for old_index, destinations in enumerate(mapping):
            (new_index,) = destinations  # Normalization preserves each leaf.
            reordered[new_index] = values[old_index]
        values = reordered
    return Omnitree.from_discretization(disc), values


def regrid(field, budget, strategy, *, max_level=9, displacement=None):
    """Prune existing branches and predict at most one new split per old leaf.

    Directional mode downsplits multi-axis parents with adjacent leaf children
    before predicting new children. Budget selection can then merge these pairs
    independently. Isotropic mode retains whole-branch coarsening. Budget is an
    upper bound, not a required leaf count. Geometry normalization follows conservative transfer.
    Safety marking sweeps old cells downstream; it never samples shifted values.
    """
    mesh = field.mesh
    d = mesh.dimension
    if strategy not in ("directional", "isotropic"):
        raise ValueError("Strategy must be directional or isotropic")
    if budget < 2**d or max_level < 1:
        raise ValueError("Require budget >= 2^dimension and positive max_level")
    if any(max(c.level) > max_level for c in mesh):
        raise ValueError("Existing mesh exceeds max_level")
    if displacement is not None and (
        len(displacement) != d or not np.all(np.isfinite(displacement))
    ):
        raise ValueError("Require finite displacement of the mesh dimension")
    analysis = analyze(mesh, field.values)
    nodes = []
    for i, source in enumerate(mesh._nodes):
        level = tuple(
            (hi - lo).denominator.bit_length() - 1
            for lo, hi in zip(source.bounds.lower, source.bounds.upper)
        )
        nodes.append(
            _Candidate(
                source.bounds,
                level,
                analysis.coefficients[i][0],
                [j for j, active in enumerate(source.mask) if active],
            )
        )
    for i, source in enumerate(mesh._nodes):
        nodes[i].children = [nodes[j] for j in source.children]

    # Node details inform leaves beneath that node, including extrema where a
    # slope limiter returns zero. Divide energy by volume before propagation.
    signal = np.zeros((len(nodes), d))
    for i, source in enumerate(mesh._nodes):
        signal[i] = np.maximum(
            signal[i], analysis.directional_energy[i] / float(source.bounds.measure)
        )
        for child in source.children:
            signal[child] = signal[i].copy()
    targets = np.array([signal[c.descriptor_index] for c in mesh])
    if displacement is not None:
        original = targets.copy()
        for cell in mesh:
            if not np.any(original[cell.box_index] > 1e-28):
                continue
            lower = tuple(
                max(Fraction(0), lo + min(Fraction(0), Fraction(str(v))))
                for lo, v in zip(cell.bounds.lower, displacement)
            )
            upper = tuple(
                min(Fraction(1), hi + max(Fraction(0), Fraction(str(v))))
                for hi, v in zip(cell.bounds.upper, displacement)
            )
            for hit in mesh.intersect(lower, upper):
                targets[hit.cell.box_index] = np.maximum(
                    targets[hit.cell.box_index], original[cell.box_index]
                )

    root = nodes[0]
    if strategy == "directional":
        _downsplit_leaf_pairs(root)

    for cell in mesh:
        i = cell.box_index
        node = nodes[cell.descriptor_index]
        widths = np.array(
            [float(hi - lo) for lo, hi in zip(cell.bounds.lower, cell.bounds.upper)]
        )
        prediction = (field.gradients[i] * widths / 4) ** 2
        importance = np.maximum(prediction, targets[i])
        eligible = [j for j in range(d) if node.level[j] < max_level]
        if not eligible or (strategy == "isotropic" and len(eligible) != d):
            continue
        axes = (
            [max(eligible, key=lambda j: importance[j])]
            if strategy == "directional"
            else eligible
        )
        if sum(importance[j] for j in axes) <= 1e-28:
            continue
        node.axes = axes
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
            center = np.array([float((a + b) / 2) for a, b in zip(lo, hi)])
            value = node.value + field.gradients[i] @ (center - field.centers[i])
            node.children.append(
                _Candidate(Bounds(tuple(lo), tuple(hi)), tuple(level), value)
            )
        node.score = float(cell.volume) * sum(importance[j] for j in axes)

    # Descendant energy keeps a zero-detail ancestor eligible (symmetric peaks).
    for node in reversed(_preorder(root)):
        if node.children:
            block = haar_block([child.value for child in node.children])
            local = float(node.bounds.measure) * float(block[1:] @ block[1:])
            node.score = max(node.score, local + sum(c.score for c in node.children))
    pending = []
    serial = count()

    def propose(node):
        if node.children and node.score > 1e-28 * float(node.bounds.measure):
            heapq.heappush(
                pending, (-node.score / (len(node.children) - 1), next(serial), node)
            )

    # Keep the root split, including its remaining axes after downsplitting.
    leaves = 1
    if root.children and len(root.children) <= budget:
        root.retained = True
        leaves = len(root.children)
        for child in root.children:
            propose(child)
    else:
        propose(root)
    while pending:
        _, _, node = heapq.heappop(pending)
        cost = len(node.children) - 1
        if leaves + cost > budget:
            continue
        node.retained = True
        leaves += cost
        for child in node.children:
            propose(child)
    words, averages = [], []
    stack = [root]
    while stack:
        node = stack.pop()
        words.append(
            "".join("1" if node.retained and j in node.axes else "0" for j in range(d))
        )
        if node.retained:
            stack.extend(reversed(node.children))
        else:
            averages.append(node.value)
    descriptor = RefinementDescriptor.from_binary(d, bitarray("".join(words)))
    disc = Discretization(MortonOrderLinearization(), descriptor)
    return _normalize_leaf_values(disc, averages)
