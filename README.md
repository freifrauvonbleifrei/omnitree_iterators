# omnitree-iterators

Exact geometric neighborhood iterators for **normalized [DyAda](https://github.com/freifrauvonbleifrei/DyAda) omnitrees**.
A project for building numerical stencils on anisotropically refined dyadic meshes.

The geometry API provides leaf iteration, shared face patches, boundary faces, per-cell face neighbors, point location, and rectangular region intersections.
It supports arbitrary dimension, hanging interfaces, and crossed tangential refinements without imposing mesh balancing.
Bounds and interface measures use exact rational arithmetic.

## Advection on adaptive omnitrees

![Stripe and ellipsoid advection: directional omnitrees versus isotropic refinement, with mesh boundaries](docs/advection.gif)

A slice at **z = 0.65** through four 3D simulations: a translating stripe and an ellipsoidal Gaussian, each with directional or isotropic refinement and a target of **456 rectangles before balancing**. Componentwise face 2:1 balancing may add cells; panel labels show the actual count.
The right column shows the **analytical solution** on the same slice and color scale.
Numerical colors show cell averages; analytical colors show point values; lines show the mesh.
The runs continue to **t = 1.5**, with regridding every 0.1 time units.
The exact stripe center reaches x = 0.825 and the ellipsoid center x = 0.75.

The animation records actual solver states, holding each until its next numerical step.
Each panel displays its own state time: the methods use different global CFL time steps, while the analytical column follows the animation display time.
No extra steps are introduced to smooth the animation.
The first-order upwind solver is an experimental validation example, not a general PDE solver.

A [balanced MP4](docs/advection-balanced.mp4) accompanies the current GIF.
The [previous unbalanced GIF](docs/advection-unbalanced.gif) is retained for comparison.

## Install

Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python examples/crossed_faces.py
```

For development with an adjacent DyAda checkout:

```bash
python -m pip install -e ../DyAda
```

## Quick start

```python
from dyada.descriptor import RefinementDescriptor
from omnitree_iterators import Omnitree

mesh = Omnitree(RefinementDescriptor(2, [2, 1]))

# Every interior face patch appears once, including coarse/fine contacts.
for face in mesh.interfaces():
    print(face.minus.box_index, face.plus.box_index, face.axis, face.measure)

# Face neighbors on the positive x side of leaf 0.
for face in mesh.face_neighbors(0, axis=0, side=+1):
    print(face.plus.box_index, face.measure)

cell = mesh.locate((0.1, 0.2))
for hit in mesh.intersect((0, 0), (0.4, 0.6)):
    print(hit.cell.box_index, hit.bounds.measure)
```

| Operation | Result |
|---|---|
| `iter(mesh)` | Leaves in DyAda's Morton order |
| `interfaces()` | Interior face patches, oriented from minus to plus along the positive axis |
| `interfaces(include_boundary=True)` | Interior and domain-boundary patches |
| `boundary_faces()` | Boundary patches, with the exterior cell represented by `None` |
| `face_neighbors(box_index, axis, side)` | Interior contacts on one cell face; `side` is −1 or +1 |
| `locate(point)` | Containing leaf, or `None` outside the half-open unit domain |
| `intersect(lower, upper)` | Leaves and their positive-volume intersections with the query box |

A face's `measure` is its area (length in 2D), whereas `face.bounds.measure` is zero because the bounds have zero normal extent.
Edge-only and corner-only contacts are excluded.

## Mesh contract

Input must satisfy omnitree normalization: when all children split an axis, the parent must split that axis too.
Normalization does not imply balancing.
The constructor validates the descriptor and rejects unnormalized input.

```python
from dyada.refinement import normalize_discretization

normalized, mapping, _ = normalize_discretization(discretization, track_mapping="boxes")
mesh = Omnitree.from_discretization(normalized)
```

Use the returned old-to-new box mapping to reorder associated field data;
DyAda returns an empty mapping when normalization makes no changes.
`from_discretization` requires `MortonOrderLinearization`.

`Omnitree` is an indexed snapshot.
`box_index` identifies a leaf in the snapshot's data ordering; `descriptor_index` includes internal nodes.
Geometric keys `(level, index)` survive reorderings.
Queries use the unit hypercube and half-open cells, so coordinate 1 lies outside the domain.

See [geometry and implementation](docs/geometry.md) for exact coordinate semantics, traversal algorithms, and complexity limitations.
The geometry API does not prescribe stencil weights, periodic topology, boundary conditions, or balancing.
The experimental advection regridder applies the balance rule below.

## Exact 2:1 balance rule

The advection examples enforce **componentwise 2:1 balance between face-neighbor leaves**.
Two leaves are face neighbors when their boundaries share a patch of positive
(d−1)-dimensional measure: a face area in 3D, an edge length in 2D, or an endpoint
in 1D. The patch need not cover an entire face of either leaf.
Edge-only and corner-only contacts in 3D, and corner-only contacts in 2D, are excluded.

For every such pair of leaves `K`, `L`, the rule is:

```text
abs(K.level[j] - L.level[j]) <= 1   for every coordinate axis j
```

**Every axis is checked, including those tangential to the shared face.**
For example, across a face normal to x, the x, y, and z levels must each differ
by at most one. A pair with levels `(5, 2, 3)` and `(4, 4, 3)` violates the rule
because its y-level difference is two, even though its x-level difference is one.

`level[j]` is the geometric directional refinement level: on the unit domain,
the cell width along axis j is `2**(-level[j])`. Thus corresponding widths of
face neighbors have ratios between 1/2 and 2. This does not constrain a cell's
own aspect ratio: levels `(8, 2, 2)` are allowed when its face neighbors satisfy
the componentwise rule. Descriptor depth and cell volume are not used to test balance.

Balancing runs after initial adaptation and after every numerical regrid.
It repeatedly refines the directionally coarser cell along each offending axis
until every face-neighbor pair satisfies the rule. Normalization is a separate
operation and does not, by itself, establish balance.

The rectangle budget applies **before** this refinement-only closure, so the
balanced mesh can exceed it. Animation labels show actual cell counts, and
`balance_history` records the added cells. `--no-balance` disables the closure
in the benchmark and animation scripts; the geometry iterators themselves
continue to accept unbalanced normalized omnitrees.

## Numerical validation and results

The experimental solver assembles one conservative upwind flux per face patch and advances all rectangles with a common forward-Euler step constrained by outgoing fluxes.
Adaptive runs use directional Haar energy, limited conservative child prediction, a downstream safety region, partial-axis coarsening through downsplitting, and componentwise face 2:1 balance.
Normalization mappings transfer the resulting leaf averages.
See [adaptation details](docs/adaptation.md).

| Experiment | Report |
|---|---|
| Fixed uniform, hanging, and crossed-interface meshes | [Advection validation](results/advection/README.md) |
| Current 2:1 balanced advection | [Results and balancing overhead](results/balanced-advection/README.md) |
| Unbalanced advection to t = 1.5 | [Accuracy and runtime](results/haar-partial-long-advection/README.md) |
| Partial-axis Haar adaptation at t = 0.3 | [Accuracy and runtime](results/haar-partial-advection/README.md) · [Profile](results/haar-partial-profile/README.md) |
| Earlier whole-branch Haar adaptation | [Comparison with slab integration](results/haar-advection/README.md) |
| Original slab-integration adaptation | [Historical baseline](results/adaptive-advection/README.md) |

Errors are measured against the number of rectangles using the physical L² norm, including unresolved within-cell variation.
The examples show directional adaptation benefits; they do not establish asymptotic convergence orders.
Normalization dominates the measured directional ellipsoid runtime.
Historical reports identify the algorithm used; timing comparisons are single-run measurements.

Reproduce the current experiments without overwriting archived results:

```bash
python examples/adaptive_advection.py --output results/local-advection
python examples/profile_adaptive_advection.py \
  --reference results/local-advection/measurements.json --output results/local-profile
python examples/adaptive_advection.py --final-time 1.5 --output results/local-long-advection
python examples/animate_advection.py --final-time 1.5 --output docs/advection.gif
```

The GIF generator uses Matplotlib and Pillow and writes its numerical results alongside the animation as JSON. The animation uses approximately 5 display updates per second and a shared 128-color palette; compression preserves the playback duration and final state.

## Tests

```bash
ruff check .
ruff format --check .
pytest --cov=omnitree_iterators
```

CI runs on Python 3.10, 3.12, and 3.14.
Tests cover independent geometric oracles, random normalized trees, crossed 3D interfaces, deep trees, conservation, positivity, Haar energies, partial-axis coarsening, and normalization mappings.

## Background

- [The Beauty of Anisotropic Mesh Refinement: Omnitrees for Efficient Dyadic Discretizations](https://arxiv.org/abs/2508.06316)
- [Towards Fully Dynamic Omnitrees: Moment-Conserving Anisotropic Compression With Wavelets](https://arxiv.org/abs/2607.04881)
- [Reference stack-based Haar transform and wavelet compression](https://github.com/freifrauvonbleifrei/wavelets_with_omnitrees)

Largely vibecoded -- use with caution.
