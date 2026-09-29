# Geometry and implementation

## Geometry and identities

- The domain is the unit hypercube. `Cell.level` and `Cell.index` are tuples of
  arbitrary-size integers; `Cell.key == (level, index)` identifies its geometry.
- `Cell.box_index` matches DyAda's leaf/data ordering. `descriptor_index` includes
  internal nodes. They are different index spaces.
- `Bounds.lower` and `Bounds.upper` contain exact `fractions.Fraction` values.
  No epsilon comparisons or finest-grid voxelization are used.
- An `Interface` contains `minus`, `plus`, `axis`, and intersection `bounds`.
  Its normal points along the positive axis from minus to plus. Its `measure`
  is the product of tangential lengths; in 1D this is one. `bounds.measure` is
  full-dimensional volume and is therefore zero for a face.
- At a lower domain boundary `minus` is `None`; at an upper boundary `plus` is
  `None`. The outward normal is respectively negative or positive. Use
  `boundary_faces()` or `interfaces(include_boundary=True)` for these faces.
- `face_neighbors(box_index, axis, side)` accepts `side=-1` or `+1`, excludes
  exterior boundaries, and preserves the same global minus/plus orientation.
  On the negative side the neighbor is `face.minus`; on the positive side it
  is `face.plus`. Only positive-measure face contacts count, not corners/edges.
- `locate` uses half-open cells: internal split points belong to the upper child;
  coordinate 1 lies outside the domain. `intersect` returns positive-volume
  intersections only and clips regions extending beyond the domain. Regions
  must have strictly positive extent in each dimension.
- Pass `Fraction` for exact rational query coordinates. Floats are interpreted
  as their actual binary values. Iteration is deterministic, but consumers
  should not rely on a geometric sort order for interface patches.

## Structure and algorithm

Construction validates descriptor structure and normalization (property 3): if
all children split an axis, the parent must split it too. Unnormalized input is
rejected with the offending descriptor index and axis. Normalization is not
balancing: a leaf can still have arbitrarily many face neighbors.

The descriptor is read once into an indexed snapshot with parent/child links
and exact bounds, taking O(d V) storage and arithmetic work for V tree nodes,
ignoring integer bit complexity. The compact descriptor itself is not modified
or retained. This extra index trades memory for direct subtree navigation.

For every internal node, the interface iterator pairs children differing in one
Morton bit. It then traverses the two touching subtrees, pruning children away
from the face and pairs without tangential overlap. A leaf contact belongs to
its unique lowest common ancestor, so every shared patch is emitted once.
Crossed refinements are handled by intersecting rectangles during traversal.
Per-cell queries ascend to a crossing sibling and then traverse its boundary
subtree. Point and region queries similarly descend the index. All stacks are
explicit, so traversal does not depend on Python's recursion limit.

No all-pairs leaf search or permanent adjacency graph is used. Traversal cost
depends on visited subtree pairs and emitted patches, and is not claimed to be
linear or constant per cell. Boundary iteration scans leaves. Python objects
and rational arithmetic favor clarity and correctness over compact storage or
maximum throughput. See the benchmark reports linked from the project README
for measurements of the experimental advection workload.

The snapshot is independent of subsequent DyAda mutations. **Rebuild it after
adaptation**, including normalization. Geometric keys survive pure reorderings;
leaf and descriptor indices do not. No incremental cache updating is provided.

