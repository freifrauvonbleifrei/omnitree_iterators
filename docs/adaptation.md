# Experimental adaptive advection

Numerical regridding now uses `validation/haar.py`: a reverse stack Haar
transform computes every existing node average and tensor detail block once.
Mixed details contribute to every participating direction; total subtree energy
counts each detail once. A budgeted heap retains branches with large subtree
energy. Limited linear reconstruction predicts at most one new split per old
leaf, conserving its mean and respecting the reconstruction bounds. A downstream
swept-region safety layer propagates directional importance without requesting
translated averages. `--no-lookahead` disables that safety layer.

Directional regridding now visits every existing multi-axis parent with at
least one adjacent leaf-sibling pair. It pushes down the eligible axis with the
smallest mean pair merge energy, exposing pairs to independent budget selection.
This lossless rewrite also handles mixed leaf/internal-child parents; pair means
supply new scaling coefficients and the existing child subtrees are preserved.
The selector can coarsen the exposed pairs, then normalization restores property 3.
This is one restructuring pass per regrid, not an exhaustive search over possible
axis orders. Isotropic mode keeps whole-branch coarsening. Existing
node signals propagate to descendant leaves, so marking is deliberately
conservative. Budgets are upper bounds, and the two strategies may finish with
different rectangle counts. The Haar pass is linear at fixed dimension; safety
queries, reconstruction, heap selection and normalization have separate costs.
The numerical path never calls `ReconstructedField.average`. Leaf averages are
emitted in candidate traversal order and scattered through DyAda’s old-to-new
box mapping after normalization; no rectangle-keyed value lookup is needed.

Analytic initialization retains the earlier slab-based initializer, including
its optional translated initial-data probes. Subsequent adaptation uses only
the computed field. Unbalanced runs carry `adaptation: hierarchical-haar-v2-partial`;
balanced runs use `hierarchical-haar-v3-balanced` as described below.

The stack traversal is based on the algorithm in
[`transform_to_all_wavelet_coefficients`](https://github.com/freifrauvonbleifrei/wavelets_with_omnitrees/blob/main/wavelets_with_omnitrees.py);
local blocks here use butterfly operations rather than dense matrices.

The reports linked from the project README distinguish the historical slab,
whole-branch Haar, and partial-axis Haar methods.

## Componentwise face 2:1 balance

The current `hierarchical-haar-v3-balanced` experiment applies a refinement-only
balance closure after initial adaptation and after every numerical regrid.
For each pair of leaves sharing a positive-area face patch, every directional
level difference must be at most one. Edge-only and corner-only contacts do not
participate. Aspect ratios within individual cells remain unrestricted.

Each pass marks the directionally coarser leaf along every offending axis,
bisects those axes, and repeats until no violations remain. New cell averages
come from the bounded conservative linear reconstruction; normalization uses
DyAda's old-to-new leaf mapping. Initial values are evaluated analytically on
the final balanced geometry. No direction exceeds the initial mesh's global
maximum level in that direction during closure.

The cell budget constrains energy-based selection **before balancing**. Closure
may exceed it; `balance_history` records input/output counts, added cells, and
iteration counts at every adaptation. `maximum_cells` records the largest mesh
used for transport. Comparisons must use actual cell counts, not just targets.
The existing energy marking and optional velocity-based look-ahead are retained;
balancing itself has no velocity input and does not guarantee a predictive buffer.
Use `--no-balance` in the benchmark or animation CLI to reproduce the previous
unbalanced numerical method.
