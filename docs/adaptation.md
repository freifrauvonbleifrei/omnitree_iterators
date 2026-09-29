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
the computed field. Results carry `adaptation: hierarchical-haar-v2-partial`.

The stack traversal is based on the algorithm in
[`transform_to_all_wavelet_coefficients`](https://github.com/freifrauvonbleifrei/wavelets_with_omnitrees/blob/main/wavelets_with_omnitrees.py);
local blocks here use butterfly operations rather than dense matrices.

The reports linked from the project README distinguish the historical slab,
whole-branch Haar, and partial-axis Haar methods.
