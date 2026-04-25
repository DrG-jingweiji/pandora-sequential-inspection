# Bundled Old Box Pool

`legacy_selected_boxes_100_0p5.json` is a portable JSON conversion of the old
`f_selected_boxes_100_0.5` prototypical-box pool.

The old main-table experiments did not sample from all 100 boxes directly.
They first applied the old notebook filter `c_F > c_P`, which leaves 56 boxes.
The new loader applies that filter when `filter_cF_gt_cP=True`.

Only this selected pool is bundled because it is sufficient for reproducing the
main old-pool experiments, including Table 4, without depending on an external
old `Numerical` folder or storing the much larger old candidate cloud.
