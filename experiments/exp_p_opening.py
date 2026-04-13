"""Experiment 4: P-opening analysis (Figures EC.9, EC.10).

Analyzes when P-opening is worthwhile under pi^OPT:
  - P-ratio vs proportion of boxes with sigma^P > sigma^F  (Figure EC.9a)
  - Suboptimality of pi^Weitzman vs that proportion        (Figure EC.9b)
  - P-ratio vs dispersion                                  (Figure EC.10c)
  - Low/high dispersion example scatters                   (Figure EC.10a/b)
  - P-ratio vs number of i.i.d. boxes                      (Figure EC.10d)

Outputs:
  - p_opening_analysis.csv
  - figure_EC9a_p_ratio_vs_proportion.png
  - figure_EC9b_weitzman_vs_proportion.png
  - figure_EC10a_low_dispersion.png
  - figure_EC10b_high_dispersion.png
  - figure_EC10c_p_ratio_vs_dispersion.png
  - figure_EC10d_p_ratio_vs_num_boxes.png
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from pandora.solver import PandoraSolver
from pandora.policies import weitzman_policy
from pandora.instance_generator import generate_prototypical_boxes, sample_instance
from pandora.utils import sum_of_variances, p_dominant_ratio
from experiments.config import (
    SMALL_N_RANGE, SMALL_INSTANCES, SEED, NUM_PROTOTYPICAL_BOXES,
    BOX_DISTANCE, DP_CUTOFF, OUTPUT_DIR,
)


def run_p_opening_analysis(n_range=None, n_instances=None, seed=None,
                           selected_boxes=None):
    """Run Experiment 4: P-opening analysis with figure generation.

    Returns a DataFrame with per-instance metrics.
    """
    if n_range is None:
        n_range = SMALL_N_RANGE
    if n_instances is None:
        n_instances = SMALL_INSTANCES
    if seed is None:
        seed = SEED

    import random
    random.seed(seed)
    np.random.seed(seed)
    rng_np = np.random.default_rng(seed)

    if selected_boxes is None:
        print("Generating prototypical boxes...")
        selected_boxes = generate_prototypical_boxes(
            NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE
        )

    rows = []
    low_disp_example = None
    high_disp_example = None

    for N in n_range:
        if N > DP_CUTOFF:
            continue
        print(f"\n--- N = {N} ---")

        for rep in tqdm(range(n_instances), desc=f"N={N}"):
            box_list, _ = sample_instance(selected_boxes, N, rng_np)
            solver = PandoraSolver(box_list)

            opt_val = solver.solve_dp()
            n_f, n_p = solver.expected_openings('DP')

            total_openings = n_f + n_p
            p_ratio = n_p / total_openings if total_openings > 0 else 0.0

            weitz_val = solver.evaluate_policy(weitzman_policy)
            weitz_ratio = weitz_val / opt_val if opt_val > 0 else 1.0

            p_dom = p_dominant_ratio(box_list)
            dispersion = sum_of_variances(box_list)

            if low_disp_example is None and dispersion < 3:
                low_disp_example = list(box_list)
            if high_disp_example is None and dispersion > 18:
                high_disp_example = list(box_list)

            rows.append({
                'N': N,
                'p_ratio': p_ratio,
                'p_dominant_fraction': p_dom,
                'dispersion': dispersion,
                'opt_value': opt_val,
                'weitzman_value': weitz_val,
                'weitzman_ratio': weitz_ratio,
                'n_f_openings': n_f,
                'n_p_openings': n_p,
            })

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'p_opening_analysis.csv'), index=False)

    summary = df.groupby('N').agg({
        'p_ratio': 'mean',
        'weitzman_ratio': 'mean',
        'dispersion': 'mean',
    }).reset_index()
    print("\nP-opening summary by N:")
    print(summary.to_string(index=False))

    # --- Generate figures ---
    print("\nGenerating P-opening figures...")
    from experiments.figures import (
        plot_figure_EC9a, plot_figure_EC9b,
        plot_figure_EC10a, plot_figure_EC10b, plot_figure_EC10c,
    )

    p_dom_arr = df['p_dominant_fraction'].values
    p_ratio_arr = df['p_ratio'].values
    weitz_ratio_arr = df['weitzman_ratio'].values
    disp_arr = df['dispersion'].values

    plot_figure_EC9a(p_dom_arr, p_ratio_arr)
    plot_figure_EC9b(p_dom_arr, weitz_ratio_arr)
    plot_figure_EC10c(disp_arr, p_ratio_arr)

    if low_disp_example is not None:
        plot_figure_EC10a(low_disp_example)
    else:
        print("  Warning: no low-dispersion example found (dispersion < 3)")

    if high_disp_example is not None:
        plot_figure_EC10b(high_disp_example)
    else:
        print("  Warning: no high-dispersion example found (dispersion > 18)")

    return df


def run_more_boxes_experiment(selected_boxes=None, seed=None,
                              n_box_specs=10, n_range=None):
    """Run the 'more boxes' i.i.d. experiment (Figure EC.10d).

    Randomly sample a few box specifications, then for each N in n_range
    create an instance of N identical copies of that box, solve DP,
    and compute the P-ratio.
    """
    if seed is None:
        seed = SEED
    if n_range is None:
        n_range = range(2, 10)

    import random
    random.seed(seed + 1000)
    np.random.seed(seed + 1000)
    rng_np = np.random.default_rng(seed + 1000)

    if selected_boxes is None:
        print("Generating prototypical boxes...")
        selected_boxes = generate_prototypical_boxes(
            NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE
        )

    n_values = list(n_range)
    p_ratios_by_n = [[] for _ in n_values]

    indices = rng_np.integers(0, len(selected_boxes), size=n_box_specs)
    box_specs = [selected_boxes[i] for i in indices]

    print(f"\nRunning more_boxes experiment ({n_box_specs} box specs, "
          f"N={n_values[0]}..{n_values[-1]})...")

    for box_spec in tqdm(box_specs, desc="box specs"):
        for ni, N in enumerate(n_values):
            box_list = [box_spec] * N
            solver = PandoraSolver(box_list)
            solver.solve_dp()
            n_f, n_p = solver.expected_openings('DP')
            total = n_f + n_p
            p_ratio = n_p / total if total > 0 else 0.0
            p_ratios_by_n[ni].append(p_ratio)

    from experiments.figures import plot_figure_EC10d
    plot_figure_EC10d(n_values, p_ratios_by_n)

    return n_values, p_ratios_by_n
