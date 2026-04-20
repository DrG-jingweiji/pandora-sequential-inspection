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

from pandora.solver import PandoraSolver
from pandora.policies import weitzman_policy
from pandora.instance_generator import generate_prototypical_boxes
from pandora.utils import sum_of_variances, p_dominant_ratio
from experiments.config import (
    SMALL_N_RANGE, SMALL_INSTANCES, SEED, NUM_PROTOTYPICAL_BOXES,
    BOX_DISTANCE, DP_CUTOFF, OUTPUT_DIR, DEFAULT_WORKERS,
)
from experiments.parallel import (
    generate_instance_tasks, run_parallel, checkpoint_path_for, get_shared,
)


def _p_opening_worker(N, rep_idx, indices):
    """Solve one instance and compute P-opening metrics."""
    selected_boxes = get_shared('selected_boxes')
    box_list = [selected_boxes[i] for i in indices]

    solver = PandoraSolver(box_list)
    opt_val = solver.solve_dp()
    n_f, n_p = solver.expected_openings('DP')

    total = n_f + n_p
    p_ratio = n_p / total if total > 0 else 0.0

    weitz_val = solver.evaluate_policy(weitzman_policy)
    weitz_ratio = weitz_val / opt_val if opt_val > 0 else 1.0

    p_dom = p_dominant_ratio(box_list)
    dispersion = sum_of_variances(box_list)

    return {
        'p_ratio': float(p_ratio),
        'p_dominant_fraction': float(p_dom),
        'dispersion': float(dispersion),
        'opt_value': float(opt_val),
        'weitzman_value': float(weitz_val),
        'weitzman_ratio': float(weitz_ratio),
        'n_f_openings': float(n_f),
        'n_p_openings': float(n_p),
        'indices': indices,
    }


_MORE_BOXES_SPEC = {
    'value_list': [0.5, 2.2, 5, 7, 9.5, 12],
    'cond_prob_matrix': [
        [0.3, 0.15, 0.2, 0.15, 0.1, 0.1],
        [0.1, 0.15, 0.15, 0.2, 0.1, 0.3],
        [0.1, 0.1, 0.1, 0.15, 0.15, 0.4],
    ],
    'type_probs': [0.6, 0.2, 0.2],
    'c_F': 3.0,
}
_CP_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _more_boxes_worker(c_P_str, N):
    """Create N identical boxes with the given c_P, solve DP, return P-ratio."""
    from pandora.box import Box

    spec = get_shared('box_spec')
    c_P = float(c_P_str)
    box = Box(spec['value_list'], spec['cond_prob_matrix'],
              spec['type_probs'], c_P, spec['c_F'])
    box_list = [box] * N

    solver = PandoraSolver(box_list)
    solver.solve_dp()
    n_f, n_p = solver.expected_openings('DP')
    total = n_f + n_p
    return {'p_ratio': float(n_p / total if total > 0 else 0.0)}


def run_p_opening_analysis(n_range=None, n_instances=None, seed=None,
                           selected_boxes=None, n_workers=None):
    """Run Experiment 4: P-opening analysis with figure generation.

    Returns a DataFrame with per-instance metrics.
    """
    if n_range is None:
        n_range = SMALL_N_RANGE
    if n_instances is None:
        n_instances = SMALL_INSTANCES
    if seed is None:
        seed = SEED
    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    if selected_boxes is None:
        print("Generating prototypical boxes (unfiltered for P-opening)...")
        selected_boxes = generate_prototypical_boxes(
            NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE, require_p_dominant=False,
        )

    effective_range = [N for N in n_range if N <= DP_CUTOFF]

    tasks = generate_instance_tasks(
        effective_range, lambda N: n_instances, len(selected_boxes), seed,
    )

    ckpt = checkpoint_path_for(OUTPUT_DIR, 'p_opening')
    all_results = run_parallel(
        _p_opening_worker, tasks,
        shared_data={'selected_boxes': selected_boxes},
        n_workers=n_workers,
        checkpoint_path=ckpt,
        desc="P-opening",
    )

    # Build per-instance DataFrame (in deterministic order)
    output_keys = ['p_ratio', 'p_dominant_fraction', 'dispersion',
                   'opt_value', 'weitzman_value', 'weitzman_ratio',
                   'n_f_openings', 'n_p_openings']
    rows = []
    for N in effective_range:
        for rep in range(n_instances):
            key = f"{N}_{rep}"
            if key not in all_results:
                continue
            r = all_results[key]
            rows.append({'N': N, **{k: r[k] for k in output_keys}})

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

    # ── Find low/high dispersion examples ──────────────────────────
    # Require N >= 7 so the scatter plots look comparable to the paper
    # (which shows instances with 9 and 10 boxes).
    low_disp_example = None
    high_disp_example = None
    for N in effective_range:
        if N < 7:
            continue
        for rep in range(n_instances):
            key = f"{N}_{rep}"
            if key not in all_results:
                continue
            r = all_results[key]
            if low_disp_example is None and r['dispersion'] < 5:
                low_disp_example = [selected_boxes[i] for i in r['indices']]
            if high_disp_example is None and r['dispersion'] > 18:
                high_disp_example = [selected_boxes[i] for i in r['indices']]
            if low_disp_example is not None and high_disp_example is not None:
                break
        if low_disp_example is not None and high_disp_example is not None:
            break

    # ── Generate figures ─────────────────────────────────────────────
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


def run_more_boxes_experiment(n_range=None, n_workers=None):
    """Run the 'more boxes' i.i.d. experiment (Figure EC.10d).

    Uses a single fixed box specification (from the old code's
    OneBoxExperiment.ipynb) and varies c_P from 0.1 to 0.6.  For each
    c_P level, creates N = 1..7 identical copies and plots P-ratio vs N.
    """
    if n_range is None:
        n_range = range(1, 8)
    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    n_values = list(n_range)

    tasks = []
    for c_P in _CP_VALUES:
        c_P_str = f"{c_P:.1f}"
        for N in n_values:
            tasks.append((f"{c_P_str}_{N}", c_P_str, N))

    print(f"\nRunning more_boxes experiment ({len(_CP_VALUES)} c_P levels, "
          f"N={n_values[0]}..{n_values[-1]})...")

    ckpt = checkpoint_path_for(OUTPUT_DIR, 'more_boxes')
    all_results = run_parallel(
        _more_boxes_worker, tasks,
        shared_data={'box_spec': _MORE_BOXES_SPEC},
        n_workers=n_workers,
        checkpoint_path=ckpt,
        desc="More boxes",
    )

    results_by_cp = {}
    for c_P in _CP_VALUES:
        c_P_str = f"{c_P:.1f}"
        p_ratios = []
        for N in n_values:
            key = f"{c_P_str}_{N}"
            if key in all_results:
                p_ratios.append(all_results[key]['p_ratio'])
        results_by_cp[c_P] = p_ratios

    from experiments.figures import plot_figure_EC10d
    plot_figure_EC10d(n_values, results_by_cp)

    return n_values, results_by_cp
