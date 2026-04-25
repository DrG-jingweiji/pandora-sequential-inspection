"""Experiment 3: Policy benchmark (Tables 3, 4, EC.1).

Compare π^OPT, π^{F*,P*}, π^index, π^W, π^STP on:
  - Small instances: N=2..9, 1000 instances (with OPT)
  - Large instances: N=10..16, 300 instances (heuristics only)

Report normalized performance (mean/std/worst) and exact optimality frequency.

Outputs:
  - table_3_performance.csv / .tex
  - table_4_exact_optimality.csv / .tex
  - table_EC1_runtime.csv / .tex
"""

import os
import time
import numpy as np
import pandas as pd

from pandora.solver import PandoraSolver
from pandora.policies import (
    index_policy, whittle_policy, stp_policy,
    best_committing_policy, weitzman_policy,
)
from pandora.instance_generator import generate_prototypical_boxes
from experiments.config import (
    SMALL_N_RANGE, LARGE_N_RANGE, SMALL_INSTANCES, LARGE_INSTANCES,
    SEED, NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE, DP_CUTOFF, OUTPUT_DIR,
    DEFAULT_WORKERS,
)
from experiments.parallel import (
    generate_instance_tasks, run_parallel, checkpoint_path_for, get_shared,
)


POLICY_NAMES = ['OPT', 'INDEX', 'WHITTLE', 'STP', 'COM', 'WEITZMAN']
HEURISTIC_NAMES = ['INDEX', 'WHITTLE', 'STP', 'COM', 'WEITZMAN']


def _benchmark_worker(N, rep_idx, indices):
    """Evaluate all policies on one instance, return values and times."""
    selected_boxes = get_shared('selected_boxes')
    box_list = [selected_boxes[i] for i in indices]
    solver = PandoraSolver(box_list)

    result = {}

    if N <= DP_CUTOFF:
        t0 = time.time()
        opt_val = solver.solve_dp()
        result['OPT_value'] = float(opt_val)
        result['OPT_time'] = time.time() - t0

    for name, policy_fn in [('INDEX', index_policy),
                             ('WHITTLE', whittle_policy),
                             ('STP', stp_policy),
                             ('WEITZMAN', weitzman_policy)]:
        t0 = time.time()
        val = solver.evaluate_policy(policy_fn)
        result[f'{name}_value'] = float(val)
        result[f'{name}_time'] = time.time() - t0

    t0 = time.time()
    _, com_val = best_committing_policy(solver)
    result['COM_value'] = float(com_val)
    result['COM_time'] = time.time() - t0

    return result


def run_policy_benchmark(n_range=None, n_instances_small=None,
                         n_instances_large=None, seed=None,
                         selected_boxes=None, n_workers=None,
                         legacy_sampling=False,
                         write_exact_table=True):
    """Run Experiment 3: policy performance comparison.

    Returns three DataFrames: performance_df, time_df, exact_df.
    """
    if seed is None:
        seed = SEED
    if n_range is None:
        n_range = list(SMALL_N_RANGE) + list(LARGE_N_RANGE)
    if n_instances_small is None:
        n_instances_small = SMALL_INSTANCES
    if n_instances_large is None:
        n_instances_large = LARGE_INSTANCES
    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    if selected_boxes is None:
        print("Generating prototypical boxes...")
        selected_boxes = generate_prototypical_boxes(
            NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE
        )

    def _n_instances(N):
        return n_instances_small if N <= DP_CUTOFF else n_instances_large

    tasks = generate_instance_tasks(
        n_range, _n_instances, len(selected_boxes), seed,
        legacy_sampling=legacy_sampling,
    )

    ckpt_name = 'policy_benchmark_legacy' if legacy_sampling else 'policy_benchmark'
    ckpt = checkpoint_path_for(OUTPUT_DIR, ckpt_name)
    all_results = run_parallel(
        _benchmark_worker, tasks,
        shared_data={'selected_boxes': selected_boxes},
        n_workers=n_workers,
        checkpoint_path=ckpt,
        desc="Benchmark",
    )

    # ── Aggregate per N ──────────────────────────────────────────────
    perf_rows = []
    time_rows = []
    exact_opt_rows = []

    for N in n_range:
        n_reps = _n_instances(N)
        reps = [all_results[f"{N}_{rep}"]
                for rep in range(n_reps) if f"{N}_{rep}" in all_results]
        if not reps:
            continue

        perf_row = {'N': N}
        time_row = {'N': N}
        exact_row = {'N': N}

        for policy in POLICY_NAMES:
            if policy == 'OPT' and N > DP_CUTOFF:
                for m in ['mean', 'std', 'worst']:
                    perf_row[f'{policy}_{m}'] = None
                time_row[f'{policy}_mean'] = None
                time_row[f'{policy}_std'] = None
                continue

            v_arr = np.array([r[f'{policy}_value'] for r in reps])

            if N <= DP_CUTOFF:
                opt_arr = np.array([r['OPT_value'] for r in reps])
                normalized = np.where(opt_arr > 0, v_arr / opt_arr, 1.0)
            else:
                best_arr = np.maximum.reduce([
                    np.array([r[f'{p}_value'] for r in reps])
                    for p in HEURISTIC_NAMES
                ])
                normalized = np.where(best_arr > 0, v_arr / best_arr, 1.0)

            perf_row[f'{policy}_mean'] = float(np.mean(normalized))
            perf_row[f'{policy}_std'] = float(np.std(normalized))
            perf_row[f'{policy}_worst'] = float(np.min(normalized))

            t_arr = np.array([r[f'{policy}_time'] for r in reps])
            time_row[f'{policy}_mean'] = float(np.mean(t_arr))
            time_row[f'{policy}_std'] = float(np.std(t_arr))

            if N <= DP_CUTOFF and policy != 'OPT':
                exact_count = np.sum((1.0 - normalized) < 1e-4)
                exact_row[f'{policy}_exact_pct'] = float(exact_count / len(v_arr))

        perf_rows.append(perf_row)
        time_rows.append(time_row)
        if N <= DP_CUTOFF:
            exact_opt_rows.append(exact_row)

    perf_df = pd.DataFrame(perf_rows)
    time_df = pd.DataFrame(time_rows)
    exact_df = pd.DataFrame(exact_opt_rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    perf_df.to_csv(os.path.join(OUTPUT_DIR, 'table_3_performance.csv'), index=False)
    time_df.to_csv(os.path.join(OUTPUT_DIR, 'table_EC1_runtime.csv'), index=False)
    if write_exact_table:
        exact_df.to_csv(os.path.join(OUTPUT_DIR, 'table_4_exact_optimality.csv'), index=False)

    from experiments.formatting import (
        format_table_3, format_table_4, format_table_EC1, save_latex,
    )
    save_latex(os.path.join(OUTPUT_DIR, 'table_3_performance.tex'),
               format_table_3(perf_df, dp_cutoff=DP_CUTOFF))
    if write_exact_table:
        save_latex(os.path.join(OUTPUT_DIR, 'table_4_exact_optimality.tex'),
                   format_table_4(exact_df))
    save_latex(os.path.join(OUTPUT_DIR, 'table_EC1_runtime.tex'),
               format_table_EC1(time_df, dp_cutoff=DP_CUTOFF))

    print("\nPerformance results:")
    print(perf_df.to_string(index=False))
    if write_exact_table:
        print("\nExact optimality:")
    else:
        print("\nExact optimality from benchmark pool (not written as Table 4):")
    print(exact_df.to_string(index=False))

    return perf_df, time_df, exact_df
