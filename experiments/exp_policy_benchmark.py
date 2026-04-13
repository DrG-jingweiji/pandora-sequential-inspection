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
from tqdm import tqdm

from pandora.solver import PandoraSolver
from pandora.policies import (
    index_policy, whittle_policy, stp_policy,
    best_committing_policy, weitzman_policy, make_committing_policy,
)
from pandora.instance_generator import generate_prototypical_boxes, sample_instance
from experiments.config import (
    SMALL_N_RANGE, LARGE_N_RANGE, SMALL_INSTANCES, LARGE_INSTANCES,
    SEED, NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE, DP_CUTOFF, OUTPUT_DIR,
)


POLICY_NAMES = ['OPT', 'INDEX', 'WHITTLE', 'STP', 'COM', 'WEITZMAN']


def run_policy_benchmark(n_range=None, n_instances_small=None,
                         n_instances_large=None, seed=None,
                         selected_boxes=None):
    """Run Experiment 3: policy performance comparison.

    Returns three DataFrames: performance_df, time_df, exact_df.
    """
    if seed is None:
        seed = SEED

    import random
    random.seed(seed)
    np.random.seed(seed)
    rng_np = np.random.default_rng(seed)

    if n_range is None:
        n_range = list(SMALL_N_RANGE) + list(LARGE_N_RANGE)
    if n_instances_small is None:
        n_instances_small = SMALL_INSTANCES
    if n_instances_large is None:
        n_instances_large = LARGE_INSTANCES

    if selected_boxes is None:
        print("Generating prototypical boxes...")
        selected_boxes = generate_prototypical_boxes(
            NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE
        )

    perf_rows = []
    time_rows = []
    exact_opt_rows = []

    for N in n_range:
        n_reps = n_instances_small if N <= DP_CUTOFF else n_instances_large
        print(f"\n--- N = {N}, {n_reps} instances ---")

        values = {p: [] for p in POLICY_NAMES}
        times = {p: [] for p in POLICY_NAMES}

        for rep in tqdm(range(n_reps), desc=f"N={N}"):
            box_list, _ = sample_instance(selected_boxes, N, rng_np)
            solver = PandoraSolver(box_list)

            # OPT (only for small N)
            if N <= DP_CUTOFF:
                t0 = time.time()
                opt_val = solver.solve_dp()
                t_opt = time.time() - t0
                values['OPT'].append(opt_val)
                times['OPT'].append(t_opt)

            # INDEX
            t0 = time.time()
            idx_val = solver.evaluate_policy(index_policy)
            times['INDEX'].append(time.time() - t0)
            values['INDEX'].append(idx_val)

            # WHITTLE
            t0 = time.time()
            wh_val = solver.evaluate_policy(whittle_policy)
            times['WHITTLE'].append(time.time() - t0)
            values['WHITTLE'].append(wh_val)

            # STP
            t0 = time.time()
            stp_val = solver.evaluate_policy(stp_policy)
            times['STP'].append(time.time() - t0)
            values['STP'].append(stp_val)

            # COM (best committing)
            t0 = time.time()
            _, com_val = best_committing_policy(solver)
            times['COM'].append(time.time() - t0)
            values['COM'].append(com_val)

            # WEITZMAN
            t0 = time.time()
            weitz_val = solver.evaluate_policy(weitzman_policy)
            times['WEITZMAN'].append(time.time() - t0)
            values['WEITZMAN'].append(weitz_val)

        # Compute normalized performance
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

            v_arr = np.array(values[policy])
            if N <= DP_CUTOFF:
                opt_arr = np.array(values['OPT'])
                normalized = np.where(opt_arr > 0, v_arr / opt_arr, 1.0)
            else:
                best_arr = np.maximum.reduce([
                    np.array(values[p]) for p in POLICY_NAMES if p != 'OPT' and values[p]
                ])
                normalized = np.where(best_arr > 0, v_arr / best_arr, 1.0)

            perf_row[f'{policy}_mean'] = float(np.mean(normalized))
            perf_row[f'{policy}_std'] = float(np.std(normalized))
            perf_row[f'{policy}_worst'] = float(np.min(normalized))

            t_arr = np.array(times[policy])
            time_row[f'{policy}_mean'] = float(np.mean(t_arr))
            time_row[f'{policy}_std'] = float(np.std(t_arr))

            # Exact optimality (small N only)
            if N <= DP_CUTOFF and policy != 'OPT':
                opt_arr = np.array(values['OPT'])
                exact_count = np.sum(np.abs(v_arr - opt_arr) < 1e-6)
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
    exact_df.to_csv(os.path.join(OUTPUT_DIR, 'table_4_exact_optimality.csv'), index=False)

    from experiments.formatting import (
        format_table_3, format_table_4, format_table_EC1, save_latex,
    )
    save_latex(os.path.join(OUTPUT_DIR, 'table_3_performance.tex'),
               format_table_3(perf_df, dp_cutoff=DP_CUTOFF))
    save_latex(os.path.join(OUTPUT_DIR, 'table_4_exact_optimality.tex'),
               format_table_4(exact_df))
    save_latex(os.path.join(OUTPUT_DIR, 'table_EC1_runtime.tex'),
               format_table_EC1(time_df, dp_cutoff=DP_CUTOFF))

    print("\nPerformance results:")
    print(perf_df.to_string(index=False))
    print("\nExact optimality:")
    print(exact_df.to_string(index=False))

    return perf_df, time_df, exact_df
