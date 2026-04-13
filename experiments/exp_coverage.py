"""Experiment 1: Coverage of analytical sufficient conditions (Table 1).

For each small instance (N=2..9, 1000 instances):
  - Solve optimal policy via naive DP
  - For each system state, check which theorem identifies the optimal action
  - Report coverage (fraction of states where some theorem applies) and
    recall for F-opening (Theorem 2) and P-opening (Theorem 3)

Outputs:
  - table_1_coverage.csv / .tex
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from pandora.solver import PandoraSolver, COND_STOP, COND_F_OPEN, COND_P_OPEN
from pandora.instance_generator import generate_prototypical_boxes, sample_instance
from experiments.config import (
    SMALL_N_RANGE, SMALL_INSTANCES, SEED, NUM_PROTOTYPICAL_BOXES,
    BOX_DISTANCE, DP_CUTOFF, OUTPUT_DIR,
)


def run_coverage_experiment(n_range=None, n_instances=None, seed=None,
                            selected_boxes=None):
    """Run Experiment 1: theorem coverage analysis.

    Returns a DataFrame with columns:
      N, coverage_stop, coverage_f, coverage_p, coverage_total,
      recall_f, recall_p
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

    results = []

    for N in n_range:
        print(f"\n--- N = {N} ---")
        cov_stop_list = []
        cov_f_list = []
        cov_p_list = []
        cov_total_list = []
        recall_f_list = []
        recall_p_list = []

        for rep in tqdm(range(n_instances), desc=f"N={N}"):
            box_list, _ = sample_instance(selected_boxes, N, rng_np)
            solver = PandoraSolver(box_list)
            solver.solve_dp()
            stats = solver.get_dp_stats()
            dp_stats = stats['dp_stats']
            action_dict = solver.dp_action_dict

            n_states = len(dp_stats)
            if n_states == 0:
                continue

            n_stop = 0
            n_f_cond = 0
            n_p_cond = 0
            n_f_optimal = 0
            n_p_optimal = 0

            for state_tuple, sinfo in dp_stats.items():
                conds = sinfo['conditions']
                action = action_dict.get(state_tuple, "STOP")

                if conds[COND_STOP]:
                    n_stop += 1
                if conds[COND_F_OPEN]:
                    n_f_cond += 1
                if conds[COND_P_OPEN]:
                    n_p_cond += 1

                if action.startswith("F"):
                    n_f_optimal += 1
                elif action.startswith("P"):
                    n_p_optimal += 1

            cov_total = (n_stop + n_f_cond + n_p_cond) / n_states
            cov_stop = n_stop / n_states
            cov_f = n_f_cond / n_states
            cov_p = n_p_cond / n_states
            recall_f = n_f_cond / n_f_optimal if n_f_optimal > 0 else 1.0
            recall_p = n_p_cond / n_p_optimal if n_p_optimal > 0 else 1.0

            cov_stop_list.append(cov_stop)
            cov_f_list.append(cov_f)
            cov_p_list.append(cov_p)
            cov_total_list.append(cov_total)
            recall_f_list.append(recall_f)
            recall_p_list.append(recall_p)

        results.append({
            'N': N,
            'coverage_stop': np.mean(cov_stop_list),
            'coverage_f': np.mean(cov_f_list),
            'coverage_p': np.mean(cov_p_list),
            'coverage_total': np.mean(cov_total_list),
            'recall_f': np.mean(recall_f_list),
            'recall_p': np.mean(recall_p_list),
        })

    df = pd.DataFrame(results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_1_coverage.csv'), index=False)

    from experiments.formatting import format_table_1, save_latex
    save_latex(os.path.join(OUTPUT_DIR, 'table_1_coverage.tex'),
               format_table_1(df))

    print("\nCoverage results:")
    print(df.to_string(index=False))
    return df
