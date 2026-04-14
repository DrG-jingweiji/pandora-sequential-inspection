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
import numpy as np
import pandas as pd

from pandora.solver import PandoraSolver, COND_STOP, COND_F_OPEN, COND_P_OPEN
from pandora.instance_generator import generate_prototypical_boxes
from experiments.config import (
    SMALL_N_RANGE, SMALL_INSTANCES, SEED, NUM_PROTOTYPICAL_BOXES,
    BOX_DISTANCE, OUTPUT_DIR, DEFAULT_WORKERS,
)
from experiments.parallel import (
    generate_instance_tasks, run_parallel, checkpoint_path_for, get_shared,
)


def _coverage_worker(N, rep_idx, indices):
    """Solve one instance and compute theorem coverage metrics."""
    selected_boxes = get_shared('selected_boxes')
    box_list = [selected_boxes[i] for i in indices]

    solver = PandoraSolver(box_list)
    solver.solve_dp()
    stats = solver.get_dp_stats()
    dp_stats = stats['dp_stats']
    action_dict = solver.dp_action_dict

    n_states = len(dp_stats)
    if n_states == 0:
        return {
            'coverage_stop': 0.0, 'coverage_f': 0.0, 'coverage_p': 0.0,
            'coverage_total': 0.0, 'recall_f': 1.0, 'recall_p': 1.0,
        }

    n_stop = n_f_cond = n_p_cond = n_f_optimal = n_p_optimal = 0

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

    return {
        'coverage_stop': n_stop / n_states,
        'coverage_f': n_f_cond / n_states,
        'coverage_p': n_p_cond / n_states,
        'coverage_total': (n_stop + n_f_cond + n_p_cond) / n_states,
        'recall_f': n_f_cond / n_f_optimal if n_f_optimal > 0 else 1.0,
        'recall_p': n_p_cond / n_p_optimal if n_p_optimal > 0 else 1.0,
    }


def run_coverage_experiment(n_range=None, n_instances=None, seed=None,
                            selected_boxes=None, n_workers=None):
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
    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    if selected_boxes is None:
        print("Generating prototypical boxes...")
        selected_boxes = generate_prototypical_boxes(
            NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE
        )

    tasks = generate_instance_tasks(
        n_range, lambda N: n_instances, len(selected_boxes), seed,
    )

    ckpt = checkpoint_path_for(OUTPUT_DIR, 'coverage')
    all_results = run_parallel(
        _coverage_worker, tasks,
        shared_data={'selected_boxes': selected_boxes},
        n_workers=n_workers,
        checkpoint_path=ckpt,
        desc="Coverage",
    )

    metrics = list(next(iter(all_results.values())).keys())
    rows = []
    for N in n_range:
        n_results = [
            all_results[f"{N}_{rep}"]
            for rep in range(n_instances)
            if f"{N}_{rep}" in all_results
        ]
        if not n_results:
            continue
        rows.append({
            'N': N,
            **{k: float(np.mean([r[k] for r in n_results])) for k in metrics},
        })

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_1_coverage.csv'), index=False)

    from experiments.formatting import format_table_1, save_latex
    save_latex(os.path.join(OUTPUT_DIR, 'table_1_coverage.tex'),
               format_table_1(df))

    print("\nCoverage results:")
    print(df.to_string(index=False))
    return df
