"""Experiment 2: Naive DP vs Structured DP comparison (Table 2).

For each small instance (N=2..9, 1000 instances):
  - Run naive DP and structured DP
  - Compare: states created, states revisited, runtime, runtime ratio

Outputs:
  - table_2_dp_comparison.csv / .tex
"""

import os
import time
import numpy as np
import pandas as pd

from pandora.solver import PandoraSolver
from pandora.instance_generator import generate_prototypical_boxes
from experiments.config import (
    SMALL_N_RANGE, SMALL_INSTANCES, SEED, NUM_PROTOTYPICAL_BOXES,
    BOX_DISTANCE, OUTPUT_DIR, DEFAULT_WORKERS,
)
from experiments.parallel import (
    generate_instance_tasks, run_parallel, checkpoint_path_for, get_shared,
)


def _dp_comparison_worker(N, rep_idx, indices):
    """Run naive and structured DP on one instance, return timing/stats."""
    selected_boxes = get_shared('selected_boxes')
    box_list = [selected_boxes[i] for i in indices]

    solver = PandoraSolver(box_list)
    t0 = time.time()
    solver.solve_dp()
    t_naive = time.time() - t0
    stats_naive = solver.get_dp_stats()

    solver2 = PandoraSolver(box_list)
    t0 = time.time()
    solver2.solve_dp_structured()
    t_plus = time.time() - t0
    stats_plus = solver2.get_dp_plus_stats()

    return {
        'states_naive': int(stats_naive['num_states_created']),
        'revisits_naive': int(stats_naive['num_revisits']),
        'time_naive': t_naive,
        'states_plus': int(stats_plus['num_states_created']),
        'revisits_plus': int(stats_plus['num_revisits']),
        'time_plus': t_plus,
    }


def run_dp_comparison(n_range=None, n_instances=None, seed=None,
                      selected_boxes=None, n_workers=None):
    """Run Experiment 2: naive vs structured DP comparison.

    Returns a DataFrame with columns:
      N, states_naive, revisits_naive, time_naive,
      states_plus, revisits_plus, time_plus, runtime_ratio
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

    ckpt = checkpoint_path_for(OUTPUT_DIR, 'dp_comparison')
    all_results = run_parallel(
        _dp_comparison_worker, tasks,
        shared_data={'selected_boxes': selected_boxes},
        n_workers=n_workers,
        checkpoint_path=ckpt,
        desc="DP comparison",
    )

    metric_keys = ['states_naive', 'revisits_naive', 'time_naive',
                   'states_plus', 'revisits_plus', 'time_plus']
    rows = []
    for N in n_range:
        n_results = [
            all_results[f"{N}_{rep}"]
            for rep in range(n_instances)
            if f"{N}_{rep}" in all_results
        ]
        if not n_results:
            continue

        row = {'N': N}
        for k in metric_keys:
            row[k] = float(np.mean([r[k] for r in n_results]))

        row['runtime_ratio'] = (row['time_plus'] / row['time_naive']
                                if row['time_naive'] > 0 else float('inf'))
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_2_dp_comparison.csv'), index=False)

    from experiments.formatting import format_table_2, save_latex
    save_latex(os.path.join(OUTPUT_DIR, 'table_2_dp_comparison.tex'),
               format_table_2(df))

    print("\nDP comparison results:")
    print(df.to_string(index=False))
    return df
