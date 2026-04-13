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
from tqdm import tqdm

from pandora.solver import PandoraSolver
from pandora.instance_generator import generate_prototypical_boxes, sample_instance
from experiments.config import (
    SMALL_N_RANGE, SMALL_INSTANCES, SEED, NUM_PROTOTYPICAL_BOXES,
    BOX_DISTANCE, OUTPUT_DIR,
)


def run_dp_comparison(n_range=None, n_instances=None, seed=None,
                      selected_boxes=None):
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
        metrics = {
            'states_naive': [], 'revisits_naive': [], 'time_naive': [],
            'states_plus': [], 'revisits_plus': [], 'time_plus': [],
        }

        for rep in tqdm(range(n_instances), desc=f"N={N}"):
            box_list, _ = sample_instance(selected_boxes, N, rng_np)

            # Naive DP
            solver = PandoraSolver(box_list)
            t0 = time.time()
            solver.solve_dp()
            t_naive = time.time() - t0
            stats_naive = solver.get_dp_stats()

            # Structured DP
            solver2 = PandoraSolver(box_list)
            t0 = time.time()
            solver2.solve_dp_structured()
            t_plus = time.time() - t0
            stats_plus = solver2.get_dp_plus_stats()

            metrics['states_naive'].append(stats_naive['num_states_created'])
            metrics['revisits_naive'].append(stats_naive['num_revisits'])
            metrics['time_naive'].append(t_naive)
            metrics['states_plus'].append(stats_plus['num_states_created'])
            metrics['revisits_plus'].append(stats_plus['num_revisits'])
            metrics['time_plus'].append(t_plus)

        ratio = (np.mean(metrics['time_plus']) / np.mean(metrics['time_naive'])
                 if np.mean(metrics['time_naive']) > 0 else float('inf'))

        results.append({
            'N': N,
            'states_naive': np.mean(metrics['states_naive']),
            'revisits_naive': np.mean(metrics['revisits_naive']),
            'time_naive': np.mean(metrics['time_naive']),
            'states_plus': np.mean(metrics['states_plus']),
            'revisits_plus': np.mean(metrics['revisits_plus']),
            'time_plus': np.mean(metrics['time_plus']),
            'runtime_ratio': ratio,
        })

    df = pd.DataFrame(results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_2_dp_comparison.csv'), index=False)

    from experiments.formatting import format_table_2, save_latex
    save_latex(os.path.join(OUTPUT_DIR, 'table_2_dp_comparison.tex'),
               format_table_2(df))

    print("\nDP comparison results:")
    print(df.to_string(index=False))
    return df
