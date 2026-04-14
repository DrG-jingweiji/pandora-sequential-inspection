"""CLI entry point for running all or selected experiments.

Generates prototypical boxes once and passes them to all experiments.
Produces all tables (CSV + LaTeX) and figures (PNG) referenced in the paper.

Usage:
    python -m experiments.run_all                        # Run all experiments
    python -m experiments.run_all --experiment coverage   # Run specific experiment
    python -m experiments.run_all --small                 # Quick run, fewer instances
    python -m experiments.run_all --workers 6             # Use 6 parallel processes
    python -m experiments.run_all --fresh                 # Discard checkpoints, start over
"""

import argparse
import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pandora.instance_generator import generate_prototypical_boxes
from experiments.config import (
    SEED, NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE, OUTPUT_DIR, DEFAULT_WORKERS,
    SMALL_N_RANGE, LARGE_N_RANGE, SMALL_INSTANCES, LARGE_INSTANCES, DP_CUTOFF,
)
from experiments.parallel import (
    clear_all_checkpoints, OverallProgress, set_overall_progress,
)
from experiments.exp_p_opening import _CP_VALUES


def _generate_boxes(seed, small=False, require_p_dominant=True):
    """Generate prototypical boxes with history for scatter plots."""
    random.seed(seed)
    np.random.seed(seed)

    n_boxes = 10 if small else NUM_PROTOTYPICAL_BOXES
    label = "P-dominant" if require_p_dominant else "mixed (unfiltered)"
    print(f"Generating {label} prototypical boxes (with candidate history)...")
    selected, all_candidates = generate_prototypical_boxes(
        n_boxes, BOX_DISTANCE, return_all=True,
        require_p_dominant=require_p_dominant,
    )
    print(f"  {len(selected)} selected from {len(all_candidates)} candidates")
    return selected, all_candidates


def _estimate_task_counts(experiment, small):
    """Pre-compute the number of parallel tasks for each experiment."""
    counts = {}

    if small:
        s_range, l_range = list(range(2, 5)), list(range(10, 12))
        n_small_inst, n_large_inst = 10, 5
        mb_range = list(range(1, 4))
    else:
        s_range = list(SMALL_N_RANGE)
        l_range = list(LARGE_N_RANGE)
        n_small_inst, n_large_inst = SMALL_INSTANCES, LARGE_INSTANCES
        mb_range = list(range(1, 8))

    dp_range = [N for N in s_range if N <= DP_CUTOFF]

    if experiment in ('all', 'coverage'):
        counts['coverage'] = len(dp_range) * n_small_inst
    if experiment in ('all', 'dp_comparison'):
        counts['dp_comparison'] = len(dp_range) * n_small_inst
    if experiment in ('all', 'policy_benchmark'):
        bench_range = s_range + l_range if not small else list(range(2, 5)) + list(range(10, 12))
        counts['policy_benchmark'] = sum(
            n_small_inst if N <= DP_CUTOFF else n_large_inst
            for N in bench_range
        )
    if experiment in ('all', 'p_opening'):
        counts['p_opening'] = len(dp_range) * n_small_inst
        counts['more_boxes'] = len(_CP_VALUES) * len(mb_range)

    return counts


def main():
    parser = argparse.ArgumentParser(description='Run PSI numerical experiments')
    parser.add_argument('--experiment', '-e', type=str, default='all',
                        choices=['all', 'coverage', 'dp_comparison',
                                 'policy_benchmark', 'p_opening',
                                 'box_scatter'],
                        help='Which experiment to run')
    parser.add_argument('--small', action='store_true',
                        help='Run with reduced instances for quick testing')
    parser.add_argument('--workers', '-w', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of parallel worker processes '
                             f'(default: {DEFAULT_WORKERS})')
    parser.add_argument('--fresh', action='store_true',
                        help='Clear all checkpoints and start from scratch')
    args = parser.parse_args()

    n_workers = args.workers
    n_small = 10 if args.small else None
    n_large = 5 if args.small else None
    n_range_small = range(2, 5) if args.small else None
    n_range_large = range(10, 12) if args.small else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.fresh:
        clear_all_checkpoints(OUTPUT_DIR)

    print(f"Using {n_workers} parallel worker(s)")

    # Main pool (σ^P > σ^F) for Tables 1-4, EC.1 and Figure 3
    selected_boxes, all_candidates = _generate_boxes(SEED, small=args.small)

    # Unfiltered pool for P-opening experiments (Figures EC.8-EC.10)
    p_opening_boxes, p_opening_candidates = _generate_boxes(
        SEED, small=args.small, require_p_dominant=False,
    )

    # ── Overall progress ─────────────────────────────────────────────
    task_counts = _estimate_task_counts(args.experiment, args.small)
    total_tasks = sum(task_counts.values())

    if total_tasks > 0:
        tracker = OverallProgress(total_tasks)
        set_overall_progress(tracker)
    else:
        tracker = None

    try:
        # ---- Figures 3 and EC.8: Prototypical box scatter ----
        if args.experiment in ('all', 'box_scatter'):
            print("\n" + "=" * 60)
            print("Figures 3 & EC.8: Prototypical box scatter plots")
            print("=" * 60)
            from experiments.figures import plot_figure_3, plot_figure_EC8
            plot_figure_3(selected_boxes)
            plot_figure_EC8(p_opening_boxes, p_opening_candidates)

        # ---- Experiment 1: Coverage (Table 1) ----
        if args.experiment in ('all', 'coverage'):
            print("\n" + "=" * 60)
            print("Experiment 1: Coverage of analytical conditions (Table 1)")
            print("=" * 60)
            from experiments.exp_coverage import run_coverage_experiment
            run_coverage_experiment(n_range=n_range_small, n_instances=n_small,
                                    selected_boxes=selected_boxes,
                                    n_workers=n_workers)

        # ---- Experiment 2: DP comparison (Table 2) ----
        if args.experiment in ('all', 'dp_comparison'):
            print("\n" + "=" * 60)
            print("Experiment 2: Naive vs Structured DP (Table 2)")
            print("=" * 60)
            from experiments.exp_dp_comparison import run_dp_comparison
            run_dp_comparison(n_range=n_range_small, n_instances=n_small,
                              selected_boxes=selected_boxes,
                              n_workers=n_workers)

        # ---- Experiment 3: Policy benchmark (Tables 3, 4, EC.1) ----
        if args.experiment in ('all', 'policy_benchmark'):
            print("\n" + "=" * 60)
            print("Experiment 3: Policy benchmark (Tables 3, 4, EC.1)")
            print("=" * 60)
            from experiments.exp_policy_benchmark import run_policy_benchmark
            n_range = None
            if args.small:
                n_range = list(range(2, 5)) + list(range(10, 12))
            run_policy_benchmark(n_range=n_range, n_instances_small=n_small,
                                 n_instances_large=n_large,
                                 selected_boxes=selected_boxes,
                                 n_workers=n_workers)

        # ---- Experiment 4: P-opening analysis (Figures EC.9, EC.10) ----
        if args.experiment in ('all', 'p_opening'):
            print("\n" + "=" * 60)
            print("Experiment 4: P-opening analysis (Figures EC.9, EC.10)")
            print("=" * 60)
            from experiments.exp_p_opening import (
                run_p_opening_analysis, run_more_boxes_experiment,
            )
            run_p_opening_analysis(n_range=n_range_small, n_instances=n_small,
                                   selected_boxes=p_opening_boxes,
                                   n_workers=n_workers)
            n_mb_range = range(1, 4) if args.small else None
            run_more_boxes_experiment(n_range=n_mb_range,
                                      n_workers=n_workers)

    finally:
        if tracker:
            set_overall_progress(None)
            tracker.close()

    print("Results saved to output/")


if __name__ == '__main__':
    main()
