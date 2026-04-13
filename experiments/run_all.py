"""CLI entry point for running all or selected experiments.

Generates prototypical boxes once and passes them to all experiments.
Produces all tables (CSV + LaTeX) and figures (PNG) referenced in the paper.

Usage:
    python -m experiments.run_all                      # Run all experiments
    python -m experiments.run_all --experiment coverage # Run specific experiment
    python -m experiments.run_all --small               # Quick run with fewer instances
"""

import argparse
import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pandora.instance_generator import generate_prototypical_boxes
from experiments.config import (
    SEED, NUM_PROTOTYPICAL_BOXES, BOX_DISTANCE, OUTPUT_DIR,
)


def _generate_boxes(seed, small=False):
    """Generate prototypical boxes with history for scatter plots."""
    random.seed(seed)
    np.random.seed(seed)

    n_boxes = 10 if small else NUM_PROTOTYPICAL_BOXES
    print("Generating prototypical boxes (with candidate history)...")
    selected, all_candidates = generate_prototypical_boxes(
        n_boxes, BOX_DISTANCE, return_all=True,
    )
    print(f"  {len(selected)} selected from {len(all_candidates)} candidates")
    return selected, all_candidates


def main():
    parser = argparse.ArgumentParser(description='Run PSI numerical experiments')
    parser.add_argument('--experiment', '-e', type=str, default='all',
                        choices=['all', 'coverage', 'dp_comparison',
                                 'policy_benchmark', 'p_opening',
                                 'box_scatter'],
                        help='Which experiment to run')
    parser.add_argument('--small', action='store_true',
                        help='Run with reduced instances for quick testing')
    args = parser.parse_args()

    n_small = 10 if args.small else None
    n_large = 5 if args.small else None
    n_range_small = range(2, 5) if args.small else None
    n_range_large = range(10, 12) if args.small else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected_boxes, all_candidates = _generate_boxes(SEED, small=args.small)

    # ---- Figures 3 and EC.8: Prototypical box scatter ----
    if args.experiment in ('all', 'box_scatter'):
        print("\n" + "=" * 60)
        print("Figures 3 & EC.8: Prototypical box scatter plots")
        print("=" * 60)
        from experiments.figures import plot_figure_3, plot_figure_EC8
        plot_figure_3(selected_boxes)
        plot_figure_EC8(selected_boxes, all_candidates)

    # ---- Experiment 1: Coverage (Table 1) ----
    if args.experiment in ('all', 'coverage'):
        print("\n" + "=" * 60)
        print("Experiment 1: Coverage of analytical conditions (Table 1)")
        print("=" * 60)
        from experiments.exp_coverage import run_coverage_experiment
        run_coverage_experiment(n_range=n_range_small, n_instances=n_small,
                                selected_boxes=selected_boxes)

    # ---- Experiment 2: DP comparison (Table 2) ----
    if args.experiment in ('all', 'dp_comparison'):
        print("\n" + "=" * 60)
        print("Experiment 2: Naive vs Structured DP (Table 2)")
        print("=" * 60)
        from experiments.exp_dp_comparison import run_dp_comparison
        run_dp_comparison(n_range=n_range_small, n_instances=n_small,
                          selected_boxes=selected_boxes)

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
                             selected_boxes=selected_boxes)

    # ---- Experiment 4: P-opening analysis (Figures EC.9, EC.10) ----
    if args.experiment in ('all', 'p_opening'):
        print("\n" + "=" * 60)
        print("Experiment 4: P-opening analysis (Figures EC.9, EC.10)")
        print("=" * 60)
        from experiments.exp_p_opening import (
            run_p_opening_analysis, run_more_boxes_experiment,
        )
        run_p_opening_analysis(n_range=n_range_small, n_instances=n_small,
                               selected_boxes=selected_boxes)
        n_mb_range = range(2, 5) if args.small else None
        n_mb_specs = 3 if args.small else 10
        run_more_boxes_experiment(selected_boxes=selected_boxes,
                                  n_box_specs=n_mb_specs,
                                  n_range=n_mb_range)

    print("\nDone! Results saved to output/")


if __name__ == '__main__':
    main()
