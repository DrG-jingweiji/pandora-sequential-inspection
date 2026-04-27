"""CLI entry point for running all or selected experiments.

Generates prototypical boxes once and passes them to all experiments.
Produces all tables (CSV + LaTeX) and figures (PNG) referenced in the paper.

Usage:
    python -m experiments.run_all                        # Run all experiments
    python -m experiments.run_all --experiment coverage   # Run specific experiment
    python -m experiments.run_all --small                 # Quick run, fewer instances
    python -m experiments.run_all --workers 6             # Use 6 parallel processes
    python -m experiments.run_all --fresh                 # Discard checkpoints, start over
    python -m experiments.run_all --pool-source generated # Use the newer generator
    python -m experiments.run_all --table4-pool-source benchmark
"""

import argparse
import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pandora.instance_generator import (
    bundled_legacy_pool_dir,
    generate_legacy_style_prototypical_boxes,
    generate_prototypical_boxes,
    load_legacy_box_pool,
)
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


def _load_old_boxes(pool_dir, filter_cF_gt_cP=True):
    """Load the bundled old selected box pool."""
    if pool_dir is None:
        pool_dir = bundled_legacy_pool_dir()
    print(f"Loading bundled old box pool from {pool_dir}...")
    selected = load_legacy_box_pool(
        pool_dir=pool_dir,
        filter_cF_gt_cP=filter_cF_gt_cP,
    )
    all_boxes = load_legacy_box_pool(pool_dir=pool_dir, filter_cF_gt_cP=False)
    print(f"  {len(selected)} selected from {len(all_boxes)} bundled old boxes")
    return selected, all_boxes


def _generate_legacy_style_boxes(seed, pool_source, n_boxes, max_attempts):
    """Generate the recovered old-code prototype pool without reading old data."""
    require_p_dominant = pool_source == 'legacy-generated'
    label = (
        "legacy-style P-dominant"
        if require_p_dominant else
        "legacy-style mixed"
    )
    print(f"Generating {label} prototypical boxes from code...")
    selected_unfiltered, all_candidates = generate_legacy_style_prototypical_boxes(
        n_boxes,
        BOX_DISTANCE,
        rng=random.Random(seed),
        max_attempts=max_attempts,
        return_all=True,
        filter_cF_gt_cP=False,
        require_p_dominant=require_p_dominant,
    )
    selected = [box for box in selected_unfiltered if box.c_F > box.c_P]
    print(
        f"  {len(selected)} selected after c_F > c_P filter "
        f"from {len(selected_unfiltered)} prototypes "
        f"({len(all_candidates)} valid candidates)"
    )
    return selected, selected_unfiltered, all_candidates


def _parse_n_range(spec):
    """Parse an inclusive range like ``2:5`` or a comma list."""
    if not spec:
        return None
    if ':' in spec:
        start, end = [int(x.strip()) for x in spec.split(':', 1)]
        if end < start:
            raise ValueError('--table4-n-range end must be >= start')
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(',') if x.strip()]


def _policy_benchmark_n_range(small):
    if small:
        return list(range(2, 5)) + list(range(10, 12))
    return list(SMALL_N_RANGE) + list(range(10, 15))


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
        dp_comp_range = [N for N in dp_range if N <= 7] if not small else dp_range
        counts['dp_comparison'] = len(dp_comp_range) * n_small_inst
    if experiment in ('all', 'policy_benchmark'):
        bench_range = _policy_benchmark_n_range(small)
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
    parser.add_argument('--pool-source', type=str,
                        default='legacy-generated-mixed',
                        choices=['legacy-generated-mixed',
                                 'legacy-generated',
                                 'generated', 'random', 'old'],
                        help=('Box pool to use. Default '
                              '"legacy-generated-mixed" generates the '
                              'recovered old-code prototype pool from code '
                              'and uses the old notebook sampling stream. '
                              '"generated" and "random" use the newer '
                              'deterministic generator; "old" loads the '
                              'optional bundled old pool.'))
    parser.add_argument('--prototype-boxes', type=int,
                        default=NUM_PROTOTYPICAL_BOXES,
                        help='Number of generated prototype boxes for '
                             'legacy-generated pool modes')
    parser.add_argument('--legacy-max-attempts', type=int, default=50000,
                        help='Maximum random draws for legacy-generated pools')
    parser.add_argument('--legacy-pool-dir', type=str, default=None,
                        help='Directory containing bundled old-pool JSON data '
                             '(default: data/legacy_box_pools)')
    parser.add_argument('--table4-pool-source', type=str, default='benchmark',
                        choices=['old', 'benchmark'],
                        help=('Pool for Table 4. Default "benchmark" uses '
                              'the same generated pool as Tables 3/EC.1; '
                              '"old" runs Table 4 from the optional bundled '
                              'old pool.'))
    parser.add_argument('--table4-n-range', type=str, default=None,
                        help='Override Table 4 N range, e.g. 2:5 or 2,3,4')
    parser.add_argument('--table4-reps', type=int, default=None,
                        help='Override Table 4 instances per N')
    args = parser.parse_args()

    pool_source = 'generated' if args.pool_source == 'random' else args.pool_source
    use_old_pool = pool_source == 'old'
    use_legacy_sampling = pool_source in (
        'old', 'legacy-generated', 'legacy-generated-mixed',
    )

    n_workers = args.workers
    n_small = 10 if args.small else None
    n_large = 5 if args.small else None
    n_range_small = range(2, 5) if args.small else None
    n_range_large = range(10, 12) if args.small else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.fresh:
        clear_all_checkpoints(OUTPUT_DIR)

    print(f"Using {n_workers} parallel worker(s)")

    # Main pool for Tables 1-4, EC.1 and Figure 3.
    if use_old_pool:
        print("Using bundled old box pool with legacy RandomState sampling")
        selected_boxes, all_candidates = _load_old_boxes(
            args.legacy_pool_dir,
            filter_cF_gt_cP=True,
        )
        p_opening_boxes = load_legacy_box_pool(
            pool_dir=args.legacy_pool_dir,
            filter_cF_gt_cP=False,
        )
        p_opening_candidates = all_candidates
    elif pool_source in ('legacy-generated', 'legacy-generated-mixed'):
        selected_boxes, p_opening_boxes, p_opening_candidates = (
            _generate_legacy_style_boxes(
                SEED,
                pool_source,
                args.prototype_boxes,
                args.legacy_max_attempts,
            )
        )
        all_candidates = p_opening_candidates
    else:
        selected_boxes, all_candidates = _generate_boxes(SEED, small=args.small)

        # Unfiltered pool for P-opening experiments (Figures EC.8-EC.10)
        p_opening_boxes, p_opening_candidates = _generate_boxes(
            SEED, small=args.small, require_p_dominant=False,
        )

    # ── Overall progress ─────────────────────────────────────────────
    task_counts = _estimate_task_counts(args.experiment, args.small)
    if args.experiment in ('all', 'policy_benchmark') and args.table4_pool_source == 'old':
        table4_n_range = _parse_n_range(args.table4_n_range)
        if table4_n_range is None:
            table4_n_range = [
                N for N in _policy_benchmark_n_range(args.small)
                if N <= DP_CUTOFF
            ]
        table4_reps = args.table4_reps
        if table4_reps is None:
            table4_reps = n_small if args.small else SMALL_INSTANCES
        task_counts['table4_old'] = len(table4_n_range) * table4_reps
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
                                    n_workers=n_workers,
                                    legacy_sampling=use_legacy_sampling)

        # ---- Experiment 2: DP comparison (Table 2) ----
        if args.experiment in ('all', 'dp_comparison'):
            print("\n" + "=" * 60)
            print("Experiment 2: Naive vs Structured DP (Table 2)")
            print("=" * 60)
            from experiments.exp_dp_comparison import run_dp_comparison
            dp_n_range = n_range_small if args.small else range(2, 8)
            run_dp_comparison(n_range=dp_n_range, n_instances=n_small,
                              selected_boxes=selected_boxes,
                              n_workers=n_workers,
                              legacy_sampling=use_legacy_sampling)

        # ---- Experiment 3: Policy benchmark (Tables 3, 4, EC.1) ----
        if args.experiment in ('all', 'policy_benchmark'):
            print("\n" + "=" * 60)
            print("Experiment 3: Policy benchmark (Tables 3, 4, EC.1)")
            print("=" * 60)
            from experiments.exp_policy_benchmark import run_policy_benchmark
            n_range = _policy_benchmark_n_range(args.small)
            table4_from_benchmark = args.table4_pool_source == 'benchmark'
            run_policy_benchmark(n_range=n_range, n_instances_small=n_small,
                                 n_instances_large=n_large,
                                 selected_boxes=selected_boxes,
                                 n_workers=n_workers,
                                 legacy_sampling=use_legacy_sampling,
                                 write_exact_table=table4_from_benchmark)

            if args.table4_pool_source == 'old':
                print("\n" + "=" * 60)
                print("Table 4: old bundled box pool")
                print("=" * 60)
                from experiments.replicate_table4 import run_table4_replication

                table4_n_range = _parse_n_range(args.table4_n_range)
                if table4_n_range is None:
                    table4_n_range = [N for N in n_range if N <= DP_CUTOFF]
                table4_reps = args.table4_reps
                if table4_reps is None:
                    table4_reps = n_small if args.small else SMALL_INSTANCES

                run_table4_replication(
                    pool_source='old',
                    legacy_pool_dir=args.legacy_pool_dir,
                    n_range=table4_n_range,
                    reps=table4_reps,
                    policies=['INDEX', 'WHITTLE', 'COM'],
                    seed=SEED,
                    output_dir=OUTPUT_DIR,
                    n_workers=n_workers,
                    fresh=args.fresh,
                    output_prefix='table_4_exact_optimality',
                    save_raw=False,
                )

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
                                   n_workers=n_workers,
                                   legacy_sampling=use_legacy_sampling)
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
