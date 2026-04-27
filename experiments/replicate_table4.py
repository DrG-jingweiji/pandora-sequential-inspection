"""Replicate Table 4 exact-optimality rates.

This focused driver evaluates the policies in Table 4 against the exact DP
value.  It can use either the new/generated prototypical box pool or the stored
old pool bundled under ``data/legacy_box_pools``.  With ``--pool-source old``
it also uses the old notebook's ``np.random.RandomState(seed).randint``
sampling stream.

Examples
--------
    python -m experiments.replicate_table4 --pool-source old --n-range 2:5
    python -m experiments.replicate_table4 --pool-source legacy-generated
    python -m experiments.replicate_table4 --pool-source legacy-generated-mixed
    python -m experiments.replicate_table4 --pool-source generated --reps 200
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pandora.instance_generator import (
    bundled_legacy_pool_dir,
    generate_legacy_style_prototypical_boxes,
    generate_prototypical_boxes,
    load_legacy_box_pool,
)
from pandora.policies import (
    best_committing_policy,
    index_policy,
    stp_policy,
    weitzman_policy,
    whittle_policy,
)
from pandora.solver import PandoraSolver
from experiments.config import (
    BOX_DISTANCE,
    DP_CUTOFF,
    NUM_PROTOTYPICAL_BOXES,
    OUTPUT_DIR,
    SEED,
    SMALL_INSTANCES,
)
from experiments.formatting import (
    format_table_4,
    save_latex,
    table4_paper_columns,
)
from experiments.parallel import (
    checkpoint_path_for,
    clear_checkpoint,
    generate_instance_tasks,
    get_shared,
    run_parallel,
)


POLICY_FUNCS = {
    'INDEX': index_policy,
    'WHITTLE': whittle_policy,
    'STP': stp_policy,
    'WEITZMAN': weitzman_policy,
}

# Values transcribed from Table 4 in the paper.
PAPER_TABLE_4 = {
    2: {'INDEX': 0.543, 'WHITTLE': 0.238, 'COM': 0.835},
    3: {'INDEX': 0.614, 'WHITTLE': 0.065, 'COM': 0.850},
    4: {'INDEX': 0.622, 'WHITTLE': 0.023, 'COM': 0.853},
    5: {'INDEX': 0.661, 'WHITTLE': 0.002, 'COM': 0.892},
    6: {'INDEX': 0.686, 'WHITTLE': 0.001, 'COM': 0.904},
    7: {'INDEX': 0.701, 'WHITTLE': 0.000, 'COM': 0.927},
    8: {'INDEX': 0.715, 'WHITTLE': 0.001, 'COM': 0.944},
    9: {'INDEX': 0.733, 'WHITTLE': 0.000, 'COM': 0.943},
}


def _parse_n_range(text):
    if ':' in text:
        start, end = [int(x.strip()) for x in text.split(':', 1)]
        if end < start:
            raise ValueError('--n-range end must be >= start')
        return list(range(start, end + 1))
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _parse_policies(text):
    raw = [p.strip().upper() for p in text.split(',') if p.strip()]
    if raw == ['ALL']:
        raw = ['INDEX', 'WHITTLE', 'COM', 'STP', 'WEITZMAN']
    valid = set(POLICY_FUNCS) | {'COM'}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f'Unknown policies: {", ".join(unknown)}')
    return raw


def _load_box_pool(pool_source, seed, legacy_pool_dir,
                   n_prototype_boxes=NUM_PROTOTYPICAL_BOXES,
                   legacy_max_attempts=50000):
    if pool_source == 'old':
        boxes = load_legacy_box_pool(
            pool_dir=legacy_pool_dir,
            filter_cF_gt_cP=True,
        )
        return boxes, True, {}

    if pool_source == 'legacy-generated':
        boxes, candidates = generate_legacy_style_prototypical_boxes(
            n_prototype_boxes,
            BOX_DISTANCE,
            rng=random.Random(seed),
            max_attempts=legacy_max_attempts,
            return_all=True,
            filter_cF_gt_cP=True,
            require_p_dominant=True,
        )
        return boxes, True, {'valid_candidates_generated': len(candidates)}

    if pool_source == 'legacy-generated-mixed':
        boxes, candidates = generate_legacy_style_prototypical_boxes(
            n_prototype_boxes,
            BOX_DISTANCE,
            rng=random.Random(seed),
            max_attempts=legacy_max_attempts,
            return_all=True,
            filter_cF_gt_cP=True,
            require_p_dominant=False,
        )
        return boxes, True, {'valid_candidates_generated': len(candidates)}

    random.seed(seed)
    np.random.seed(seed)
    boxes = generate_prototypical_boxes(
        n_prototype_boxes,
        BOX_DISTANCE,
        require_p_dominant=True,
    )
    return boxes, False, {}


def _table4_worker(N, rep_idx, indices):
    selected_boxes = get_shared('selected_boxes')
    policies = get_shared('policies')
    box_list = [selected_boxes[i] for i in indices]
    solver = PandoraSolver(box_list)

    result = {'N': N, 'rep': rep_idx}

    t0 = time.time()
    opt_value = solver.solve_dp()
    result['OPT_value'] = float(opt_value)
    result['OPT_time'] = time.time() - t0

    for policy_name in policies:
        if policy_name == 'COM':
            t0 = time.time()
            partition, value = best_committing_policy(solver)
            result['COM_value'] = float(value)
            result['COM_time'] = time.time() - t0
            result['COM_partition'] = ''.join(partition)
            continue

        t0 = time.time()
        value = solver.evaluate_policy(POLICY_FUNCS[policy_name])
        result[f'{policy_name}_value'] = float(value)
        result[f'{policy_name}_time'] = time.time() - t0

    return result


def _aggregate(results, n_range, reps, policies, tol):
    rows = []
    raw_rows = []
    for N in n_range:
        n_reps = reps
        n_results = [
            results[f'{N}_{rep}']
            for rep in range(n_reps)
            if f'{N}_{rep}' in results
        ]
        if not n_results:
            continue

        opt = np.array([r['OPT_value'] for r in n_results])
        row = {'N': N, 'instances': len(n_results)}
        for policy in policies:
            values = np.array([r[f'{policy}_value'] for r in n_results])
            normalized = np.where(opt > 0, values / opt, 1.0)
            exact = (1.0 - normalized) < tol
            row[f'{policy}_exact_pct'] = float(np.mean(exact))
            row[f'{policy}_mean_ratio'] = float(np.mean(normalized))
            row[f'{policy}_worst_ratio'] = float(np.min(normalized))

            for rep_idx, r in enumerate(n_results):
                raw_rows.append({
                    'N': N,
                    'rep': r.get('rep', rep_idx),
                    'policy': policy,
                    'OPT_value': r['OPT_value'],
                    'policy_value': r[f'{policy}_value'],
                    'normalized': normalized[rep_idx],
                    'exact': bool(exact[rep_idx]),
                })
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(raw_rows)


def _build_pdf_comparison(exact_df, policies):
    rows = []
    for _, row in exact_df.iterrows():
        N = int(row['N'])
        if N not in PAPER_TABLE_4:
            continue
        for policy in policies:
            if policy not in PAPER_TABLE_4[N]:
                continue
            actual = float(row[f'{policy}_exact_pct'])
            pdf = PAPER_TABLE_4[N][policy]
            rows.append({
                'N': N,
                'policy': policy,
                'actual': actual,
                'pdf': pdf,
                'diff': actual - pdf,
                'abs_diff': abs(actual - pdf),
            })
    return pd.DataFrame(rows)


def _checkpoint_name(pool_source, n_range, reps, policies, tol, seed,
                     n_prototype_boxes):
    n_token = '-'.join(str(n) for n in n_range)
    p_token = '-'.join(policies)
    tol_token = str(tol).replace('-', 'm').replace('.', 'p')
    return (f'table4_{pool_source}_seed{seed}_pool{n_prototype_boxes}_'
            f'n{n_token}_r{reps}_{p_token}_{tol_token}')


def run_table4_replication(pool_source='old', legacy_pool_dir=None, n_range=None,
                           reps=SMALL_INSTANCES, policies=None, seed=SEED,
                           tol=1e-4, output_dir=None, n_workers=1,
                           fresh=False, output_prefix=None, save_raw=True,
                           n_prototype_boxes=NUM_PROTOTYPICAL_BOXES,
                           legacy_max_attempts=50000,
                           save_comparison=False):
    if pool_source == 'random':
        pool_source = 'generated'
    if legacy_pool_dir is None:
        legacy_pool_dir = bundled_legacy_pool_dir()
    if n_range is None:
        n_range = list(range(2, DP_CUTOFF + 1))
    if policies is None:
        policies = ['INDEX', 'WHITTLE', 'COM']
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'table4_replication')

    if max(n_range) > DP_CUTOFF:
        raise ValueError(
            f'Table 4 requires exact DP; requested max N={max(n_range)} '
            f'but DP_CUTOFF={DP_CUTOFF}.'
        )

    selected_boxes, legacy_sampling, pool_meta = _load_box_pool(
        pool_source,
        seed,
        legacy_pool_dir,
        n_prototype_boxes=n_prototype_boxes,
        legacy_max_attempts=legacy_max_attempts,
    )
    print(f'Pool source: {pool_source}')
    if pool_source == 'old':
        print(f'Legacy pool directory: {legacy_pool_dir}')
    print(f'Box pool size: {len(selected_boxes)}')
    for key, value in pool_meta.items():
        print(f'{key}: {value}')
    print(f'Legacy sampling: {legacy_sampling}')
    print(f'N range: {n_range}; reps per N: {reps}; policies: {policies}')

    tasks = generate_instance_tasks(
        n_range,
        lambda _N: reps,
        len(selected_boxes),
        seed,
        legacy_sampling=legacy_sampling,
    )

    os.makedirs(output_dir, exist_ok=True)
    ckpt = checkpoint_path_for(
        output_dir,
        _checkpoint_name(pool_source, n_range, reps, policies, tol, seed,
                         n_prototype_boxes),
    )
    if fresh:
        clear_checkpoint(ckpt)

    results = run_parallel(
        _table4_worker,
        tasks,
        shared_data={'selected_boxes': selected_boxes, 'policies': policies},
        n_workers=n_workers,
        checkpoint_path=ckpt,
        desc='Table 4',
    )

    exact_df, raw_df = _aggregate(results, n_range, reps, policies, tol)
    paper_df = table4_paper_columns(exact_df)
    comparison_df = _build_pdf_comparison(exact_df, policies)

    prefix = output_prefix or f'table_4_exact_optimality_{pool_source}'
    exact_path = os.path.join(output_dir, f'{prefix}.csv')
    raw_path = os.path.join(output_dir, f'{prefix}_raw.csv')
    latex_path = os.path.join(output_dir, f'{prefix}.tex')
    comparison_path = os.path.join(output_dir, f'{prefix}_vs_pdf.csv')

    paper_df.to_csv(exact_path, index=False, float_format='%.3f')
    if save_raw:
        raw_df.to_csv(raw_path, index=False)
    save_latex(latex_path, format_table_4(paper_df))
    if save_comparison:
        comparison_df.to_csv(comparison_path, index=False)

    print('\nExact optimality rates:')
    print(paper_df.to_string(
        index=False,
        formatters={col: '{:.3f}'.format for col in paper_df.columns
                    if col != 'N'},
    ))

    print(f'\nSaved exact rates to {os.path.abspath(exact_path)}')
    if save_raw:
        print(f'Saved per-instance rates to {os.path.abspath(raw_path)}')
    if save_comparison:
        print(f'Saved PDF comparison to {os.path.abspath(comparison_path)}')
    return exact_df, raw_df, comparison_df


def main():
    parser = argparse.ArgumentParser(description='Replicate Table 4')
    parser.add_argument('--pool-source', default='old',
                        choices=['old', 'legacy-generated',
                                 'legacy-generated-mixed', 'generated',
                                 'random'],
                        help='Use the bundled old pool, a legacy-style '
                             'generated pool, or the generated pool')
    parser.add_argument('--legacy-pool-dir', default=None,
                        help='Directory containing bundled old-pool JSON data '
                             '(default: data/legacy_box_pools)')
    parser.add_argument('--n-range', default='2:9',
                        help='Inclusive range like 2:5, or comma list')
    parser.add_argument('--reps', type=int, default=SMALL_INSTANCES,
                        help='Instances per N')
    parser.add_argument('--policies', default='INDEX,WHITTLE,COM',
                        help='Comma list: INDEX,WHITTLE,COM,STP,WEITZMAN,ALL')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--prototype-boxes', type=int,
                        default=NUM_PROTOTYPICAL_BOXES,
                        help='Number of generated prototype boxes '
                             '(ignored for --pool-source old)')
    parser.add_argument('--legacy-max-attempts', type=int, default=50000,
                        help='Maximum random draws for legacy-generated pool')
    parser.add_argument('--tol', type=float, default=1e-4,
                        help='Exact-optimality tolerance in normalized gap')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Parallel worker processes')
    parser.add_argument('--fresh', action='store_true',
                        help='Clear this script checkpoint before running')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for CSV/LaTeX outputs')
    parser.add_argument('--output-prefix', default=None,
                        help='Filename prefix for output files')
    parser.add_argument('--no-raw', action='store_true',
                        help='Do not write per-instance raw output')
    parser.add_argument('--save-comparison', action='store_true',
                        help='Write a diagnostic comparison against the '
                             'values transcribed from the paper')
    args = parser.parse_args()

    run_table4_replication(
        pool_source=args.pool_source,
        legacy_pool_dir=args.legacy_pool_dir,
        n_range=_parse_n_range(args.n_range),
        reps=args.reps,
        policies=_parse_policies(args.policies),
        seed=args.seed,
        tol=args.tol,
        output_dir=args.output_dir,
        n_workers=args.workers,
        fresh=args.fresh,
        output_prefix=args.output_prefix,
        save_raw=not args.no_raw,
        n_prototype_boxes=args.prototype_boxes,
        legacy_max_attempts=args.legacy_max_attempts,
        save_comparison=args.save_comparison,
    )


if __name__ == '__main__':
    main()
