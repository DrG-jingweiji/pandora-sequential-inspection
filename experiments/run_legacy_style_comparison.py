"""Short legacy-style pool experiment and comparison PDF builder.

This runner is intentionally separate from ``run_all``.  It is for diagnostic
experiments where every pool-based output should use boxes generated from the
recovered old ``Experiment.py`` prototype-selection logic, while keeping the run
small enough to finish in a few hours.

The default mode is ``legacy-generated-mixed``:
  * generate 100 prototypes from scratch,
  * use the old candidate-blocking rule in (sigma^P, sigma^F) space,
  * do not impose sigma^P > sigma^F before prototype selection,
  * use the c_F > c_P filtered subset for Tables 1-4/EC.1/Figure 3,
  * use the unfiltered mixed legacy-style prototypes for P-opening figures.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pandora.instance_generator import generate_legacy_style_prototypical_boxes
from experiments import config
from experiments.formatting import (
    format_table_1,
    format_table_2,
    format_table_3,
    format_table_4,
    save_latex,
)
from experiments.parallel import clear_all_checkpoints


def _parse_range(spec):
    if ':' in spec:
        start, end = [int(x.strip()) for x in spec.split(':', 1)]
        if end < start:
            raise ValueError(f'invalid range {spec!r}')
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(',') if x.strip()]


def _generate_legacy_pool(pool_source, prototype_boxes, max_attempts):
    require_p_dominant = pool_source == 'legacy-generated'
    selected_unfiltered, all_candidates = generate_legacy_style_prototypical_boxes(
        prototype_boxes,
        config.BOX_DISTANCE,
        rng=random.Random(config.SEED),
        max_attempts=max_attempts,
        return_all=True,
        filter_cF_gt_cP=False,
        require_p_dominant=require_p_dominant,
    )
    selected_main = [box for box in selected_unfiltered if box.c_F > box.c_P]
    return selected_main, selected_unfiltered, all_candidates


def _write_metadata(path, args, main_pool, p_opening_pool, candidates):
    metadata = {
        'pool_source': args.pool_source,
        'seed': config.SEED,
        'prototype_boxes_requested': args.prototype_boxes,
        'main_pool_after_cF_gt_cP': len(main_pool),
        'p_opening_pool_unfiltered': len(p_opening_pool),
        'valid_candidates_generated': len(candidates),
        'n_range': _parse_range(args.n_range),
        'instances': args.instances,
        'large_n_range': _parse_range(args.large_n_range),
        'large_instances': args.large_instances,
        'workers': args.workers,
        'legacy_sampling': True,
        'notes': (
            'Pool-based outputs use legacy-style generated prototypes. '
            'Figure EC.10d is the paper fixed-i.i.d.-box experiment and does '
            'not use a prototype pool.'
        ),
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)


def _paper_table_1_df():
    from tests.validate_against_paper import PAPER_TABLE1, TABLE1_COLS

    rows = []
    for n, vals in sorted(PAPER_TABLE1.items()):
        row = {'N': n}
        row.update(dict(zip(TABLE1_COLS, vals)))
        rows.append(row)
    return pd.DataFrame(rows)


def _paper_table_2_df():
    from tests.validate_against_paper import PAPER_TABLE2

    # The validator stores counts and ratios.  Add the published runtime
    # values from the paper/comparison source for display only.
    paper_times = {
        2: (0.04, 0.03),
        3: (0.01, 0.00),
        4: (0.10, 0.02),
        5: (0.64, 0.10),
        6: (3.99, 0.51),
        7: (23.59, 3.00),
        8: (143.80, 15.93),
        9: (810.50, 82.81),
    }
    rows = []
    for n, vals in sorted(PAPER_TABLE2.items()):
        time_naive, time_plus = paper_times[n]
        rows.append({
            'N': n,
            'states_naive': vals[0],
            'revisits_naive': vals[1],
            'time_naive': time_naive,
            'states_plus': vals[2],
            'revisits_plus': vals[3],
            'time_plus': time_plus,
            'runtime_ratio': vals[4],
        })
    return pd.DataFrame(rows)


def _paper_table_3_df():
    from tests.validate_against_paper import PAPER_TABLE3

    policies = ['OPT', 'INDEX', 'WHITTLE', 'COM', 'STP']
    rows = []
    for n, pdata in sorted(PAPER_TABLE3.items()):
        row = {'N': n}
        for policy in policies:
            vals = pdata.get(policy)
            if vals is None:
                row[f'{policy}_mean'] = np.nan
                row[f'{policy}_std'] = np.nan
                row[f'{policy}_worst'] = np.nan
            else:
                row[f'{policy}_mean'] = vals[0]
                row[f'{policy}_std'] = vals[1]
                row[f'{policy}_worst'] = vals[2]
        rows.append(row)
    return pd.DataFrame(rows)


def _paper_table_4_df():
    from tests.validate_against_paper import PAPER_TABLE4

    rows = []
    for n, pdata in sorted(PAPER_TABLE4.items()):
        rows.append({
            'N': n,
            'OPT': 1.0,
            'INDEX': pdata['INDEX'],
            'WHITTLE': pdata['WHITTLE'],
            'COM': pdata['COM'],
        })
    return pd.DataFrame(rows)


def _write_paper_tables(output_dir):
    paper_dir = output_dir / 'paper_reference_tables'
    paper_dir.mkdir(parents=True, exist_ok=True)
    save_latex(paper_dir / 'paper_table_1.tex', format_table_1(_paper_table_1_df()))
    save_latex(paper_dir / 'paper_table_2.tex', format_table_2(_paper_table_2_df()))
    save_latex(paper_dir / 'paper_table_3.tex', format_table_3(_paper_table_3_df()))
    save_latex(paper_dir / 'paper_table_4.tex', format_table_4(_paper_table_4_df()))
    return paper_dir


def _copy_if_exists(src, dst):
    if not src:
        return None
    src = Path(src)
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _copy_reference_figures(output_dir, reference_figure_dir, reference_figure3):
    ref_out = output_dir / 'paper_reference_figures'
    ref_dir = Path(reference_figure_dir) if reference_figure_dir else None

    copied = {}
    copied['figure_3'] = _copy_if_exists(
        reference_figure3,
        ref_out / 'figure_3_prototypical_boxes_paper.png',
    )

    mapping = {
        'figure_EC8': 'boxes_scatter.png',
        'figure_EC9a': 'when_P_is_important.png',
        'figure_EC9b': 'Weitzman_performance.png',
        'figure_EC10a': 'boxes_scatter_small_dispersion.png',
        'figure_EC10b': 'boxes_scatter_big_dispersion.png',
        'figure_EC10c': 'P_F_dispersion.png',
        'figure_EC10d': 'more_boxes.png',
    }
    for key, fname in mapping.items():
        src = ref_dir / fname if ref_dir else None
        copied[key] = _copy_if_exists(src, ref_out / fname)
    return copied


def _tex_path(path, base_dir):
    if path is None:
        return None
    return Path(path).resolve().relative_to(base_dir.resolve()).as_posix()


def _input_block(path, base_dir):
    rel = _tex_path(path, base_dir)
    if rel is None or not Path(path).exists():
        return r'\noindent\emph{Not available in this short run.}'
    return rf'\input{{{rel}}}'


def _figure_block(path, base_dir, width='0.58\\textwidth'):
    rel = _tex_path(path, base_dir)
    if rel is None or not Path(path).exists():
        return r'\noindent\emph{Figure not available in this short run.}'
    return (
        r'\begin{figure}[H]' '\n'
        r'\centering' '\n'
        rf'\includegraphics[width={width}]{{{rel}}}' '\n'
        r'\end{figure}'
    )


def _build_comparison_tex(output_dir, paper_tables, paper_figures):
    generated = output_dir
    sections = []

    table_sections = [
        ('Table 1: Coverage of Analytical Conditions',
         paper_tables / 'paper_table_1.tex', generated / 'table_1_coverage.tex',
         'Short run uses fewer replications and skips N=8,9 by default.'),
        ('Table 2: Naive vs. Structured DP Comparison',
         paper_tables / 'paper_table_2.tex', generated / 'table_2_dp_comparison.tex',
         'Short run skips the most expensive DP rows by default.'),
        ('Table 3: Normalized Policy Performance',
         paper_tables / 'paper_table_3.tex', generated / 'table_3_performance.tex',
         'Large-N rows are reduced to the configured short-run range.'),
        ('Table 4: Exact Optimality',
         paper_tables / 'paper_table_4.tex', generated / 'table_4_exact_optimality.tex',
         'This Table 4 uses the same legacy-style generated pool as the benchmark.'),
        ('Table EC.1: Runtime',
         None, generated / 'table_EC1_runtime.tex',
         'Runtime values are hardware- and run-size-dependent; only generated values are shown.'),
    ]
    for title, paper_path, generated_path, note in table_sections:
        sections.append(r'\newpage')
        sections.append(rf'\section{{{title}}}')
        sections.append(r'\paperlabel')
        sections.append(_input_block(paper_path, output_dir))
        sections.append(r'\genlabel')
        sections.append(_input_block(generated_path, output_dir))
        sections.append(rf'\noteslabel{{{note}}}')

    figure_sections = [
        ('Figure 3: Prototypical Boxes Visualization',
         paper_figures.get('figure_3'), generated / 'figure_3_prototypical_boxes.png'),
        ('Figure EC.8: Prototypical Box Selection Process',
         paper_figures.get('figure_EC8'), generated / 'figure_EC8_prototypical_boxes_selection.png'),
        ('Figure EC.9a: P-ratio vs. Proportion of P-dominant Boxes',
         paper_figures.get('figure_EC9a'), generated / 'figure_EC9a_p_ratio_vs_proportion.png'),
        ('Figure EC.9b: Weitzman Suboptimality vs. Proportion',
         paper_figures.get('figure_EC9b'), generated / 'figure_EC9b_weitzman_vs_proportion.png'),
        ('Figure EC.10a: Low Dispersion Example',
         paper_figures.get('figure_EC10a'), generated / 'figure_EC10a_low_dispersion.png'),
        ('Figure EC.10b: High Dispersion Example',
         paper_figures.get('figure_EC10b'), generated / 'figure_EC10b_high_dispersion.png'),
        ('Figure EC.10c: P-ratio vs. Dispersion',
         paper_figures.get('figure_EC10c'), generated / 'figure_EC10c_p_ratio_vs_dispersion.png'),
        ('Figure EC.10d: P-ratio vs. Number of Boxes',
         paper_figures.get('figure_EC10d'), generated / 'figure_EC10d_p_ratio_vs_num_boxes.png'),
    ]
    for title, paper_path, generated_path in figure_sections:
        sections.append(r'\newpage')
        sections.append(rf'\section{{{title}}}')
        sections.append(r'\paperlabel')
        sections.append(_figure_block(paper_path, output_dir))
        sections.append(r'\genlabel')
        sections.append(_figure_block(generated_path, output_dir))

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{caption}
\usepackage{adjustbox}
\usepackage{xcolor}
\usepackage{float}
\usepackage{hyperref}

\hypersetup{colorlinks=true, linkcolor=blue!60!black}
\newcommand{\paperlabel}{\medskip{\large\bfseries\color{red!70!black} Original (Paper)}\par\medskip}
\newcommand{\genlabel}{\bigskip{\large\bfseries\color{green!50!black} Generated (Legacy-Style Short Run)}\par\medskip}
\newcommand{\noteslabel}[1]{\bigskip\noindent\textbf{Notes:} #1}

\title{\textbf{Paper vs. Legacy-Style Generated Output}\\[0.3em]
\large Short-run comparison using generated legacy-style prototypical boxes}
\author{}
\date{\today}

\begin{document}
\maketitle

\noindent This diagnostic run uses legacy-style generated prototypical boxes
for all pool-based outputs.  It is intentionally smaller than the full paper
run, so rows outside the configured ranges are omitted.

\tableofcontents
""" + '\n'.join(sections) + r"""

\end{document}
"""
    tex_path = output_dir / 'comparison_legacy_style_short.tex'
    tex_path.write_text(tex, encoding='utf-8')
    return tex_path


def _compile_pdf(tex_path):
    cmd = ['pdflatex', '-interaction=nonstopmode', tex_path.name]
    for _ in range(2):
        subprocess.run(
            cmd,
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return tex_path.with_suffix('.pdf')


def main():
    parser = argparse.ArgumentParser(
        description='Run a short comparison using legacy-style generated boxes.'
    )
    parser.add_argument('--pool-source', default='legacy-generated-mixed',
                        choices=['legacy-generated', 'legacy-generated-mixed'])
    parser.add_argument('--prototype-boxes', type=int, default=100)
    parser.add_argument('--legacy-max-attempts', type=int, default=50000)
    parser.add_argument('--n-range', default='2:7',
                        help='DP N range for short-run tables/figures')
    parser.add_argument('--instances', type=int, default=100,
                        help='Replications per DP N')
    parser.add_argument('--large-n-range', default='10:11',
                        help='Heuristic-only N range for Table 3/EC.1')
    parser.add_argument('--large-instances', type=int, default=20)
    parser.add_argument('--workers', type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument('--output-dir', default=os.path.join(
        os.path.dirname(__file__), '..', 'output', 'legacy_style_short',
    ))
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--skip-pdf', action='store_true')
    parser.add_argument('--reference-figure-dir', default=None,
                        help='Optional directory containing paper figure PNGs')
    parser.add_argument('--reference-figure3', default=None,
                        help='Optional paper Figure 3 PNG')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = str(output_dir)

    if args.fresh:
        clear_all_checkpoints(config.OUTPUT_DIR)

    print(f'Output directory: {output_dir}')
    print(f'Pool source: {args.pool_source}')
    main_pool, p_opening_pool, candidates = _generate_legacy_pool(
        args.pool_source,
        args.prototype_boxes,
        args.legacy_max_attempts,
    )
    print(f'Main pool after c_F > c_P filter: {len(main_pool)}')
    print(f'P-opening unfiltered pool: {len(p_opening_pool)}')
    print(f'Valid candidates generated: {len(candidates)}')

    _write_metadata(output_dir / 'legacy_style_short_metadata.json',
                    args, main_pool, p_opening_pool, candidates)

    n_range = _parse_range(args.n_range)
    large_n_range = _parse_range(args.large_n_range)
    benchmark_range = n_range + [n for n in large_n_range if n not in n_range]

    # Import experiment modules after config.OUTPUT_DIR is redirected.
    from experiments.figures import plot_figure_3, plot_figure_EC8
    from experiments.exp_coverage import run_coverage_experiment
    from experiments.exp_dp_comparison import run_dp_comparison
    from experiments.exp_policy_benchmark import run_policy_benchmark
    from experiments.exp_p_opening import (
        run_p_opening_analysis,
        run_more_boxes_experiment,
    )

    plot_figure_3(main_pool)
    plot_figure_EC8(p_opening_pool, candidates)

    run_coverage_experiment(
        n_range=n_range,
        n_instances=args.instances,
        selected_boxes=main_pool,
        n_workers=args.workers,
        legacy_sampling=True,
    )
    run_dp_comparison(
        n_range=n_range,
        n_instances=args.instances,
        selected_boxes=main_pool,
        n_workers=args.workers,
        legacy_sampling=True,
    )
    run_policy_benchmark(
        n_range=benchmark_range,
        n_instances_small=args.instances,
        n_instances_large=args.large_instances,
        selected_boxes=main_pool,
        n_workers=args.workers,
        legacy_sampling=True,
        write_exact_table=True,
    )
    run_p_opening_analysis(
        n_range=n_range,
        n_instances=args.instances,
        selected_boxes=p_opening_pool,
        n_workers=args.workers,
        legacy_sampling=True,
    )
    run_more_boxes_experiment(n_range=range(1, 4), n_workers=args.workers)

    if not args.skip_pdf:
        paper_tables = _write_paper_tables(output_dir)
        paper_figures = _copy_reference_figures(
            output_dir,
            args.reference_figure_dir,
            args.reference_figure3,
        )
        tex_path = _build_comparison_tex(output_dir, paper_tables, paper_figures)
        try:
            pdf_path = _compile_pdf(tex_path)
            print(f'Comparison PDF: {pdf_path}')
        except subprocess.CalledProcessError as exc:
            log_path = tex_path.with_suffix('.build.log')
            log_path.write_text(exc.stdout or '', encoding='utf-8')
            print(f'PDF build failed. Log: {log_path}')
            raise

    print('Legacy-style short comparison complete.')


if __name__ == '__main__':
    main()
