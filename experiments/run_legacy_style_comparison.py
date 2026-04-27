"""Legacy-style pool experiment and comparison PDF builder.

This runner is intentionally separate from ``run_all``.  It is the default
public-replication entry point for generating every pool-based output from the
recovered old ``Experiment.py`` prototype-selection logic.

The default preset is ``full`` for public replication.  Use ``--preset check``
for a stable no-large-instance check report, or ``--preset short`` for a quick
diagnostic run.  The full preset is expensive; it is designed to be resumed
from checkpoints if interrupted.

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
    format_table_EC1,
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
        'preset': args.preset,
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
        'more_boxes_n_range': _parse_range(args.more_boxes_n_range),
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


def _preset_defaults(preset):
    if preset == 'full':
        return {
            'n_range': '2:9',
            'instances': 200,
            'large_n_range': '10:16',
            'large_instances': 100,
            'more_boxes_n_range': '1:7',
            'workers': 12,
            'output_subdir': 'legacy_style_full',
            'run_label': 'Full Run',
            'comparison_stem': 'comparison_legacy_style_full',
        }
    if preset == 'check':
        return {
            'n_range': '2:6',
            'instances': 100,
            'large_n_range': '',
            'large_instances': 0,
            'more_boxes_n_range': '1:6',
            'workers': 12,
            'output_subdir': 'legacy_style_check',
            'run_label': 'Check Run',
            'comparison_stem': 'comparison_legacy_style_check',
        }
    return {
        'n_range': '2:7',
        'instances': 100,
        'large_n_range': '10:11',
        'large_instances': 20,
        'more_boxes_n_range': '1:3',
        'workers': config.DEFAULT_WORKERS,
        'output_subdir': 'legacy_style_short',
        'run_label': 'Short Run',
        'comparison_stem': 'comparison_legacy_style_short',
    }


def _apply_preset_defaults(args):
    defaults = _preset_defaults(args.preset)
    for key, value in defaults.items():
        if key == 'output_subdir':
            continue
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), '..', 'output', defaults['output_subdir']
        )
    return args


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


def _paper_table_ec1_df():
    rows = [
        (2, 0.002, 0.002, 0.001, 0.001, 0.005, 0.004, 0.001, 0.001, 0.000, 0.000),
        (3, 0.014, 0.013, 0.002, 0.002, 0.021, 0.024, 0.004, 0.004, 0.001, 0.001),
        (4, 0.101, 0.078, 0.004, 0.005, 0.084, 0.121, 0.013, 0.014, 0.002, 0.002),
        (5, 0.738, 0.577, 0.007, 0.010, 0.276, 0.466, 0.032, 0.049, 0.004, 0.003),
        (6, 5.698, 5.128, 0.012, 0.017, 0.884, 1.729, 0.081, 0.144, 0.007, 0.008),
        (7, 41.049, 39.646, 0.025, 0.050, 2.933, 6.081, 0.248, 0.563, 0.015, 0.020),
        (8, 316.648, 311.921, 0.047, 0.114, 10.831, 30.604, 0.707, 1.902, 0.028, 0.031),
        (9, 2472.509, 2585.495, 0.088, 0.323, 29.127, 72.949, 1.662, 3.831, 0.052, 0.060),
        (10, np.nan, np.nan, 0.128, 0.316, 75.226, 228.431, 3.292, 7.010, 0.072, 0.104),
        (11, np.nan, np.nan, 0.261, 1.018, 152.678, 1817.640, 9.082, 25.463, 0.141, 0.220),
        (12, np.nan, np.nan, 0.372, 0.778, 718.722, 3458.725, 20.250, 40.365, 0.227, 0.382),
        (13, np.nan, np.nan, 0.823, 1.851, 1613.774, 4178.379, 79.878, 266.576, 0.438, 0.594),
        (14, np.nan, np.nan, 1.556, 6.139, 3668.967, 10860.112, 245.827, 1494.645, 0.793, 1.365),
        (15, np.nan, np.nan, 2.977, 3.895, 10325.838, 13620.728, 810.565, 1658.033, 1.885, 2.392),
        (16, np.nan, np.nan, 5.475, 11.173, 10168.432, 15753.618, 869.394, 5183.688, 2.025, 3.310),
    ]
    columns = [
        'N',
        'OPT_mean', 'OPT_std',
        'INDEX_mean', 'INDEX_std',
        'WHITTLE_mean', 'WHITTLE_std',
        'COM_mean', 'COM_std',
        'STP_mean', 'STP_std',
    ]
    return pd.DataFrame(rows, columns=columns)


def _write_paper_tables(output_dir):
    paper_dir = output_dir / 'paper_reference_tables'
    paper_dir.mkdir(parents=True, exist_ok=True)
    save_latex(paper_dir / 'paper_table_1.tex', format_table_1(_paper_table_1_df()))
    save_latex(paper_dir / 'paper_table_2.tex', format_table_2(_paper_table_2_df()))
    save_latex(paper_dir / 'paper_table_3.tex', format_table_3(_paper_table_3_df()))
    save_latex(paper_dir / 'paper_table_4.tex', format_table_4(_paper_table_4_df()))
    save_latex(paper_dir / 'paper_table_EC1.tex',
               format_table_EC1(_paper_table_ec1_df()))
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


def _input_block(path, base_dir, missing_text='Not available in this run.'):
    rel = _tex_path(path, base_dir)
    if rel is None or not Path(path).exists():
        return rf'\noindent\emph{{{missing_text}}}'
    return rf'\input{{{rel}}}'


def _figure_block(path, base_dir, width='0.58\\textwidth',
                  missing_text='Figure not available in this run.'):
    rel = _tex_path(path, base_dir)
    if rel is None or not Path(path).exists():
        return rf'\noindent\emph{{{missing_text}}}'
    return (
        r'\begin{figure}[H]' '\n'
        r'\centering' '\n'
        rf'\includegraphics[width={width}]{{{rel}}}' '\n'
        r'\end{figure}'
    )


def _build_comparison_tex(output_dir, paper_tables, paper_figures, run_label,
                          comparison_stem, is_full):
    generated = output_dir
    sections = []
    row_note = (
        'Full paper-scale run: Tables use N=2--9 for DP-based rows and '
        'N=10--16 for heuristic-only benchmark rows.'
        if is_full else
        'Short diagnostic run: fewer replications are used and rows outside '
        'the configured ranges are omitted.'
    )

    table_sections = [
        ('Table 1: Coverage of Analytical Conditions',
         paper_tables / 'paper_table_1.tex', generated / 'table_1_coverage.tex',
         row_note),
        ('Table 2: Naive vs. Structured DP Comparison',
         paper_tables / 'paper_table_2.tex', generated / 'table_2_dp_comparison.tex',
         row_note),
        ('Table 3: Normalized Policy Performance',
         paper_tables / 'paper_table_3.tex', generated / 'table_3_performance.tex',
         row_note),
        ('Table 4: Exact Optimality',
         paper_tables / 'paper_table_4.tex', generated / 'table_4_exact_optimality.tex',
         'This Table 4 uses the same legacy-style generated pool as the benchmark.'),
        ('Table EC.1: Runtime',
         paper_tables / 'paper_table_EC1.tex', generated / 'table_EC1_runtime.tex',
         'Runtime values are hardware- and run-size-dependent; compare ordering and scale qualitatively.'),
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
        sections.append(_figure_block(
            paper_path,
            output_dir,
            missing_text=(
                'Paper reference PNG was not provided. Pass '
                '--reference-figure-dir and --reference-figure3 to include it.'
            ),
        ))
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
\newcommand{\genlabel}{\bigskip{\large\bfseries\color{green!50!black} Generated (Legacy-Style """ + run_label + r""")}\par\medskip}
\newcommand{\noteslabel}[1]{\bigskip\noindent\textbf{Notes:} #1}

\title{\textbf{Paper vs. Legacy-Style Generated Output}\\[0.3em]
\large """ + run_label + r""" comparison using generated legacy-style prototypical boxes}
\author{}
\date{\today}

\begin{document}
\maketitle

\noindent This run uses legacy-style generated prototypical boxes
for all pool-based outputs.  """ + row_note + r"""

\tableofcontents
""" + '\n'.join(sections) + r"""

\end{document}
"""
    tex_path = output_dir / f'{comparison_stem}.tex'
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
        description='Run a comparison using legacy-style generated boxes.'
    )
    parser.add_argument('--preset', default='full',
                        choices=['check', 'short', 'full'],
                        help=('Use "full" for the paper-scale run or "short" '
                              'for a diagnostic run. "check" runs N=2..6 '
                              'with no large-instance rows and 100 '
                              'replications per N. Explicit range/count '
                              'options override the preset.'))
    parser.add_argument('--pool-source', default='legacy-generated-mixed',
                        choices=['legacy-generated', 'legacy-generated-mixed'])
    parser.add_argument('--prototype-boxes', type=int, default=100)
    parser.add_argument('--legacy-max-attempts', type=int, default=50000)
    parser.add_argument('--n-range', default=None,
                        help='DP N range for tables/figures')
    parser.add_argument('--instances', type=int, default=None,
                        help='Replications per DP N')
    parser.add_argument('--large-n-range', default=None,
                        help='Heuristic-only N range for Table 3/EC.1')
    parser.add_argument('--large-instances', type=int, default=None)
    parser.add_argument('--more-boxes-n-range', default=None,
                        help='N range for Figure EC.10d identical-box run')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--comparison-stem', default=None,
                        help='Base filename for comparison .tex/.pdf')
    parser.add_argument('--run-label', default=None,
                        help='Label used in the comparison PDF')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--skip-pdf', action='store_true')
    parser.add_argument('--skip-experiments', action='store_true',
                        help=('Only rebuild the comparison PDF from existing '
                              'output files.'))
    parser.add_argument('--reference-figure-dir', default=None,
                        help='Optional directory containing paper figure PNGs')
    parser.add_argument('--reference-figure3', default=None,
                        help='Optional paper Figure 3 PNG')
    args = parser.parse_args()
    args = _apply_preset_defaults(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = str(output_dir)

    if args.fresh:
        clear_all_checkpoints(config.OUTPUT_DIR)

    print(f'Output directory: {output_dir}')
    if not args.skip_experiments:
        print(f'Pool source: {args.pool_source}')
        main_pool, p_opening_pool, candidates = _generate_legacy_pool(
            args.pool_source,
            args.prototype_boxes,
            args.legacy_max_attempts,
        )
        print(f'Main pool after c_F > c_P filter: {len(main_pool)}')
        print(f'P-opening unfiltered pool: {len(p_opening_pool)}')
        print(f'Valid candidates generated: {len(candidates)}')

        _write_metadata(output_dir / 'legacy_style_run_metadata.json',
                        args, main_pool, p_opening_pool, candidates)

        n_range = _parse_range(args.n_range)
        large_n_range = _parse_range(args.large_n_range)
        more_boxes_n_range = _parse_range(args.more_boxes_n_range)
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
        run_more_boxes_experiment(n_range=more_boxes_n_range,
                                  n_workers=args.workers)

    if not args.skip_pdf:
        paper_tables = _write_paper_tables(output_dir)
        paper_figures = _copy_reference_figures(
            output_dir,
            args.reference_figure_dir,
            args.reference_figure3,
        )
        tex_path = _build_comparison_tex(
            output_dir,
            paper_tables,
            paper_figures,
            args.run_label,
            args.comparison_stem,
            args.preset == 'full',
        )
        try:
            pdf_path = _compile_pdf(tex_path)
            print(f'Comparison PDF: {pdf_path}')
        except subprocess.CalledProcessError as exc:
            log_path = tex_path.with_suffix('.build.log')
            log_path.write_text(exc.stdout or '', encoding='utf-8')
            print(f'PDF build failed. Log: {log_path}')
            raise

    print(f'Legacy-style {args.preset} comparison complete.')


if __name__ == '__main__':
    main()
