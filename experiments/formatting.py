"""LaTeX table formatting utilities matching the paper's exact layout.

Each format_table_N function takes a DataFrame and returns a LaTeX string
that reproduces the corresponding table from the paper.
"""

import os
import numpy as np
import pandas as pd


def save_latex(path, latex_str):
    """Write a LaTeX string to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(latex_str)


def _fmt(val, decimals=2, width=0):
    """Format a numeric value to a fixed number of decimal places."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '-'
    s = f'{val:.{decimals}f}'
    if width:
        s = s.rjust(width)
    return s


def _fmt_int(val, decimals=1):
    """Format a large number with one decimal place (e.g. 38.4)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '-'
    return f'{val:.{decimals}f}'


# ======================================================================
# Table 1: Coverage rates (tab:util_suff_thms)
# ======================================================================

def format_table_1(df):
    """Format coverage results as LaTeX matching Table 1 in the paper.

    Expected DataFrame columns: N, coverage_stop, coverage_f, coverage_p,
    coverage_total, recall_f, recall_p.
    """
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Average weighted coverage rates of Theorems 1--3 '
                 r'and recall rates of Theorems 2 and~3.}')
    lines.append(r'    \begin{tabular}{ccccccccccccc}')
    lines.append(r'    \toprule')
    lines.append(r'    \multirow{2}[4]{*}{N} &    & '
                 r'\multicolumn{7}{c}{Coverage rate} &    & '
                 r'\multicolumn{3}{c}{Recall } \\')
    lines.append(r'\cmidrule{2-9}\cmidrule{11-13}'
                 r'       &    & Thm.~1 &    & Thm.~2 &    & '
                 r'Thm.~3 &    & Overall &    & Thm.~2 &    & Thm.~3 \\')
    lines.append(r'\cmidrule{1-1}\cmidrule{3-3}\cmidrule{5-5}'
                 r'\cmidrule{7-7}\cmidrule{9-9}\cmidrule{11-11}'
                 r'\cmidrule{13-13}')

    for _, row in df.iterrows():
        n = int(row['N'])
        vals = [
            _fmt(row['coverage_stop']),
            _fmt(row['coverage_f']),
            _fmt(row['coverage_p']),
            _fmt(row['coverage_total']),
            _fmt(row['recall_f']),
            _fmt(row['recall_p']),
        ]
        line = (f'    {n}  &    & {vals[0]} &    & {vals[1]} &    & '
                f'{vals[2]} &    & {vals[3]} &    & {vals[4]} &    & '
                f'{vals[5]} \\\\')
        lines.append(line)

    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}%')
    lines.append(r'  \label{tab:util_suff_thms}%')
    lines.append(r'\end{table}%')
    return '\n'.join(lines)


# ======================================================================
# Table 2: DP comparison (tab:comparison_DP_JAY)
# ======================================================================

def format_table_2(df):
    r"""Format DP comparison results as LaTeX matching Table 2.

    Expected columns: N, states_naive, revisits_naive, time_naive,
    states_plus, revisits_plus, time_plus, runtime_ratio.
    """
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\scriptsize')
    lines.append(r'  \centering')
    lines.append(r'  \caption{A comparison between a DP implementation that '
                 r'utilizes the sufficient optimality conditions and one that '
                 r'does not, in terms of the average number of states created '
                 r'and revisited, and average runtimes (in seconds).}')
    lines.append(r'    \begin{tabular}{ccccrcccc}')
    lines.append(r'    \toprule')
    lines.append(r'    \multirow{2}[4]{*}{$N$} & '
                 r'\multicolumn{3}{c}{Na\"ive DP} &    & '
                 r'\multicolumn{3}{c}{DP that utilizes Theorems 1--3} & '
                 r'\multirow{2}[4]{*}{runtime ratio} \\')
    lines.append(r'\cmidrule{2-4}\cmidrule{6-8}'
                 r'       & \# states created &'
                 r'\# states revisited & runtime (s) &    & '
                 r'\# states created &\# states revisited & runtime (s) &  \\')
    lines.append(r'    \midrule')

    for _, row in df.iterrows():
        n = int(row['N'])
        line = (f'    {n}  & '
                f'{_fmt_int(row["states_naive"])} & '
                f'{_fmt_int(row["revisits_naive"])} & '
                f'{_fmt(row["time_naive"])} &    & '
                f'{_fmt_int(row["states_plus"])} & '
                f'{_fmt(row["revisits_plus"])} & '
                f'{_fmt(row["time_plus"])} & '
                f'{_fmt(row["runtime_ratio"], 3)} \\\\')
        lines.append(line)

    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}%')
    lines.append(r'  \label{tab:comparison_DP_JAY}%')
    lines.append(r'\end{table}%')
    return '\n'.join(lines)


# ======================================================================
# Table 3: Normalized performance (tab:performance_with_DP)
# ======================================================================

def format_table_3(df, dp_cutoff=9):
    r"""Format policy performance results as LaTeX matching Table 3.

    Expected columns: N, and for each policy (OPT, INDEX, WHITTLE, COM, STP):
    {POLICY}_mean, {POLICY}_std, {POLICY}_worst.
    """
    policies = ['OPT', 'INDEX', 'WHITTLE', 'COM', 'STP']
    policy_labels = [
        r'$\pi^\textrm{OPT}$',
        r'$\pi^{\textrm{index}}$',
        r'$\pi^{\textrm{W}}$',
        r'$\pi^{F^*,P^*}$',
        r'$\pi^{STP}$',
    ]

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\scriptsize')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Normalized performance of different policies. }')
    lines.append(r'    \begin{tabular}{cccccccccccccccccccc}')
    lines.append(r'     \toprule')

    header1 = r'    \multirow{2}[2]{*}{$N$}'
    for i, label in enumerate(policy_labels):
        sep = ' &    & ' if i > 0 else ' & '
        header1 += sep + r'\multicolumn{3}{c}{' + label + '}'
    header1 += r' \\'
    lines.append(header1)

    lines.append(r'\cmidrule{2-4}\cmidrule{6-8}\cmidrule{10-12}'
                 r'\cmidrule{14-16}\cmidrule{18-20}'
                 r'       & mean  & std & worst &    '
                 r'& mean  & std & worst &    '
                 r'& mean  & std & worst &    '
                 r'& mean  & std & worst &    '
                 r'& mean  & std & worst \\')
    lines.append(r'\midrule')

    prev_large = False
    for _, row in df.iterrows():
        n = int(row['N'])
        if n > dp_cutoff and not prev_large:
            lines.append(r'\cmidrule{1-1}')
            prev_large = True

        parts = [f'    {n} ']
        for i, pol in enumerate(policies):
            sep = ' &    & ' if i > 0 else ' & '
            m_key = f'{pol}_mean'
            s_key = f'{pol}_std'
            w_key = f'{pol}_worst'
            m_val = row.get(m_key)
            if m_val is None or (isinstance(m_val, float) and np.isnan(m_val)):
                parts.append(sep + '-  & -  & - ')
            else:
                parts.append(sep + f'{_fmt(m_val, 3)} & '
                             f'{_fmt(row[s_key], 3)} & '
                             f'{_fmt(row[w_key], 3)}')
        parts.append(' \\\\')
        lines.append(''.join(parts))

    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}%')
    lines.append(r'  \label{tab:performance_with_DP}%')
    lines.append(r'\end{table}%')
    return '\n'.join(lines)


# ======================================================================
# Table 4: Exact optimality (tab:comparison_of_J)
# ======================================================================

def format_table_4(df):
    r"""Format exact optimality percentages as LaTeX matching Table 4.

    Expected columns: either paper columns (N, OPT, INDEX, WHITTLE, COM) or
    internal exact_pct columns.
    """
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'     \scriptsize')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Percentage of instances in which the policies '
                 r'attained the optimal objective value.}')
    lines.append(r'   \begin{adjustbox}{width=0.4\textwidth}')
    lines.append(r'    \begin{tabular}{ccccc}')
    lines.append(r'    \toprule')
    lines.append(r'    $N$ & ')
    lines.append(r'    $\pi^\textrm{OPT}$ &')
    lines.append(r'    $\pi^{\textrm{index}}$ & ')
    lines.append(r'    $\pi^{\textrm{W}}$ &    ')
    lines.append(r'    $\pi^{F^*,P^*}$ \\')
    lines.append(r'    \midrule')

    for _, row in df.iterrows():
        n = int(row['N'])
        opt = _fmt(row.get('OPT', 1.0), 3)
        idx_val = _fmt(row.get('INDEX', row.get('INDEX_exact_pct', 0)), 3)
        wh_val = _fmt(row.get('WHITTLE', row.get('WHITTLE_exact_pct', 0)), 3)
        com_val = _fmt(row.get('COM', row.get('COM_exact_pct', 0)), 3)
        line = (f'    {n}  & {opt} & {idx_val} & {wh_val}  &  {com_val}  \\\\')
        lines.append(line)

    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}%')
    lines.append(r'      \end{adjustbox}')
    lines.append(r'  \label{tab:comparison_of_J}%')
    lines.append(r'\end{table}%')
    return '\n'.join(lines)


def table4_paper_columns(df):
    """Return Table 4 with only columns shown in the paper."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'N': int(row['N']),
            'OPT': float(row.get('OPT', 1.0)),
            'INDEX': float(row.get('INDEX', row.get('INDEX_exact_pct', np.nan))),
            'WHITTLE': float(row.get('WHITTLE', row.get('WHITTLE_exact_pct', np.nan))),
            'COM': float(row.get('COM', row.get('COM_exact_pct', np.nan))),
        })
    return pd.DataFrame(rows, columns=['N', 'OPT', 'INDEX', 'WHITTLE', 'COM'])


# ======================================================================
# Table EC.1: Runtime (fig:optimality_boxplot)
# ======================================================================

def format_table_EC1(df, dp_cutoff=9):
    r"""Format runtime results as LaTeX matching Table EC.1.

    Expected columns: N, and for each policy (OPT, INDEX, WHITTLE, COM, STP):
    {POLICY}_mean, {POLICY}_std.
    """
    policies = ['OPT', 'INDEX', 'WHITTLE', 'COM', 'STP']
    policy_labels = [
        r'$\pi^\textrm{OPT}$',
        r'$\pi^{\textrm{index}}$',
        r'$\pi^{\textrm{W}}$',
        r'$\pi^{F^*,P^*}$',
        r'$\pi^{STP}$',
    ]

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r' \footnotesize')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Runtime (in seconds) of different policies. }')
    lines.append(r'    \begin{tabular}{cccrccccccccccc}')
    lines.append(r'    \toprule')

    header1 = r'    \multirow{2}[3]{*}{$N$}'
    for i, label in enumerate(policy_labels):
        sep = ' &    & ' if i > 0 else ' & '
        header1 += sep + r'\multicolumn{2}{c}{' + label + '}'
    header1 += r' \\'
    lines.append(header1)

    lines.append(r'\cmidrule{2-3}\cmidrule{5-6}\cmidrule{8-9}'
                 r'\cmidrule{11-12}\cmidrule{14-15}'
                 r'       & mean  & std &    '
                 r'& mean  & std &    '
                 r'& mean  & std &    '
                 r'& mean  & std &    '
                 r'& mean & std \\')
    lines.append(r'\midrule')

    prev_large = False
    for _, row in df.iterrows():
        n = int(row['N'])
        if n > dp_cutoff and not prev_large:
            lines.append(r'\cmidrule{1-1}')
            prev_large = True

        parts = [f'    {n} ']
        for i, pol in enumerate(policies):
            sep = ' &    & ' if i > 0 else ' & '
            m_key = f'{pol}_mean'
            s_key = f'{pol}_std'
            m_val = row.get(m_key)
            if m_val is None or (isinstance(m_val, float) and np.isnan(m_val)):
                parts.append(sep + '-  & - ')
            else:
                parts.append(sep + f'{_fmt(m_val, 3)} & '
                             f'{_fmt(row[s_key], 3)}')
        parts.append(' \\\\')
        lines.append(''.join(parts))

    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}%')
    lines.append(r' \label{fig:optimality_boxplot}')
    lines.append(r'\end{table}%')
    return '\n'.join(lines)
