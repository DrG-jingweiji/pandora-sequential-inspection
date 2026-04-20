"""Validate experiment output against the paper's published tables.

Reads the generated CSVs from output/ and compares each metric to the
values in Tables 1-4 and EC.1 from the paper's LaTeX source.  Reports
absolute deviations and flags cells that exceed tolerance.

Results won't match exactly because the prototypical box pool is
regenerated (same seed, different code path), so instances differ.
We compare statistical patterns with generous tolerances.

Usage:
    python tests/validate_against_paper.py
"""

import os
import sys
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

# =====================================================================
# Paper reference values (from tables_n_figures.tex and
# numerical_appendix.tex)
# =====================================================================

# Table 1: Coverage of analytical conditions
# Columns: N, coverage_stop, coverage_f, coverage_p, coverage_total,
#           recall_f, recall_p
PAPER_TABLE1 = {
    2: (0.48, 0.38, 0.06, 0.92, 0.90, 0.53),
    3: (0.47, 0.38, 0.09, 0.94, 0.95, 0.66),
    4: (0.46, 0.39, 0.10, 0.95, 0.97, 0.72),
    5: (0.46, 0.39, 0.11, 0.95, 0.97, 0.77),
    6: (0.45, 0.40, 0.11, 0.96, 0.97, 0.81),
    7: (0.45, 0.40, 0.11, 0.96, 0.97, 0.83),
    8: (0.44, 0.41, 0.11, 0.96, 0.96, 0.86),
    9: (0.44, 0.41, 0.11, 0.96, 0.96, 0.87),
}
TABLE1_COLS = ['coverage_stop', 'coverage_f', 'coverage_p',
               'coverage_total', 'recall_f', 'recall_p']

# Table 2: Naive vs Structured DP
# Columns: N, states_naive, revisits_naive, states_plus, revisits_plus,
#           runtime_ratio  (absolute runtimes skipped)
PAPER_TABLE2 = {
    2: (38.4, 145.0, 24.8, 32.37, 0.731),
    3: (209.0, 1384.0, 115.3, 195.09, 0.225),
    4: (1101.4, 10513.9, 575.1, 1204.03, 0.180),
    5: (5567.0, 69796.4, 2798.7, 6703.55, 0.156),
    6: (28158.6, 440669.7, 12943.3, 35306.54, 0.128),
    7: (137652.0, 2570720.3, 65826.3, 208193.38, 0.121),
    8: (680310.1, 14825094.6, 298534.1, 1067723.38, 0.106),
    9: (3238243.1, 80398698.6, 1368893.5, 5450797.57, 0.094),
}
TABLE2_COUNT_COLS = ['states_naive', 'revisits_naive',
                     'states_plus', 'revisits_plus']

# Table 3: Policy performance (normalized)
# Columns per policy: mean, std, worst
# Paper policies: OPT, INDEX, WHITTLE (W), COM (F*,P*), STP
PAPER_TABLE3 = {
    2:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.987, 0.036, 0.605),
         'WHITTLE': (0.922, 0.123, 0.033), 'COM': (0.999, 0.005, 0.933),
         'STP': (0.963, 0.057, 0.709)},
    3:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.995, 0.017, 0.804),
         'WHITTLE': (0.901, 0.101, 0.298), 'COM': (1.000, 0.002, 0.960),
         'STP': (0.947, 0.062, 0.601)},
    4:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.998, 0.007, 0.882),
         'WHITTLE': (0.893, 0.088, 0.403), 'COM': (1.000, 0.001, 0.976),
         'STP': (0.943, 0.063, 0.611)},
    5:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.999, 0.003, 0.958),
         'WHITTLE': (0.890, 0.082, 0.373), 'COM': (1.000, 0.000, 0.993),
         'STP': (0.938, 0.062, 0.652)},
    6:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.999, 0.002, 0.966),
         'WHITTLE': (0.890, 0.085, 0.438), 'COM': (1.000, 0.000, 0.993),
         'STP': (0.940, 0.059, 0.691)},
    7:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.999, 0.002, 0.984),
         'WHITTLE': (0.888, 0.084, 0.503), 'COM': (1.000, 0.000, 0.996),
         'STP': (0.933, 0.063, 0.575)},
    8:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.999, 0.002, 0.989),
         'WHITTLE': (0.879, 0.088, 0.494), 'COM': (1.000, 0.000, 0.996),
         'STP': (0.937, 0.061, 0.564)},
    9:  {'OPT': (1.000, 0.000, 1.000), 'INDEX': (0.999, 0.001, 0.991),
         'WHITTLE': (0.877, 0.090, 0.522), 'COM': (1.000, 0.000, 0.997),
         'STP': (0.935, 0.061, 0.653)},
    10: {'INDEX': (0.999, 0.001, 0.992), 'WHITTLE': (0.876, 0.099, 0.515),
         'COM': (1.000, 0.000, 1.000), 'STP': (0.941, 0.061, 0.709)},
    11: {'INDEX': (0.999, 0.001, 0.993), 'WHITTLE': (0.871, 0.110, 0.463),
         'COM': (1.000, 0.000, 1.000), 'STP': (0.939, 0.063, 0.686)},
    12: {'INDEX': (0.999, 0.001, 0.994), 'WHITTLE': (0.870, 0.104, 0.451),
         'COM': (1.000, 0.000, 1.000), 'STP': (0.940, 0.061, 0.693)},
    13: {'INDEX': (0.999, 0.001, 0.992), 'WHITTLE': (0.869, 0.099, 0.568),
         'COM': (1.000, 0.000, 1.000), 'STP': (0.945, 0.055, 0.710)},
    14: {'INDEX': (0.999, 0.001, 0.996), 'WHITTLE': (0.877, 0.102, 0.464),
         'COM': (1.000, 0.000, 1.000), 'STP': (0.940, 0.058, 0.691)},
    15: {'INDEX': (0.999, 0.001, 0.996), 'WHITTLE': (0.847, 0.136, 0.512),
         'COM': (1.000, 0.000, 1.000), 'STP': (0.957, 0.064, 0.779)},
    16: {'INDEX': (0.999, 0.001, 0.997), 'WHITTLE': (0.884, 0.096, 0.633),
         'COM': (1.000, 0.000, 1.000), 'STP': (0.944, 0.055, 0.780)},
}
PERF_STATS = ['mean', 'std', 'worst']

# Table 4: Exact optimality (fraction of instances hitting optimal)
# Paper columns: INDEX, WHITTLE (W), COM (F*,P*)
PAPER_TABLE4 = {
    2: {'INDEX': 0.543, 'WHITTLE': 0.238, 'COM': 0.835},
    3: {'INDEX': 0.614, 'WHITTLE': 0.065, 'COM': 0.850},
    4: {'INDEX': 0.622, 'WHITTLE': 0.023, 'COM': 0.853},
    5: {'INDEX': 0.661, 'WHITTLE': 0.002, 'COM': 0.892},
    6: {'INDEX': 0.686, 'WHITTLE': 0.001, 'COM': 0.904},
    7: {'INDEX': 0.701, 'WHITTLE': 0.000, 'COM': 0.927},
    8: {'INDEX': 0.715, 'WHITTLE': 0.001, 'COM': 0.944},
    9: {'INDEX': 0.733, 'WHITTLE': 0.000, 'COM': 0.943},
}

# =====================================================================
# Tolerances
# =====================================================================

TOL_COVERAGE = 0.05       # absolute on fractions
TOL_PERF_MEAN = 0.03      # absolute on normalized means
TOL_PERF_STD = 0.05       # absolute on std
TOL_PERF_WORST = 0.15     # absolute on worst-case (high variance tail)
TOL_EXACT = 0.10          # absolute on exact optimality fractions
TOL_DP_COUNT_REL = 0.30   # 30% relative on state/revisit counts
TOL_DP_RATIO = 0.10       # absolute on runtime ratio


# =====================================================================
# Helpers
# =====================================================================

def _load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _check_abs(label, paper_val, our_val, tol):
    diff = our_val - paper_val
    ok = abs(diff) <= tol
    tag = "OK" if ok else "FAIL"
    return ok, f"  {label:<45s} paper={paper_val:>8.3f}  ours={our_val:>8.3f}  diff={diff:>+8.3f}  {tag}"


def _check_rel(label, paper_val, our_val, tol):
    if paper_val == 0:
        diff_rel = 0.0 if our_val == 0 else float('inf')
    else:
        diff_rel = (our_val - paper_val) / abs(paper_val)
    ok = abs(diff_rel) <= tol
    tag = "OK" if ok else "FAIL"
    return ok, (f"  {label:<45s} paper={paper_val:>12.1f}  "
                f"ours={our_val:>12.1f}  rel={diff_rel:>+7.1%}  {tag}")


# =====================================================================
# Table validators
# =====================================================================

def validate_table1():
    print("\n" + "=" * 70)
    print("TABLE 1: Coverage of analytical conditions")
    print("=" * 70)

    df = _load_csv('table_1_coverage.csv')
    if df is None:
        print("  SKIP: table_1_coverage.csv not found")
        return None

    n_ok, n_total = 0, 0
    for N, paper_row in sorted(PAPER_TABLE1.items()):
        row = df[df['N'] == N]
        if row.empty:
            continue
        for col, pval in zip(TABLE1_COLS, paper_row):
            oval = float(row[col].iloc[0])
            ok, line = _check_abs(f"N={N}  {col}", pval, oval, TOL_COVERAGE)
            print(line)
            n_ok += ok
            n_total += 1

    passed = n_ok == n_total
    print(f"\n  {'PASS' if passed else 'FAIL'}: {n_ok}/{n_total} cells "
          f"within +/-{TOL_COVERAGE}")
    return passed


def validate_table2():
    print("\n" + "=" * 70)
    print("TABLE 2: Naive vs Structured DP comparison")
    print("=" * 70)

    df = _load_csv('table_2_dp_comparison.csv')
    if df is None:
        print("  SKIP: table_2_dp_comparison.csv not found")
        return None

    n_ok, n_total = 0, 0
    for N, paper_row in sorted(PAPER_TABLE2.items()):
        row = df[df['N'] == N]
        if row.empty:
            continue
        for col, pval in zip(TABLE2_COUNT_COLS, paper_row[:4]):
            oval = float(row[col].iloc[0])
            ok, line = _check_rel(f"N={N}  {col}", pval, oval,
                                  TOL_DP_COUNT_REL)
            print(line)
            n_ok += ok
            n_total += 1
        # Runtime ratio (absolute tolerance)
        pval_ratio = paper_row[4]
        oval_ratio = float(row['runtime_ratio'].iloc[0])
        ok, line = _check_abs(f"N={N}  runtime_ratio",
                              pval_ratio, oval_ratio, TOL_DP_RATIO)
        print(line)
        n_ok += ok
        n_total += 1

    n_paper_rows = len(PAPER_TABLE2)
    n_our_rows = len(df)
    if n_our_rows < n_paper_rows:
        print(f"\n  Note: {n_paper_rows - n_our_rows} paper row(s) not "
              f"computed (N=8,9 skipped to save time)")

    passed = n_ok == n_total
    print(f"\n  {'PASS' if passed else 'FAIL'}: {n_ok}/{n_total} cells "
          f"within tolerance")
    return passed


def validate_table3():
    print("\n" + "=" * 70)
    print("TABLE 3: Policy performance (normalized)")
    print("=" * 70)

    df = _load_csv('table_3_performance.csv')
    if df is None:
        print("  SKIP: table_3_performance.csv not found")
        return None

    tols = {'mean': TOL_PERF_MEAN, 'std': TOL_PERF_STD,
            'worst': TOL_PERF_WORST}

    n_ok, n_total = 0, 0
    for N, policies in sorted(PAPER_TABLE3.items()):
        row = df[df['N'] == N]
        if row.empty:
            continue
        for policy, (p_mean, p_std, p_worst) in policies.items():
            paper_vals = {'mean': p_mean, 'std': p_std, 'worst': p_worst}
            for stat, pval in paper_vals.items():
                col = f"{policy}_{stat}"
                oval_raw = row[col].iloc[0]
                if pd.isna(oval_raw):
                    continue
                oval = float(oval_raw)
                tol = tols[stat]
                ok, line = _check_abs(f"N={N}  {policy}_{stat}",
                                      pval, oval, tol)
                print(line)
                n_ok += ok
                n_total += 1

    passed = n_ok == n_total
    print(f"\n  {'PASS' if passed else 'FAIL'}: {n_ok}/{n_total} cells "
          f"within tolerance (mean +/-{TOL_PERF_MEAN}, "
          f"std +/-{TOL_PERF_STD}, worst +/-{TOL_PERF_WORST})")
    return passed


def validate_table4():
    print("\n" + "=" * 70)
    print("TABLE 4: Exact optimality rates")
    print("=" * 70)

    df = _load_csv('table_4_exact_optimality.csv')
    if df is None:
        print("  SKIP: table_4_exact_optimality.csv not found")
        return None

    n_ok, n_total = 0, 0
    for N, policies in sorted(PAPER_TABLE4.items()):
        row = df[df['N'] == N]
        if row.empty:
            continue
        for policy, pval in policies.items():
            col = f"{policy}_exact_pct"
            oval_raw = row[col].iloc[0]
            if pd.isna(oval_raw):
                continue
            oval = float(oval_raw)
            ok, line = _check_abs(f"N={N}  {policy}", pval, oval, TOL_EXACT)
            print(line)
            n_ok += ok
            n_total += 1

    passed = n_ok == n_total
    print(f"\n  {'PASS' if passed else 'FAIL'}: {n_ok}/{n_total} cells "
          f"within +/-{TOL_EXACT}")
    return passed


def validate_table_ec1():
    print("\n" + "=" * 70)
    print("TABLE EC.1: Runtime ordering (qualitative)")
    print("=" * 70)

    df = _load_csv('table_EC1_runtime.csv')
    if df is None:
        print("  SKIP: table_EC1_runtime.csv not found")
        return None

    n_ok, n_total = 0, 0
    for _, row in df.iterrows():
        N = int(row['N'])
        times = {}
        for col in row.index:
            if col.endswith('_mean') and col != 'N':
                policy = col.replace('_mean', '')
                val = row[col]
                if not pd.isna(val):
                    times[policy] = float(val)

        if not times:
            continue

        if 'OPT' in times and 'INDEX' in times:
            ok = times['OPT'] > times['INDEX']
            tag = "OK" if ok else "FAIL"
            print(f"  N={N}  OPT ({times['OPT']:.2f}s) > "
                  f"INDEX ({times['INDEX']:.2f}s)  {tag}")
            n_ok += ok
            n_total += 1

        if 'WHITTLE' in times and 'INDEX' in times:
            ok = times['WHITTLE'] > times['INDEX']
            tag = "OK" if ok else "FAIL"
            print(f"  N={N}  WHITTLE ({times['WHITTLE']:.2f}s) > "
                  f"INDEX ({times['INDEX']:.2f}s)  {tag}")
            n_ok += ok
            n_total += 1

        if 'STP' in times and 'INDEX' in times:
            ok = times['STP'] < times['WHITTLE'] if 'WHITTLE' in times else True
            tag = "OK" if ok else "FAIL"
            print(f"  N={N}  STP ({times['STP']:.2f}s) < "
                  f"WHITTLE ({times.get('WHITTLE', 0):.2f}s)  {tag}")
            n_ok += ok
            n_total += 1

    passed = n_ok == n_total
    print(f"\n  {'PASS' if passed else 'FAIL'}: {n_ok}/{n_total} "
          f"ordering checks")
    return passed


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("VALIDATION: Comparing experiment output to paper's tables")
    print("=" * 70)
    print(f"Output directory: {os.path.realpath(OUTPUT_DIR)}")
    print(f"Tolerances: coverage={TOL_COVERAGE}, perf_mean={TOL_PERF_MEAN}, "
          f"perf_std={TOL_PERF_STD}, perf_worst={TOL_PERF_WORST}, "
          f"exact={TOL_EXACT}")

    results = {}
    results['Table 1 (coverage)'] = validate_table1()
    results['Table 2 (DP comparison)'] = validate_table2()
    results['Table 3 (performance)'] = validate_table3()
    results['Table 4 (exact optimality)'] = validate_table4()
    results['Table EC.1 (runtime order)'] = validate_table_ec1()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_pass = 0
    n_total = 0
    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
            n_pass += 1
            n_total += 1
        else:
            status = "FAIL"
            n_total += 1
        print(f"  {name:<35s}  {status}")

    print(f"\n  OVERALL: {n_pass}/{n_total} tables PASS")
    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
