# Pandora's Box Problem with Sequential Inspections

This repository contains Python code for the numerical experiments in:

> **The Pandora's Box Problem with Sequential Inspections**
> Ali Aouad, Jingwei Ji, and Yaron Shaposhnik
> *Operations Research*

The code provides:
- A **solver library** (`pandora/`) implementing optimal and heuristic policies for the PSI problem
- **Experiment runners** (`experiments/`) that reproduce all tables and figures from the paper
- **Tests and validation** (`tests/`) comparing generated output against the paper's published results

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the tutorial to see how the library works
python tutorial.py

# Run the full public-replication experiment suite.
# This uses the generated legacy-style box pool by default.
python -m experiments.run_legacy_style_comparison --fresh

# Stable check report up to N=6, with large instances skipped
python -m experiments.run_legacy_style_comparison --preset check --fresh

# Quick diagnostic run
python -m experiments.run_legacy_style_comparison --preset short --fresh
```

## Repository Structure

```
pandora/                    Core solver library
  box.py                    Box class with opening thresholds
  solver.py                 DP solver: naive Bellman and structured DP
  policies.py               Heuristic policies: index, Whittle, STP, committing, Weitzman
  instance_generator.py     Random instance generation and prototypical box pools
  utils.py                  Helper functions

data/
  legacy_box_pools/         Optional old selected box pool for diagnostics

experiments/                Experiment runners
  config.py                 Parameters matching the paper's experimental setup
  parallel.py               Parallel execution engine with crash-recovery checkpointing
  exp_coverage.py           Theorem coverage analysis -> Table 1
  exp_dp_comparison.py      Naive vs structured DP -> Table 2
  exp_policy_benchmark.py   Policy benchmark -> Tables 3, 4, EC.1
  replicate_table4.py       Focused Table 4 replication driver
  exp_p_opening.py          P-opening analysis -> Figures EC.9, EC.10
  figures.py                Matplotlib figure generators
  formatting.py             LaTeX table formatters
  run_all.py                CLI entry point
  run_legacy_style_comparison.py
                            Default public-replication runner and comparison PDF builder

tests/                      Validation and comparison
  test_box_thresholds.py    Unit tests for threshold computations
  test_dp_values.py         Unit tests for DP optimal values
  test_dp_efficiency.py     Tests for structured DP efficiency gains
  test_policies.py          Unit tests for heuristic policies
  validate_against_paper.py Compares generated tables to paper values
  comparison.tex            LaTeX source for visual comparison document
  comparison.pdf            Paper figures/tables vs generated output

tutorial.py                 Step-by-step walkthrough for solving a single instance
requirements.txt            Python dependencies
```

## Using the Library

The tutorial (`tutorial.py`) demonstrates the full workflow. Here is a condensed version:

```python
from pandora.box import Box
from pandora.solver import PandoraSolver
from pandora.policies import index_policy, best_committing_policy

# Define a box: 2 values, 2 types, with inspection costs
box = Box(
    value_list=[2.0, 8.0],
    cond_prob_matrix=[[0.8, 0.2], [0.3, 0.7]],  # Pr[V=v | Type=t]
    type_probs=[0.6, 0.4],
    c_P=0.5,   # cost of partial inspection (reveals type)
    c_F=1.5,   # cost of full inspection (reveals value)
)

# Check thresholds (Eqs. 2–5 in the paper)
print(box.f_threshold)   # σ^F
print(box.p_threshold)   # σ^P
print(box.fp_threshold)  # σ^{F/P}

# Solve a 3-box instance optimally via DP
solver = PandoraSolver([box1, box2, box3])
opt_value = solver.solve_dp()

# Evaluate a heuristic policy
idx_value = solver.evaluate_policy(index_policy)
print(f"Index policy achieves {idx_value / opt_value:.1%} of optimal")

# Find the best committing policy (enumerates F/P partitions)
partition, com_value = best_committing_policy(solver)
```

### Key Classes

| Class / Function | Paper Reference | Description |
|---|---|---|
| `Box` | Definition 1, Eqs. 2–5 | A single box with type structure, inspection costs, and computed thresholds |
| `PandoraSolver` | Eq. 1, Theorems 1–3 | Solves a PSI instance via naive or structured DP |
| `index_policy` | Section 5.1 | Index-based heuristic using threshold ordering |
| `whittle_policy` | Section 5.2 | Whittle's integral policy |
| `stp_policy` | Section 5.3 | Single-type policy |
| `best_committing_policy` | Section 5.4 | Best committing policy π^{F*,P*} |
| `weitzman_policy` | Section 3 | Weitzman's original policy (ignores P-opening) |

### Generating Random Instances

```python
import random
from pandora.instance_generator import generate_legacy_style_prototypical_boxes

# Generate the default public-replication pool from code.
# This does not read the archived old pickle or the bundled old JSON pool.
unfiltered_boxes = generate_legacy_style_prototypical_boxes(
    n_boxes=100,
    distance=0.5,
    rng=random.Random(0),
    require_p_dominant=False,
)
boxes = [box for box in unfiltered_boxes if box.c_F > box.c_P]

# Sample N boxes from the pool for an instance
import numpy as np
rng = np.random.RandomState(42)
instance = [boxes[i] for i in rng.randint(0, len(boxes), size=5)]
```

## Reproducing the Paper's Experiments

### What Gets Generated

| Output File | Paper Reference |
|---|---|
| `table_1_coverage.csv/.tex` | Table 1 — Coverage rates of Theorems 1–3 |
| `table_2_dp_comparison.csv/.tex` | Table 2 — Naive vs structured DP (states, revisits, runtime) |
| `table_3_performance.csv/.tex` | Table 3 — Normalized policy performance (mean, std, worst) |
| `table_4_exact_optimality.csv/.tex` | Table 4 — Fraction of instances achieving optimality, with only the paper columns |
| `table_EC1_runtime.csv/.tex` | Table EC.1 — Policy runtimes |
| `figure_3_prototypical_boxes.png` | Figure 3 — Prototypical box scatter plot |
| `figure_EC8_*.png` | Figure EC.8 — Box selection visualization |
| `figure_EC9a_*.png`, `figure_EC9b_*.png` | Figure EC.9 — P-ratio and Weitzman performance |
| `figure_EC10a_*.png` through `figure_EC10d_*.png` | Figure EC.10 — Dispersion and "more boxes" analysis |

All output is written to the `output/` directory.

### Running Experiments

```bash
# Full public-replication run.
# Default: legacy-style generated mixed pool, full preset, 12 workers.
python -m experiments.run_legacy_style_comparison --fresh

# Resume the same run from checkpoints
python -m experiments.run_legacy_style_comparison

# Quick diagnostic run
python -m experiments.run_legacy_style_comparison --preset short --fresh

# Stable check report up to N=6, with no large-instance rows
python -m experiments.run_legacy_style_comparison --preset check --fresh

# Run a single experiment through the general runner.
# This also defaults to the legacy-style generated mixed pool.
python -m experiments.run_all -e coverage          # Table 1
python -m experiments.run_all -e dp_comparison      # Table 2
python -m experiments.run_all -e policy_benchmark   # Tables 3, 4, EC.1
python -m experiments.run_all -e p_opening          # Figures EC.9, EC.10
python -m experiments.run_all -e box_scatter        # Figures 3, EC.8

# Focused Table 4 replication from the default generated legacy-style pool
python -m experiments.replicate_table4 --n-range 2:9

# Focused Table 4 probe generated from scratch with the old Experiment.py
# prototype-selection rule. This does not read the bundled old pool.
python -m experiments.replicate_table4 --pool-source legacy-generated --prototype-boxes 40 --n-range 2:5

# Focused Table 4 probe generated from scratch with the old candidate-blocking
# rule and the mixed threshold order observed in the stored old pool.
python -m experiments.replicate_table4 --pool-source legacy-generated-mixed --prototype-boxes 100 --n-range 2:5

# Use the newer reorganized generator instead of the legacy-style generator
python -m experiments.run_all --pool-source generated

# Optional diagnostic: focused Table 4 replication from the bundled old pool
python -m experiments.replicate_table4 --pool-source old --n-range 2:9
```

### CLI Options

| Flag | Description |
|---|---|
| `--experiment`, `-e` | Run a specific experiment (`coverage`, `dp_comparison`, `policy_benchmark`, `p_opening`, `box_scatter`, or `all`) |
| `--small` | Quick test with reduced N range and fewer instances |
| `--workers N`, `-w N` | Number of parallel worker processes (default: 3) |
| `--fresh` | Clear all checkpoints and start from scratch |
| `--pool-source legacy-generated-mixed\|legacy-generated\|generated\|old` | Pool for Tables 1-3, EC.1, and the main figures. Default: `legacy-generated-mixed` |
| `--prototype-boxes N` | Number of generated prototypes in legacy-generated pool modes. Default: 100 |
| `--legacy-max-attempts N` | Maximum random draws for legacy-style prototype generation |
| `--table4-pool-source benchmark\|old` | Pool for Table 4. Default: `benchmark`, the same generated pool used by Tables 3/EC.1 |
| `--legacy-pool-dir PATH` | Override the bundled legacy pool directory |
| `--table4-n-range SPEC` | Override Table 4 N range, e.g. `2:5` |
| `--table4-reps N` | Override Table 4 instances per N |
| `experiments.replicate_table4 --pool-source legacy-generated` | Generate a Table 4 test pool using the old `Experiment.py` candidate-blocking rule |
| `experiments.replicate_table4 --pool-source legacy-generated-mixed` | Same candidate-blocking rule, but without the archived source's `sigma^P > sigma^F` prefilter |
| `experiments.replicate_table4 --prototype-boxes N` | Number of generated prototypes for focused Table 4 runs |
| `experiments.run_legacy_style_comparison` | Full paper-scale run using generated legacy-style boxes for all pool-based outputs and a comparison PDF |
| `experiments.run_legacy_style_comparison --preset check` | Stable no-large-instance check report: N=2..6, 100 replications per N, Figure EC.10d N=1..6 |
| `experiments.run_legacy_style_comparison --preset short` | Short diagnostic run using generated legacy-style boxes for all pool-based outputs |

### Parallel Execution and Crash Recovery

Experiments run in parallel using Python's `ProcessPoolExecutor`. Intermediate
results are checkpointed to `output/.checkpoints/` as JSONL files. If a run is
interrupted (crash, Ctrl-C, SSH disconnect), re-running the same command
automatically resumes from the last checkpoint:

```bash
python -m experiments.run_legacy_style_comparison --fresh    # interrupted at 40%
python -m experiments.run_legacy_style_comparison            # resumes from checkpoint
```

Use `--fresh` to discard checkpoints and restart. This is necessary when
changing parameters (e.g., switching between `--small` and full runs).

For the legacy-style comparison runner, use the same command without `--fresh`
to resume from checkpoints. To rebuild only the comparison PDF after a completed
or partially completed run, use:

```bash
python -m experiments.run_legacy_style_comparison --skip-experiments
```

### Runtime Estimates

With 4 parallel workers on a modern machine:

| Experiment | Approximate Time |
|---|---|
| Coverage (Table 1) | ~2 hours |
| DP comparison (Table 2, N≤7) | ~3 hours |
| Policy benchmark (Tables 3, 4, EC.1, N≤14) | ~45 hours |
| P-opening analysis (Figures EC.9, EC.10) | ~2 hours |
| **Total** | **~50 hours** |

The `--small` flag reduces this to ~30 minutes for a quick sanity check.

### Full Legacy-Style Paper Run

To generate all paper and appendix tables/figures using the recovered
legacy-style prototype pool, run:

```bash
python -m experiments.run_legacy_style_comparison --fresh
```

The full preset uses:

| Setting | Value |
|---|---|
| Prototype generation | `legacy-generated-mixed` |
| Prototypes requested | 100 |
| Main pool filter | `c_F > c_P` |
| DP-based N range | 2--9 |
| DP-based replications | 200 per N |
| Heuristic-only N range | 10--16 |
| Heuristic-only replications | 100 per N |
| Figure EC.10d N range | 1--7 |
| Workers | 12 |

For a smaller but still stable check report, run:

```bash
python -m experiments.run_legacy_style_comparison --preset check --fresh
```

The check preset skips large-instance rows and uses `N=2..6` with 100
replications per `N`, so Table 4 is much less noisy than a 5-rep smoke test.
It writes to `output/legacy_style_check/` and builds
`comparison_legacy_style_check.pdf`.

Outputs are written to `output/legacy_style_full/`, including:

- `table_1_coverage.csv/.tex`
- `table_2_dp_comparison.csv/.tex`
- `table_3_performance.csv/.tex`
- `table_4_exact_optimality.csv/.tex`
- `table_EC1_runtime.csv/.tex`
- `figure_3_prototypical_boxes.png`
- `figure_EC8_prototypical_boxes_selection.png`
- `figure_EC9a_p_ratio_vs_proportion.png`
- `figure_EC9b_weitzman_vs_proportion.png`
- `figure_EC10a_low_dispersion.png`
- `figure_EC10b_high_dispersion.png`
- `figure_EC10c_p_ratio_vs_dispersion.png`
- `figure_EC10d_p_ratio_vs_num_boxes.png`
- `comparison_legacy_style_full.pdf`

The comparison PDF always includes paper reference tables. To also include the
paper's original figure PNGs side by side, pass their directory:

```bash
python -m experiments.run_legacy_style_comparison --fresh \
  --reference-figure-dir PATH_TO_PAPER_EXPERIMENTS_GRAPHS \
  --reference-figure3 PATH_TO_PAPER_FIGURE_3_PNG
```

### Prototypical Box Pools

The default public-replication pool is generated from code with
`--pool-source legacy-generated-mixed`.  This is the default for both
`experiments.run_legacy_style_comparison` and `experiments.run_all`; it does not
read the archived old pickle, the old folder, or the optional bundled old JSON
pool.

The default pool construction is:

1. Draw candidate boxes using the recovered old `Experiment.py` generators. The
   positive-threshold candidate counter determines the mixture: every third
   valid candidate uses the 2-value/2-type generator; the other valid
   candidates use the 5-value/3-type generator.
2. Keep a candidate as a prototype only if no previous valid candidate is within
   Euclidean distance 0.5 in `(sigma^P, sigma^F)` space.  The default mixed mode
   does not impose `sigma^P > sigma^F` before prototype selection.
3. Stop after selecting 100 prototypes.  Tables 1-4, Table EC.1, and Figure 3
   use the subset satisfying `c_F > c_P`.  The P-opening figures use the
   unfiltered selected prototypes.
4. Sample instance boxes from the selected pool with replacement using the old
   notebook stream, `np.random.RandomState(seed).randint(...)`.

Other pool modes are available for diagnostics:

- `--pool-source legacy-generated` uses the same old candidate-blocking rule
  but also imposes the archived source condition `sigma^P > sigma^F`.
- `--pool-source generated` uses the newer reorganized generator.
- `--pool-source old` loads the optional bundled old selected pool under
  `data/legacy_box_pools/`.
- `--table4-pool-source old` runs Table 4 from the optional bundled old pool
  instead of the benchmark pool.

The paper describes the high-level experiment as sampling randomized copies from
a prototypical pool with `c_F >= c_P`.  The threshold-space blocking rule, the
5-value/3-type versus 2-value/2-type mixture, and the old RNG stream are
recovered implementation details from the old code and are now generated
directly in this repository.

To force Table 4 to use the same benchmark pool explicitly, run:

```bash
python -m experiments.run_all --table4-pool-source benchmark
```

## Validating Results

After running experiments, you can compare the output against the paper's
published values:

```bash
# Automated tolerance-based comparison against paper tables
python tests/validate_against_paper.py

# Visual side-by-side comparison (precompiled PDF in tests/)
# To regenerate after re-running experiments:
cd tests && tectonic comparison.tex
```

The visual comparison document (`tests/comparison.pdf`) places each paper table
and figure above the corresponding generated output for easy manual inspection.

**Note on reproducibility:** The default public-replication path uses the
legacy-style pool generated from code, then samples from that pool with the old
notebook RNG stream.  The optional bundled old pool is retained only for
diagnostics; it is not required to reproduce the default results.

## Running Unit Tests

```bash
pytest tests/ -v
```

The unit tests validate threshold computations, DP values, structured DP
efficiency, and heuristic policy implementations against reference instances.

## Notation Reference

The code uses the paper's notation throughout:

| Symbol | Code | Meaning |
|---|---|---|
| c_F | `c_F` | F-opening (full inspection) cost |
| c_P | `c_P` | P-opening (partial inspection) cost |
| σ^F | `f_threshold` | F-threshold (Eq. 2) |
| σ^{F\|t} | `f_thresholds_by_type[t]` | Conditional F-threshold (Eq. 3) |
| σ^P | `p_threshold` | P-threshold (Eq. 4) |
| σ^{F/P} | `fp_threshold` | FP-threshold (Eq. 5) |

### Dispersion Metric

The dispersion measure used in Figures EC.10a–c is Var(σ^F) + Var(σ^P) computed
across the boxes in each instance. The paper's text describes this as "sum of
standard deviations," but the code computes the sum of variances, which matches
the formula used to generate the published figures.

## Experimental Parameters

All parameters are defined in `experiments/config.py` and match the paper's
Section 6:

| Parameter | Value | Description |
|---|---|---|
| Support size | 5 | Number of possible prize values per box |
| Max types | 3 | Maximum number of types per box |
| Values | U(0, 10) | Prize value distribution |
| c_F | U(0, 5) | F-opening cost distribution |
| c_P | U(0, 3) | P-opening cost distribution, with c_P ≤ c_F |
| Small instances | 1000 | Instances per N for N = 2, ..., 9 |
| Large instances | 300 | Instances per N for N = 10, ..., 16 |
| Prototypical boxes | 100 | Size of the box pool |
| Seed | 0 | Random seed for reproducibility |
