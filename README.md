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

# Run all numerical experiments (takes ~50 hours with 4 workers)
python -m experiments.run_all --workers 4

# Quick test run (~30 min) with smaller instances
python -m experiments.run_all --small
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
  legacy_box_pools/         Bundled old selected box pool used by Table 4

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
from pandora.instance_generator import generate_prototypical_boxes

# Generate a pool of diverse prototypical boxes (same method as the paper)
boxes = generate_prototypical_boxes(
    n_boxes=100,          # pool size
    min_distance=0.5,     # diversity threshold in (σ^F, σ^P) space
    require_p_dominant=True,  # only keep boxes with σ^P > σ^F
)

# Sample N boxes from the pool for an instance
import numpy as np
rng = np.random.default_rng(42)
instance = [boxes[i] for i in rng.integers(0, len(boxes), size=5)]
```

## Reproducing the Paper's Experiments

### What Gets Generated

| Output File | Paper Reference |
|---|---|
| `table_1_coverage.csv/.tex` | Table 1 — Coverage rates of Theorems 1–3 |
| `table_2_dp_comparison.csv/.tex` | Table 2 — Naive vs structured DP (states, revisits, runtime) |
| `table_3_performance.csv/.tex` | Table 3 — Normalized policy performance (mean, std, worst) |
| `table_4_exact_optimality.csv/.tex` | Table 4 — Fraction of instances achieving optimality |
| `table_EC1_runtime.csv/.tex` | Table EC.1 — Policy runtimes |
| `figure_3_prototypical_boxes.png` | Figure 3 — Prototypical box scatter plot |
| `figure_EC8_*.png` | Figure EC.8 — Box selection visualization |
| `figure_EC9a_*.png`, `figure_EC9b_*.png` | Figure EC.9 — P-ratio and Weitzman performance |
| `figure_EC10a_*.png` through `figure_EC10d_*.png` | Figure EC.10 — Dispersion and "more boxes" analysis |

All output is written to the `output/` directory.

### Running Experiments

```bash
# Full run (all experiments)
python -m experiments.run_all

# Run a single experiment
python -m experiments.run_all -e coverage          # Table 1
python -m experiments.run_all -e dp_comparison      # Table 2
python -m experiments.run_all -e policy_benchmark   # Tables 3, 4, EC.1
python -m experiments.run_all -e p_opening          # Figures EC.9, EC.10
python -m experiments.run_all -e box_scatter        # Figures 3, EC.8

# Focused Table 4 replication from the bundled old pool
python -m experiments.replicate_table4 --pool-source old --n-range 2:5
```

### CLI Options

| Flag | Description |
|---|---|
| `--experiment`, `-e` | Run a specific experiment (`coverage`, `dp_comparison`, `policy_benchmark`, `p_opening`, `box_scatter`, or `all`) |
| `--small` | Quick test with reduced N range and fewer instances |
| `--workers N`, `-w N` | Number of parallel worker processes (default: 3) |
| `--fresh` | Clear all checkpoints and start from scratch |
| `--pool-source generated\|old` | Pool for Tables 1-3, EC.1, and the main figures. Default: `generated` |
| `--table4-pool-source old\|benchmark` | Pool for Table 4. Default: `old`, the bundled legacy pool |
| `--legacy-pool-dir PATH` | Override the bundled legacy pool directory |
| `--table4-n-range SPEC` | Override Table 4 N range, e.g. `2:5` |
| `--table4-reps N` | Override Table 4 instances per N |

### Parallel Execution and Crash Recovery

Experiments run in parallel using Python's `ProcessPoolExecutor`. Intermediate
results are checkpointed to `output/.checkpoints/` as JSONL files. If a run is
interrupted (crash, Ctrl-C, SSH disconnect), re-running the same command
automatically resumes from the last checkpoint:

```bash
python -m experiments.run_all --workers 4    # interrupted at 40%
python -m experiments.run_all --workers 4    # resumes from checkpoint
```

Use `--fresh` to discard checkpoints and restart. This is necessary when
changing parameters (e.g., switching between `--small` and full runs).

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

### Prototypical Box Pools

By default, `experiments.run_all` uses generated boxes for every experiment
except Table 4:

1. **Generated P-dominant pool** (`require_p_dominant=True`) — Used for
   Tables 1, 2, 3, EC.1, and Figure 3.
2. **Generated mixed pool** (`require_p_dominant=False`) — Used for the
   P-opening analysis and Figures EC.8–EC.10.
3. **Bundled old selected pool** (`data/legacy_box_pools/`) — Used for
   Table 4 by default. The bundled file contains the old 100 selected
   prototypical boxes; the loader applies the old `c_F > c_P` filter, leaving
   the 56-box pool used in the original Table 4 run.

This default keeps the reorganized/generated experiments for most outputs, but
uses the old box pool for Table 4 because exact-optimality rates are especially
sensitive to the selected boxes.

To make Table 4 use the same generated pool as Table 3/EC.1, run:

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

**Note on reproducibility:** Tables 1, 2, 3, EC.1, and the figures use generated
box pools by default, so they may differ from the historical run when a result is
sensitive to the exact selected boxes. Table 4 uses the bundled old selected pool
by default and is intended to match the original Table 4 much more closely.

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
