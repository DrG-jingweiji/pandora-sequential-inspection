# Pandora's Box Problem with Sequential Inspections (PSI)

Numerical experiments for *The Pandora's Box Problem with Sequential Inspections*
by Ali Aouad, Jingwei Ji, and Yaron Shaposhnik.

## Structure

- `pandora/` — Core library
  - `box.py` — Box class with opening thresholds (Definition 1, Eqs. 2–5)
  - `solver.py` — DP solver: naive Bellman (Eq. 1) and structured DP (Theorems 1–3)
  - `policies.py` — Heuristic policies: index, Whittle, STP, committing, Weitzman
  - `instance_generator.py` — Instance generation and prototypical box pools
  - `utils.py` — Helper functions (dispersion, P-dominant ratio)
- `experiments/` — Experiment runners replicating paper tables/figures
  - `config.py` — Shared configuration (instance counts, ranges, defaults)
  - `parallel.py` — Parallel execution engine with crash-recovery checkpointing
  - `exp_coverage.py` — Theorem coverage analysis (Table 1)
  - `exp_dp_comparison.py` — Naive vs structured DP (Table 2)
  - `exp_policy_benchmark.py` — Policy benchmark (Tables 3, 4, EC.1)
  - `exp_p_opening.py` — P-opening analysis (Figures EC.9, EC.10)
  - `formatting.py` — LaTeX table formatters
  - `figures.py` — Matplotlib figure generators
  - `run_all.py` — CLI entry point
- `tests/` — Validation tests comparing old and new code output
- `output/` — Generated results (CSVs, LaTeX, figures)
- `tutorial.py` — Walkthrough: define boxes, solve an instance, compare policies

## Output

The experiments produce the following files in `output/`:

| File | Paper reference |
|------|----------------|
| `table_1_coverage.csv/.tex` | Table 1 — Coverage of analytical conditions |
| `table_2_dp_comparison.csv/.tex` | Table 2 — Naive vs structured DP |
| `table_3_performance.csv/.tex` | Table 3 — Policy performance |
| `table_4_exact_optimality.csv/.tex` | Table 4 — Exact optimality rates |
| `table_EC1_runtime.csv/.tex` | Table EC.1 — Runtime comparison |
| `figure_3_prototypical_boxes.png` | Figure 3 — Prototypical boxes (P-dominant pool) |
| `figure_EC8_prototypical_boxes_selection.png` | Figure EC.8 — Box selection (mixed pool) |
| `figure_EC9a_p_ratio_vs_proportion.png` | Figure EC.9 left — P-ratio vs proportion |
| `figure_EC9b_weitzman_vs_proportion.png` | Figure EC.9 right — Weitzman suboptimality |
| `figure_EC10a_low_dispersion.png` | Figure EC.10 top-left — Low-dispersion example |
| `figure_EC10b_high_dispersion.png` | Figure EC.10 top-right — High-dispersion example |
| `figure_EC10c_p_ratio_vs_dispersion.png` | Figure EC.10 bottom-left — P-ratio vs dispersion |
| `figure_EC10d_p_ratio_vs_num_boxes.png` | Figure EC.10 bottom-right — P-ratio vs N (i.i.d.) |
| `p_opening_analysis.csv` | Per-instance P-opening metrics |

## Usage

```bash
pip install -r requirements.txt

# Solve a single instance (tutorial walkthrough)
python tutorial.py

# Run all experiments (3 parallel workers by default)
python -m experiments.run_all

# Run a specific experiment
python -m experiments.run_all --experiment coverage

# Quick test run with fewer instances
python -m experiments.run_all --small

# Run tests
pytest tests/ -v
```

### Parallel Execution

Problem instances are solved in parallel using Python's `ProcessPoolExecutor`.
The number of worker processes defaults to 3 and can be changed with `--workers`:

```bash
# Use 6 parallel workers
python -m experiments.run_all --workers 6

# Run sequentially (useful for debugging)
python -m experiments.run_all --workers 1
```

### Crash Recovery

Intermediate results are checkpointed to `output/.checkpoints/` as JSONL files.
If a run is interrupted (crash, Ctrl-C, system restart), simply re-run the same
command and it will **automatically resume** from where it stopped:

```bash
# First run — interrupted at 40%
python -m experiments.run_all
# ^C

# Second run — resumes from the checkpoint
python -m experiments.run_all
#  Checkpoint: 3200/8000 done, 4800 remaining
```

To discard all checkpoints and start fresh:

```bash
python -m experiments.run_all --fresh
```

**Note:** Checkpoints are tied to the experiment parameters (N range, instance
count, seed). If you change parameters (e.g., switch between `--small` and full
runs), use `--fresh` to avoid mixing results from different configurations.

## CLI Reference

| Flag | Description |
|------|-------------|
| `--experiment`, `-e` | Run a specific experiment: `coverage`, `dp_comparison`, `policy_benchmark`, `p_opening`, `box_scatter`, or `all` (default) |
| `--small` | Quick test with reduced N range and fewer instances |
| `--workers N`, `-w N` | Number of parallel processes (default: 3) |
| `--fresh` | Clear checkpoints before running |

## Prototypical Box Pools

The experiments use two separate pools of prototypical boxes, generated with
the same seed but different selection criteria:

1. **P-dominant pool** (`require_p_dominant=True`) — Only boxes with σ^P > σ^F
   are selected. Used for the main experiments (Tables 1–4, EC.1) and Figure 3.
2. **Mixed pool** (`require_p_dominant=False`) — Boxes are selected by distance
   in (σ^F, σ^P) space regardless of which threshold is larger. Used for the
   P-opening analysis (Figures EC.8–EC.10) where the relationship between σ^P
   and σ^F is the quantity of interest.

Both pools apply the same distance-based diversity criterion and positive-threshold
filter. The distinction matches the original experimental setup in the paper.

## Experiment Details

### "More Boxes" Experiment (Figure EC.10d)

This experiment uses a single fixed box specification (3 types, 6 values) with
c_F = 3.0 and varies c_P from 0.1 to 0.6 in 6 steps. For each c_P level,
N = 1, ..., 7 identical copies of the box are created, the optimal policy is
computed via DP, and the P-ratio N_P / (N_P + N_F) is plotted. The resulting
figure shows one line per c_P level.

### Dispersion Metric

The dispersion measure used in Figures EC.10a–c is Var(σ^F) + Var(σ^P) computed
across the boxes in each instance. The paper's text describes this as "sum of
standard deviations", but both the original and new code compute the sum of
*variances*, which is the formula used to generate the published figures.

## Notation

The code uses the paper's F/P notation throughout:

| Symbol | Meaning |
|--------|---------|
| `c_F` | F-opening (full inspection) cost |
| `c_P` | P-opening (partial inspection) cost |
| `f_threshold` | F-threshold σ^F (Eq. 2) |
| `f_threshold_given_type[t]` | Conditional F-threshold σ^{F\|t} (Eq. 3) |
| `p_threshold` | P-threshold σ^P (Eq. 4) |
| `fp_threshold` | FP-threshold σ^{F/P} (Eq. 5) |
