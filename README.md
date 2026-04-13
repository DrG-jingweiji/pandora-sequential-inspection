# Pandora's Box Problem with Sequential Inspections (PSI)

Numerical experiments for *The Pandora's Box Problem with Sequential Inspections*
by Ali Aouad, Jingwei Ji, and Yaron Shaposhnik.

## Structure

- `pandora/` — Core library
  - `box.py` — Box class with opening thresholds (Definition 1, Eqs. 2–5)
  - `solver.py` — DP solver: naive Bellman (Eq. 1) and structured DP (Theorems 1–3)
  - `policies.py` — Heuristic policies: index, Whittle, STP, committing, Weitzman
  - `instance_generator.py` — Instance generation and JSON I/O
  - `utils.py` — Helper functions
- `experiments/` — Experiment runners replicating paper tables/figures
  - `exp_coverage.py` — Theorem coverage analysis (Table 5)
  - `exp_dp_comparison.py` — Naive vs structured DP (Table 6)
  - `exp_policy_benchmark.py` — Policy benchmark (Tables 7–8)
  - `exp_p_opening.py` — P-opening analysis (Appendix)
  - `run_all.py` — CLI entry point
- `tests/` — Validation tests comparing old and new code output
- `output/` — Generated results (CSVs, figures)

## Usage

```bash
pip install -r requirements.txt

# Run all experiments
python -m experiments.run_all

# Run a specific experiment
python -m experiments.run_all --experiment coverage

# Run tests
pytest tests/ -v
```

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
