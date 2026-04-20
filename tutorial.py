"""
Tutorial: Solving a Single PSI Problem Instance
================================================

This script walks through how to use the library to define and solve
a single instance of the Pandora's Box Problem with Sequential
Inspections (PSI), as formulated in the paper:

    "The Pandora's Box Problem with Sequential Inspections"
    by Ali Aouad, Jingwei Ji, and Yaron Shaposhnik

It demonstrates:
  1. Defining boxes with type structure and inspection costs (Definition 1)
  2. Computing opening thresholds σ^F, σ^P, σ^{F/P} (Eqs. 2–5)
  3. Solving an instance optimally via dynamic programming (Eq. 1)
  4. Evaluating heuristic policies (Section 5)
  5. Tracing the optimal policy on a sample realization

Run from the project root:
    python tutorial.py
"""

import numpy as np
from pandora.box import Box
from pandora.solver import PandoraSolver
from pandora.policies import (
    index_policy, whittle_policy, stp_policy,
    weitzman_policy, best_committing_policy,
)


# =====================================================================
# Step 1: Define boxes
# =====================================================================
#
# Each box has:
#   - value_list:        possible prize values (support of V_i)
#   - cond_prob_matrix:  Pr[V = v | Type = t] for each type t (rows)
#                        and each value v (columns)
#   - type_probs:        prior probability of each type
#   - c_P:               cost of P-opening (partial inspection, reveals type)
#   - c_F:               cost of F-opening (full inspection, reveals value)

# Box 1: Two possible values, two types.
#   Values: 2.0 or 8.0
#   Type A (prob 0.6): likely low value    — Pr[V=2]=0.8, Pr[V=8]=0.2
#   Type B (prob 0.4): likely high value   — Pr[V=2]=0.3, Pr[V=8]=0.7
box1 = Box(
    value_list=[2.0, 8.0],
    cond_prob_matrix=[
        [0.8, 0.2],   # Type A
        [0.3, 0.7],   # Type B
    ],
    type_probs=[0.6, 0.4],
    c_P=0.5,
    c_F=1.5,
)

# Box 2: Three possible values, two types.
box2 = Box(
    value_list=[1.0, 5.0, 9.0],
    cond_prob_matrix=[
        [0.5, 0.3, 0.2],  # Type A
        [0.1, 0.3, 0.6],  # Type B
    ],
    type_probs=[0.5, 0.5],
    c_P=0.4,
    c_F=1.0,
)

# Box 3: A simple box with one type (no informative partial inspection).
box3 = Box(
    value_list=[3.0, 7.0],
    cond_prob_matrix=[
        [0.4, 0.6],
    ],
    type_probs=[1.0],
    c_P=0.2,
    c_F=0.8,
)

print("=" * 60)
print("Box Thresholds")
print("=" * 60)
for i, box in enumerate([box1, box2, box3], 1):
    print(f"\n  Box {i}: {box}")
    print(f"    F-threshold (σ^F):  {box.f_threshold:.4f}")
    print(f"    P-threshold (σ^P):  {box.p_threshold:.4f}")
    print(f"    FP-threshold (σ^{{F/P}}): {box.fp_threshold:.4f}")
    for t in range(box.T):
        print(f"    F-threshold | type {t+1} (σ^{{F|{t+1}}}): "
              f"{box.f_thresholds_by_type[t]:.4f}")


# =====================================================================
# Step 2: Solve the instance optimally via dynamic programming
# =====================================================================

instance = [box1, box2, box3]
solver = PandoraSolver(instance)

opt_value = solver.solve_dp()

print("\n" + "=" * 60)
print("Optimal Solution (Dynamic Programming)")
print("=" * 60)
print(f"\n  Optimal expected value: {opt_value:.4f}")

# Show the optimal first action from the initial state [y=0, s1=0, s2=0, s3=0]
initial_state = [0, 0, 0, 0]
first_action = solver.get_dp_action(initial_state)
print(f"  Optimal first action:  {first_action}")

# Count expected number of openings under the optimal policy
n_f, n_p = solver.expected_openings('DP')
print(f"  Expected F-openings:   {n_f:.4f}")
print(f"  Expected P-openings:   {n_p:.4f}")


# =====================================================================
# Step 3: Evaluate heuristic policies
# =====================================================================

policies = {
    'Index':   index_policy,
    'Whittle': whittle_policy,
    'STP':     stp_policy,
    'Weitzman': weitzman_policy,
}

print("\n" + "=" * 60)
print("Heuristic Policy Comparison")
print("=" * 60)
print(f"\n  {'Policy':<12} {'Value':>10} {'% of OPT':>10}")
print(f"  {'-'*12} {'-'*10} {'-'*10}")
print(f"  {'OPT':<12} {opt_value:10.4f} {'100.00%':>10}")

for name, policy_fn in policies.items():
    val = solver.evaluate_policy(policy_fn)
    pct = 100.0 * val / opt_value if opt_value > 0 else 100.0
    print(f"  {name:<12} {val:10.4f} {pct:9.2f}%")

# Best committing policy (enumerates all F/P partitions)
best_partition, best_com_val = best_committing_policy(solver)
pct = 100.0 * best_com_val / opt_value if opt_value > 0 else 100.0
print(f"  {'Committing':<12} {best_com_val:10.4f} {pct:9.2f}%")
print(f"\n  Best committing partition: {best_partition}")
print(f"  (F = always F-open, P = P-open first then F-open)")


# =====================================================================
# Step 4: Trace the optimal policy on a sample realization
# =====================================================================
#
# Simulate one realization to see the decision sequence.
# We draw random types and values, then follow the optimal policy.

print("\n" + "=" * 60)
print("Sample Realization (following the optimal policy)")
print("=" * 60)

rng = np.random.default_rng(42)

# Draw types and values for each box
realized_types = []
realized_values = []
for box in instance:
    t = int(rng.choice(box.T, p=box.type_probs))
    v = float(rng.choice(box.value_list, p=box.cond_prob_matrix[t]))
    realized_types.append(t)
    realized_values.append(v)

print(f"\n  Realized types:  {[t+1 for t in realized_types]}")
print(f"  Realized values: {realized_values}")

state = list(initial_state)
step = 0
total_cost = 0.0

print(f"\n  {'Step':<6} {'State':<24} {'Action':<14} {'Cost':>6} {'y':>6}")
print(f"  {'-'*6} {'-'*24} {'-'*14} {'-'*6} {'-'*6}")

while True:
    action = solver.get_dp_action(state)
    if action is None:
        action = "STOP"

    step += 1
    step_cost = 0.0
    print(f"  {step:<6} {str(state):<24} {action:<14}", end="")

    if action == "STOP":
        print(f" {'':>6} {state[0]:6.1f}")
        break

    if action.startswith("P"):
        box_idx = int(action.split("_")[1]) - 1
        step_cost = instance[box_idx].c_P
        total_cost += step_cost
        state[box_idx + 1] = realized_types[box_idx] + 1
        print(f" {step_cost:6.2f} {state[0]:6.1f}")

    elif action.startswith("F"):
        if "|" in action:
            box_idx = int(action.split("_")[-1]) - 1
        else:
            box_idx = int(action.split("_")[1]) - 1
        step_cost = instance[box_idx].c_F
        total_cost += step_cost
        state[box_idx + 1] = -1
        state[0] = float(max(state[0], realized_values[box_idx]))
        print(f" {step_cost:6.2f} {state[0]:6.1f}")

prize = state[0]
print(f"\n  Final prize: {prize:.1f}")
print(f"  Total inspection cost: {total_cost:.2f}")
print(f"  Net payoff: {prize - total_cost:.2f}")
