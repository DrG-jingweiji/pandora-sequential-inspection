"""
Heuristic policies for the PSI problem.

Each policy is a callable: policy_fn(solver, state) -> action string.
Action format: "STOP", "F_{i}", "F|{t}_{i}", "P_{i}" (i is 1-indexed).

Policies implemented:
  - index_policy: Greedy threshold index (Section 6)
  - whittle_policy: One-step lookahead using Whittle integral J^W (Eq. 11)
  - stp_policy: Single Test Policy — myopic single-box improvement (Section 6)
  - committing_policy: Evaluate a fixed (F, P) partition (Algorithm 1)
  - best_committing_policy: Enumerate all 2^N partitions, return best
  - weitzman_policy: Committing with all boxes in F (ignore P-opening)
"""

import copy
import itertools
import numpy as np


# ======================================================================
# Index Policy (Section 6)
# ======================================================================

def index_policy(solver, state):
    """At each state, pick the action with the largest threshold index.

    Compare y (stop) against:
      - σ^F_i for F-opening closed box i
      - σ^{F|t}_i for F-opening partially opened box i
      - σ^P_i for P-opening closed box i

    Old code: Pandora_Unified.Pandora.largest_threshold + INDEX branch.
    """
    lead_idx, lead_mode, lead_th = solver.sigma_M(state)

    if lead_idx is None or state[0] >= lead_th:
        return "STOP"

    i = lead_idx
    s_i = state[i + 1]

    if lead_mode == 'P':
        return f"P_{i + 1}"
    else:
        if s_i == 0:
            return f"F_{i + 1}"
        else:
            return f"F|{s_i}_{i + 1}"


# ======================================================================
# Whittle One-Step Lookahead (Section 5.2 / Section 6)
# ======================================================================

def whittle_policy(solver, state):
    """One-step lookahead using Whittle integral J^W (Eq. 11).

    For each possible action, compute -cost + E[J^W(successor)].
    Pick the action that maximizes this, or stop with y.
    """
    y = state[0]
    best_val = y
    best_action = "STOP"

    for i in range(solver.N):
        box = solver.box_list[i]
        s_i = state[i + 1]

        # P-open closed box
        if s_i == 0:
            val = -box.c_P
            for t in range(box.T):
                ns = list(state)
                ns[i + 1] = t + 1
                val += box.type_probs[t] * _whittle_integral(solver, ns)
            if val > best_val:
                best_val = val
                best_action = f"P_{i + 1}"

        # F-open closed or partial box
        if s_i >= 0:
            val = -box.c_F
            for k in range(box.S):
                ns = list(state)
                ns[i + 1] = -1
                ns[0] = max(ns[0], box.value_list[k])
                val += box.prob_matrix[s_i, k] * _whittle_integral(solver, ns)
            if val > best_val:
                best_val = val
                if s_i == 0:
                    best_action = f"F_{i + 1}"
                else:
                    best_action = f"F|{s_i}_{i + 1}"

    return best_action


def _omega(box, box_state, y):
    """Probability that prize y is ultimately selected for a single box.

    Paper Section 5.2: ω_i(y).
    Old code: Pandora_Unified.Pandora.omega.
    """
    if box_state < 0:
        return 1.0

    if box_state == 0:
        if y >= box.p_threshold and y >= box.f_threshold:
            return 1.0
        elif (box.p_threshold <= box.f_threshold and y < box.f_threshold) or \
             (box.p_threshold > box.f_threshold and y <= box.fp_threshold):
            # F-open regime: Pr[min(V, σ^F) <= y] = Pr[V <= y]
            p = sum(box.prob_matrix[0, k] for k in range(box.S) if box.value_list[k] <= y)
            return p
        elif box.p_threshold > box.f_threshold and box.fp_threshold < y <= box.p_threshold:
            # P-open regime: Pr[min(V, σ^{F|T}, σ^P) <= y]
            p = 0.0
            for t in range(box.T):
                f_t = box.f_thresholds_all[t + 1]
                if f_t <= y:
                    p += box.type_probs[t]
                else:
                    q = sum(box.prob_matrix[t + 1, k] for k in range(box.S) if box.value_list[k] <= y)
                    p += box.type_probs[t] * q
            return p
    else:
        # Partially opened with type box_state
        f_t = box.f_thresholds_all[box_state]
        if y >= f_t:
            return 1.0
        p = sum(box.prob_matrix[box_state, k] for k in range(box.S) if box.value_list[k] <= y)
        return p

    return 1.0


def _whittle_integral(solver, state):
    """Compute J^W(C, P, y) — Whittle's integral (Eq. 11).

    J^W = σ_M - ∫_y^{σ_M} ∏_i ω_i(u) du

    The integral is computed via piecewise-constant approximation over
    breaking points (F-thresholds + value support points).
    Old code: Pandora_Unified.Pandora.Whittle_integral_J_hat.
    """
    y = state[0]
    lead = solver.sigma_M(state)
    if lead[0] is None or y >= lead[2]:
        return y
    sigma_m = lead[2]

    # Collect breaking points matching old code's approach:
    # For each box: add F-thresholds >= y, sort, then add values in range.
    break_points = [y]
    for box in solver.box_list:
        for sigma in box.f_thresholds_all:
            if sigma >= y:
                break_points.append(sigma)
        break_points = sorted(break_points)
        for v in box.value_list:
            if v >= y and v <= break_points[-1]:
                break_points.append(v)
    break_points = sorted(set(break_points))

    # Integrate using left-endpoint rule
    integral = 0.0
    for j in range(1, len(break_points)):
        u = break_points[j - 1]
        width = break_points[j] - break_points[j - 1]
        prod = 1.0
        for idx in range(solver.N):
            s_i = state[idx + 1]
            if s_i >= 0:
                prod *= _omega(solver.box_list[idx], s_i, u)
        integral += prod * width

    return sigma_m - integral


# ======================================================================
# STP: Single Test Policy (Section 6)
# ======================================================================

def stp_policy(solver, state):
    """Single Test Policy: myopic single-box improvement vs stopping.

    For each possible action on each box, compute the single-box expected
    value (ignoring other closed boxes) and pick the best.
    Old code: Pandora_Unified.Pandora.J (policy="STP").
    """
    y = state[0]
    best_val = y
    best_action = "STOP"

    for i in range(solver.N):
        box = solver.box_list[i]
        s_i = state[i + 1]

        # P-open closed box: single-box value
        if s_i == 0:
            val = -box.c_P
            for t in range(box.T):
                # After P-opening, single-box optimal: max(y, F-open if worthwhile)
                j_f = -box.c_F
                for k in range(box.S):
                    j_f += box.prob_matrix[t + 1, k] * max(y, box.value_list[k])
                val += box.type_probs[t] * max(y, j_f)
            if val > best_val:
                best_val = val
                best_action = f"P_{i + 1}"

        # F-open closed or partial box: single-box value
        if s_i >= 0:
            val = -box.c_F
            for k in range(box.S):
                val += box.prob_matrix[s_i, k] * max(y, box.value_list[k])
            if val > best_val:
                best_val = val
                if s_i == 0:
                    best_action = f"F_{i + 1}"
                else:
                    best_action = f"F|{s_i}_{i + 1}"

    return best_action


# ======================================================================
# Committing Policies (Algorithm 1, Section 5.3)
# ======================================================================

def make_committing_policy(partition):
    """Create a committing policy function for a given partition.

    Parameters
    ----------
    partition : list[str]
        partition[i] = 'F' or 'P' for each box i (0-indexed).

    Returns
    -------
    callable : policy_fn(solver, state) -> action
    """
    def policy_fn(solver, state):
        return _committing_action(solver, state, partition)
    return policy_fn


def _committing_action(solver, state, partition):
    """Determine action under committing policy π^{F,P} (Algorithm 1).

    Boxes in F: follow F-threshold ordering.
    Boxes in P: follow P-threshold for closed, F-threshold for partial.
    Always pick highest threshold action; stop when y exceeds it.
    """
    y = state[0]
    best_th = -np.inf
    best_idx = None
    best_mode = None

    for i in range(solver.N):
        box = solver.box_list[i]
        s_i = state[i + 1]

        if partition[i] == 'F':
            if s_i >= 0:
                th = box.f_thresholds_all[s_i]
                if th > best_th:
                    best_th = th
                    best_idx = i
                    best_mode = 'F'
        elif partition[i] == 'P':
            if s_i == 0:
                th = box.p_threshold
                if th > best_th:
                    best_th = th
                    best_idx = i
                    best_mode = 'P'
            elif s_i > 0:
                th = box.f_thresholds_all[s_i]
                if th > best_th:
                    best_th = th
                    best_idx = i
                    best_mode = 'F'

    if best_idx is None or y >= best_th:
        return "STOP"

    s_i = state[best_idx + 1]
    if best_mode == 'P':
        return f"P_{best_idx + 1}"
    else:
        if s_i == 0:
            return f"F_{best_idx + 1}"
        else:
            return f"F|{s_i}_{best_idx + 1}"


def best_committing_policy(solver, state=None):
    """Enumerate all 2^N committing partitions, return best value and partition.

    Skips dominated partitions where a box is assigned to P but σ^P < σ^F.
    Old code: Pandora_Unified.Pandora.enumerate_committing_policies.

    Returns
    -------
    best_partition : list[str]
    best_value : float
    """
    if state is None:
        state = [0] * (solver.N + 1)

    best_value = -np.inf
    best_partition = None

    for bits in itertools.product(['F', 'P'], repeat=solver.N):
        partition = list(bits)

        # Prune dominated: assigning P when σ^P < σ^F
        dominated = False
        for i in range(solver.N):
            if partition[i] == 'P' and solver.box_list[i].p_threshold < solver.box_list[i].f_threshold:
                dominated = True
                break
        if dominated:
            continue

        policy_fn = make_committing_policy(partition)
        val = solver.evaluate_policy(policy_fn, list(state))
        if val > best_value:
            best_value = val
            best_partition = partition

    return best_partition, best_value


def weitzman_policy(solver, state):
    """Weitzman policy: committing with all boxes in F (no P-opening).

    This is π^{[N], ∅} in the paper.
    Old code: policy="COM" with partition_list=["S"]*N.
    """
    partition = ['F'] * solver.N
    return _committing_action(solver, state, partition)
