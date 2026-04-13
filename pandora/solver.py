"""
Dynamic programming solver for the PSI problem.

Implements:
  - Naive DP: Full Bellman recursion (Eq. 1)
  - Structured DP: Accelerated recursion using Theorems 1–3

State representation: [y, s_1, ..., s_N]
  - y: current best prize found so far
  - s_i = 0: box i is closed
  - s_i > 0: box i is partially opened, type = s_i (1-indexed into prob_matrix)
  - s_i = -1: box i is fully opened
"""

import copy
import numpy as np

# Condition flags for tracking which theorem applied
COND_STOP = 0
COND_F_OPEN = 1
COND_P_OPEN = 2


class PandoraSolver:
    """Multi-box PSI solver.

    Parameters
    ----------
    box_list : list[Box]
        The boxes in the problem instance.
    """

    def __init__(self, box_list):
        self.box_list = list(box_list)
        self.N = len(self.box_list)

        # Naive DP caches
        self._dp_value = {}
        self._dp_action = {}
        self._dp_stats = {}
        self._dp_num_revisits = 0

        # Structured DP caches
        self._dp_plus_value = {}
        self._dp_plus_action = {}
        self._dp_plus_num_revisits = 0

    # ==================================================================
    # Public interface
    # ==================================================================

    def solve_dp(self, state=None):
        """Solve via naive Bellman DP (Eq. 1). Returns optimal value.

        Parameters
        ----------
        state : list, optional
            Starting state. Defaults to initial state [0, 0, ..., 0].
        """
        if state is None:
            state = [0] * (self.N + 1)
        self._dp_value.clear()
        self._dp_action.clear()
        self._dp_stats.clear()
        self._dp_num_revisits = 0
        return self._bellman(state)

    def solve_dp_structured(self, state=None):
        """Solve via structured/accelerated DP using Theorems 1–3.

        Returns optimal value. Also populates stats for efficiency analysis.
        """
        if state is None:
            state = [0] * (self.N + 1)
        self._dp_plus_value.clear()
        self._dp_plus_action.clear()
        self._dp_plus_num_revisits = 0
        return self._bellman_plus(state)

    def get_optimal_value(self, state=None):
        """Convenience: solve naive DP and return value at initial state."""
        return self.solve_dp(state)

    def get_dp_stats(self):
        """Return stats from the most recent naive DP solve.

        Returns
        -------
        dict with keys:
          - num_states_created: number of unique states visited
          - num_revisits: number of state revisits
          - dp_stats: per-state condition flags
        """
        return {
            'num_states_created': len(self._dp_value),
            'num_revisits': self._dp_num_revisits,
            'dp_stats': self._dp_stats,
        }

    def get_dp_plus_stats(self):
        """Return stats from the most recent structured DP solve."""
        return {
            'num_states_created': len(self._dp_plus_value),
            'num_revisits': self._dp_plus_num_revisits,
        }

    def get_dp_action(self, state):
        """Get optimal action at a state (after solve_dp)."""
        return self._dp_action.get(tuple(state))

    def get_dp_value_at(self, state):
        """Get optimal value at a state (after solve_dp)."""
        return self._dp_value.get(tuple(state))

    @property
    def dp_value_dict(self):
        return self._dp_value

    @property
    def dp_action_dict(self):
        return self._dp_action

    # ==================================================================
    # Sigma_M: largest opening threshold (Eq. 6)
    # ==================================================================

    def sigma_M(self, state):
        """Compute σ_M(C, P) — the largest opening threshold (Eq. 6).

        Returns (box_index, mode, threshold) or (None, None, -inf).
        box_index is 0-based index into box_list.
        mode is 'F' or 'P'.
        """
        best_idx = None
        best_mode = None
        best_th = -np.inf

        for i in range(self.N):
            box = self.box_list[i]
            s_i = state[i + 1]
            th, mode = box.active_threshold(s_i)
            if th > best_th:
                best_th = th
                best_idx = i
                best_mode = mode

        return best_idx, best_mode, best_th

    def sigma_M_excluding(self, state, exclude_idx):
        """σ_M(C\\{i}, P) — largest threshold excluding box exclude_idx."""
        best_th = -np.inf
        for i in range(self.N):
            if i == exclude_idx:
                continue
            box = self.box_list[i]
            s_i = state[i + 1]
            th, _ = box.active_threshold(s_i)
            if th > best_th:
                best_th = th
        return best_th

    # ==================================================================
    # Naive DP: Bellman recursion (Eq. 1)
    # ==================================================================

    def _bellman(self, state_list):
        """Recursive memoized Bellman equation (Eq. 1).

        J(C, P, y) = max{
          y,                                                    [stop]
          -c_F + E[J(C\\{i}, P, max(V_i, y))]   for i in C,   [F-open closed]
          -c_F + E[J(C, P\\{i}, max(V_i, y))]   for i in P,   [F-open partial]
          -c_P + E[J(C\\{i}, P+{(i,T)}, y)]     for i in C    [P-open closed]
        }
        """
        state_tuple = tuple(state_list)

        if state_tuple in self._dp_value:
            self._dp_num_revisits += 1
            self._dp_stats[state_tuple]['num_visits'] += 1
            return self._dp_value[state_tuple]

        self._dp_stats[state_tuple] = {'num_visits': 1, 'conditions': [0, 0, 0]}

        # Terminal check: all boxes fully opened
        if all(state_list[i + 1] < 0 for i in range(self.N)):
            self._dp_value[state_tuple] = state_list[0]
            self._dp_action[state_tuple] = "STOP"
            return state_list[0]

        y = state_list[0]

        # Check and record sufficient conditions (for coverage analysis)
        self._record_conditions(state_list, state_tuple)

        # Enumerate all feasible actions
        j_values = []
        actions = []

        for i in range(self.N):
            box = self.box_list[i]
            s_i = state_list[i + 1]

            # P-open closed box (paper: P-open box i ∈ C)
            if s_i == 0:
                val = -box.c_P
                for t in range(box.T):
                    ns = list(state_list)
                    ns[i + 1] = t + 1
                    val += box.type_probs[t] * self._bellman(ns)
                j_values.append(val)
                actions.append(f"P_{i + 1}")

            # F-open closed or partial box (paper: F-open box i ∈ C or i ∈ P)
            if s_i >= 0:
                val = -box.c_F
                prob_row = s_i  # 0 for marginal, >0 for conditional
                for k in range(box.S):
                    ns = list(state_list)
                    ns[i + 1] = -1
                    ns[0] = max(ns[0], box.value_list[k])
                    val += box.prob_matrix[prob_row, k] * self._bellman(ns)
                j_values.append(val)
                if s_i == 0:
                    actions.append(f"F_{i + 1}")
                else:
                    actions.append(f"F|{s_i}_{i + 1}")

        # Stop action
        j_values.append(y)
        actions.append("STOP")

        best_idx = int(np.argmax(j_values))
        self._dp_value[state_tuple] = j_values[best_idx]
        self._dp_action[state_tuple] = actions[best_idx]
        return j_values[best_idx]

    def _record_conditions(self, state_list, state_tuple):
        """Record which sufficient conditions apply at this state."""
        y = state_list[0]

        # Theorem 1: stopping condition — all boxes expired
        all_expired = True
        for i in range(self.N):
            if not self.box_list[i].is_expired(y, state_list[i + 1]):
                all_expired = False
                break

        if all_expired:
            self._dp_stats[state_tuple]['conditions'][COND_STOP] = 1
            return

        # Determine leading box
        lead_idx, lead_mode, lead_th = self.sigma_M(state_list)

        # Theorem 2: F-opening the leading box
        if lead_mode == 'F':
            self._dp_stats[state_tuple]['conditions'][COND_F_OPEN] = 1
            return

        # Theorem 3: P-opening a well-classified leading box
        if lead_mode == 'P' and state_list[lead_idx + 1] == 0:
            sigma_neg_i = self.sigma_M_excluding(state_list, lead_idx)
            box = self.box_list[lead_idx]
            if box.is_well_classified(y, sigma_neg_i):
                self._dp_stats[state_tuple]['conditions'][COND_P_OPEN] = 1

    # ==================================================================
    # Structured DP: Bellman + Theorems 1–3 (J_plus)
    # ==================================================================

    def _bellman_plus(self, state_list):
        """Accelerated DP using structural properties (Theorems 1–3).

        1. If all boxes expired (Theorem 1): STOP.
        2. If σ_M is an F-threshold (Theorem 2): F-open leading box only.
        3. If σ_M is a P-threshold and well-classified (Theorem 3): P-open only.
        4. Otherwise: full Bellman recursion.
        """
        state_tuple = tuple(state_list)

        if state_tuple in self._dp_plus_value:
            self._dp_plus_num_revisits += 1
            return self._dp_plus_value[state_tuple]

        y = state_list[0]

        # Theorem 1 + terminal: check if all boxes are expired or opened
        all_done = True
        for i in range(self.N):
            s_i = state_list[i + 1]
            if not self.box_list[i].is_expired(y, s_i):
                all_done = False
                break

        if all_done:
            self._dp_plus_value[state_tuple] = y
            self._dp_plus_action[state_tuple] = "STOP"
            return y

        # Find leading box and second-largest threshold
        active_thresholds = []
        fp_encoding = []
        for i in range(self.N):
            box = self.box_list[i]
            s_i = state_list[i + 1]
            th, mode = box.active_threshold(s_i)
            active_thresholds.append(th)
            fp_encoding.append(mode)

        lead_idx = int(np.argmax(active_thresholds))
        if self.N >= 2:
            sorted_ths = sorted(active_thresholds)
            second_largest = sorted_ths[-2]
        else:
            second_largest = -np.inf

        lead_mode = fp_encoding[lead_idx]
        box = self.box_list[lead_idx]

        # Theorem 2: F-open leading box
        if lead_mode == 'F':
            s_i = state_list[lead_idx + 1]
            val = -box.c_F
            prob_row = s_i
            for k in range(box.S):
                ns = list(state_list)
                ns[lead_idx + 1] = -1
                ns[0] = max(ns[0], box.value_list[k])
                val += box.prob_matrix[prob_row, k] * self._bellman_plus(ns)

            if s_i == 0:
                action = f"F_{lead_idx + 1}"
            else:
                action = f"F|{s_i}_{lead_idx + 1}"

            self._dp_plus_value[state_tuple] = val
            self._dp_plus_action[state_tuple] = action
            return val

        # Theorem 3: P-open well-classified leading box
        if lead_mode == 'P' and state_list[lead_idx + 1] == 0:
            if box.is_well_classified(y, second_largest):
                val = -box.c_P
                for t in range(box.T):
                    ns = list(state_list)
                    ns[lead_idx + 1] = t + 1
                    val += box.type_probs[t] * self._bellman_plus(ns)

                self._dp_plus_value[state_tuple] = val
                self._dp_plus_action[state_tuple] = f"P_{lead_idx + 1}"
                return val

        # Fallback: full Bellman recursion
        j_values = []
        actions = []

        for i in range(self.N):
            box_i = self.box_list[i]
            s_i = state_list[i + 1]

            if s_i == 0:
                val = -box_i.c_P
                for t in range(box_i.T):
                    ns = list(state_list)
                    ns[i + 1] = t + 1
                    val += box_i.type_probs[t] * self._bellman_plus(ns)
                j_values.append(val)
                actions.append(f"P_{i + 1}")

            if s_i >= 0:
                val = -box_i.c_F
                for k in range(box_i.S):
                    ns = list(state_list)
                    ns[i + 1] = -1
                    ns[0] = max(ns[0], box_i.value_list[k])
                    val += box_i.prob_matrix[s_i, k] * self._bellman_plus(ns)
                j_values.append(val)
                if s_i == 0:
                    actions.append(f"F_{i + 1}")
                else:
                    actions.append(f"F|{s_i}_{i + 1}")

        j_values.append(y)
        actions.append("STOP")

        best_idx = int(np.argmax(j_values))
        self._dp_plus_value[state_tuple] = j_values[best_idx]
        self._dp_plus_action[state_tuple] = actions[best_idx]
        return j_values[best_idx]

    # ==================================================================
    # Policy evaluation
    # ==================================================================

    def evaluate_policy(self, policy_fn, state=None):
        """Evaluate a given policy from the initial state.

        Parameters
        ----------
        policy_fn : callable
            policy_fn(solver, state) -> action string.
        state : list, optional
            Starting state. Defaults to initial state.

        Returns
        -------
        float : expected value under the policy.
        """
        if state is None:
            state = [0] * (self.N + 1)
        return self._eval_policy_recursive(policy_fn, state, {})

    def _eval_policy_recursive(self, policy_fn, state_list, cache):
        state_tuple = tuple(state_list)
        if state_tuple in cache:
            return cache[state_tuple]

        if all(state_list[i + 1] < 0 for i in range(self.N)):
            cache[state_tuple] = state_list[0]
            return state_list[0]

        action = policy_fn(self, state_list)
        val = self._take_action(action, state_list, policy_fn, cache)
        cache[state_tuple] = val
        return val

    def _take_action(self, action, state_list, policy_fn, cache):
        """Execute an action and return the expected value."""
        y = state_list[0]
        if action == "STOP":
            return y

        # Parse action string
        if action.startswith("F"):
            if "|" in action:
                parts = action.split("_")
                box_idx = int(parts[-1]) - 1
                type_str = action[action.find("|") + 1:action.find("_")]
                prob_row = int(type_str)
            else:
                box_idx = int(action.split("_")[1]) - 1
                prob_row = 0 if state_list[box_idx + 1] == 0 else state_list[box_idx + 1]

            box = self.box_list[box_idx]
            val = -box.c_F
            for k in range(box.S):
                ns = list(state_list)
                ns[box_idx + 1] = -1
                ns[0] = max(ns[0], box.value_list[k])
                val += box.prob_matrix[prob_row, k] * self._eval_policy_recursive(
                    policy_fn, ns, cache)
            return val

        elif action.startswith("P"):
            box_idx = int(action.split("_")[1]) - 1
            box = self.box_list[box_idx]
            val = -box.c_P
            for t in range(box.T):
                ns = list(state_list)
                ns[box_idx + 1] = t + 1
                val += box.type_probs[t] * self._eval_policy_recursive(
                    policy_fn, ns, cache)
            return val

        raise ValueError(f"Unknown action: {action}")

    # ==================================================================
    # Number of tests (expected F and P openings under a policy)
    # ==================================================================

    def expected_openings(self, policy_name='DP', state=None):
        """Expected number of F-openings and P-openings under a policy.

        Parameters
        ----------
        policy_name : str
            'DP' uses the optimal policy from solve_dp.
        state : list, optional

        Returns
        -------
        n_f : float
            Expected number of F-openings.
        n_p : float
            Expected number of P-openings.
        """
        if state is None:
            state = [0] * (self.N + 1)
        action_dict = self._dp_action
        return self._count_openings(state, action_dict)

    def _count_openings(self, state_list, action_dict):
        state_tuple = tuple(state_list)
        action = action_dict.get(state_tuple)
        if action is None or action == "STOP":
            return 0.0, 0.0

        if action.startswith("F"):
            if "|" in action:
                box_idx = int(action.split("_")[-1]) - 1
                type_str = action[action.find("|") + 1:action.find("_")]
                prob_row = int(type_str)
            else:
                box_idx = int(action.split("_")[1]) - 1
                prob_row = state_list[box_idx + 1]

            box = self.box_list[box_idx]
            n_f, n_p = 1.0, 0.0
            for k in range(box.S):
                ns = list(state_list)
                ns[box_idx + 1] = -1
                ns[0] = max(ns[0], box.value_list[k])
                sf, sp = self._count_openings(ns, action_dict)
                n_f += box.prob_matrix[prob_row, k] * sf
                n_p += box.prob_matrix[prob_row, k] * sp
            return n_f, n_p

        elif action.startswith("P"):
            box_idx = int(action.split("_")[1]) - 1
            box = self.box_list[box_idx]
            n_f, n_p = 0.0, 1.0
            for t in range(box.T):
                ns = list(state_list)
                ns[box_idx + 1] = t + 1
                sf, sp = self._count_openings(ns, action_dict)
                n_f += box.type_probs[t] * sf
                n_p += box.type_probs[t] * sp
            return n_f, n_p

        return 0.0, 0.0
