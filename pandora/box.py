"""
Box model for the Pandora's Box Problem with Sequential Inspections (PSI).

Each box i has (Definition 1 in the paper):
  - Random prize V_i with discrete support (value_list)
  - Discrete type T_i with support Gamma_i (type_probs)
  - Conditional distributions V_i | T_i = t  (cond_prob_matrix)
  - F-opening cost c_F (full inspection, reveals value)
  - P-opening cost c_P (partial inspection, reveals type only)

Opening thresholds (Eqs. 2–5):
  - σ^F  (f_threshold):          F-opening threshold
  - σ^{F|t} (f_thresholds_by_type): conditional F-threshold given type t
  - σ^P  (p_threshold):          P-opening threshold
  - σ^{F/P} (fp_threshold):      FP-threshold
"""

import numpy as np


class Box:
    """A single box in the PSI problem.

    Parameters
    ----------
    value_list : list[float]
        Support of V_i in ascending order, size S.
    cond_prob_matrix : array-like, shape (T, S)
        cond_prob_matrix[t][k] = Pr[V_i = value_list[k] | T_i = type t].
    type_probs : list[float]
        type_probs[t] = Pr[T_i = type t], length T.
    c_P : float
        P-opening (partial) cost.  Old code: c_W.
    c_F : float
        F-opening (full) cost.  Old code: c_S.
    """

    def __init__(self, value_list, cond_prob_matrix, type_probs, c_P, c_F):
        self.value_list = list(value_list)
        self.cond_prob_matrix = np.array(cond_prob_matrix, dtype=float)
        self.type_probs = np.array(type_probs, dtype=float)
        self.c_P = float(c_P)
        self.c_F = float(c_F)

        self.S = len(self.value_list)
        self.T = len(self.type_probs)

        # prob_matrix: row 0 = marginal, rows 1..T = conditional V|T=t
        marginal = self.cond_prob_matrix.T @ self.type_probs
        self.prob_matrix = np.vstack([marginal, self.cond_prob_matrix])

        # Precompute all thresholds
        self._f_thresholds = self._compute_f_thresholds()
        self._p_threshold = self._compute_p_threshold()
        self._fp_threshold = self._compute_fp_threshold()
        self._sw_threshold = self._compute_sw_threshold()

    # ------------------------------------------------------------------
    # Public threshold properties
    # ------------------------------------------------------------------

    @property
    def f_threshold(self):
        """σ^F_i: F-threshold for closed box (Eq. 2).

        Solves c_F = E[(V_i - σ)^+] under the marginal distribution.
        Old code: strongThresholds_list[0].
        """
        return self._f_thresholds[0]

    @property
    def f_thresholds_by_type(self):
        """σ^{F|t}_i for each type t (Eq. 3).

        f_thresholds_by_type[t] solves c_F = E[(V_i - σ)^+ | T_i = t].
        Old code: strongThresholds_list[t+1].
        """
        return self._f_thresholds[1:]

    @property
    def f_thresholds_all(self):
        """All F-thresholds: [σ^F, σ^{F|t_0}, σ^{F|t_1}, ...].

        Index 0 = marginal, index t+1 = conditional on type t.
        Old code: strongThresholds_list.
        """
        return self._f_thresholds

    @property
    def p_threshold(self):
        """σ^P_i: P-threshold for closed box (Eq. 4).

        Solves c_P = E_T[max{0, -c_F + E_{V|T}[(V - σ)^+]}].
        Old code: weakThreshold.
        """
        return self._p_threshold

    @property
    def fp_threshold(self):
        """σ^{F/P}_i: FP-threshold (Eq. 5).

        Solves c_P = E_T[max{0, c_F - E_{V|T}[(V - σ)^+]}].
        Determines whether P-opening or F-opening is preferred.
        Old code: sw_threshold (from SW_threshold method).
        """
        return self._fp_threshold

    @property
    def s_w_threshold(self):
        """Internal s-w threshold used for well-classified check.

        Old code: s_w_threshold_.
        """
        return self._sw_threshold

    # ------------------------------------------------------------------
    # Threshold computation internals
    # ------------------------------------------------------------------

    def _f_rhs(self, sigma, state):
        """E[(V - sigma)^+] - c_F under prob_matrix[state].

        Paper Eq. 2/3: the RHS minus LHS of the threshold equation.
        Old code: s_rhs.
        """
        clipped = [max(0.0, v - sigma) for v in self.value_list]
        return float(np.dot(clipped, self.prob_matrix[state])) - self.c_F

    def _p_rhs(self, sigma):
        """E_T[max{0, -c_F + E_{V|T}[(V - sigma)^+]}] - c_P.

        Paper Eq. 4: the RHS minus LHS of the P-threshold equation.
        Old code: w_rhs.
        """
        per_type = [max(0.0, self._f_rhs(sigma, t + 1)) for t in range(self.T)]
        return float(np.dot(per_type, self.type_probs)) - self.c_P

    def _fp_rhs(self, sigma):
        """E_T[max{0, c_F - E_{V|T}[(V - sigma)^+]}] - c_P.

        Paper Eq. 5: the RHS minus LHS of the FP-threshold equation.
        Old code: sw_rhs. Note the sign flip vs _f_rhs.
        """
        per_type = [max(0.0, -self._f_rhs(sigma, t + 1)) for t in range(self.T)]
        return float(np.dot(per_type, self.type_probs)) - self.c_P

    @staticmethod
    def _bisect(func, left, right, eps=1e-4, max_iter=200):
        """Generic bisection solver. Returns mid when |func(mid)| < eps."""
        mid = (left + right) / 2.0
        r = func(mid)
        for _ in range(max_iter):
            if abs(r) < eps:
                break
            if r > 0:
                left = mid
            else:
                right = mid
            mid = (left + right) / 2.0
            r = func(mid)
        return mid

    def _compute_f_thresholds(self, eps=1e-4):
        """Compute F-thresholds for marginal (state=0) and each type (state=1..T)."""
        thresholds = []
        for state in range(self.T + 1):
            mu = float(np.dot(self.value_list, self.prob_matrix[state]))
            left = -self.c_F + mu
            right = self.value_list[-1]
            thresholds.append(self._bisect(lambda s: self._f_rhs(s, state), left, right, eps))
        return thresholds

    def _compute_p_threshold(self, eps=1e-4):
        """Compute P-threshold σ^P (Eq. 4) via bisection."""
        mu = float(np.dot(self.value_list, self.prob_matrix[0]))
        left = -self.c_F - self.c_P + mu
        right = self.value_list[-1]
        return self._bisect(self._p_rhs, left, right, eps)

    def _compute_fp_threshold(self, eps=1e-4):
        """Compute FP-threshold σ^{F/P} (Eq. 5) via bisection.

        Old code: SW_threshold.
        If c_P > c_F, returns a large value (P-opening never preferred).
        """
        if self.c_P > self.c_F:
            return self.value_list[-1] + self.c_P + self.c_F

        left = -(self.value_list[-1] + self.c_P + self.c_F)
        right = self.value_list[-1] + self.c_P + self.c_F
        return self._bisect(lambda s: -self._fp_rhs(s), left, right, eps)

    def _compute_sw_threshold(self, eps=1e-4):
        """Compute s-w threshold used for well-classified check.

        Old code: s_w_threshold. Uses same _fp_rhs but different bounds.
        """
        left = -self.c_F - self.c_P
        right = self.value_list[-1]
        return self._bisect(lambda s: -self._fp_rhs(s), left, right, eps)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_expired(self, y, box_state):
        """Check if this box is expired (cannot improve on y).

        A box is expired when y exceeds all its active thresholds.
        Used in Theorem 1 (stopping condition).

        Parameters
        ----------
        y : float
            Current best prize.
        box_state : int
            0 = closed, >0 = type revealed (1-indexed), -1 = fully opened.
        """
        if box_state < 0:
            return True
        if box_state == 0:
            return y >= self.f_threshold and y >= self.p_threshold
        # box_state > 0: partially opened with type = box_state
        return y >= self._f_thresholds[box_state]

    def active_threshold(self, box_state):
        """Return the largest active threshold for this box and whether it is F or P.

        Returns
        -------
        threshold : float
            The largest active threshold value.
        mode : str
            'F' if the largest threshold is an F-threshold, 'P' if P-threshold.
        """
        if box_state < 0:
            return -np.inf, None
        if box_state == 0:
            f_th = self.f_threshold
            p_th = self.p_threshold
            if f_th >= p_th:
                return f_th, 'F'
            else:
                return p_th, 'P'
        # Partially opened: only F-threshold applies
        return self._f_thresholds[box_state], 'F'

    def is_well_classified(self, y, sigma_minus_i):
        """Check if this box is well-classified (Definition 3, Section 3.3).

        A closed box i with σ^P_i > σ^F_i is well-classified if:
          1. y > σ^{F/P}_i, AND
          2. For every type t: σ^{F|t}_i >= σ_{-i} OR y > σ^{F|t}_i

        Parameters
        ----------
        y : float
            Current best prize.
        sigma_minus_i : float
            σ_M(C\\{i}, P) — largest threshold among all other boxes.
        """
        if y < self.s_w_threshold:
            return False
        for t in range(self.T):
            f_t = self._f_thresholds[t + 1]
            if f_t < sigma_minus_i and y < f_t:
                return False
        return True

    def __repr__(self):
        return (
            f"Box(c_F={self.c_F:.4f}, c_P={self.c_P:.4f}, "
            f"σ^F={self.f_threshold:.4f}, σ^P={self.p_threshold:.4f}, "
            f"σ^{{F/P}}={self.fp_threshold:.4f})"
        )
