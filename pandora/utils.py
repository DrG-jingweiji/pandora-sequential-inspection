"""Utility functions for the PSI codebase."""

import numpy as np


def sum_of_variances(box_list):
    """Dispersion measure: sum of variance of F-thresholds and P-thresholds.

    Used in the P-opening analysis (Appendix) to measure heterogeneity
    of thresholds across boxes. Old code: sum_of_variances in Experiment.py.
    """
    f_thresholds = [box.f_threshold for box in box_list]
    p_thresholds = [box.p_threshold for box in box_list]
    return float(np.std(f_thresholds) ** 2 + np.std(p_thresholds) ** 2)


def p_dominant_ratio(box_list):
    """Fraction of boxes where σ^P >= σ^F."""
    if not box_list:
        return 0.0
    count = sum(1 for box in box_list if box.p_threshold >= box.f_threshold)
    return count / len(box_list)


def state_to_sets(state):
    """Convert internal state list to paper notation (C, P_dict, y).

    Parameters
    ----------
    state : list
        [y, s_1, ..., s_N] where s_i=0 (closed), >0 (type), -1 (opened).

    Returns
    -------
    y : float
    closed : set of int
        Set of 0-indexed box indices that are closed.
    partial : dict of int -> int
        Maps 0-indexed box index to revealed type (1-indexed into prob_matrix).
    """
    y = state[0]
    closed = set()
    partial = {}
    for i in range(1, len(state)):
        if state[i] == 0:
            closed.add(i - 1)
        elif state[i] > 0:
            partial[i - 1] = state[i]
    return y, closed, partial


def all_opened(state):
    """Check if all boxes are fully opened (terminal state)."""
    return all(s < 0 for s in state[1:])
