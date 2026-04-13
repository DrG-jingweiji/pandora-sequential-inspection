"""Test that new DP solver matches old code's optimal values."""

import pytest
from pandora.box import Box
from pandora.solver import PandoraSolver


def _build_solver(inst):
    """Build a PandoraSolver from an old instance dict."""
    box_list = []
    for box_dict in inst['boxes']:
        box_list.append(Box(
            value_list=box_dict['value_list'],
            cond_prob_matrix=box_dict['condProb_matrix'],
            type_probs=box_dict['type_list'],
            c_P=box_dict['c_W'],
            c_F=box_dict['c_S'],
        ))
    return PandoraSolver(box_list)


class TestDPValues:
    """Test naive DP optimal values against stored old code results."""

    def test_dp_matches_yaron_opt(self, small_instances):
        """DP value should match (PandoraSolver.py) OPT J from PandoraSolutionYaron.json."""
        for inst, _, yaron in small_instances:
            if yaron is None:
                continue
            opt_key = '(PandoraSolver.py) OPT J'
            if opt_key not in yaron:
                continue

            solver = _build_solver(inst)
            new_val = solver.solve_dp()
            old_val = yaron[opt_key]

            assert abs(new_val - old_val) < 1e-6, (
                f"Instance {inst.get('instance id')}: "
                f"new DP={new_val:.8f}, old OPT={old_val:.8f}"
            )

    def test_structured_dp_matches_naive(self, small_instances):
        """Structured DP should return same value as naive DP."""
        for inst, _, _ in small_instances:
            solver = _build_solver(inst)
            naive_val = solver.solve_dp()

            solver2 = _build_solver(inst)
            plus_val = solver2.solve_dp_structured()

            assert abs(naive_val - plus_val) < 1e-6, (
                f"Instance {inst.get('instance id')}: "
                f"naive={naive_val:.8f}, structured={plus_val:.8f}"
            )
