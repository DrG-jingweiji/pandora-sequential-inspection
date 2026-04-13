"""Test that heuristic policy values match old code's stored results."""

import pytest
from pandora.box import Box
from pandora.solver import PandoraSolver
from pandora.policies import (
    index_policy, whittle_policy, stp_policy,
    best_committing_policy, weitzman_policy,
)


def _build_solver(inst):
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


class TestPolicies:
    """Test policy values against stored solutions in instance.json."""

    def test_index_policy(self, small_instances):
        """INDEX policy value should match old code's solutions.INDEX.J."""
        for inst, _, _ in small_instances:
            solutions = inst.get('solutions', {})
            if 'INDEX' not in solutions:
                continue

            solver = _build_solver(inst)
            new_val = solver.evaluate_policy(index_policy)
            old_val = solutions['INDEX']['J']

            assert abs(new_val - old_val) < 1e-5, (
                f"Instance {inst.get('instance id')}: "
                f"INDEX new={new_val:.8f}, old={old_val:.8f}"
            )

    def test_stp_policy(self, small_instances):
        """STP policy value should match old code's solutions.STP.J."""
        for inst, _, _ in small_instances:
            solutions = inst.get('solutions', {})
            if 'STP' not in solutions:
                continue

            solver = _build_solver(inst)
            new_val = solver.evaluate_policy(stp_policy)
            old_val = solutions['STP']['J']

            assert abs(new_val - old_val) < 1e-5, (
                f"Instance {inst.get('instance id')}: "
                f"STP new={new_val:.8f}, old={old_val:.8f}"
            )

    def test_committing_policy(self, small_instances):
        """Best committing policy should match old code's solutions.COM.J."""
        for inst, _, _ in small_instances:
            solutions = inst.get('solutions', {})
            if 'COM' not in solutions:
                continue

            solver = _build_solver(inst)
            _, new_val = best_committing_policy(solver)
            old_val = solutions['COM']['J']

            assert abs(new_val - old_val) < 1e-5, (
                f"Instance {inst.get('instance id')}: "
                f"COM new={new_val:.8f}, old={old_val:.8f}"
            )

    def test_whittle_policy(self, small_instances):
        """WHITTLE policy value should match old code's solutions.WHITTLE.J."""
        for inst, _, _ in small_instances:
            solutions = inst.get('solutions', {})
            if 'WHITTLE' not in solutions:
                continue

            solver = _build_solver(inst)
            new_val = solver.evaluate_policy(whittle_policy)
            old_val = solutions['WHITTLE']['J']

            assert abs(new_val - old_val) < 1e-4, (
                f"Instance {inst.get('instance id')}: "
                f"WHITTLE new={new_val:.8f}, old={old_val:.8f}"
            )

    def test_policies_leq_optimal(self, small_instances):
        """All policies should yield values <= optimal DP value."""
        for inst, _, yaron in small_instances:
            if yaron is None:
                continue
            opt_key = '(PandoraSolver.py) OPT J'
            if opt_key not in yaron:
                continue

            solver = _build_solver(inst)
            opt_val = yaron[opt_key]

            for policy_fn in [index_policy, stp_policy, whittle_policy]:
                val = solver.evaluate_policy(policy_fn)
                assert val <= opt_val + 1e-6, (
                    f"Instance {inst.get('instance id')}: "
                    f"policy value {val:.8f} > OPT {opt_val:.8f}"
                )
