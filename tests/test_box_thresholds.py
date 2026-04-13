"""Test that new Box thresholds match old code's stored values."""

import pytest
from pandora.box import Box


def _make_box_from_old(box_dict):
    """Construct a new Box from old instance.json box dict."""
    return Box(
        value_list=box_dict['value_list'],
        cond_prob_matrix=box_dict['condProb_matrix'],
        type_probs=box_dict['type_list'],
        c_P=box_dict['c_W'],
        c_F=box_dict['c_S'],
    )


class TestBoxThresholds:
    """Compare new Box thresholds against stored values from old code."""

    def test_f_thresholds_match(self, all_instances):
        """σ^F and σ^{F|t} should match strongThresholds_list."""
        for inst, _, _ in all_instances:
            for box_dict in inst['boxes']:
                box = _make_box_from_old(box_dict)
                old_thresholds = box_dict['strongThresholds_list']
                for idx, old_val in enumerate(old_thresholds):
                    assert abs(box.f_thresholds_all[idx] - old_val) < 1e-3, (
                        f"F-threshold mismatch at idx={idx}: "
                        f"new={box.f_thresholds_all[idx]:.6f}, old={old_val:.6f}"
                    )

    def test_p_threshold_match(self, all_instances):
        """σ^P should match weakThreshold."""
        for inst, _, _ in all_instances:
            for box_dict in inst['boxes']:
                box = _make_box_from_old(box_dict)
                old_val = box_dict['weakThreshold']
                assert abs(box.p_threshold - old_val) < 1e-3, (
                    f"P-threshold mismatch: new={box.p_threshold:.6f}, old={old_val:.6f}"
                )

    def test_fp_threshold_match(self, all_instances):
        """σ^{F/P} should match sw_threshold."""
        for inst, _, _ in all_instances:
            for box_dict in inst['boxes']:
                box = _make_box_from_old(box_dict)
                old_val = box_dict['sw_threshold']
                assert abs(box.fp_threshold - old_val) < 1e-3, (
                    f"FP-threshold mismatch: new={box.fp_threshold:.6f}, old={old_val:.6f}"
                )

    def test_prob_matrix_matches(self, all_instances):
        """prob_matrix[0] (marginal) should match old stored prob_matrix[0]."""
        for inst, _, _ in all_instances:
            for box_dict in inst['boxes']:
                box = _make_box_from_old(box_dict)
                old_pm = box_dict['prob_matrix']
                for row in range(len(old_pm)):
                    for col in range(len(old_pm[row])):
                        assert abs(box.prob_matrix[row, col] - old_pm[row][col]) < 1e-10, (
                            f"prob_matrix mismatch at [{row},{col}]"
                        )
