"""
Instance generation and JSON I/O for PSI.

Generates prototypical boxes and random instances following the paper's
experimental setup (Section 6, "Problem instances"):
  - Support of V_i: 5 values from U(0,10)
  - Types: |Γ_i| <= 3
  - Costs: c_F ~ U(0,5), c_P ~ U(0,3), with c_F >= c_P
  - Instances: randomized copies of prototypical boxes
"""

import json
import os
import random

import numpy as np

from pandora.box import Box


# ======================================================================
# Prototypical box generation
# ======================================================================

def generate_one_box_v5_t3(rng=None):
    """Generate one prototypical box with 5 support points and 3 types.

    Old code: Experiment.py generate_one_box_v_5_t_3.
    """
    if rng is None:
        rng = random

    v1 = rng.uniform(0, 2)
    v2 = rng.uniform(2, 4)
    v3 = rng.uniform(4, 6)
    v4 = rng.uniform(6, 8)
    v5 = rng.uniform(8, 10)
    value_list = [v1, v2, v3, v4, v5]

    cond_prob_matrix = []
    for _ in range(3):
        cuts = sorted([rng.random(), rng.random(), rng.random(), rng.random()])
        row = [
            cuts[0],
            cuts[1] - cuts[0],
            cuts[2] - cuts[1],
            cuts[3] - cuts[2],
            1.0 - cuts[3],
        ]
        cond_prob_matrix.append(row)

    a = rng.uniform(0, 0.35)
    b = rng.random()
    p_a = min(a, b)
    p_b = max(a, b) - min(a, b)
    p_c = 1.0 - p_a - p_b
    type_probs = [p_a, p_b, p_c]

    c_F = rng.uniform(0, 5)
    c_P = rng.uniform(0, 3)

    return Box(value_list, cond_prob_matrix, type_probs, c_P, c_F)


def generate_one_box_v2_t2(rng=None):
    """Generate one prototypical box with 2 support points and 2 types.

    Old code: Experiment.py generate_one_box_v_2_t_2.
    """
    if rng is None:
        rng = random

    v1 = rng.uniform(0, 3)
    v2 = rng.uniform(7, 10)
    value_list = [v1, v2]

    a = rng.uniform(0, 0.2)
    b = rng.uniform(0.8, 1)
    cond_prob_matrix = [[a, 1 - a], [b, 1 - b]]

    p_a = rng.random()
    type_probs = [p_a, 1.0 - p_a]

    c_F = rng.uniform(0, 4)
    c_P = rng.uniform(0, 2)

    return Box(value_list, cond_prob_matrix, type_probs, c_P, c_F)


def generate_prototypical_boxes(n_boxes, distance=0.5, rng=None,
                                max_attempts=500000, return_all=False):
    """Generate a diverse set of prototypical boxes.

    Selects boxes that are spread out in (σ^F, σ^P) space among the
    selected set. Filters to positive thresholds and σ^P > σ^F.
    Old code: Experiment.py generate.

    Parameters
    ----------
    return_all : bool
        If True, also return a list of all candidate boxes generated
        (for scatter plot visualization in Figure EC.8).

    Returns
    -------
    selected : list[Box]
        The prototypical boxes.
    all_candidates : list[Box]
        Only returned when return_all=True. Every box generated during
        the search (including those not selected).
    """
    if rng is None:
        rng = random

    selected = []
    all_candidates = [] if return_all else None

    for attempt in range(max_attempts):
        if len(selected) >= n_boxes:
            break

        if attempt % 3 != 0:
            box = generate_one_box_v5_t3(rng)
        else:
            box = generate_one_box_v2_t2(rng)

        if box.p_threshold <= 0 or box.f_threshold <= 0:
            continue

        if return_all:
            all_candidates.append(box)

        if box.p_threshold <= box.f_threshold:
            continue

        is_different = True
        for existing in selected:
            dist = np.sqrt(
                (box.p_threshold - existing.p_threshold) ** 2
                + (box.f_threshold - existing.f_threshold) ** 2
            )
            if dist < distance:
                is_different = False
                break

        if is_different:
            selected.append(box)

    if return_all:
        return selected, all_candidates
    return selected


def sample_instance(selected_boxes, n_boxes, rng=None):
    """Sample an instance of n_boxes from a pool of prototypical boxes.

    Old code: sampling in (Yaron) Generate instances.ipynb.
    """
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.integers(0, len(selected_boxes), size=n_boxes)
    return [selected_boxes[i] for i in indices], indices.tolist()


# ======================================================================
# JSON I/O (backward compatible with old instance.json schema)
# ======================================================================

def box_to_dict(box):
    """Serialize a Box to a dict matching old instance.json format."""
    return {
        'value_list': box.value_list,
        'condProb_matrix': box.cond_prob_matrix.tolist(),
        'type_list': box.type_probs.tolist(),
        'T': box.T,
        'c_W': box.c_P,       # Old code name
        'c_S': box.c_F,       # Old code name
        'prob_matrix': box.prob_matrix.tolist(),
        'strongThresholds_list': box.f_thresholds_all,
        'weakThreshold': box.p_threshold,
        'sw_threshold': box.fp_threshold,
    }


def box_from_dict(d):
    """Deserialize a Box from a dict (old instance.json format)."""
    return Box(
        value_list=d['value_list'],
        cond_prob_matrix=d['condProb_matrix'],
        type_probs=d['type_list'],
        c_P=d['c_W'],
        c_F=d['c_S'],
    )


def save_instance(path, instance_id, box_list, boxes_choice=None, solutions=None):
    """Save an instance to JSON (old format compatible)."""
    data = {
        'instance id': instance_id,
        'N': len(box_list),
        'boxes_choice': boxes_choice or [],
        'boxes': [box_to_dict(b) for b in box_list],
        'solutions': solutions or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_instance(path):
    """Load an instance from JSON.

    Returns
    -------
    dict with keys:
      - 'instance_id': int
      - 'N': int
      - 'box_list': list[Box]
      - 'boxes_choice': list[int]
      - 'solutions': dict
      - 'raw': the raw JSON dict
    """
    with open(path) as f:
        data = json.load(f)

    box_list = [box_from_dict(b) for b in data['boxes']]
    return {
        'instance_id': data.get('instance id'),
        'N': data['N'],
        'box_list': box_list,
        'boxes_choice': data.get('boxes_choice', []),
        'solutions': data.get('solutions', {}),
        'raw': data,
    }


def load_jingwei_computations(path):
    """Load JingweiComputations.json for DP efficiency comparison."""
    with open(path) as f:
        return json.load(f)
