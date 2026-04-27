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
                                max_attempts=500000, return_all=False,
                                require_p_dominant=True):
    """Generate a diverse set of prototypical boxes.

    Selects boxes that are spread out in (σ^F, σ^P) space among the
    selected set.  Filters to positive thresholds; optionally also
    requires σ^P > σ^F.
    Old code: Experiment.py generate.

    Parameters
    ----------
    return_all : bool
        If True, also return a list of all candidate boxes generated
        (for scatter plot visualization in Figure EC.8).
    require_p_dominant : bool
        If True (default), only select boxes with σ^P > σ^F.  Set to
        False for the P-opening experiments (Appendix), which need a
        mixed pool containing both σ^P > σ^F and σ^P ≤ σ^F boxes.

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

        if require_p_dominant and box.p_threshold <= box.f_threshold:
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


def generate_legacy_style_prototypical_boxes(n_boxes, distance=0.5, rng=None,
                                             max_attempts=500000,
                                             return_all=False,
                                             filter_cF_gt_cP=False,
                                             require_p_dominant=True):
    """Generate boxes with the old ``Experiment.py`` selection rule.

    The default generator compares distance only against previously selected
    boxes.  The old generator compared each candidate against every valid
    candidate generated so far, including candidates that were not selected.
    It also advanced the 2-type/5-type alternation only after a candidate had
    positive thresholds.

    ``require_p_dominant=True`` matches the archived source line
    ``weakThreshold > strongThresholds_list[0]``.  The stored old pickle itself
    is mixed in that threshold order, so diagnostic runs can set it to False.
    """
    if rng is None:
        rng = random

    selected = []
    all_candidates = []
    candidate_grid = {}
    valid_count = 0
    attempts = 0

    def cell_for(box):
        return (
            int(np.floor(box.p_threshold / distance)),
            int(np.floor(box.f_threshold / distance)),
        )

    def has_close_candidate(box):
        cell_p, cell_f = cell_for(box)
        for dp in (-1, 0, 1):
            for df in (-1, 0, 1):
                for existing in candidate_grid.get((cell_p + dp,
                                                    cell_f + df), []):
                    dist = np.sqrt(
                        (box.p_threshold - existing.p_threshold) ** 2
                        + (box.f_threshold - existing.f_threshold) ** 2
                    )
                    if dist < distance:
                        return True
        return False

    while len(selected) < n_boxes and attempts < max_attempts:
        attempts += 1
        if valid_count % 3 != 0:
            box = generate_one_box_v5_t3(rng)
        else:
            box = generate_one_box_v2_t2(rng)

        if box.p_threshold <= 0 or box.f_threshold <= 0:
            continue

        threshold_order_ok = (
            not require_p_dominant or box.p_threshold > box.f_threshold
        )
        if not has_close_candidate(box) and threshold_order_ok:
            selected.append(box)

        all_candidates.append(box)
        candidate_grid.setdefault(cell_for(box), []).append(box)
        valid_count += 1

    if len(selected) < n_boxes:
        raise RuntimeError(
            f'Generated only {len(selected)} legacy-style boxes after '
            f'{max_attempts} attempts; requested {n_boxes}.'
        )

    if filter_cF_gt_cP:
        selected = [box for box in selected if box.c_F > box.c_P]

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
# Legacy old-folder box pools
# ======================================================================

LEGACY_SELECTED_POOL_FILE = 'legacy_selected_boxes_100_0p5.json'


def bundled_legacy_pool_dir():
    """Return the project-local directory containing bundled old box pools."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(repo_root, 'data', 'legacy_box_pools')


def load_legacy_box_pool(pool_dir=None, pool_file=LEGACY_SELECTED_POOL_FILE,
                         filter_cF_gt_cP=False):
    """Load a bundled old box pool as new ``Box`` objects.

    Parameters
    ----------
    pool_dir : str, optional
        Directory containing project-local legacy pool JSON files. If omitted,
        uses ``data/legacy_box_pools`` in this repository.
    pool_file : str
        JSON file containing the converted old selected pool.
    filter_cF_gt_cP : bool
        Apply the old Yaron notebook filter ``c_S > c_W``.  This is the
        effective pool used for Tables 1--4/EC.1 in the old experiments.
    """
    if pool_dir is None:
        pool_dir = bundled_legacy_pool_dir()

    path = os.path.join(pool_dir, pool_file)
    with open(path) as fh:
        payload = json.load(fh)

    raw_boxes = payload['boxes'] if isinstance(payload, dict) else payload
    boxes = [box_from_dict(box) for box in raw_boxes]
    if filter_cF_gt_cP:
        boxes = [box for box in boxes if box.c_F > box.c_P]
    return boxes


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
