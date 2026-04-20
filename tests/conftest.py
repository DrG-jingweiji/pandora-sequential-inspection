"""Shared test fixtures for loading reference test instances.

The unit tests compare against pre-computed reference instances stored
in the old code directory. If the reference data is not available,
tests that depend on it are skipped automatically.
"""

import json
import os
import sys
import pytest

# Add parent directory to path so we can import pandora
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OLD_INSTANCES_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'old code', 'Jingwei', 'Numerical', 'test_instances'
)


def _list_instance_ids():
    """List all instance IDs in the test_instances directory."""
    if not os.path.isdir(OLD_INSTANCES_DIR):
        return []
    return sorted([
        d for d in os.listdir(OLD_INSTANCES_DIR)
        if os.path.isdir(os.path.join(OLD_INSTANCES_DIR, d))
    ])


def _load_instance(instance_id):
    """Load instance.json, JingweiComputations.json, and PandoraSolutionYaron.json."""
    base = os.path.join(OLD_INSTANCES_DIR, str(instance_id))

    inst_path = os.path.join(base, 'instance.json')
    with open(inst_path) as f:
        inst = json.load(f)

    jc_path = os.path.join(base, 'JingweiComputations.json')
    jc = None
    if os.path.exists(jc_path):
        with open(jc_path) as f:
            jc = json.load(f)

    yaron_path = os.path.join(base, 'PandoraSolutionYaron.json')
    yaron = None
    if os.path.exists(yaron_path):
        with open(yaron_path) as f:
            yaron = json.load(f)

    return inst, jc, yaron


@pytest.fixture(scope='session')
def all_instance_ids():
    return _list_instance_ids()


@pytest.fixture(scope='session')
def all_instances():
    """Load all test instances. Returns list of (inst_dict, jingwei_dict, yaron_dict)."""
    ids = _list_instance_ids()
    instances = []
    for iid in ids:
        try:
            instances.append(_load_instance(iid))
        except Exception:
            pass
    return instances


@pytest.fixture(scope='session')
def small_instances():
    """Load a subset of small instances (N <= 4) for fast tests."""
    ids = _list_instance_ids()
    instances = []
    for iid in ids:
        try:
            inst, jc, yaron = _load_instance(iid)
            if inst['N'] <= 4:
                instances.append((inst, jc, yaron))
        except Exception:
            pass
    return instances
