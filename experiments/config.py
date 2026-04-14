"""Shared configuration for numerical experiments.

Matches the paper's experimental setup (Section 6):
  - Support size 5, up to 3 types per box
  - Values from U(0,10), c_F from U(0,5), c_P from U(0,3)
  - c_F >= c_P filter
  - 1000 small instances (N=2..9), 300 large instances (N=10..16)
"""

import os

SMALL_N_RANGE = range(2, 10)       # N = 2..9
LARGE_N_RANGE = range(10, 17)      # N = 10..16
SMALL_INSTANCES = 1000
LARGE_INSTANCES = 300
SUPPORT_SIZE = 5
MAX_TYPES = 3
VALUE_RANGE = (0, 10)
CF_RANGE = (0, 5)
CP_RANGE = (0, 3)
SEED = 0
DP_CUTOFF = 9                      # OPT only computed for N <= DP_CUTOFF
NUM_PROTOTYPICAL_BOXES = 100
BOX_DISTANCE = 0.5
DEFAULT_WORKERS = 3                # parallel processes (override with --workers)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
