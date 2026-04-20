"""Shared configuration for numerical experiments.

All parameters match the paper's experimental setup (Section 6,
"The Pandora's Box Problem with Sequential Inspections" by Aouad,
Ji, and Shaposhnik):

  - Support size 5, up to 3 types per box
  - Values drawn from U(0,10), c_F from U(0,5), c_P from U(0,3)
  - c_F >= c_P filter (partial inspection is never more expensive)
  - 1000 instances for small N (2..9), 300 for large N (10..16)
  - DP cutoff at N=9 (optimal policy only computed for N <= 9)
  - 100 prototypical boxes with min distance 0.5 in (σ^F, σ^P) space

To adjust the experiments (e.g., fewer instances for a quick test),
modify these constants or use the --small CLI flag.
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
