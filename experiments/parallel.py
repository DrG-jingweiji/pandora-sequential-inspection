"""Parallel execution engine with crash-recovery checkpointing.

Runs experiment instances across multiple CPU cores using
ProcessPoolExecutor. Completed results are checkpointed to JSONL files
so that interrupted runs can resume from where they stopped.

Usage by experiment modules:
    1. Define a top-level worker function that calls get_shared() for data
    2. Call generate_instance_tasks() to create deterministic task lists
    3. Call run_parallel() to execute with checkpointing
"""

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


# ── Shared worker state ──────────────────────────────────────────────

_shared_data = {}


def _init_worker(data):
    """Process initializer: called once per worker to set shared data."""
    global _shared_data
    _shared_data = data


def get_shared(key=None):
    """Access shared data inside a worker process.

    Call without arguments to get the full dict, or with a key string.
    """
    if key is None:
        return _shared_data
    return _shared_data[key]


# ── Overall progress tracking ────────────────────────────────────────

class OverallProgress:
    """Tracks cumulative progress across multiple experiments.

    Creates a persistent tqdm bar (position=1) that updates as tasks
    complete from any experiment, showing total ETA.
    """

    def __init__(self, total_tasks):
        self.total = total_tasks
        self.completed = 0
        self.start_time = time.time()
        self.bar = tqdm(
            total=total_tasks,
            desc="Overall",
            position=1,
            leave=True,
            bar_format=(
                '{desc}: {percentage:3.0f}%|{bar}| '
                '{n_fmt}/{total_fmt} tasks '
                '[{elapsed}<{remaining}, {rate_fmt}]'
            ),
        )

    def advance(self, n=1):
        """Record n completed tasks."""
        self.completed += n
        self.bar.update(n)

    def skip_done(self, n):
        """Account for tasks loaded from checkpoint (already done)."""
        self.completed += n
        self.bar.update(n)

    def close(self):
        self.bar.close()
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        parts = []
        if h:
            parts.append(f"{h}h")
        if m:
            parts.append(f"{m}m")
        parts.append(f"{s}s")
        print(f"\nAll experiments finished in {''.join(parts)}"
              f"  ({self.completed} tasks)")


_overall = None


def set_overall_progress(tracker):
    """Set (or clear) the global overall-progress tracker."""
    global _overall
    _overall = tracker


# ── Checkpointing ────────────────────────────────────────────────────

def _checkpoint_dir(output_dir):
    return os.path.join(output_dir, '.checkpoints')


def checkpoint_path_for(output_dir, experiment_name):
    """Return the canonical checkpoint file path for an experiment."""
    return os.path.join(_checkpoint_dir(output_dir), f'{experiment_name}.jsonl')


def _load_checkpoint(path):
    """Load completed results from a JSONL checkpoint file.

    Each line is ``{"key": str, "result": dict}``.  Corrupted trailing
    lines (from mid-write crashes) are silently skipped.

    Returns (meta_dict_or_None, {key: result}).
    """
    meta = None
    completed = {}
    if not os.path.exists(path):
        return meta, completed
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry['key'] == '__meta__':
                    meta = entry['result']
                else:
                    completed[entry['key']] = entry['result']
            except (json.JSONDecodeError, KeyError):
                continue
    return meta, completed


def _append_checkpoint(path, key, result):
    """Append one completed result to the JSONL checkpoint."""
    with open(path, 'a') as f:
        f.write(json.dumps({'key': key, 'result': result}) + '\n')


def clear_checkpoint(path):
    """Remove a single checkpoint file."""
    if os.path.exists(path):
        os.remove(path)
        print(f"  Cleared checkpoint: {os.path.basename(path)}")


def clear_all_checkpoints(output_dir):
    """Remove every checkpoint file in the output directory."""
    ckpt_dir = _checkpoint_dir(output_dir)
    if not os.path.exists(ckpt_dir):
        return
    removed = 0
    for fname in os.listdir(ckpt_dir):
        if fname.endswith('.jsonl'):
            os.remove(os.path.join(ckpt_dir, fname))
            removed += 1
    if removed:
        print(f"  Cleared {removed} checkpoint file(s)")


# ── Task generation ──────────────────────────────────────────────────

def generate_instance_tasks(n_range, n_instances_fn, n_pool, seed,
                            legacy_sampling=False):
    """Pre-generate deterministic (key, N, rep, indices) task tuples.

    Uses a single RNG consumed in the same sequential order as the
    original code so that instances are reproducible regardless of the
    parallel execution order.

    Parameters
    ----------
    n_range : iterable of int
        Box counts to iterate over.
    n_instances_fn : callable(int) -> int
        Given N, return the number of instances.
    n_pool : int
        Size of the prototypical box pool (``len(selected_boxes)``).
    seed : int
        RNG seed for reproducibility.
    legacy_sampling : bool
        If True, use ``np.random.RandomState(seed).randint`` in the same
        stream order as the old notebooks.  If False, use NumPy's modern
        ``default_rng``.

    Returns
    -------
    list of (key_str, N, rep_idx, indices_list)
    """
    if legacy_sampling:
        rng = np.random.RandomState(seed)
        draw_indices = lambda n: rng.randint(0, n_pool, size=n).tolist()
    else:
        rng = np.random.default_rng(seed)
        draw_indices = lambda n: rng.integers(0, n_pool, size=n).tolist()

    tasks = []
    for N in n_range:
        n_inst = n_instances_fn(N)
        for rep in range(n_inst):
            indices = draw_indices(N)
            tasks.append((f"{N}_{rep}", N, rep, indices))
    return tasks


# ── Parallel runner ──────────────────────────────────────────────────

def run_parallel(worker_fn, tasks, shared_data, n_workers=3,
                 checkpoint_path=None, desc="Processing"):
    """Run tasks in parallel with optional crash-recovery checkpointing.

    Parameters
    ----------
    worker_fn : callable
        Top-level function receiving ``(task[1], task[2], ...)`` and
        returning a JSON-serializable dict.  Must access shared data
        via ``get_shared()``.
    tasks : list of tuples
        Each tuple is ``(key, *worker_args)``.  *key* is a string used
        for checkpoint lookup.
    shared_data : dict
        Passed to every worker process via the pool initializer.
    n_workers : int
        Number of parallel processes.  Use 1 for sequential execution.
    checkpoint_path : str or None
        Path to a JSONL file for crash recovery.  ``None`` disables
        checkpointing.
    desc : str
        Label for the progress bar.

    Returns
    -------
    dict : ``{key: result_dict}``
    """
    completed = {}
    if checkpoint_path:
        ckpt_dir = os.path.dirname(checkpoint_path)
        os.makedirs(ckpt_dir, exist_ok=True)
        meta, completed = _load_checkpoint(checkpoint_path)

        expected_total = len(tasks)
        if meta and meta.get('total_tasks') != expected_total:
            print(f"  WARNING: Checkpoint expects {meta['total_tasks']} tasks "
                  f"but current run has {expected_total}. "
                  f"Use --fresh to start over.")

        if meta is None:
            _append_checkpoint(checkpoint_path, '__meta__',
                               {'total_tasks': expected_total})

    remaining = [t for t in tasks if t[0] not in completed]
    n_total = len(tasks)
    n_done = n_total - len(remaining)
    results = dict(completed)

    if n_done > 0:
        print(f"  Checkpoint: {n_done}/{n_total} done, "
              f"{len(remaining)} remaining")
        if _overall:
            _overall.skip_done(n_done)

    if not remaining:
        print(f"  All {n_total} tasks already completed (from checkpoint)")
        return results

    if n_workers <= 1:
        _init_worker(shared_data)
        for task in tqdm(remaining, desc=desc, position=0, leave=True):
            key, *args = task
            try:
                result = worker_fn(*args)
                results[key] = result
                if checkpoint_path:
                    _append_checkpoint(checkpoint_path, key, result)
                if _overall:
                    _overall.advance(1)
            except Exception as e:
                print(f"\n  Error on {key}: {e}")
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(shared_data,),
        ) as executor:
            future_to_key = {}
            for task in remaining:
                key, *args = task
                future = executor.submit(worker_fn, *args)
                future_to_key[future] = key

            for future in tqdm(
                as_completed(future_to_key),
                total=len(remaining),
                desc=desc,
                position=0,
                leave=True,
            ):
                key = future_to_key[future]
                try:
                    result = future.result()
                    results[key] = result
                    if checkpoint_path:
                        _append_checkpoint(checkpoint_path, key, result)
                    if _overall:
                        _overall.advance(1)
                except Exception as e:
                    print(f"\n  Error on {key}: {e}")
                    traceback.print_exc()

    return results
