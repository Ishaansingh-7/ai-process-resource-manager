"""
anomaly.py
==========
Phase 3, part 1: is this process behaving unlike itself?

Two independent checks run on every process:

1. **Statistical spike (3-sigma).**  From ``data/process_data.csv`` we build a
   baseline for each *program name* - the mean and standard deviation of its
   CPU and memory.  A process is flagged when its current reading sits more
   than three standard deviations above its own historical mean.  Grouping by
   name rather than PID is deliberate: PIDs change every time a program
   restarts, but ``chrome.exe`` should behave roughly like ``chrome.exe`` did
   yesterday.

2. **Monotonic memory growth (leak heuristic).**  Independently of any
   history, if a process's memory went up in five consecutive live samples
   with no drop at all, that is the shape of a leak.  This uses the live
   rolling window from history.py, not the CSV, because we want to know what
   is leaking *now*.

Why three sigma?  For a roughly normal distribution about 99.7 % of samples
fall within three standard deviations of the mean, so "beyond 3σ" means
"something this program has essentially never done before".

The two traps this module has to avoid
--------------------------------------
* **Zero-variance baselines.**  Most processes sit at exactly 0.00 % CPU
  forever, so their standard deviation is 0 and *any* activity at all is
  infinitely many sigmas away.  Every threshold therefore has a floor
  (``CPU_STD_FLOOR``, ``MEMORY_STD_FLOOR``) and a minimum absolute value worth
  reporting (``CPU_MIN_TO_FLAG``, ``MEMORY_MIN_TO_FLAG``).  Without those, a
  quiet machine produces hundreds of meaningless alerts.
* **Not enough history.**  A program we have only seen twice has no meaningful
  baseline.  Those are reported as ``insufficient_data``, never as anomalies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

# pandas is only needed to build the baselines.  As in Phase 2, the dashboard
# must survive without the scientific stack installed.
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_DIR / "data" / "process_data.csv"


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

SIGMA = 3.0                     # how many standard deviations counts as a spike
MIN_BASELINE_SAMPLES = 20       # rows needed before a name's baseline is trusted

# Floors that stop zero-variance baselines from flagging everything.
CPU_STD_FLOOR = 0.5             # percent
MEMORY_STD_FLOOR = 5.0          # MB

# A spike below these is technically unusual but not worth anyone's attention.
CPU_MIN_TO_FLAG = 1.0           # percent
MEMORY_MIN_TO_FLAG = 50.0       # MB

# Leak heuristic: N rises in a row (so N+1 samples) totalling at least this
# much growth.  The growth floor filters out normal working-set jitter, which
# wanders by a fraction of a MB on every sample.
LEAK_CONSECUTIVE_RISES = 5
LEAK_MIN_TOTAL_GROWTH_MB = 10.0

# Kinds, used by the GUI to pick an icon and a colour.
KIND_NORMAL = "normal"
KIND_CPU_SPIKE = "cpu_spike"
KIND_MEMORY_SPIKE = "memory_spike"
KIND_MEMORY_LEAK = "memory_leak"
KIND_INSUFFICIENT = "insufficient_data"

# Used only to order the Alerts panel: a suspected leak is more actionable
# than a one-off spike, so it goes to the top.  This changes nothing about
# *what* counts as an anomaly.
KIND_PRIORITY = {
    KIND_MEMORY_LEAK: 3,
    KIND_CPU_SPIKE: 2,
    KIND_MEMORY_SPIKE: 2,
    KIND_NORMAL: 0,
    KIND_INSUFFICIENT: 0,
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Baseline:
    """What one program normally does, learned from the CSV."""

    name: str
    samples: int
    cpu_mean: float
    cpu_std: float
    memory_mean: float
    memory_std: float

    def cpu_limit(self, sigma: float = SIGMA) -> float:
        """The CPU reading above which this program is behaving unusually."""
        return self.cpu_mean + sigma * max(self.cpu_std, CPU_STD_FLOOR)

    def memory_limit(self, sigma: float = SIGMA) -> float:
        return self.memory_mean + sigma * max(self.memory_std, MEMORY_STD_FLOOR)


@dataclass
class AnomalyResult:
    """The verdict for one process."""

    is_anomaly: bool
    reason: str
    kind: str = KIND_NORMAL
    # How far past the line this went - sigmas for a spike, MB grown for a
    # leak.  Only used to rank the Alerts panel, never to decide is_anomaly.
    score: float = 0.0

    @property
    def has_baseline(self) -> bool:
        return self.kind != KIND_INSUFFICIENT

    @property
    def rank(self) -> tuple:
        """Sort key for the Alerts panel: worst first."""
        return (KIND_PRIORITY.get(self.kind, 0), self.score)


NO_BASELINE = AnomalyResult(False, "Insufficient history to judge this process",
                            KIND_INSUFFICIENT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe_spike(metric: str, current: float, mean: float,
                    limit: float, unit: str) -> str:
    """
    Phrase a spike in the most informative way available.

    When the baseline mean is a real number, "3.2x above baseline" is the
    clearest way to say it.  When the mean is ~0 (a process that normally does
    nothing at all) a ratio would be meaningless or infinite, so we fall back
    to stating the actual numbers.
    """
    if mean > 0.05:
        ratio = current / mean
        return (f"{metric} {ratio:.1f}x above baseline "
                f"({current:,.1f}{unit} vs {mean:,.1f}{unit} average)")
    return (f"{metric} {current:,.1f}{unit}, baseline is "
            f"{mean:,.2f}{unit} (limit {limit:,.1f}{unit})")


def detect_memory_leak(samples: Sequence,
                       rises: int = LEAK_CONSECUTIVE_RISES,
                       min_growth_mb: float = LEAK_MIN_TOTAL_GROWTH_MB
                       ) -> Optional[tuple]:
    """
    Look for memory that only ever goes up.

    Needs ``rises + 1`` samples to see ``rises`` steps.  Every step must be a
    strict increase - a single dip means the process gave memory back, which
    is exactly what a leaking process never does.

    Returns ``(reason, growth_mb)`` or None.
    """
    if len(samples) < rises + 1:
        return None

    window = list(samples)[-(rises + 1):]
    for earlier, later in zip(window, window[1:]):
        if later.memory_mb <= earlier.memory_mb:
            return None

    growth = window[-1].memory_mb - window[0].memory_mb
    if growth < min_growth_mb:
        return None    # real growth, but too small to mean anything

    return (f"Memory increasing for {rises} consecutive samples - "
            f"possible leak (+{growth:,.1f} MB)", growth)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """
    Flags processes behaving unlike their own history.

    Like the Phase 2 classifier this never raises: if the dataset is missing
    or pandas is not installed it reports itself unavailable through
    :attr:`available` / :attr:`status` and the dashboard carries on.
    """

    def __init__(self,
                 csv_path: Optional[Path] = None,
                 sigma: float = SIGMA,
                 min_samples: int = MIN_BASELINE_SAMPLES) -> None:
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        self.sigma = sigma
        self.min_samples = min_samples

        self.available = False
        self.status = "not loaded"
        self._baselines: Dict[str, Baseline] = {}

        self._load_baselines()

    # ------------------------------------------------------------------
    # Start-up
    # ------------------------------------------------------------------

    def _load_baselines(self) -> None:
        """Summarise the CSV into one Baseline per program name."""
        if not PANDAS_AVAILABLE:
            self.status = "pandas not installed (pip install -r requirements.txt)"
            return
        if not self.csv_path.exists():
            self.status = "no history yet - run: python main.py --collect-only"
            return

        try:
            frame = pd.read_csv(self.csv_path,
                                usecols=["name", "cpu_percent", "memory_mb"])
            # The CSV is appended to across runs, is readable while the
            # collector is still writing it, and is easy to open in Excel and
            # save back.  Any of those can leave a value that is not a number,
            # which would turn the whole column into text and make the
            # aggregation below raise.  Coerce first and drop what will not
            # convert - the same thing model.load_dataset() does.
            for column in ("cpu_percent", "memory_mb"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        except Exception as exc:
            self.status = f"could not read history: {exc}"
            log.warning("Failed to read %s: %s", self.csv_path, exc)
            return

        frame = frame.dropna(subset=["name", "cpu_percent", "memory_mb"])
        if frame.empty:
            self.status = "history file is empty"
            return

        # One row per program name: how many times we saw it, and the mean and
        # spread of its CPU and memory.
        grouped = frame.groupby("name").agg(
            samples=("cpu_percent", "size"),
            cpu_mean=("cpu_percent", "mean"),
            cpu_std=("cpu_percent", "std"),
            memory_mean=("memory_mb", "mean"),
            memory_std=("memory_mb", "std"),
        )
        # std() is NaN for a name seen exactly once; treat that as no spread.
        grouped = grouped.fillna(0.0)

        # Names we have barely seen get no baseline at all, so they can only
        # ever be reported as "insufficient data".
        grouped = grouped[grouped["samples"] >= self.min_samples]

        self._baselines = {
            str(name): Baseline(
                name=str(name),
                samples=int(row.samples),
                cpu_mean=float(row.cpu_mean),
                cpu_std=float(row.cpu_std),
                memory_mean=float(row.memory_mean),
                memory_std=float(row.memory_std),
            )
            for name, row in grouped.iterrows()
        }

        if not self._baselines:
            self.status = (f"not enough history yet "
                           f"(need {self.min_samples}+ samples per program)")
            return

        self.available = True
        self.status = (f"{len(self._baselines)} baselines from "
                       f"{len(frame):,} rows")
        log.info("Anomaly baselines ready: %s", self.status)

    def reload(self) -> None:
        """Rebuild the baselines - useful after collecting more data."""
        self.available = False
        self._baselines = {}
        self._load_baselines()

    def baseline_for(self, name: str) -> Optional[Baseline]:
        return self._baselines.get(name)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, processes: Sequence, history=None) -> Dict[int, AnomalyResult]:
        """
        Judge every process in a snapshot, keyed by PID.

        ``history`` is a :class:`history.ProcessHistory`; without it the leak
        check is skipped but the 3-sigma check still runs.
        """
        results: Dict[int, AnomalyResult] = {}
        for process in processes:
            results[process.pid] = self._evaluate_one(process, history)
        return results

    def _evaluate_one(self, process, history) -> AnomalyResult:
        reasons: List[str] = []
        kind = KIND_NORMAL
        score = 0.0

        # --- check 1: does this look like a leak, right now? --------------
        # This runs even with no CSV baseline, because it only needs the live
        # window - which is also why it is checked first.
        if history is not None:
            # Ask only for the samples the check reads: the window now holds
            # ~5 minutes for the charts, and copying all of it 340 times per
            # refresh would be wasted work.
            leak = detect_memory_leak(
                history.samples(process.pid, last=LEAK_CONSECUTIVE_RISES + 1))
            if leak:
                reason, growth = leak
                reasons.append(reason)
                kind = KIND_MEMORY_LEAK
                score = growth

        # --- check 2: is this reading far outside the norm? ---------------
        baseline = self._baselines.get(process.name)
        if baseline is None:
            if reasons:                       # leak found, but no baseline yet
                return AnomalyResult(True, "; ".join(reasons), kind, score)
            return NO_BASELINE

        cpu_limit = baseline.cpu_limit(self.sigma)
        if process.cpu_percent > cpu_limit and process.cpu_percent >= CPU_MIN_TO_FLAG:
            reasons.append(_describe_spike("CPU", process.cpu_percent,
                                           baseline.cpu_mean, cpu_limit, " %"))
            sigmas = ((process.cpu_percent - baseline.cpu_mean)
                      / max(baseline.cpu_std, CPU_STD_FLOOR))
            if kind == KIND_NORMAL:
                kind = KIND_CPU_SPIKE
                score = sigmas

        memory_limit = baseline.memory_limit(self.sigma)
        if process.memory_mb > memory_limit and process.memory_mb >= MEMORY_MIN_TO_FLAG:
            reasons.append(_describe_spike("Memory", process.memory_mb,
                                           baseline.memory_mean, memory_limit, " MB"))
            sigmas = ((process.memory_mb - baseline.memory_mean)
                      / max(baseline.memory_std, MEMORY_STD_FLOOR))
            if kind == KIND_NORMAL:
                kind = KIND_MEMORY_SPIKE
                score = sigmas

        if reasons:
            return AnomalyResult(True, "; ".join(reasons), kind, score)
        return AnomalyResult(
            False,
            f"Normal for {process.name} "
            f"(baseline {baseline.cpu_mean:.2f} % CPU, "
            f"{baseline.memory_mean:,.0f} MB from {baseline.samples:,} samples)",
            KIND_NORMAL)


# ---------------------------------------------------------------------------
# Manual test:  python anomaly.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from history import ProcessHistory
    from monitor import ProcessMonitor

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    detector = AnomalyDetector()
    print(f"detector: {detector.status}\n")

    monitor = ProcessMonitor()
    history = ProcessHistory()
    monitor.list_processes()          # first pass establishes the CPU baseline

    # Take several samples so the leak check has a window to work with.
    print("sampling", end="", flush=True)
    processes: list = []
    for _ in range(7):
        time.sleep(1.0)
        processes = monitor.list_processes()
        history.update(processes)
        print(".", end="", flush=True)
    print("\n")

    started = time.perf_counter()
    results = detector.evaluate(processes, history)
    print(f"evaluated {len(results)} processes in "
          f"{(time.perf_counter() - started) * 1000:.1f} ms\n")

    flagged = [(p, results[p.pid]) for p in processes if results[p.pid].is_anomaly]
    print(f"{len(flagged)} anomaly(ies):")
    for process, result in flagged:
        print(f"  [{result.kind}] {process.name} (PID {process.pid})\n"
              f"      {result.reason}")
    if not flagged:
        print("  none - the machine is behaving normally")

    kinds: Dict[str, int] = {}
    for result in results.values():
        kinds[result.kind] = kinds.get(result.kind, 0) + 1
    print("\nverdict breakdown:", kinds)
