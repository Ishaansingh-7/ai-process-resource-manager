"""
predictor_trend.py
==================
Phase 3, part 2: where is this process heading?

anomaly.py answers "is this reading strange *now*?".  This module answers the
forward-looking question: "if the last few seconds continue, what happens
next?"

The method
----------
For each process we take its last 5-10 readings from the live rolling window
(history.py) and fit a straight line through them with ``numpy.polyfit``:

    value(t) = slope * t + intercept

The slope is the rate of change - percent of CPU per second, or MB of memory
per second.  Extrapolating that line forward gives the time until the process
would cross a high-usage threshold:

    seconds_to_limit = (limit - current_value) / slope        (slope > 0)

If that lands inside the warning horizon (60 s by default) we raise a warning.
This is deliberately the simplest possible forecast: a straight line through a
handful of points.  It cannot predict a process that is about to *start* doing
something, only one that is already on its way.

A note on the CPU threshold
---------------------------
``CPU_WARN_PERCENT`` is 90 %, and monitor.py reports CPU as a share of the
*whole machine* (Task Manager's convention), so 90 % means "nearly the entire
CPU".  On a 16-thread machine a single-threaded process maxes out around 6 %,
so this threshold will essentially never fire for one process.  Two ways to
make it meaningful, depending on what you want to demonstrate:

* collect and run with ``--raw-cpu``, where 100 % means one full core, or
* lower ``CPU_WARN_PERCENT`` below (5 % is roughly "saturating one core" on a
  16-thread machine).

The memory warnings fire on ordinary desktop behaviour and need no tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import psutil

log = logging.getLogger(__name__)

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

CPU_WARN_PERCENT = 90.0          # see the note in the docstring above
MEMORY_WARN_FRACTION = 0.25      # of total RAM (4 GB on a 16 GB machine)
HORIZON_SECONDS = 60.0           # only warn about the next minute
MIN_SAMPLES = 5                  # fewer than this is not a trend, it is noise
TREND_WINDOW = 10                # how many recent samples the line is fitted to

# Sustained growth worth mentioning even when the absolute threshold is far
# away - this is the "check for leak" warning.
MEMORY_GROWTH_WARN_MB_PER_MIN = 30.0

# Below these, a series counts as flat and we skip the fit entirely.  Working
# sets jitter by a fraction of a MB constantly; fitting a line to that would
# produce meaningless slopes (and 300+ pointless polyfit calls per refresh).
CPU_FLAT_EPSILON = 0.05          # percent
MEMORY_FLAT_EPSILON = 0.5        # MB

KIND_NONE = "none"
KIND_INSUFFICIENT = "insufficient_data"
KIND_CPU_THRESHOLD = "cpu_threshold"
KIND_MEMORY_THRESHOLD = "memory_threshold"
KIND_MEMORY_GROWTH = "memory_growth"

# Ordering for the Alerts panel: an imminent threshold crossing beats a slow
# upward drift.
KIND_PRIORITY = {
    KIND_CPU_THRESHOLD: 3,
    KIND_MEMORY_THRESHOLD: 3,
    KIND_MEMORY_GROWTH: 2,
    KIND_NONE: 0,
    KIND_INSUFFICIENT: 0,
}


@dataclass
class Forecast:
    """Where one process is heading, based on its recent readings."""

    has_warning: bool = False
    message: str = ""
    kind: str = KIND_NONE
    cpu_rate_per_min: float = 0.0        # percentage points per minute
    memory_rate_per_min: float = 0.0     # MB per minute
    seconds_to_cpu_limit: Optional[float] = None
    seconds_to_memory_limit: Optional[float] = None
    score: float = 0.0                   # ranking only, bigger = more urgent

    @property
    def rank(self) -> tuple:
        return (KIND_PRIORITY.get(self.kind, 0), self.score)


INSUFFICIENT = Forecast(False, "Not enough samples yet to establish a trend",
                        KIND_INSUFFICIENT)
STEADY = Forecast(False, "No significant trend", KIND_NONE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fit_slope(times: Sequence[float], values: Sequence[float],
              flat_epsilon: float) -> float:
    """
    Least-squares slope of ``values`` against ``times``, in units per second.

    Returns 0.0 for a series that barely moves, which skips the fit for the
    hundreds of idle processes that sit at a constant value.
    """
    if len(values) < 2:
        return 0.0
    if max(values) - min(values) < flat_epsilon:
        return 0.0
    if max(times) - min(times) <= 0:
        # Every reading carries the same timestamp, so there is no time axis
        # to fit against and polyfit raises LinAlgError.  "No measurable
        # slope" is the honest answer, and it matches the flat-series case.
        return 0.0
    # polyfit(..., 1) fits a straight line and returns [slope, intercept].
    slope, _intercept = np.polyfit(times, values, 1)
    return float(slope)


def seconds_until(current: float, limit: float, slope_per_second: float) -> Optional[float]:
    """
    How long until a rising value reaches ``limit``.

    None when it is not rising, or when it is already past the limit (that is
    a fact, not a forecast - anomaly.py reports those).
    """
    if slope_per_second <= 0 or current >= limit:
        return None
    return (limit - current) / slope_per_second


# ---------------------------------------------------------------------------
# The forecaster
# ---------------------------------------------------------------------------


class TrendPredictor:
    """
    Fits a short linear trend per process and warns about the near future.

    Stateless apart from its thresholds: it reads the rolling window owned by
    the GUI, so nothing has to be kept in sync.
    """

    def __init__(self,
                 cpu_threshold: float = CPU_WARN_PERCENT,
                 memory_threshold_mb: Optional[float] = None,
                 horizon_seconds: float = HORIZON_SECONDS,
                 min_samples: int = MIN_SAMPLES,
                 window: int = TREND_WINDOW) -> None:
        self.cpu_threshold = cpu_threshold
        self.horizon_seconds = horizon_seconds
        self.min_samples = min_samples
        self.window = window

        if memory_threshold_mb is None:
            total_mb = psutil.virtual_memory().total / (1024 * 1024)
            memory_threshold_mb = MEMORY_WARN_FRACTION * total_mb
        self.memory_threshold_mb = memory_threshold_mb

        self.available = NUMPY_AVAILABLE
        self.status = (
            f"linear fit over last {window} samples, {horizon_seconds:.0f}s horizon"
            if NUMPY_AVAILABLE else
            "numpy not installed (pip install -r requirements.txt)")

    # ------------------------------------------------------------------

    def evaluate(self, processes: Sequence, history) -> Dict[int, Forecast]:
        """Forecast every process in a snapshot, keyed by PID."""
        if not self.available or history is None:
            return {}
        # Only the newest `window` samples are fitted, so only those are
        # copied - the stored history is much longer now (it feeds the charts).
        return {p.pid: self._evaluate_one(p, history.samples(p.pid, last=self.window))
                for p in processes}

    def _evaluate_one(self, process, samples: List) -> Forecast:
        if len(samples) < self.min_samples:
            return INSUFFICIENT

        window = samples[-self.window:]
        start = window[0].t
        times = [s.t - start for s in window]
        if times[-1] <= 0:          # every sample carries the same timestamp
            return INSUFFICIENT

        cpu_slope = fit_slope(times, [s.cpu_percent for s in window],
                              CPU_FLAT_EPSILON)
        memory_slope = fit_slope(times, [s.memory_mb for s in window],
                                 MEMORY_FLAT_EPSILON)

        to_cpu_limit = seconds_until(process.cpu_percent, self.cpu_threshold,
                                     cpu_slope)
        to_memory_limit = seconds_until(process.memory_mb,
                                        self.memory_threshold_mb, memory_slope)

        forecast = Forecast(
            cpu_rate_per_min=cpu_slope * 60.0,
            memory_rate_per_min=memory_slope * 60.0,
            seconds_to_cpu_limit=to_cpu_limit,
            seconds_to_memory_limit=to_memory_limit,
        )

        # --- warning 1: CPU about to saturate -----------------------------
        if to_cpu_limit is not None and to_cpu_limit <= self.horizon_seconds:
            forecast.has_warning = True
            forecast.kind = KIND_CPU_THRESHOLD
            forecast.message = (f"CPU may exceed {self.cpu_threshold:.0f}% soon "
                                f"(~{to_cpu_limit:.0f}s at "
                                f"+{forecast.cpu_rate_per_min:.1f}%/min)")
            forecast.score = self.horizon_seconds - to_cpu_limit
            return forecast

        # --- warning 2: memory about to cross the high-water mark ---------
        if to_memory_limit is not None and to_memory_limit <= self.horizon_seconds:
            forecast.has_warning = True
            forecast.kind = KIND_MEMORY_THRESHOLD
            forecast.message = (
                f"Memory may exceed {self.memory_threshold_mb / 1024:.1f} GB soon "
                f"(~{to_memory_limit:.0f}s at "
                f"+{forecast.memory_rate_per_min:,.0f} MB/min)")
            forecast.score = self.horizon_seconds - to_memory_limit
            return forecast

        # --- warning 3: climbing steadily, threshold still far away -------
        if forecast.memory_rate_per_min >= MEMORY_GROWTH_WARN_MB_PER_MIN:
            forecast.has_warning = True
            forecast.kind = KIND_MEMORY_GROWTH
            forecast.message = (f"Memory trending upward "
                                f"(+{forecast.memory_rate_per_min:,.0f} MB/min) "
                                f"- check for leak")
            forecast.score = forecast.memory_rate_per_min
            return forecast

        forecast.message = "No significant trend"
        return forecast


# ---------------------------------------------------------------------------
# Manual test:  python predictor_trend.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from history import ProcessHistory
    from monitor import ProcessMonitor

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    trend = TrendPredictor()
    print(f"trend predictor: {trend.status}")
    print(f"thresholds     : CPU {trend.cpu_threshold:.0f} %, "
          f"memory {trend.memory_threshold_mb:,.0f} MB\n")

    monitor = ProcessMonitor()
    history = ProcessHistory()
    monitor.list_processes()

    print("sampling", end="", flush=True)
    processes: list = []
    for _ in range(8):
        time.sleep(1.0)
        processes = monitor.list_processes()
        history.update(processes)
        print(".", end="", flush=True)
    print("\n")

    started = time.perf_counter()
    forecasts = trend.evaluate(processes, history)
    print(f"forecast {len(forecasts)} processes in "
          f"{(time.perf_counter() - started) * 1000:.1f} ms\n")

    warned = [(p, forecasts[p.pid]) for p in processes if forecasts[p.pid].has_warning]
    print(f"{len(warned)} warning(s):")
    for process, forecast in sorted(warned, key=lambda pf: pf[1].rank, reverse=True):
        print(f"  [{forecast.kind}] {process.name} (PID {process.pid})\n"
              f"      {forecast.message}")
    if not warned:
        print("  none")

    movers = sorted(processes,
                    key=lambda p: abs(forecasts[p.pid].memory_rate_per_min),
                    reverse=True)[:5]
    print("\nfastest-moving memory (MB/min):")
    for process in movers:
        forecast = forecasts[process.pid]
        print(f"  {process.name[:32]:<32} {forecast.memory_rate_per_min:>+9,.1f}  "
              f"cpu {forecast.cpu_rate_per_min:>+6.2f} %/min")
