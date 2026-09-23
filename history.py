"""
history.py
==========
Short rolling windows of recent measurements - per process, and for the
machine as a whole.

Phases 1 and 2 only ever look at *one* snapshot at a time.  Phases 3 and 5 ask
questions a single snapshot cannot answer:

    "has this process's memory gone up five times in a row?"     (anomaly.py)
    "where will its memory be in 60 seconds?"                    (predictor_trend.py)
    "draw the last five minutes of CPU usage"                    (charts.py)

They all read from here rather than from the CSV.  Re-parsing a 1.6 MB
``process_data.csv`` on every 2.5-second refresh would cost more than every
other part of a refresh combined; a deque of floats costs nothing and is
already in the right order.

Both classes cap themselves with ``maxlen``, so memory use stays flat no
matter how long the dashboard runs - the deque discards the oldest sample
automatically as each new one arrives.
"""

from __future__ import annotations

import time
from collections import deque
from itertools import islice
from typing import Deque, Dict, Iterable, List, NamedTuple, Optional

# 120 samples is about five minutes at the default 2.5 s refresh, which is the
# span the per-process detail charts draw.  The cost is roughly 340 processes
# x 120 samples x ~120 bytes, so about 5 MB - the price of having any history
# to plot at all.  The Phase 3 analysers only ever ask for the last handful.
DEFAULT_WINDOW = 120

# The system-wide buffer is one sample per refresh rather than one per
# process, so it can afford to be longer: ~10 minutes at 2.5 s.
DEFAULT_SYSTEM_WINDOW = 240


class Sample(NamedTuple):
    """One measurement of one process."""

    t: float            # time.monotonic() seconds - immune to clock changes
    cpu_percent: float
    memory_mb: float
    thread_count: int
    # Context switches per second at this instant (see monitor.py).  Defaulted
    # so any older code building a Sample by hand keeps working.
    ctx_per_sec: float = 0.0


class SystemSample(NamedTuple):
    """One measurement of the machine as a whole."""

    t: float
    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float


class ProcessHistory:
    """
    Keeps the last N samples for each live PID.

    Dead processes are dropped on the next update, so the window never grows
    without bound on a machine that starts and stops a lot of processes.
    """

    def __init__(self, maxlen: int = DEFAULT_WINDOW) -> None:
        self.maxlen = maxlen
        self._windows: Dict[int, Deque[Sample]] = {}
        self._names: Dict[int, str] = {}

    # ------------------------------------------------------------------

    def update(self, processes: Iterable, now: Optional[float] = None) -> None:
        """Append the current reading for every process and forget the dead ones."""
        if now is None:
            now = time.monotonic()

        seen = set()
        for process in processes:
            seen.add(process.pid)

            # Two cases handled by one check: a PID we have never seen, and a
            # PID the OS has recycled for a different program.  Either way the
            # old readings belong to somebody else, so start fresh.
            if self._names.get(process.pid) != process.name:
                self._windows[process.pid] = deque(maxlen=self.maxlen)
                self._names[process.pid] = process.name

            self._windows[process.pid].append(
                Sample(now, process.cpu_percent, process.memory_mb,
                       process.thread_count,
                       getattr(process, "ctx_switches_per_sec", 0.0)))

        for pid in list(self._windows):
            if pid not in seen:
                del self._windows[pid]
                self._names.pop(pid, None)

    # ------------------------------------------------------------------

    def samples(self, pid: int, last: Optional[int] = None) -> List[Sample]:
        """
        The stored samples for one PID, oldest first (empty if unknown).

        Pass ``last`` to copy only the newest few.  The analysers do that on
        every refresh for every process, and copying 120 samples 340 times
        when six are needed is pure waste.
        """
        window = self._windows.get(pid)
        if not window:
            return []
        if last is not None and last < len(window):
            # islice avoids materialising the whole deque just to slice it.
            return list(islice(window, len(window) - last, len(window)))
        return list(window)

    def count(self, pid: int) -> int:
        """How many samples we have for this PID so far."""
        window = self._windows.get(pid)
        return len(window) if window else 0

    def __len__(self) -> int:
        return len(self._windows)

    def __contains__(self, pid: object) -> bool:
        return pid in self._windows


class SystemHistory:
    """
    Machine-wide CPU and RAM over time, for the live chart.

    One sample per refresh, appended by the GUI as each snapshot arrives.
    """

    def __init__(self, maxlen: int = DEFAULT_SYSTEM_WINDOW) -> None:
        self.maxlen = maxlen
        self._samples: Deque[SystemSample] = deque(maxlen=maxlen)

    def add(self, system_info, now: Optional[float] = None) -> None:
        """Record one SystemInfo (see monitor.SystemInfo)."""
        self._samples.append(SystemSample(
            t=time.monotonic() if now is None else now,
            cpu_percent=system_info.cpu_percent,
            memory_used_mb=system_info.memory_used_mb,
            memory_total_mb=system_info.memory_total_mb,
        ))

    def samples(self, last: Optional[int] = None) -> List[SystemSample]:
        if last is not None and last < len(self._samples):
            return list(islice(self._samples, len(self._samples) - last,
                               len(self._samples)))
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
