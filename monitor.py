"""
monitor.py
==========
The OS-facing layer of the AI-Based Process Resource Manager.

The GUI and the CSV collector never talk to the operating system themselves.
They ask a :class:`ProcessMonitor` for a snapshot and get back plain Python
objects (:class:`ProcessInfo`, :class:`SystemInfo`, :class:`Snapshot`).
Keeping every OS call behind one small API is what makes the rest of the
project easy to read and easy to test.

Where the numbers come from
---------------------------
Two data sources, same output:

* **Windows** -> ``winprobe.WindowsProcessSource``: one native system call
  returns the whole process table (~6 ms).  See winprobe.py for why psutil is
  ~700x slower for this particular job on Windows.
* **Anything else** (or if the native call fails) -> psutil, which is fast on
  Linux/macOS because /proc is cheap to read.

System-wide totals (CPU %, RAM) always come from psutil: those are single
calls, so there is nothing to optimise.

How CPU % is calculated
-----------------------
The OS never hands out a ready-made "CPU %".  It only exposes a *counter*: how
many seconds of CPU time each process has consumed since it started.  We turn
that counter into a percentage by remembering the previous reading and
dividing the difference by the wall-clock time that passed in between:

    cpu% = (cpu_time_now - cpu_time_before) / (t_now - t_before) * 100

That is exactly what Task Manager does.  Because the baseline lives on the
ProcessMonitor instance, the GUI (sampling every 2.5 s) and the collector
(every 3 s) each keep their own and never disturb each other.

Context switches: what they are and why we count them
-----------------------------------------------------
A CPU core can run exactly one thread at a time.  A **context switch** is the
moment the operating system takes the core away from one thread and gives it
to another: it saves the first thread's registers, program counter and stack
pointer into its process control block, loads the next thread's, and jumps.
The switch itself does no useful work, so it is pure overhead - which is why
"context switches per second" is a direct, visible measure of how hard the
scheduler is working.

There are two reasons a switch happens, and the distinction is the whole
point of the two columns:

* **Voluntary** - the thread gave the CPU up *by itself*, before its turn was
  over, because it cannot continue: it asked to read a file, waited on a
  network socket, slept, or blocked on a lock.  The thread says "I have
  nothing to do until this finishes, take the CPU".  Lots of voluntary
  switches is the signature of an **I/O-bound** process.

* **Involuntary** (preemption) - the thread was still perfectly happy to run,
  but the scheduler *took* the CPU away: its time slice (quantum) expired, or
  a higher-priority thread became ready.  The thread did not consent and does
  not even notice.  Lots of involuntary switches is the signature of a
  **CPU-bound** process competing with others for the core.

So: voluntary = "I stopped", involuntary = "I was stopped".

An honest limitation, worth knowing before you are asked
--------------------------------------------------------
Linux exposes both counts separately (``/proc/<pid>/status`` gives
``voluntary_ctxt_switches`` and ``nonvoluntary_ctxt_switches``).  **Windows
does not.**  The kernel keeps a single ``ContextSwitches`` counter per thread
in ``SYSTEM_THREAD_INFORMATION`` and never records *why* the switch happened.
psutil papers over this by reporting that single total as ``voluntary`` and
hard-coding ``involuntary`` to 0 - measured on this machine, every one of the
~320 running processes comes back with ``involuntary=0``.

This project does not invent the split.  On Windows we report the true total
and leave the involuntary count as "not available"; on Linux/macOS the psutil
fallback fills in both for real.  A number that is always zero would look
like "nothing is ever preempted", which is false and the opposite of what the
counter is for.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import psutil

import winprobe
from winprobe import BASE_PRIORITY_LABELS, RawProcess

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BYTES_PER_MB = 1024 * 1024

# Windows PID 0 is the "System Idle Process": its CPU time *is* the machine's
# idle time, so leaving it in would put a fake 90 %-CPU row at the top of the
# table.  Task Manager hides it from its Processes tab for the same reason.
IDLE_PID = 0

# A process that is schedulable but used less CPU than this during the last
# interval is really just waiting for work, so we label it "sleeping".
CPU_ACTIVE_THRESHOLD = 0.1  # percent

# Column order of data/process_data.csv -- keep in sync with ProcessInfo.to_row().
CSV_COLUMNS = [
    "timestamp",
    "pid",
    "name",
    "cpu_percent",
    "memory_mb",
    "thread_count",
    "priority",
    "status",
    # Added in the final phase.  The cumulative counters are logged as well as
    # the rate because a rate is only meaningful next to the interval it was
    # measured over, and a later analysis may want to re-derive it.
    "ctx_switches_vol",
    "ctx_switches_invol",
    "ctx_switches_per_sec",
]

# What we ask psutil for in the fallback path, in one pass per process.
# Fields we are not allowed to read come back as None instead of raising.
_PSUTIL_ATTRS = ["pid", "name", "num_threads", "status", "memory_info", "nice",
                 "cpu_times", "num_ctx_switches"]

# psutil status values are plain lowercase strings.  We fold them into the
# four states this project cares about.
_STATUS_LABELS = {
    "running": "running",
    "sleeping": "sleeping",
    "idle": "sleeping",
    "disk-sleep": "waiting",
    "waiting": "waiting",
    "locked": "waiting",
    "stopped": "suspended",
    "suspended": "suspended",
    "tracing-stop": "suspended",
    "zombie": "zombie",
    "dead": "dead",
}

# psutil's nice() on Windows returns a priority *class* rather than a nice
# value.  These constants only exist on Windows, hence the guard.
if hasattr(psutil, "NORMAL_PRIORITY_CLASS"):
    _PRIORITY_CLASS_LABELS = {
        psutil.IDLE_PRIORITY_CLASS: "Idle",
        psutil.BELOW_NORMAL_PRIORITY_CLASS: "Below Normal",
        psutil.NORMAL_PRIORITY_CLASS: "Normal",
        psutil.ABOVE_NORMAL_PRIORITY_CLASS: "Above Normal",
        psutil.HIGH_PRIORITY_CLASS: "High",
        psutil.REALTIME_PRIORITY_CLASS: "Realtime",
    }
else:  # Linux / macOS: nice() is the classic -20..19 integer
    _PRIORITY_CLASS_LABELS = {}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ProcessInfo:
    """One row of the dashboard: everything we track about a single process."""

    pid: int
    name: str
    cpu_percent: float      # share of total CPU capacity (0-100 when normalised)
    memory_mb: float        # physical RAM in use (working set / RSS)
    thread_count: int
    priority: str           # human-readable label, e.g. "Normal"
    status: str             # running / sleeping / waiting / suspended
    priority_value: Optional[int] = None  # raw OS value, kept for ML features

    # Context switches (see the module docstring for what these mean).
    # Cumulative counters straight from the OS...
    ctx_switches_vol: int = 0
    ctx_switches_invol: Optional[int] = None   # None = this OS does not say
    # ...and the derived per-second rate, which is the number worth watching:
    # the totals only ever grow, so they say nothing about right now.
    ctx_switches_per_sec: float = 0.0

    @property
    def ctx_switches_total(self) -> int:
        """Both kinds together, for platforms that separate them."""
        return self.ctx_switches_vol + (self.ctx_switches_invol or 0)

    def to_row(self, timestamp: str) -> List[object]:
        """Flatten into a CSV row matching CSV_COLUMNS."""
        return [
            timestamp,
            self.pid,
            self.name,
            self.cpu_percent,
            self.memory_mb,
            self.thread_count,
            self.priority,
            self.status,
            self.ctx_switches_vol,
            # Empty rather than 0: "we do not know" and "it happened zero
            # times" are different claims, and only one of them is true here.
            "" if self.ctx_switches_invol is None else self.ctx_switches_invol,
            self.ctx_switches_per_sec,
        ]


@dataclass
class SystemInfo:
    """Machine-wide totals shown in the dashboard header."""

    cpu_percent: float
    memory_total_mb: float
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float
    process_count: int


@dataclass
class Snapshot:
    """One complete sample of the machine at a point in time."""

    timestamp: datetime
    processes: List[ProcessInfo]
    system: SystemInfo


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _priority_label(value: Optional[int], native: bool) -> str:
    """Turn a raw priority value into something a human can read."""
    if value is None:
        return "n/a"                    # protected process, access denied
    if native:
        # Windows base priority level, 0-31.
        return BASE_PRIORITY_LABELS.get(value, f"Base {value}")
    if value in _PRIORITY_CLASS_LABELS:
        return _PRIORITY_CLASS_LABELS[value]
    return f"nice {int(value)}"         # POSIX


def _derive_status(raw: RawProcess, cpu_percent: float) -> str:
    """
    Reduce everything to running / sleeping / waiting / suspended.

    Windows has no cheap "is this process running right now?" flag -- a
    process is either schedulable or fully suspended -- so we use the CPU
    reading to tell the two apart: no measurable CPU during the last interval
    means the process is blocked waiting for something, which is what the
    dashboard shows as "sleeping".
    """
    if raw.suspended:
        return "suspended"
    if raw.status_hint:  # psutil path: trust the kernel where it is specific
        label = _STATUS_LABELS.get(raw.status_hint, raw.status_hint)
        if label not in ("running", "sleeping"):
            return label
    return "running" if cpu_percent >= CPU_ACTIVE_THRESHOLD else "sleeping"


# ---------------------------------------------------------------------------
# The monitor
# ---------------------------------------------------------------------------


class ProcessMonitor:
    """
    Takes snapshots of the running processes.

    Create one instance per consumer -- the GUI has one, the collector has
    another -- because each keeps the CPU-time baseline for its own sampling
    interval.
    """

    def __init__(self, normalize_cpu: bool = True, use_native: bool = True) -> None:
        # normalize_cpu=True  -> 100 % means "the whole machine", like Task
        #                        Manager.  False -> 100 % means "one full
        #                        core", so an 8-thread process can hit 800 %.
        self.normalize_cpu = normalize_cpu
        self.cpu_count = psutil.cpu_count(logical=True) or 1

        # Pick the data source once, at start-up.
        self._native: Optional[winprobe.WindowsProcessSource] = None
        if use_native and winprobe.is_available():
            try:
                winprobe.self_test()
                self._native = winprobe.WindowsProcessSource()
                log.debug("Using the native Windows process probe.")
            except Exception as exc:
                log.warning("Native probe unavailable (%s); falling back to psutil.", exc)
        if self._native is None and psutil.WINDOWS:
            log.warning("Falling back to psutil on Windows - refreshes will be slow.")

        # pid -> CPU seconds consumed as of the previous snapshot
        self._prev_cpu_time: Dict[int, float] = {}
        # pid -> cumulative context switches as of the previous snapshot
        self._prev_ctx: Dict[int, int] = {}
        self._prev_clock: Optional[float] = None

        # psutil.cpu_percent() also works on deltas, so prime its baseline.
        psutil.cpu_percent(interval=None)
        # ...but the first reading still spans construction to first snapshot,
        # and the caller spends that gap loading the model and building
        # anomaly baselines - busy work that reads as ~100 % CPU and then sits
        # in the chart for five minutes.  Report the first sample as unknown,
        # exactly as per-process CPU does before it has a previous value.
        self._cpu_primed = False

    @property
    def source(self) -> str:
        """"native" or "psutil" - shown in the dashboard status bar."""
        return "native" if self._native is not None else "psutil"

    # -- raw data ----------------------------------------------------------

    def _raw_processes(self) -> List[RawProcess]:
        """Read the process table using whichever source is available."""
        if self._native is not None:
            try:
                return self._native.list_processes()
            except OSError as exc:
                # Extremely unlikely, but never let the dashboard die for it.
                log.warning("Native probe failed (%s); using psutil instead.", exc)
                self._native = None
        return self._psutil_processes()

    @staticmethod
    def _psutil_processes() -> List[RawProcess]:
        """Portable fallback: the same tuples, built from psutil."""
        rows: List[RawProcess] = []
        # process_iter() silently skips processes that exit mid-iteration.
        for proc in psutil.process_iter(attrs=_PSUTIL_ATTRS, ad_value=None):
            info = proc.info
            times = info["cpu_times"]
            memory = info["memory_info"]
            status = info["status"]
            # On Linux/macOS - where this path is the *primary* one - psutil
            # reads both counts straight out of /proc and they are genuinely
            # separate.  On Windows this path is the slow emergency fallback
            # and psutil reports the total as "voluntary" with involuntary
            # hard-coded to 0, so we discard that fake zero below.
            switches = info["num_ctx_switches"]
            if switches is None:
                voluntary, involuntary = 0, None
            elif psutil.WINDOWS:
                voluntary, involuntary = switches.voluntary, None
            else:
                voluntary, involuntary = switches.voluntary, switches.involuntary

            rows.append(RawProcess(
                pid=info["pid"],
                name=info["name"] or f"pid-{info['pid']}",
                thread_count=info["num_threads"] or 0,
                cpu_seconds=(times.user + times.system) if times else 0.0,
                memory_bytes=memory.rss if memory else 0,
                priority_value=info["nice"],
                suspended=status in ("stopped", "suspended"),
                status_hint=status,
                ctx_switches=voluntary,
                ctx_switches_invol=involuntary,
            ))
        return rows

    # -- processes ---------------------------------------------------------

    def list_processes(self, sort_by_cpu: bool = True) -> List[ProcessInfo]:
        """
        Take one snapshot and return a ProcessInfo per process.

        The very first call has no previous reading to compare against, so
        every CPU % comes back as 0.0.  From the second call onwards the
        numbers are real.
        """
        now = time.monotonic()
        elapsed = 0.0 if self._prev_clock is None else now - self._prev_clock
        native = self._native is not None

        current_cpu_time: Dict[int, float] = {}
        current_ctx: Dict[int, int] = {}
        processes: List[ProcessInfo] = []

        for raw in self._raw_processes():
            if raw.pid == IDLE_PID:
                continue
            current_cpu_time[raw.pid] = raw.cpu_seconds

            # --- CPU %: how much of the interval did this process consume? --
            cpu_percent = 0.0
            previous = self._prev_cpu_time.get(raw.pid)
            if previous is not None and elapsed > 0:
                # max(0, ...) guards against a recycled PID making the counter
                # look like it went backwards.
                cpu_percent = max(0.0, (raw.cpu_seconds - previous) / elapsed * 100.0)
                ceiling = 100.0 * self.cpu_count
                if self.normalize_cpu:
                    cpu_percent /= self.cpu_count
                    ceiling = 100.0
                cpu_percent = min(cpu_percent, ceiling)

            # --- context switches: same counter-to-rate trick as CPU % ------
            # The OS only ever gives a total that grows for the life of the
            # process, so "switches per second" is the difference between two
            # readings divided by the time between them.
            ctx_total = raw.ctx_switches + (raw.ctx_switches_invol or 0)
            current_ctx[raw.pid] = ctx_total
            ctx_per_sec = 0.0
            previous_ctx = self._prev_ctx.get(raw.pid)
            if previous_ctx is not None and elapsed > 0:
                # max(0, ...) for the same reason as CPU %: a recycled PID
                # makes the counter appear to run backwards.
                ctx_per_sec = max(0.0, (ctx_total - previous_ctx) / elapsed)

            processes.append(ProcessInfo(
                pid=raw.pid,
                name=raw.name,
                cpu_percent=round(cpu_percent, 2),
                memory_mb=round(raw.memory_bytes / BYTES_PER_MB, 2),
                thread_count=raw.thread_count,
                priority=_priority_label(raw.priority_value, native),
                status=_derive_status(raw, cpu_percent),
                priority_value=raw.priority_value,
                ctx_switches_vol=raw.ctx_switches,
                ctx_switches_invol=raw.ctx_switches_invol,
                ctx_switches_per_sec=round(ctx_per_sec, 1),
            ))

        # Replace (not update) the baseline so dead PIDs are forgotten.
        self._prev_cpu_time = current_cpu_time
        self._prev_ctx = current_ctx
        self._prev_clock = now

        if sort_by_cpu:
            processes.sort(key=lambda p: p.cpu_percent, reverse=True)
        return processes

    # -- system ------------------------------------------------------------

    def system_stats(self, process_count: Optional[int] = None) -> SystemInfo:
        """Machine-wide CPU and RAM totals for the dashboard header."""
        vm = psutil.virtual_memory()
        # interval=None -> non-blocking: measures since the previous call.
        cpu = psutil.cpu_percent(interval=None)
        if not self._cpu_primed:
            self._cpu_primed = True     # see __init__: this one covers startup
            cpu = 0.0
        return SystemInfo(
            cpu_percent=round(cpu, 1),
            memory_total_mb=round(vm.total / BYTES_PER_MB, 1),
            memory_used_mb=round(vm.used / BYTES_PER_MB, 1),
            memory_available_mb=round(vm.available / BYTES_PER_MB, 1),
            memory_percent=vm.percent,
            process_count=len(psutil.pids()) if process_count is None else process_count,
        )

    # -- both --------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        """Processes + system totals + a timestamp, in one object."""
        processes = self.list_processes()
        return Snapshot(
            timestamp=datetime.now(),
            processes=processes,
            system=self.system_stats(process_count=len(processes)),
        )


# ---------------------------------------------------------------------------
# Manual test:  python monitor.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    monitor = ProcessMonitor()
    monitor.list_processes()      # first pass only establishes the baseline
    time.sleep(1.0)

    started = time.perf_counter()
    snap = monitor.snapshot()
    took_ms = (time.perf_counter() - started) * 1000

    sysinfo = snap.system
    print(f"source: {monitor.source}   snapshot took {took_ms:.1f} ms")
    print(f"CPU {sysinfo.cpu_percent:5.1f} %   "
          f"RAM {sysinfo.memory_used_mb:,.0f} / {sysinfo.memory_total_mb:,.0f} MB   "
          f"{sysinfo.process_count} processes\n")
    print(f"{'PID':>7}  {'NAME':<28} {'CPU%':>6} {'MEM(MB)':>10} "
          f"{'THR':>4}  {'PRIORITY':<13} STATUS")
    for p in snap.processes[:15]:
        print(f"{p.pid:>7}  {p.name[:28]:<28} {p.cpu_percent:>6.1f} "
              f"{p.memory_mb:>10,.1f} {p.thread_count:>4}  "
              f"{p.priority:<13} {p.status}")
