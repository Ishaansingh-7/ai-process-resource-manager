"""
scheduler.py
============
CPU scheduling algorithms, implemented from scratch.

The problem being solved
------------------------
A CPU core runs one process at a time.  When several processes are ready to
run, something has to decide which one goes next - that decision is the
*scheduling algorithm*, and different algorithms optimise for different
things.  This module implements the four classic ones and measures them with
the four standard metrics.

The four times, defined precisely
---------------------------------
For one process, with ``arrival`` = when it became ready and ``burst`` = how
much CPU time it needs:

    completion time  - the instant it finishes, once and for all
    turnaround time  = completion - arrival
                       total time from becoming ready to being finished,
                       including all the time it spent waiting
    waiting time     = turnaround - burst
                       time spent ready but not running.  This is the number
                       most algorithms are judged on, because it is the time
                       the process got nothing done through no fault of its own
    response time    = first time it ran - arrival
                       how long until it *starts*, as opposed to finishes.
                       This is what feels like lag on an interactive system:
                       a text editor that starts instantly but finishes slowly
                       feels fine; one that takes two seconds to react does not

Note ``waiting = turnaround - burst`` is an identity, not a separate
measurement: whatever time was not spent running or waiting to arrive was
spent waiting.  The tests check it holds for every process in every algorithm.

The four algorithms
-------------------
* **FCFS** - run them in the order they arrived.  Completely fair in the
  queueing sense and trivial to implement, but suffers the *convoy effect*:
  one long job at the front makes every short job behind it wait, which is
  terrible for average waiting time.

* **SJF (non-preemptive)** - of the processes that have arrived, run the one
  with the shortest burst.  This is *provably optimal* for average waiting
  time among non-preemptive algorithms.  Its fatal flaw is that it needs to
  know the burst length in advance, which a real OS never does - it can only
  estimate from history.  It can also starve a long job indefinitely if short
  ones keep arriving.

* **Priority (non-preemptive)** - of the processes that have arrived, run the
  one with the best (numerically lowest) priority number.  Same starvation
  problem: a low-priority process can wait forever.  Real systems fix this
  with *ageing* - slowly improving the priority of anything that has waited a
  long time.  This implementation does not age, because the textbook version
  does not.

* **Round Robin** - each process gets at most one *quantum* of CPU, then goes
  to the back of the ready queue.  The only preemptive one here, and the only
  one that guarantees a bounded response time, which is why interactive
  systems use it.  The quantum is the whole trade-off: too large and it
  degenerates into FCFS; too small and the CPU spends its time context
  switching instead of working.

Everything here is plain Python with no third-party imports, so the logic can
be tested and explained on its own - see test_simulators.py, which checks all
four against worked examples whose answers are published.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_DIR / "data" / "process_data.csv"

FCFS = "FCFS"
SJF = "SJF"
PRIORITY = "Priority"
ROUND_ROBIN = "Round Robin"
ALGORITHM_NAMES = (FCFS, SJF, PRIORITY, ROUND_ROBIN)

DEFAULT_QUANTUM = 2.0

# Label used in the timeline for stretches where no process is ready to run.
IDLE_LABEL = "idle"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SimProcess:
    """One process as the simulator sees it: when it arrives and what it needs."""

    pid: str
    arrival: float
    burst: float
    priority: int = 0          # lower number = more important, as in the textbook

    def __post_init__(self) -> None:
        if self.burst <= 0:
            raise ValueError(f"{self.pid}: burst time must be greater than 0")
        if self.arrival < 0:
            raise ValueError(f"{self.pid}: arrival time cannot be negative")


@dataclass
class Slice:
    """
    One unbroken stretch of the timeline, used to draw the Gantt chart.

    Round Robin produces several slices per process; the non-preemptive
    algorithms produce exactly one each.  ``pid`` is IDLE_LABEL for a gap
    where the CPU had nothing to run.
    """

    pid: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_idle(self) -> bool:
        return self.pid == IDLE_LABEL


@dataclass
class ProcessMetrics:
    """The four times, for one process."""

    pid: str
    arrival: float
    burst: float
    priority: int
    start: float               # first moment it was given the CPU
    completion: float

    @property
    def turnaround(self) -> float:
        return self.completion - self.arrival

    @property
    def waiting(self) -> float:
        return self.turnaround - self.burst

    @property
    def response(self) -> float:
        return self.start - self.arrival


@dataclass
class ScheduleResult:
    """Everything one run of one algorithm produced."""

    algorithm: str
    metrics: List[ProcessMetrics]
    timeline: List[Slice]
    quantum: Optional[float] = None
    # Filled in by __post_init__ so callers never have to average by hand.
    avg_waiting: float = field(init=False, default=0.0)
    avg_turnaround: float = field(init=False, default=0.0)
    avg_response: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        n = len(self.metrics)
        if not n:
            return
        self.avg_waiting = sum(m.waiting for m in self.metrics) / n
        self.avg_turnaround = sum(m.turnaround for m in self.metrics) / n
        self.avg_response = sum(m.response for m in self.metrics) / n

    @property
    def makespan(self) -> float:
        """When the last process finished - the length of the whole schedule."""
        return max((m.completion for m in self.metrics), default=0.0)

    @property
    def cpu_busy(self) -> float:
        return sum(s.duration for s in self.timeline if not s.is_idle)

    @property
    def cpu_utilisation(self) -> float:
        """Percentage of the schedule during which the CPU was doing work."""
        span = self.makespan
        return (self.cpu_busy / span * 100.0) if span > 0 else 0.0

    @property
    def context_switches(self) -> int:
        """
        How many times the CPU was handed from one process to another.

        This is what ties the simulator back to the live dashboard: Round
        Robin's lower response time is paid for in exactly this currency, and
        the monitor measures the same event on real processes.
        """
        running = [s.pid for s in self.timeline if not s.is_idle]
        return sum(1 for a, b in zip(running, running[1:]) if a != b)


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------


def _append_slice(timeline: List[Slice], pid: str, start: float, end: float) -> None:
    """
    Add a slice, merging it into the previous one if it is the same process.

    Round Robin can give a process two quanta back to back when it is the only
    one in the queue.  Drawing that as two bars would suggest a context switch
    that never happened, so they are merged.
    """
    if end <= start:
        return
    if timeline and timeline[-1].pid == pid and timeline[-1].end == start:
        timeline[-1].end = end
    else:
        timeline.append(Slice(pid, start, end))


def _run_non_preemptive(processes: Sequence[SimProcess],
                        choose_key: Callable[[SimProcess], tuple],
                        algorithm: str) -> ScheduleResult:
    """
    FCFS, SJF and Priority differ *only* in which ready process they pick next.

    So the loop is written once and each algorithm supplies a sort key.  Once
    a process starts it runs to completion - that is what "non-preemptive"
    means - so each one produces a single slice and finishes at
    ``start + burst``.
    """
    pending = list(processes)
    clock = 0.0
    timeline: List[Slice] = []
    metrics: List[ProcessMetrics] = []

    while pending:
        ready = [p for p in pending if p.arrival <= clock]
        if not ready:
            # Nothing has arrived yet: the CPU sits idle until the next one
            # does.  Recording the gap keeps the Gantt chart honest and makes
            # the utilisation figure meaningful.
            next_arrival = min(p.arrival for p in pending)
            _append_slice(timeline, IDLE_LABEL, clock, next_arrival)
            clock = next_arrival
            continue

        chosen = min(ready, key=choose_key)
        pending.remove(chosen)

        start = clock
        clock = start + chosen.burst
        _append_slice(timeline, chosen.pid, start, clock)
        metrics.append(ProcessMetrics(
            pid=chosen.pid, arrival=chosen.arrival, burst=chosen.burst,
            priority=chosen.priority, start=start, completion=clock))

    return ScheduleResult(algorithm, metrics, timeline)


def _ordered(processes: Sequence[SimProcess]) -> List[Tuple[int, SimProcess]]:
    """Pair each process with its input position, used to break ties stably."""
    return list(enumerate(processes))


# ---------------------------------------------------------------------------
# The algorithms
# ---------------------------------------------------------------------------


def fcfs(processes: Sequence[SimProcess]) -> ScheduleResult:
    """
    First Come First Served: whoever arrived earliest runs next.

    Ties are broken by input order, which is the convention worked examples
    use when two processes share an arrival time.
    """
    position = {id(p): i for i, p in enumerate(processes)}
    return _run_non_preemptive(
        processes,
        choose_key=lambda p: (p.arrival, position[id(p)]),
        algorithm=FCFS)


def sjf(processes: Sequence[SimProcess]) -> ScheduleResult:
    """
    Shortest Job First, non-preemptive.

    Of everything that has arrived, run the shortest.  Ties go to whoever
    arrived first, then to input order.
    """
    position = {id(p): i for i, p in enumerate(processes)}
    return _run_non_preemptive(
        processes,
        choose_key=lambda p: (p.burst, p.arrival, position[id(p)]),
        algorithm=SJF)


def priority_scheduling(processes: Sequence[SimProcess]) -> ScheduleResult:
    """
    Priority scheduling, non-preemptive, **lower number = higher priority**.

    That convention is worth stating out loud because it is the opposite of
    what most people guess, and it is what the textbook examples use.
    """
    position = {id(p): i for i, p in enumerate(processes)}
    return _run_non_preemptive(
        processes,
        choose_key=lambda p: (p.priority, p.arrival, position[id(p)]),
        algorithm=PRIORITY)


def round_robin(processes: Sequence[SimProcess],
                quantum: float = DEFAULT_QUANTUM) -> ScheduleResult:
    """
    Round Robin with a fixed time quantum - the only preemptive one here.

    Each process runs for at most ``quantum``, then is put at the back of the
    ready queue whether it is finished or not.

    The one subtle rule
    -------------------
    When a process's quantum expires at exactly the same instant another
    process arrives, **the arriving process is queued first** and the
    preempted one goes behind it.  This is the standard convention, and it is
    not arbitrary: the arriving process has been waiting since it arrived,
    while the preempted one has just had its turn.  Worked examples assume it,
    and getting it backwards changes the answer - so it is done explicitly
    below rather than left to luck.
    """
    if quantum <= 0:
        raise ValueError("The time quantum must be greater than 0")

    # Process the arrivals in time order; ties keep their input order.
    arrivals = sorted(_ordered(processes), key=lambda pair: (pair[1].arrival, pair[0]))
    remaining: Dict[str, float] = {p.pid: p.burst for p in processes}
    by_pid: Dict[str, SimProcess] = {p.pid: p for p in processes}
    first_start: Dict[str, float] = {}
    completion: Dict[str, float] = {}

    queue: deque = deque()
    timeline: List[Slice] = []
    clock = 0.0
    next_arrival = 0          # index into `arrivals` of the next one to admit

    def admit_up_to(moment: float) -> None:
        """Move every process that has arrived by `moment` into the queue."""
        nonlocal next_arrival
        while (next_arrival < len(arrivals)
               and arrivals[next_arrival][1].arrival <= moment):
            queue.append(arrivals[next_arrival][1].pid)
            next_arrival += 1

    admit_up_to(clock)

    while len(completion) < len(processes):
        if not queue:
            # Idle: jump forward to the next arrival rather than ticking.
            if next_arrival >= len(arrivals):
                break                      # nothing left (cannot normally happen)
            moment = arrivals[next_arrival][1].arrival
            _append_slice(timeline, IDLE_LABEL, clock, moment)
            clock = moment
            admit_up_to(clock)
            continue

        pid = queue.popleft()
        run_for = min(quantum, remaining[pid])
        start = clock
        clock = start + run_for
        remaining[pid] -= run_for
        _append_slice(timeline, pid, start, clock)
        first_start.setdefault(pid, start)

        # Admit everything that arrived *during* this slice, including exactly
        # at its end, BEFORE putting the preempted process back - see the note
        # in the docstring.  This ordering is the whole subtlety of RR.
        admit_up_to(clock)

        if remaining[pid] > 1e-9:
            queue.append(pid)
        else:
            completion[pid] = clock

    metrics = [
        ProcessMetrics(pid=p.pid, arrival=p.arrival, burst=p.burst,
                       priority=p.priority,
                       start=first_start.get(p.pid, 0.0),
                       completion=completion.get(p.pid, 0.0))
        for p in processes
    ]
    return ScheduleResult(ROUND_ROBIN, metrics, timeline, quantum=quantum)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run(algorithm: str, processes: Sequence[SimProcess],
        quantum: float = DEFAULT_QUANTUM) -> ScheduleResult:
    """Run one algorithm by name.  ``quantum`` is ignored except by Round Robin."""
    if not processes:
        raise ValueError("Add at least one process before running")
    if algorithm == FCFS:
        return fcfs(processes)
    if algorithm == SJF:
        return sjf(processes)
    if algorithm == PRIORITY:
        return priority_scheduling(processes)
    if algorithm == ROUND_ROBIN:
        return round_robin(processes, quantum)
    raise ValueError(f"Unknown algorithm: {algorithm!r}")


def compare_all(processes: Sequence[SimProcess],
                quantum: float = DEFAULT_QUANTUM) -> Dict[str, ScheduleResult]:
    """Run all four on the same input, for the side-by-side comparison."""
    return {name: run(name, processes, quantum) for name in ALGORITHM_NAMES}


# ---------------------------------------------------------------------------
# Turning real measurements into simulator input
# ---------------------------------------------------------------------------
#
# This is the part that connects the simulator to the rest of the project, and
# it is also the part with the most assumptions in it, so they are all written
# down here rather than hidden in the code.
#
# HOW BURST TIME IS DERIVED
# -------------------------
# monitor.py records cpu_percent as a share of the *whole machine*, so a
# reading of 6.25 % on a 16-thread machine means "one core, fully busy".  For
# one sample covering `interval` seconds, the CPU time that process actually
# consumed is:
#
#     cpu_seconds = (cpu_percent / 100) * logical_cores * interval
#
# Summing that over every sample of a process gives the total CPU time it was
# observed to use.  That total is its burst time: not a guess, but the
# measured amount of CPU work it did while the collector was watching.  The
# numbers are then scaled so the largest burst is SCALE_TARGET units, because
# a Gantt chart of 4,000-second bars is unreadable and the algorithms only
# care about the ratios between bursts, not their absolute size.
#
# WHAT THIS IS NOT - the honest part
# ----------------------------------
# 1. A real process is not one burst.  It alternates between using the CPU and
#    waiting for I/O for its entire life - dozens or millions of short CPU
#    bursts.  Collapsing all of that into a single number is exactly the
#    textbook model, and exactly what real processes do not look like.
# 2. SJF and Priority need the burst length *before* running the process.  We
#    only have these numbers because we already watched it run.  That is not a
#    flaw in the derivation - it is the reason SJF cannot be implemented as-is
#    in a real OS, which is worth saying out loud rather than hiding.
# 3. Arrival time here is "when the collector first saw this process", not
#    when it was created.  Most of these processes were already running when
#    the collector started, so their arrival is 0 - that is an artefact of the
#    observation window, not a fact about the machine.
# 4. Sampling every 2.5-3 seconds cannot see a burst shorter than that.  A
#    process that woke, used 5 ms of CPU and slept contributes almost nothing,
#    even though a real scheduler dealt with it thousands of times.
# 5. Priority is mapped from the observed Windows priority class, which is a
#    genuine measurement - but Windows priorities are not the small integers
#    the textbook algorithm assumes, so they are compressed onto 0-5.
#
# So: the *burst times are real measurements*, and the *scheduling model they
# are fed into is a simplification*.  Both halves of that sentence matter.

SCALE_TARGET = 10.0             # longest derived burst becomes this many units
MIN_BURST = 0.1                 # nothing may round down to a zero-length job
DEFAULT_SAMPLE_INTERVAL = 3.0   # collector.DEFAULT_INTERVAL, if we cannot measure it

# Windows priority classes compressed onto the textbook's small integers,
# keeping the direction the algorithms expect: lower number = higher priority.
PRIORITY_MAP = {
    "Realtime": 0,
    "High": 1,
    "Above Normal": 2,
    "Normal": 3,
    "Below Normal": 4,
    "Idle": 5,
}
DEFAULT_PRIORITY = 3


def _logical_cores() -> int:
    """Cores on this machine, used to undo monitor.py's normalisation."""
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 1
    except Exception:
        import os
        return os.cpu_count() or 1


def _parse_timestamps(stamps: Sequence[str]) -> float:
    """
    Work out the seconds between consecutive snapshots from the timestamps.

    Better than assuming the default: whoever collected the data may have run
    with --interval, and the burst calculation is directly proportional to it.
    """
    from datetime import datetime
    unique = sorted(set(stamps))
    if len(unique) < 2:
        return DEFAULT_SAMPLE_INTERVAL
    gaps = []
    for earlier, later in zip(unique, unique[1:]):
        try:
            a = datetime.strptime(earlier, "%Y-%m-%d %H:%M:%S")
            b = datetime.strptime(later, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        gap = (b - a).total_seconds()
        # Ignore the long gaps between separate collection runs.
        if 0 < gap <= 60:
            gaps.append(gap)
    if not gaps:
        return DEFAULT_SAMPLE_INTERVAL
    gaps.sort()
    return gaps[len(gaps) // 2]     # median, so one odd gap cannot skew it


@dataclass
class DerivedProcess:
    """A simulator process plus the real numbers it was derived from."""

    process: SimProcess
    cpu_seconds: float          # measured CPU time before scaling
    samples: int
    mean_cpu_percent: float


def load_from_csv(csv_path: Optional[Path] = None,
                  limit: int = 6,
                  scale_target: float = SCALE_TARGET) -> List[DerivedProcess]:
    """
    Build simulator input from real collected data.

    Picks the ``limit`` processes that used the most CPU (the interesting ones
    - a Gantt chart of 300 idle processes says nothing), derives each one's
    burst time as documented above, and returns them ready to simulate.

    Raises FileNotFoundError / ValueError with an explanation the GUI shows
    directly, rather than returning something empty and mysterious.
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}.\n"
            "Collect some first:  python main.py --collect-only")

    totals: Dict[str, float] = {}       # name -> summed cpu_percent
    counts: Dict[str, int] = {}
    first_seen: Dict[str, str] = {}
    priorities: Dict[str, str] = {}
    stamps: List[str] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "cpu_percent" not in reader.fieldnames:
            raise ValueError(f"{path.name} does not look like a process dataset")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            try:
                cpu = float(row["cpu_percent"])
            except (TypeError, ValueError):
                continue            # blank or text cell - skip this row only
            stamp = row.get("timestamp") or ""
            totals[name] = totals.get(name, 0.0) + cpu
            counts[name] = counts.get(name, 0) + 1
            if name not in first_seen:
                first_seen[name] = stamp
                priorities[name] = (row.get("priority") or "").strip()
            if len(stamps) < 5000:
                stamps.append(stamp)

    if not totals:
        raise ValueError(f"{path.name} contains no usable rows")

    interval = _parse_timestamps(stamps)
    cores = _logical_cores()

    # cpu_seconds = mean_fraction_of_machine * cores * observed_seconds,
    # which is the same thing as summing each sample's contribution.
    measured: List[Tuple[str, float, int, float]] = []
    for name, total_cpu in totals.items():
        n = counts[name]
        cpu_seconds = (total_cpu / 100.0) * cores * interval
        measured.append((name, cpu_seconds, n, total_cpu / n))

    measured.sort(key=lambda item: item[1], reverse=True)
    top = measured[:max(1, limit)]

    biggest = max(item[1] for item in top) or 1.0
    scale = scale_target / biggest

    # Arrival: the earliest timestamp any of the chosen processes was seen at
    # becomes time 0, and the others are offset from it.  For a dataset
    # collected in one run this is 0 for nearly everything (see assumption 3).
    earliest = min((first_seen[name] for name, *_ in top), default="")

    from datetime import datetime

    def arrival_of(name: str) -> float:
        try:
            a = datetime.strptime(earliest, "%Y-%m-%d %H:%M:%S")
            b = datetime.strptime(first_seen[name], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return 0.0
        return round(max(0.0, (b - a).total_seconds()) * scale / interval, 1)

    derived: List[DerivedProcess] = []
    for name, cpu_seconds, n, mean_cpu in top:
        burst = max(MIN_BURST, round(cpu_seconds * scale, 1))
        derived.append(DerivedProcess(
            process=SimProcess(
                pid=name,
                arrival=arrival_of(name),
                burst=burst,
                priority=PRIORITY_MAP.get(priorities.get(name, ""), DEFAULT_PRIORITY),
            ),
            cpu_seconds=cpu_seconds,
            samples=n,
            mean_cpu_percent=mean_cpu,
        ))
    return derived


# ---------------------------------------------------------------------------
# Manual demo:  python scheduler.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = [
        SimProcess("P1", arrival=0, burst=5, priority=2),
        SimProcess("P2", arrival=1, burst=3, priority=1),
        SimProcess("P3", arrival=2, burst=8, priority=4),
        SimProcess("P4", arrival=3, burst=2, priority=3),
    ]
    print("Input")
    print(f"  {'PID':<5}{'arrival':>9}{'burst':>8}{'priority':>10}")
    for p in demo:
        print(f"  {p.pid:<5}{p.arrival:>9g}{p.burst:>8g}{p.priority:>10d}")

    for name, result in compare_all(demo, quantum=2).items():
        print(f"\n{'=' * 62}\n{name}"
              f"{f'  (quantum {result.quantum:g})' if result.quantum else ''}")
        print(f"  {'PID':<5}{'start':>8}{'finish':>8}{'turnaround':>12}"
              f"{'waiting':>9}{'response':>10}")
        for m in sorted(result.metrics, key=lambda m: m.pid):
            print(f"  {m.pid:<5}{m.start:>8g}{m.completion:>8g}"
                  f"{m.turnaround:>12g}{m.waiting:>9g}{m.response:>10g}")
        print(f"  average waiting {result.avg_waiting:.2f}   "
              f"turnaround {result.avg_turnaround:.2f}   "
              f"response {result.avg_response:.2f}")
        print("  Gantt: " + " | ".join(
            f"{s.pid} {s.start:g}-{s.end:g}" for s in result.timeline))
        print(f"  CPU busy {result.cpu_utilisation:.0f} %   "
              f"context switches {result.context_switches}")

    print(f"\n{'=' * 62}\nDerived from real data")
    try:
        for d in load_from_csv(limit=5):
            p = d.process
            print(f"  {p.pid[:28]:<28} burst={p.burst:>6g}  priority={p.priority}  "
                  f"({d.cpu_seconds:,.1f} CPU-seconds over {d.samples:,} samples, "
                  f"mean {d.mean_cpu_percent:.2f} %)")
    except (FileNotFoundError, ValueError) as exc:
        print(f"  unavailable: {exc}")
