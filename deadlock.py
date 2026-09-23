"""
deadlock.py
===========
Banker's Algorithm and deadlock detection, implemented from scratch.

The problem being solved
------------------------
Processes hold resources and ask for more.  If four processes each hold one
resource and each waits for one another is holding, none of them can ever
proceed - they are **deadlocked**, and the only way out is to kill something.
Deadlock needs all four Coffman conditions to hold at once: mutual exclusion,
hold-and-wait, no preemption, and circular wait.

There are two different questions here, and confusing them is the classic
exam mistake, so they are two separate functions:

* **AVOIDANCE** (:func:`is_safe`, :func:`request_resources`) - deadlock has
  not happened.  Before granting a request we ask "if I say yes, could I still
  guarantee everybody finishes?"  If not, the requester waits even though the
  resources are sitting right there.  Needs to know each process's *maximum*
  future demand in advance.

* **DETECTION** (:func:`detect_deadlock`) - we granted whatever was asked for
  and now something is stuck.  Which processes are deadlocked?  Needs no
  advance knowledge, only what each process holds and what it is asking for
  *right now* - but by the time it says yes, the damage is done.

Safe vs unsafe vs deadlocked
----------------------------
    SAFE       there is at least one order in which every process can be run
               to completion.  Deadlock is impossible from here.
    UNSAFE     no such order can be guaranteed.  This does NOT mean deadlocked
               - the processes might never actually ask for their stated
               maximum, and everything might be fine.  It means the system can
               no longer *promise* it will be fine.  Banker's refuses to enter
               an unsafe state precisely because it will not gamble.
    DEADLOCKED actually stuck, right now.

That distinction - unsafe is not the same as deadlocked - is the single point
most worth being able to state clearly.

The safety algorithm, in plain terms
------------------------------------
    Work = Available            (what is free right now)
    Finish[i] = False for all i

    repeat:
        find a process i where Finish[i] is False and Need[i] <= Work
            "this process could get everything it might still ask for,
             out of what is free"
        if found:
            Work = Work + Allocation[i]
            "pretend it runs to completion and hands back everything it holds,
             which leaves more free for the others"
            Finish[i] = True
        else:
            stop
    the state is SAFE if every Finish[i] is True

The pessimism is deliberate: it assumes every process will demand its full
maximum.  If the system survives even that, it is genuinely safe.

Everything here is plain Python with no third-party imports.  The worked
examples in test_simulators.py are from Silberschatz, *Operating System
Concepts*, whose answers are published, so the implementation is checked
against them rather than against itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

Matrix = List[List[int]]
Vector = List[int]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """
    One decision the algorithm made, recorded so it can be walked through.

    The point of keeping this is that "UNSAFE" on its own teaches nobody
    anything.  Being able to say "it got through P1 and P3, then no remaining
    process could be satisfied from (2,1,0)" is the actual explanation.
    """

    number: int
    process: Optional[str]      # None when the step is a dead end
    need: Optional[Vector]
    work_before: Vector
    work_after: Optional[Vector]
    allocation: Optional[Vector]
    granted: bool
    note: str

    def describe(self) -> str:
        """A human-readable line for the GUI's trace panel."""
        if self.granted:
            return (f"Step {self.number}: {self.process} can finish.  "
                    f"Need {_fmt(self.need)} <= Work {_fmt(self.work_before)}. "
                    f"It runs, releases {_fmt(self.allocation)}, "
                    f"Work becomes {_fmt(self.work_after)}.")
        return f"Step {self.number}: {self.note}"


@dataclass
class SafetyResult:
    """The verdict from the safety check, with the reasoning that produced it."""

    safe: bool
    sequence: List[str]                     # e.g. ["P1", "P3", "P4", "P2", "P0"]
    steps: List[TraceStep] = field(default_factory=list)
    need: Matrix = field(default_factory=list)
    unfinished: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def sequence_text(self) -> str:
        return " -> ".join(self.sequence) if self.sequence else "(none)"


@dataclass
class RequestDecision:
    """Whether a resource request may be granted, and why."""

    granted: bool
    reason: str
    # The state that would result, filled in only when the request is
    # legitimate enough to have been simulated at all.
    resulting_available: Optional[Vector] = None
    resulting_allocation: Optional[Matrix] = None
    safety: Optional[SafetyResult] = None


@dataclass
class DetectionResult:
    """Which processes, if any, are deadlocked right now."""

    deadlocked: List[str]
    sequence: List[str]                     # the order the others could finish in
    steps: List[TraceStep] = field(default_factory=list)
    message: str = ""

    @property
    def has_deadlock(self) -> bool:
        return bool(self.deadlocked)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fmt(vector: Optional[Sequence[int]]) -> str:
    return "(" + ", ".join(str(v) for v in vector) + ")" if vector else "()"


def _le(left: Sequence[int], right: Sequence[int]) -> bool:
    """Vector <=: true only when *every* element is <=, which is the whole point."""
    return all(a <= b for a, b in zip(left, right))


def default_names(count: int) -> List[str]:
    return [f"P{i}" for i in range(count)]


def validate(allocation: Matrix, maximum: Optional[Matrix],
             available: Vector) -> None:
    """
    Reject impossible input early, with a message worth reading.

    The GUI lets people type numbers into a grid, so this catches the
    mistakes that produce nonsense rather than a crash.
    """
    if not allocation:
        raise ValueError("There are no processes to check")
    width = len(available)
    if width == 0:
        raise ValueError("There are no resource types")
    for i, row in enumerate(allocation):
        if len(row) != width:
            raise ValueError(
                f"Allocation row {i} has {len(row)} entries but there are "
                f"{width} resource types")
        if any(v < 0 for v in row):
            raise ValueError(f"Allocation row {i} contains a negative number")
    if any(v < 0 for v in available):
        raise ValueError("Available contains a negative number")
    if maximum is None:
        return
    if len(maximum) != len(allocation):
        raise ValueError("Max and Allocation must have the same number of rows")
    for i, row in enumerate(maximum):
        if len(row) != width:
            raise ValueError(
                f"Max row {i} has {len(row)} entries but there are "
                f"{width} resource types")
        for j, value in enumerate(row):
            if value < allocation[i][j]:
                raise ValueError(
                    f"Process {i} holds {allocation[i][j]} of resource {j} but "
                    f"its maximum is only {value} - a process cannot hold more "
                    f"than it ever claimed it would need")


# ---------------------------------------------------------------------------
# Need = Max - Allocation
# ---------------------------------------------------------------------------


def need_matrix(allocation: Matrix, maximum: Matrix) -> Matrix:
    """
    What each process could still ask for: Need = Max - Allocation.

    Every part of Banker's works off this rather than off Max, because what
    matters is not how much a process might ever want in total, but how much
    more it might want *given what it is already holding*.
    """
    return [[mx - al for mx, al in zip(max_row, alloc_row)]
            for max_row, alloc_row in zip(maximum, allocation)]


# ---------------------------------------------------------------------------
# The safety algorithm
# ---------------------------------------------------------------------------


def is_safe(allocation: Matrix, maximum: Matrix, available: Vector,
            names: Optional[Sequence[str]] = None) -> SafetyResult:
    """
    Is this state safe?  If so, in what order can everybody finish?

    Returns the safe sequence and a step-by-step trace of how it was found.
    See the module docstring for the algorithm in plain terms.
    """
    validate(allocation, maximum, available)
    labels = list(names) if names else default_names(len(allocation))
    need = need_matrix(allocation, maximum)

    work = list(available)
    finished = [False] * len(allocation)
    sequence: List[str] = []
    steps: List[TraceStep] = []

    # At most one process is added per pass, so n passes settles it.
    for _ in range(len(allocation)):
        progressed = False
        for i in range(len(allocation)):
            if finished[i] or not _le(need[i], work):
                continue
            work_before = list(work)
            # It can finish, so assume it does and give back what it holds.
            work = [w + a for w, a in zip(work, allocation[i])]
            finished[i] = True
            sequence.append(labels[i])
            steps.append(TraceStep(
                number=len(steps) + 1, process=labels[i], need=list(need[i]),
                work_before=work_before, work_after=list(work),
                allocation=list(allocation[i]), granted=True,
                note=f"{labels[i]} satisfied from available resources"))
            progressed = True
            break                   # restart the scan, lowest index first
        if not progressed:
            break

    stuck = [labels[i] for i, done in enumerate(finished) if not done]
    if not stuck:
        return SafetyResult(
            safe=True, sequence=sequence, steps=steps, need=need,
            message=("SAFE - every process can finish in the order "
                     + " -> ".join(sequence)))

    # Explain *why* it stopped: which processes are left and what they needed
    # that the remaining free pool could not cover.
    detail = "; ".join(
        f"{labels[i]} still needs {_fmt(need[i])}"
        for i, done in enumerate(finished) if not done)
    steps.append(TraceStep(
        number=len(steps) + 1, process=None, need=None, work_before=list(work),
        work_after=None, allocation=None, granted=False,
        note=(f"No remaining process can be satisfied from Work {_fmt(work)}. "
              f"{detail}.")))
    return SafetyResult(
        safe=False, sequence=sequence, steps=steps, need=need, unfinished=stuck,
        message=(f"UNSAFE - got as far as {' -> '.join(sequence) or '(nobody)'}, "
                 f"then {', '.join(stuck)} could not be guaranteed to finish. "
                 f"Note this is not the same as being deadlocked: it means the "
                 f"system can no longer promise everybody will finish."))


# ---------------------------------------------------------------------------
# The resource-request handler
# ---------------------------------------------------------------------------


def verify_sequence(sequence: Sequence[str], allocation: Matrix,
                    maximum: Matrix, available: Vector,
                    names: Optional[Sequence[str]] = None) -> bool:
    """
    Is this particular ordering a genuine safe sequence?

    Worth having for a reason that trips people up: a safe state usually has
    **several** valid safe sequences, and which one an implementation finds
    depends only on the order it scans processes in.  Scanning lowest-index
    first (as here, and as the textbook's pseudocode does) gives
    P1 -> P3 -> P0 -> P2 -> P4 for the standard example, while the sequence
    printed in the book is P1 -> P3 -> P4 -> P2 -> P0.  Both are correct.

    So the tests check that our answer *is a valid safe sequence*, rather than
    that it matches one particular published string - which would be testing
    the scan order, not the algorithm.
    """
    labels = list(names) if names else default_names(len(allocation))
    index_of = {label: i for i, label in enumerate(labels)}
    if sorted(sequence) != sorted(labels):
        return False                # must be a permutation of every process

    need = need_matrix(allocation, maximum)
    work = list(available)
    for label in sequence:
        i = index_of[label]
        if not _le(need[i], work):
            return False            # this process could not have run here
        work = [w + a for w, a in zip(work, allocation[i])]
    return True


def request_resources(process_index: int, request: Vector,
                      allocation: Matrix, maximum: Matrix, available: Vector,
                      names: Optional[Sequence[str]] = None) -> RequestDecision:
    """
    Process ``process_index`` asks for ``request``.  Grant it or make it wait?

    Three checks, in this order - the order matters and is the textbook's:

    1. **Is the request within what it claimed?**  ``Request <= Need``.  If
       not, the process has exceeded its own declared maximum, which is a
       programming error, not a scheduling decision.
    2. **Are the resources even free?**  ``Request <= Available``.  If not it
       simply waits; nothing is wrong.
    3. **Would granting it still leave a safe state?**  Pretend to grant it,
       run the safety check, and only commit if the answer is yes.  This is
       the step that makes it *avoidance* rather than just bookkeeping - the
       resources may be sitting free and the answer can still be "wait".
    """
    validate(allocation, maximum, available)
    labels = list(names) if names else default_names(len(allocation))
    if not 0 <= process_index < len(allocation):
        raise ValueError(f"No such process: index {process_index}")
    if len(request) != len(available):
        raise ValueError("The request must have one entry per resource type")
    if any(v < 0 for v in request):
        raise ValueError("A request cannot be negative")

    who = labels[process_index]
    need = need_matrix(allocation, maximum)[process_index]

    if not _le(request, need):
        return RequestDecision(
            granted=False,
            reason=(f"DENIED - {who} asked for {_fmt(request)} but may still "
                    f"claim only {_fmt(need)}. It has exceeded the maximum it "
                    f"declared, which is an error in the process itself."))

    if not _le(request, available):
        return RequestDecision(
            granted=False,
            reason=(f"WAIT - {who} asked for {_fmt(request)} but only "
                    f"{_fmt(available)} is free. Nothing is wrong; it simply "
                    f"has to wait until enough is released."))

    # Tentatively grant it, then test the resulting state.
    new_available = [a - r for a, r in zip(available, request)]
    new_allocation = [list(row) for row in allocation]
    new_allocation[process_index] = [a + r for a, r
                                     in zip(allocation[process_index], request)]

    safety = is_safe(new_allocation, maximum, new_available, labels)
    if safety.safe:
        return RequestDecision(
            granted=True,
            reason=(f"GRANTED - after giving {who} {_fmt(request)} the state is "
                    f"still safe ({safety.sequence_text})."),
            resulting_available=new_available,
            resulting_allocation=new_allocation,
            safety=safety)

    return RequestDecision(
        granted=False,
        reason=(f"DENIED - {who} could have {_fmt(request)} right now, but "
                f"granting it would leave an unsafe state, so it must wait. "
                f"{safety.message}"),
        resulting_available=new_available,
        resulting_allocation=new_allocation,
        safety=safety)


# ---------------------------------------------------------------------------
# Detection: for processes that are already stuck
# ---------------------------------------------------------------------------


def detect_deadlock(allocation: Matrix, request: Matrix, available: Vector,
                    names: Optional[Sequence[str]] = None) -> DetectionResult:
    """
    Find the processes that are deadlocked *now*.

    Nearly the same loop as :func:`is_safe`, with one crucial difference: it
    works from ``Request`` - what each process is actually asking for at this
    instant - instead of ``Need``, the worst case it might ever ask for.  That
    is what makes it detection rather than avoidance, and why it needs no
    advance declaration of maximum demand.

    One more difference worth noticing: a process holding nothing and
    requesting nothing finishes immediately, so it never counts as
    deadlocked.

    Whatever cannot be finished at the end is deadlocked - not "might be",
    *is*: it is waiting for something that nobody remaining will ever release.
    """
    validate(allocation, None, available)
    labels = list(names) if names else default_names(len(allocation))
    if len(request) != len(allocation):
        raise ValueError("Request and Allocation must have the same number of rows")
    for i, row in enumerate(request):
        if len(row) != len(available):
            raise ValueError(
                f"Request row {i} has {len(row)} entries but there are "
                f"{len(available)} resource types")

    work = list(available)
    finished = [False] * len(allocation)
    sequence: List[str] = []
    steps: List[TraceStep] = []

    for _ in range(len(allocation)):
        progressed = False
        for i in range(len(allocation)):
            if finished[i] or not _le(request[i], work):
                continue
            work_before = list(work)
            work = [w + a for w, a in zip(work, allocation[i])]
            finished[i] = True
            sequence.append(labels[i])
            steps.append(TraceStep(
                number=len(steps) + 1, process=labels[i], need=list(request[i]),
                work_before=work_before, work_after=list(work),
                allocation=list(allocation[i]), granted=True,
                note=f"{labels[i]}'s current request can be met"))
            progressed = True
            break
        if not progressed:
            break

    deadlocked = [labels[i] for i, done in enumerate(finished) if not done]
    if not deadlocked:
        return DetectionResult(
            deadlocked=[], sequence=sequence, steps=steps,
            message=("No deadlock - every process can complete, in the order "
                     + " -> ".join(sequence)))

    detail = "; ".join(
        f"{labels[i]} is waiting for {_fmt(request[i])} while holding "
        f"{_fmt(allocation[i])}"
        for i, done in enumerate(finished) if not done)
    steps.append(TraceStep(
        number=len(steps) + 1, process=None, need=None, work_before=list(work),
        work_after=None, allocation=None, granted=False,
        note=(f"No remaining process can proceed with Work {_fmt(work)}. "
              f"{detail}.")))
    return DetectionResult(
        deadlocked=deadlocked, sequence=sequence, steps=steps,
        message=(f"DEADLOCK - {', '.join(deadlocked)} "
                 f"{'is' if len(deadlocked) == 1 else 'are'} deadlocked. "
                 f"Each is waiting for a resource that only another member of "
                 f"the group holds, so none of them can ever proceed."))


# ---------------------------------------------------------------------------
# The worked example everyone knows
# ---------------------------------------------------------------------------
#
# Silberschatz, Galvin & Gagne, "Operating System Concepts", the Banker's
# Algorithm example: 5 processes, 3 resource types (A=10, B=5, C=7 in total).
# The published safe sequence is P1 -> P3 -> P4 -> P2 -> P0.  Kept here so the
# GUI's "Load Example" button and the unit tests use the same known-good data.

TEXTBOOK_NAMES = ["P0", "P1", "P2", "P3", "P4"]
TEXTBOOK_RESOURCES = ["A", "B", "C"]
TEXTBOOK_ALLOCATION: Matrix = [
    [0, 1, 0],
    [2, 0, 0],
    [3, 0, 2],
    [2, 1, 1],
    [0, 0, 2],
]
TEXTBOOK_MAX: Matrix = [
    [7, 5, 3],
    [3, 2, 2],
    [9, 0, 2],
    [2, 2, 2],
    [4, 3, 3],
]
TEXTBOOK_AVAILABLE: Vector = [3, 3, 2]
TEXTBOOK_SAFE_SEQUENCE = ["P1", "P3", "P4", "P2", "P0"]

# The detection example from the same chapter: 3 resource types (A=7, B=2,
# C=6).  With these requests nothing is deadlocked; if P2 asks for one more C
# instead, P1-P4 all deadlock.
DETECTION_ALLOCATION: Matrix = [
    [0, 1, 0],
    [2, 0, 0],
    [3, 0, 3],
    [2, 1, 1],
    [0, 0, 2],
]
DETECTION_REQUEST: Matrix = [
    [0, 0, 0],
    [2, 0, 2],
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 2],
]
DETECTION_AVAILABLE: Vector = [0, 0, 0]


# ---------------------------------------------------------------------------
# Manual demo:  python deadlock.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def show_matrix(title, matrix, names, resources):
        print(f"\n{title}")
        print("       " + "".join(f"{r:>4}" for r in resources))
        for name, row in zip(names, matrix):
            print(f"  {name:<5}" + "".join(f"{v:>4}" for v in row))

    print("=" * 68)
    print("BANKER'S ALGORITHM - Silberschatz worked example")
    print("=" * 68)
    show_matrix("Allocation", TEXTBOOK_ALLOCATION, TEXTBOOK_NAMES, TEXTBOOK_RESOURCES)
    show_matrix("Max", TEXTBOOK_MAX, TEXTBOOK_NAMES, TEXTBOOK_RESOURCES)
    show_matrix("Need = Max - Allocation",
                need_matrix(TEXTBOOK_ALLOCATION, TEXTBOOK_MAX),
                TEXTBOOK_NAMES, TEXTBOOK_RESOURCES)
    print(f"\nAvailable: {_fmt(TEXTBOOK_AVAILABLE)}")

    result = is_safe(TEXTBOOK_ALLOCATION, TEXTBOOK_MAX, TEXTBOOK_AVAILABLE,
                     TEXTBOOK_NAMES)
    print(f"\n{result.message}\n")
    for step in result.steps:
        print("  " + step.describe())
    # A safe state normally has several safe sequences; ours and the book's
    # differ only because of the order processes are scanned in.  Both are
    # checked here rather than compared to each other - see verify_sequence().
    print(f"\n  published in the book : {' -> '.join(TEXTBOOK_SAFE_SEQUENCE)}")
    print(f"  found by this code    : {result.sequence_text}")
    for label, seq in (("book's", TEXTBOOK_SAFE_SEQUENCE),
                       ("ours ", result.sequence)):
        valid = verify_sequence(seq, TEXTBOOK_ALLOCATION, TEXTBOOK_MAX,
                                TEXTBOOK_AVAILABLE, TEXTBOOK_NAMES)
        print(f"  {label} sequence verified valid: {valid}")
    print("  (both are correct - a safe state can have more than one safe order)")

    print("\n" + "=" * 68)
    print("RESOURCE REQUESTS")
    print("=" * 68)
    print("  The book asks these one after another, each from the state the")
    print("  previous one left behind - so they are chained here too.")

    allocation = [list(r) for r in TEXTBOOK_ALLOCATION]
    availability = list(TEXTBOOK_AVAILABLE)
    for index, req in ((1, [1, 0, 2]), (4, [3, 3, 0]), (0, [0, 2, 0])):
        decision = request_resources(index, req, allocation, TEXTBOOK_MAX,
                                     availability, TEXTBOOK_NAMES)
        print(f"\n  {TEXTBOOK_NAMES[index]} requests {_fmt(req)}   "
              f"(Available is {_fmt(availability)})")
        print(f"    {decision.reason}")
        if decision.granted:
            # Commit it, so the next request is judged from the new state.
            allocation = decision.resulting_allocation
            availability = decision.resulting_available

    print("\n" + "=" * 68)
    print("DEADLOCK DETECTION")
    print("=" * 68)
    detection = detect_deadlock(DETECTION_ALLOCATION, DETECTION_REQUEST,
                                DETECTION_AVAILABLE, TEXTBOOK_NAMES)
    print(f"\n  {detection.message}")

    stuck_request = [list(r) for r in DETECTION_REQUEST]
    stuck_request[2] = [0, 0, 1]        # P2 asks for one more C
    detection2 = detect_deadlock(DETECTION_ALLOCATION, stuck_request,
                                 DETECTION_AVAILABLE, TEXTBOOK_NAMES)
    print(f"\n  ...now P2 requests one more C:")
    print(f"  {detection2.message}")
    for step in detection2.steps:
        print("    " + step.describe())
