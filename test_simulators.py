"""
test_simulators.py
==================
Unit tests for scheduler.py and deadlock.py.

Run them with::

    python test_simulators.py            # or: python -m unittest test_simulators

Why these particular examples
-----------------------------
Every numbered example below is a worked exercise from Silberschatz, Galvin &
Gagne, *Operating System Concepts*, where the correct answer is printed in the
book.  Testing against those means the implementations are checked against an
outside authority rather than against themselves - a test that just records
whatever the code happens to do today proves nothing.

Alongside the fixed examples there are *property* tests, which check things
that must hold for every possible input rather than for one worked example:

* ``waiting == turnaround - burst`` for every process under every algorithm
* the timeline must account for every unit of CPU time, with no gaps or overlaps
* SJF must never produce a worse average waiting time than FCFS
* a safe sequence must actually be executable

One deliberate choice: where a safe state has several valid safe sequences,
the test checks that ours *is valid* rather than that it matches the book's
string.  Matching the string would test the order the code scans processes in,
which is an implementation detail, not the algorithm.
"""

from __future__ import annotations

import random
import unittest

import deadlock
import scheduler
from deadlock import (detect_deadlock, is_safe, need_matrix, request_resources,
                      verify_sequence)
from scheduler import (FCFS, PRIORITY, ROUND_ROBIN, SJF, IDLE_LABEL, SimProcess,
                       compare_all, fcfs, priority_scheduling, round_robin, run,
                       sjf)

PLACES = 6          # float comparison tolerance for exact arithmetic


def procs(*rows) -> list:
    """Build a process list from (pid, arrival, burst[, priority]) tuples."""
    return [SimProcess(pid=r[0], arrival=r[1], burst=r[2],
                       priority=r[3] if len(r) > 3 else 0) for r in rows]


def waiting_of(result) -> dict:
    return {m.pid: m.waiting for m in result.metrics}


# ---------------------------------------------------------------------------
# Scheduling - worked examples with published answers
# ---------------------------------------------------------------------------


class TestFCFS(unittest.TestCase):
    """Silberschatz ch.5: P1=24, P2=3, P3=3, all arriving at time 0."""

    def setUp(self):
        self.demo = procs(("P1", 0, 24), ("P2", 0, 3), ("P3", 0, 3))

    def test_average_waiting_time_is_17(self):
        result = fcfs(self.demo)
        self.assertAlmostEqual(result.avg_waiting, 17.0, places=PLACES)

    def test_individual_waiting_times(self):
        # The book's numbers: P1 waits 0, P2 waits 24, P3 waits 27.
        self.assertEqual(waiting_of(fcfs(self.demo)),
                         {"P1": 0, "P2": 24, "P3": 27})

    def test_gantt_order(self):
        timeline = [(s.pid, s.start, s.end) for s in fcfs(self.demo).timeline]
        self.assertEqual(timeline, [("P1", 0, 24), ("P2", 24, 27), ("P3", 27, 30)])

    def test_convoy_effect(self):
        """
        The same three jobs, short ones first, average 17 -> 3.

        This is the convoy effect and the whole argument against FCFS: the
        work is identical, only the order changed.
        """
        reordered = procs(("P2", 0, 3), ("P3", 0, 3), ("P1", 0, 24))
        self.assertAlmostEqual(fcfs(reordered).avg_waiting, 3.0, places=PLACES)


class TestSJF(unittest.TestCase):
    """Silberschatz ch.5: bursts 6, 8, 7, 3 all arriving at time 0."""

    def setUp(self):
        self.demo = procs(("P1", 0, 6), ("P2", 0, 8), ("P3", 0, 7), ("P4", 0, 3))

    def test_average_waiting_time_is_7(self):
        self.assertAlmostEqual(sjf(self.demo).avg_waiting, 7.0, places=PLACES)

    def test_runs_shortest_first(self):
        order = [s.pid for s in sjf(self.demo).timeline]
        self.assertEqual(order, ["P4", "P1", "P3", "P2"])

    def test_individual_waiting_times(self):
        self.assertEqual(waiting_of(sjf(self.demo)),
                         {"P1": 3, "P2": 16, "P3": 9, "P4": 0})


class TestPriority(unittest.TestCase):
    """
    Silberschatz ch.5: five jobs, lower number = higher priority.

        P1 burst 10 priority 3      P4 burst 1 priority 5
        P2 burst  1 priority 1      P5 burst 5 priority 2
        P3 burst  2 priority 4
    """

    def setUp(self):
        self.demo = procs(("P1", 0, 10, 3), ("P2", 0, 1, 1), ("P3", 0, 2, 4),
                          ("P4", 0, 1, 5), ("P5", 0, 5, 2))

    def test_average_waiting_time_is_8_2(self):
        self.assertAlmostEqual(priority_scheduling(self.demo).avg_waiting,
                               8.2, places=PLACES)

    def test_order_is_by_priority(self):
        order = [s.pid for s in priority_scheduling(self.demo).timeline]
        self.assertEqual(order, ["P2", "P5", "P1", "P3", "P4"])

    def test_lower_number_really_means_higher_priority(self):
        two = procs(("Low", 0, 5, 9), ("High", 0, 5, 1))
        self.assertEqual(priority_scheduling(two).timeline[0].pid, "High")


class TestRoundRobin(unittest.TestCase):
    """Silberschatz ch.5: P1=24, P2=3, P3=3, quantum 4."""

    def setUp(self):
        self.demo = procs(("P1", 0, 24), ("P2", 0, 3), ("P3", 0, 3))

    def test_average_waiting_time(self):
        result = round_robin(self.demo, quantum=4)
        # The book's figure is 17/3 = 5.66.
        self.assertAlmostEqual(result.avg_waiting, 17 / 3, places=PLACES)

    def test_individual_waiting_times(self):
        self.assertEqual(waiting_of(round_robin(self.demo, quantum=4)),
                         {"P1": 6, "P2": 4, "P3": 7})

    def test_response_time_is_its_selling_point(self):
        """RR starts every process quickly; FCFS makes the last one wait 27."""
        rr = round_robin(self.demo, quantum=4)
        self.assertAlmostEqual(rr.avg_response, (0 + 4 + 7) / 3, places=PLACES)
        self.assertLess(rr.avg_response, fcfs(self.demo).avg_response)

    def test_consecutive_quanta_are_merged(self):
        """
        A process running two quanta back to back is one slice, not two.

        Nothing is switched away and back, so drawing two bars would claim a
        context switch that never happened.  After P2 and P3 finish, P1 is
        alone and runs 10 -> 30 without interruption.
        """
        timeline = [(s.pid, s.start, s.end)
                    for s in round_robin(self.demo, quantum=4).timeline]
        self.assertEqual(timeline, [("P1", 0, 4), ("P2", 4, 7),
                                    ("P3", 7, 10), ("P1", 10, 30)])

    def test_arrival_at_quantum_expiry_is_queued_first(self):
        """
        The tie-break that decides RR's answer.

        P1 arrives at 0 with burst 4, P2 arrives at exactly 2 - the same
        instant P1's first quantum expires.  P2 must go into the ready queue
        *before* the preempted P1, so the order is P1, P2, P1.
        """
        demo = procs(("P1", 0, 4), ("P2", 2, 2))
        order = [s.pid for s in round_robin(demo, quantum=2).timeline]
        self.assertEqual(order, ["P1", "P2", "P1"])

    def test_quantum_must_be_positive(self):
        with self.assertRaises(ValueError):
            round_robin(self.demo, quantum=0)

    def test_large_quantum_degenerates_to_fcfs(self):
        """With a quantum longer than every burst, RR *is* FCFS."""
        big = round_robin(self.demo, quantum=1000)
        self.assertAlmostEqual(big.avg_waiting, fcfs(self.demo).avg_waiting,
                               places=PLACES)


class TestArrivalsAndIdleTime(unittest.TestCase):
    """Staggered arrivals, including a stretch where the CPU has nothing to do."""

    def test_idle_gap_is_recorded(self):
        demo = procs(("P1", 0, 2), ("P2", 10, 2))
        result = fcfs(demo)
        idle = [s for s in result.timeline if s.is_idle]
        self.assertEqual(len(idle), 1)
        self.assertEqual((idle[0].start, idle[0].end), (2, 10))

    def test_idle_time_lowers_utilisation(self):
        demo = procs(("P1", 0, 2), ("P2", 10, 2))
        # 4 units of work across a 12-unit schedule.
        self.assertAlmostEqual(fcfs(demo).cpu_utilisation, 4 / 12 * 100, places=PLACES)

    def test_first_process_does_not_start_before_it_arrives(self):
        demo = procs(("P1", 5, 3))
        result = fcfs(demo)
        self.assertEqual(result.metrics[0].start, 5)
        self.assertEqual(result.metrics[0].waiting, 0)

    def test_round_robin_handles_idle_gaps(self):
        demo = procs(("P1", 0, 2), ("P2", 10, 2))
        result = round_robin(demo, quantum=1)
        self.assertEqual(result.metrics[1].completion, 12)


# ---------------------------------------------------------------------------
# Scheduling - properties that must hold for *every* input
# ---------------------------------------------------------------------------


class TestSchedulingProperties(unittest.TestCase):

    def _random_workload(self, rng, n=6):
        return [SimProcess(pid=f"P{i}",
                           arrival=rng.choice([0, 0, 1, 3, 7]),
                           burst=rng.randint(1, 12),
                           priority=rng.randint(1, 5))
                for i in range(n)]

    def test_waiting_identity_holds_everywhere(self):
        """waiting == turnaround - burst, for every process, every algorithm."""
        rng = random.Random(20260903)
        for _ in range(60):
            workload = self._random_workload(rng)
            for name, result in compare_all(workload, quantum=3).items():
                for m in result.metrics:
                    self.assertAlmostEqual(
                        m.waiting, m.turnaround - m.burst, places=PLACES,
                        msg=f"{name}/{m.pid}")

    def test_nobody_waits_a_negative_amount_of_time(self):
        rng = random.Random(7)
        for _ in range(60):
            workload = self._random_workload(rng)
            for name, result in compare_all(workload, quantum=2).items():
                for m in result.metrics:
                    self.assertGreaterEqual(m.waiting, -1e-9, f"{name}/{m.pid}")
                    self.assertGreaterEqual(m.response, -1e-9, f"{name}/{m.pid}")

    def test_timeline_is_contiguous_and_complete(self):
        """
        The Gantt chart must account for every instant with no gaps or overlaps,
        and the busy time must equal the total work submitted.
        """
        rng = random.Random(99)
        for _ in range(60):
            workload = self._random_workload(rng)
            total_burst = sum(p.burst for p in workload)
            for name, result in compare_all(workload, quantum=3).items():
                timeline = result.timeline
                for earlier, later in zip(timeline, timeline[1:]):
                    self.assertAlmostEqual(earlier.end, later.start, places=PLACES,
                                           msg=f"{name}: gap or overlap")
                self.assertAlmostEqual(result.cpu_busy, total_burst, places=PLACES,
                                       msg=f"{name}: lost or invented CPU time")

    def test_every_process_finishes_after_it_arrives(self):
        rng = random.Random(4242)
        for _ in range(40):
            workload = self._random_workload(rng)
            for name, result in compare_all(workload, quantum=4).items():
                for m in result.metrics:
                    self.assertGreaterEqual(m.start, m.arrival - 1e-9, f"{name}/{m.pid}")
                    self.assertGreaterEqual(m.completion, m.start + m.burst - 1e-9,
                                            f"{name}/{m.pid}")

    def test_sjf_is_optimal_for_average_waiting_time(self):
        """
        With everything arriving together, SJF is provably optimal - no other
        ordering can beat it.  So it must never lose to FCFS or Priority.
        """
        rng = random.Random(1234)
        for _ in range(80):
            workload = [SimProcess(pid=f"P{i}", arrival=0,
                                   burst=rng.randint(1, 20),
                                   priority=rng.randint(1, 5))
                        for i in range(rng.randint(2, 7))]
            best = sjf(workload).avg_waiting
            self.assertLessEqual(best, fcfs(workload).avg_waiting + 1e-9)
            self.assertLessEqual(best, priority_scheduling(workload).avg_waiting + 1e-9)
            self.assertLessEqual(best, round_robin(workload, 2).avg_waiting + 1e-9)

    def test_all_algorithms_do_the_same_total_work(self):
        workload = procs(("A", 0, 5), ("B", 2, 3), ("C", 4, 7))
        spans = {name: r.cpu_busy for name, r in compare_all(workload, 2).items()}
        self.assertEqual(len(set(round(v, 6) for v in spans.values())), 1, spans)


class TestSchedulerValidation(unittest.TestCase):

    def test_zero_burst_is_rejected(self):
        with self.assertRaises(ValueError):
            SimProcess("P1", 0, 0)

    def test_negative_arrival_is_rejected(self):
        with self.assertRaises(ValueError):
            SimProcess("P1", -1, 5)

    def test_empty_workload_is_rejected(self):
        with self.assertRaises(ValueError):
            run(FCFS, [])

    def test_unknown_algorithm_is_rejected(self):
        with self.assertRaises(ValueError):
            run("Shortest Remaining Time", procs(("P1", 0, 1)))

    def test_single_process(self):
        for name in (FCFS, SJF, PRIORITY, ROUND_ROBIN):
            result = run(name, procs(("Solo", 0, 5)), quantum=2)
            self.assertAlmostEqual(result.avg_waiting, 0.0, places=PLACES, msg=name)
            self.assertAlmostEqual(result.makespan, 5.0, places=PLACES, msg=name)


# ---------------------------------------------------------------------------
# Banker's algorithm - the standard worked example
# ---------------------------------------------------------------------------


class TestBankersSafety(unittest.TestCase):
    """
    Silberschatz ch.8: 5 processes, 3 resource types, Available = (3, 3, 2).
    The book gives <P1, P3, P4, P2, P0> as a safe sequence.
    """

    def setUp(self):
        self.allocation = deadlock.TEXTBOOK_ALLOCATION
        self.maximum = deadlock.TEXTBOOK_MAX
        self.available = deadlock.TEXTBOOK_AVAILABLE
        self.names = deadlock.TEXTBOOK_NAMES

    def test_need_matrix(self):
        """Need = Max - Allocation, as printed in the book."""
        self.assertEqual(
            need_matrix(self.allocation, self.maximum),
            [[7, 4, 3], [1, 2, 2], [6, 0, 0], [0, 1, 1], [4, 3, 1]])

    def test_state_is_safe(self):
        self.assertTrue(is_safe(self.allocation, self.maximum,
                                self.available, self.names).safe)

    def test_our_sequence_is_a_valid_one(self):
        result = is_safe(self.allocation, self.maximum, self.available, self.names)
        self.assertTrue(verify_sequence(result.sequence, self.allocation,
                                        self.maximum, self.available, self.names),
                        f"{result.sequence_text} is not actually executable")

    def test_the_books_sequence_is_also_valid(self):
        """Both are correct - a safe state can have several safe sequences."""
        self.assertTrue(verify_sequence(deadlock.TEXTBOOK_SAFE_SEQUENCE,
                                        self.allocation, self.maximum,
                                        self.available, self.names))

    def test_sequence_includes_every_process_exactly_once(self):
        result = is_safe(self.allocation, self.maximum, self.available, self.names)
        self.assertEqual(sorted(result.sequence), sorted(self.names))

    def test_a_bogus_sequence_is_rejected(self):
        """P0 cannot go first: it needs (7,4,3) and only (3,3,2) is free."""
        self.assertFalse(verify_sequence(["P0", "P1", "P2", "P3", "P4"],
                                         self.allocation, self.maximum,
                                         self.available, self.names))

    def test_trace_explains_each_step(self):
        result = is_safe(self.allocation, self.maximum, self.available, self.names)
        self.assertEqual(len(result.steps), 5)
        self.assertTrue(all(s.granted for s in result.steps))
        self.assertIn("P1", result.steps[0].describe())

    def test_an_unsafe_state_is_detected(self):
        """Take resources away until nobody can finish."""
        result = is_safe(self.allocation, self.maximum, [0, 0, 0], self.names)
        self.assertFalse(result.safe)
        self.assertEqual(result.sequence, [])
        self.assertEqual(sorted(result.unfinished), sorted(self.names))

    def test_unsafe_message_says_it_is_not_deadlock(self):
        result = is_safe(self.allocation, self.maximum, [0, 0, 0], self.names)
        self.assertIn("not the same as being deadlocked", result.message)


class TestBankersRequests(unittest.TestCase):
    """
    The book asks three requests in sequence, each judged from the state the
    previous one left behind:

        P1 asks (1,0,2)  -> granted, state stays safe
        P4 asks (3,3,0)  -> must wait, the resources are not free
        P0 asks (0,2,0)  -> denied, the resources ARE free but the result is unsafe
    """

    def setUp(self):
        self.allocation = [list(r) for r in deadlock.TEXTBOOK_ALLOCATION]
        self.maximum = deadlock.TEXTBOOK_MAX
        self.available = list(deadlock.TEXTBOOK_AVAILABLE)
        self.names = deadlock.TEXTBOOK_NAMES

    def _grant_p1(self):
        decision = request_resources(1, [1, 0, 2], self.allocation, self.maximum,
                                     self.available, self.names)
        self.allocation = decision.resulting_allocation
        self.available = decision.resulting_available
        return decision

    def test_p1_request_is_granted(self):
        decision = self._grant_p1()
        self.assertTrue(decision.granted)
        self.assertEqual(decision.resulting_available, [2, 3, 0])
        self.assertEqual(decision.resulting_allocation[1], [3, 0, 2])

    def test_p4_request_must_wait_for_resources(self):
        self._grant_p1()
        decision = request_resources(4, [3, 3, 0], self.allocation, self.maximum,
                                     self.available, self.names)
        self.assertFalse(decision.granted)
        self.assertIn("WAIT", decision.reason)

    def test_p0_request_is_denied_although_resources_are_free(self):
        """The distinction that makes this *avoidance* rather than bookkeeping."""
        self._grant_p1()
        decision = request_resources(0, [0, 2, 0], self.allocation, self.maximum,
                                     self.available, self.names)
        self.assertFalse(decision.granted)
        self.assertIn("DENIED", decision.reason)
        self.assertIsNotNone(decision.safety)
        self.assertFalse(decision.safety.safe)

    def test_request_beyond_declared_maximum_is_an_error(self):
        """P1's Need is (1,2,2); asking for 5 of A is a bug in the process."""
        decision = request_resources(1, [5, 0, 0], self.allocation, self.maximum,
                                     self.available, self.names)
        self.assertFalse(decision.granted)
        self.assertIn("exceeded the maximum", decision.reason)

    def test_zero_request_is_granted_and_changes_nothing(self):
        decision = request_resources(2, [0, 0, 0], self.allocation, self.maximum,
                                     self.available, self.names)
        self.assertTrue(decision.granted)
        self.assertEqual(decision.resulting_available, self.available)

    def test_negative_request_is_rejected(self):
        with self.assertRaises(ValueError):
            request_resources(0, [-1, 0, 0], self.allocation, self.maximum,
                              self.available, self.names)

    def test_unknown_process_is_rejected(self):
        with self.assertRaises(ValueError):
            request_resources(99, [0, 0, 0], self.allocation, self.maximum,
                              self.available, self.names)


class TestDeadlockDetection(unittest.TestCase):
    """
    Silberschatz ch.8 detection example: Available = (0,0,0) and nothing is
    deadlocked, because P0 and P2 can finish and release enough for the rest.
    Change P2's request to one more C and P1-P4 all deadlock.
    """

    def setUp(self):
        self.allocation = deadlock.DETECTION_ALLOCATION
        self.request = deadlock.DETECTION_REQUEST
        self.available = deadlock.DETECTION_AVAILABLE
        self.names = deadlock.TEXTBOOK_NAMES

    def test_no_deadlock_in_the_first_state(self):
        result = detect_deadlock(self.allocation, self.request,
                                 self.available, self.names)
        self.assertFalse(result.has_deadlock)
        self.assertEqual(sorted(result.sequence), sorted(self.names))

    def test_p2_asking_for_one_more_c_deadlocks_four_processes(self):
        request = [list(r) for r in self.request]
        request[2] = [0, 0, 1]
        result = detect_deadlock(self.allocation, request,
                                 self.available, self.names)
        self.assertTrue(result.has_deadlock)
        self.assertEqual(result.deadlocked, ["P1", "P2", "P3", "P4"])
        self.assertEqual(result.sequence, ["P0"])

    def test_detection_explains_who_waits_for_what(self):
        request = [list(r) for r in self.request]
        request[2] = [0, 0, 1]
        result = detect_deadlock(self.allocation, request,
                                 self.available, self.names)
        final = result.steps[-1].describe()
        self.assertIn("P1 is waiting for", final)
        self.assertIn("while holding", final)

    def test_a_process_wanting_nothing_never_deadlocks(self):
        result = detect_deadlock([[1, 0]], [[0, 0]], [0, 0], ["P0"])
        self.assertFalse(result.has_deadlock)

    def test_classic_two_process_circular_wait(self):
        """
        P0 holds A and wants B; P1 holds B and wants A; nothing is free.
        The smallest possible deadlock, and both processes are in it.
        """
        result = detect_deadlock(allocation=[[1, 0], [0, 1]],
                                 request=[[0, 1], [1, 0]],
                                 available=[0, 0],
                                 names=["P0", "P1"])
        self.assertTrue(result.has_deadlock)
        self.assertEqual(result.deadlocked, ["P0", "P1"])

    def test_detection_differs_from_safety(self):
        """
        The same state can be UNSAFE yet not deadlocked - the point most worth
        being able to explain.  Here each process holds 1 of 2 units and could
        still ask for 1 more, so the state is unsafe; but neither is actually
        requesting anything right now, so nothing is stuck.
        """
        allocation = [[1], [1]]
        maximum = [[2], [2]]
        available = [0]
        self.assertFalse(is_safe(allocation, maximum, available).safe)
        self.assertFalse(detect_deadlock(allocation, [[0], [0]],
                                         available).has_deadlock)


class TestDeadlockValidation(unittest.TestCase):

    def test_holding_more_than_the_declared_maximum_is_rejected(self):
        with self.assertRaises(ValueError):
            is_safe([[5]], [[2]], [0])

    def test_ragged_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            is_safe([[1, 2], [1]], [[2, 2], [2, 2]], [0, 0])

    def test_no_processes_is_rejected(self):
        with self.assertRaises(ValueError):
            is_safe([], [], [1, 2])

    def test_negative_allocation_is_rejected(self):
        with self.assertRaises(ValueError):
            is_safe([[-1]], [[1]], [0])

    def test_everything_free_and_nothing_held_is_safe(self):
        result = is_safe([[0, 0], [0, 0]], [[1, 1], [1, 1]], [1, 1])
        self.assertTrue(result.safe)


# ---------------------------------------------------------------------------
# Deriving simulator input from real collected data
# ---------------------------------------------------------------------------


class TestLoadFromRealData(unittest.TestCase):
    """
    These check the derivation is *sane*, not that it produces any particular
    number - the numbers depend on whatever the collector happened to record.
    """

    def setUp(self):
        try:
            self.derived = scheduler.load_from_csv(limit=5)
        except (FileNotFoundError, ValueError) as exc:
            self.skipTest(f"no usable dataset: {exc}")

    def test_returns_processes(self):
        self.assertTrue(self.derived)
        self.assertLessEqual(len(self.derived), 5)

    def test_bursts_are_positive_and_scaled(self):
        for d in self.derived:
            self.assertGreater(d.process.burst, 0)
            self.assertLessEqual(d.process.burst, scheduler.SCALE_TARGET + 1e-9)

    def test_busiest_process_sets_the_scale(self):
        biggest = max(d.process.burst for d in self.derived)
        self.assertAlmostEqual(biggest, scheduler.SCALE_TARGET, places=1)

    def test_ordered_by_measured_cpu_time(self):
        seconds = [d.cpu_seconds for d in self.derived]
        self.assertEqual(seconds, sorted(seconds, reverse=True))

    def test_derived_processes_can_actually_be_scheduled(self):
        workload = [d.process for d in self.derived]
        for name, result in compare_all(workload, quantum=2).items():
            self.assertEqual(len(result.metrics), len(workload), name)
            self.assertGreaterEqual(result.avg_waiting, 0.0, name)

    def test_missing_file_raises_something_actionable(self):
        with self.assertRaises(FileNotFoundError) as caught:
            scheduler.load_from_csv(csv_path="no/such/file.csv")
        self.assertIn("--collect-only", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
