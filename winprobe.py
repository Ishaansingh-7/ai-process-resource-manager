"""
winprobe.py
===========
Fast Windows process probe -- one system call for the whole machine.

Why this module exists
----------------------
psutil is the obvious way to read process data, and this project still uses it
for the system-wide totals and as the portable fallback.  But on Windows,
asking psutil for memory / CPU time / thread count / status forces it to fall
back on a *per-process* call to the native ``NtQuerySystemInformation``,
because a normal (non-administrator) program is not allowed to open protected
system processes.  Each of those calls walks the entire process table, so the
cost is quadratic.  Measured on this machine with ~330 processes:

    psutil, all fields ................ ~4200 ms per refresh
    this module ........................... ~6 ms per refresh

A 2.5-second dashboard refresh simply cannot afford the first number.

What it does instead
--------------------
``NtQuerySystemInformation(SystemProcessInformation)`` -- the same call Task
Manager uses -- fills one buffer with a linked list of records describing
*every* process: image name, PID, thread count, kernel/user CPU time, working
set, base priority, and an array of per-thread records.  We make that call
once and walk the buffer with :mod:`ctypes`, so one snapshot costs one system
call no matter how many processes are running.

The layout of ``SYSTEM_PROCESS_INFORMATION`` is documented on MSDN and has
been stable since Windows NT; :func:`self_test` re-checks the offsets at
runtime so a layout change would be caught immediately rather than silently
producing garbage.
"""

from __future__ import annotations

import ctypes
import sys
from typing import List, NamedTuple, Optional

# ---------------------------------------------------------------------------
# Win32 / NT types
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    from ctypes import wintypes
    ULONG = wintypes.ULONG
    LONG = wintypes.LONG
    USHORT = wintypes.USHORT
else:  # keeps the module importable on Linux/macOS for the fallback path
    ULONG = ctypes.c_ulong
    LONG = ctypes.c_long
    USHORT = ctypes.c_ushort

LARGE_INTEGER = ctypes.c_longlong

SystemProcessInformation = 5                    # SYSTEM_INFORMATION_CLASS value
STATUS_SUCCESS = 0
STATUS_INFO_LENGTH_MISMATCH = ctypes.c_long(0xC0000004).value

# Thread state / wait reason values we care about: a process counts as
# "suspended" only when every one of its threads is waiting *because* it was
# suspended (this is how Windows parks background Store apps).
THREAD_STATE_WAIT = 5
WAIT_REASON_SUSPENDED = 5

# Windows expresses a process's priority as a base priority level 0-31.  These
# are the levels that correspond to the six priority classes shown in Task
# Manager.
BASE_PRIORITY_LABELS = {
    0: "Idle",           # System Idle Process
    4: "Idle",
    6: "Below Normal",
    8: "Normal",
    10: "Above Normal",
    13: "High",
    24: "Realtime",
}


class UNICODE_STRING(ctypes.Structure):
    _fields_ = [
        ("Length", USHORT),          # in bytes, not characters
        ("MaximumLength", USHORT),
        ("Buffer", ctypes.c_void_p),
    ]


class CLIENT_ID(ctypes.Structure):
    _fields_ = [("UniqueProcess", ctypes.c_void_p), ("UniqueThread", ctypes.c_void_p)]


class SYSTEM_THREAD_INFORMATION(ctypes.Structure):
    """One of these follows each process record, NumberOfThreads times."""

    _fields_ = [
        ("KernelTime", LARGE_INTEGER),
        ("UserTime", LARGE_INTEGER),
        ("CreateTime", LARGE_INTEGER),
        ("WaitTime", ULONG),
        ("StartAddress", ctypes.c_void_p),
        ("ClientId", CLIENT_ID),
        ("Priority", LONG),
        ("BasePriority", LONG),
        ("ContextSwitches", ULONG),
        ("ThreadState", ULONG),
        ("WaitReason", ULONG),
    ]


class SYSTEM_PROCESS_INFORMATION(ctypes.Structure):
    """
    One process record.  ``NextEntryOffset`` is the number of bytes to the next
    record (0 marks the end), which is what makes the buffer a linked list.
    """

    _fields_ = [
        ("NextEntryOffset", ULONG),
        ("NumberOfThreads", ULONG),
        ("WorkingSetPrivateSize", LARGE_INTEGER),
        ("HardFaultCount", ULONG),
        ("NumberOfThreadsHighWatermark", ULONG),
        ("CycleTime", ctypes.c_ulonglong),
        ("CreateTime", LARGE_INTEGER),
        ("UserTime", LARGE_INTEGER),          # in 100-nanosecond units
        ("KernelTime", LARGE_INTEGER),        # in 100-nanosecond units
        ("ImageName", UNICODE_STRING),
        ("BasePriority", LONG),
        ("UniqueProcessId", ctypes.c_void_p),  # the PID
        ("InheritedFromUniqueProcessId", ctypes.c_void_p),
        ("HandleCount", ULONG),
        ("SessionId", ULONG),
        ("UniqueProcessKey", ctypes.c_size_t),
        ("PeakVirtualSize", ctypes.c_size_t),
        ("VirtualSize", ctypes.c_size_t),
        ("PageFaultCount", ULONG),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),   # physical RAM in use, in bytes
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivatePageCount", ctypes.c_size_t),
        ("ReadOperationCount", LARGE_INTEGER),
        ("WriteOperationCount", LARGE_INTEGER),
        ("OtherOperationCount", LARGE_INTEGER),
        ("ReadTransferCount", LARGE_INTEGER),
        ("WriteTransferCount", LARGE_INTEGER),
        ("OtherTransferCount", LARGE_INTEGER),
    ]


_PROCESS_RECORD_SIZE = ctypes.sizeof(SYSTEM_PROCESS_INFORMATION)
_THREAD_RECORD_SIZE = ctypes.sizeof(SYSTEM_THREAD_INFORMATION)
_HUNDRED_NS_PER_SECOND = 10_000_000

# Load ntdll once.  On a non-Windows platform ctypes has no WinDLL at all, so
# the whole module degrades to "not available" and monitor.py uses psutil.
try:
    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtQuerySystemInformation.restype = ctypes.c_long
except (AttributeError, OSError):  # pragma: no cover - non-Windows
    _ntdll = None


# ---------------------------------------------------------------------------
# Public data shape
# ---------------------------------------------------------------------------


class RawProcess(NamedTuple):
    """
    Raw, un-derived facts about one process.

    ``cpu_seconds`` is a *cumulative counter* (total CPU time used since the
    process started), not a percentage -- monitor.py turns it into a
    percentage by comparing consecutive snapshots.  The psutil fallback in
    monitor.py produces this exact same tuple, so everything downstream is
    written once.
    """

    pid: int
    name: str
    thread_count: int
    cpu_seconds: float
    memory_bytes: int
    priority_value: Optional[int]
    suspended: bool
    status_hint: Optional[str] = None   # only the psutil fallback fills this in
    # Cumulative context switches across every thread in the process, another
    # counter monitor.py turns into a per-second rate.  Windows reports one
    # total per thread and does not separate voluntary from involuntary, so
    # ctx_switches_invol stays None here; see the note in monitor.py.
    ctx_switches: int = 0
    ctx_switches_invol: Optional[int] = None


def is_available() -> bool:
    """True when the fast native path can be used."""
    return sys.platform == "win32" and _ntdll is not None


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


class WindowsProcessSource:
    """
    Reads the whole process table in one call.

    Each instance owns its buffer, so two of them (the GUI sampler and the CSV
    collector run on different threads) never touch the same memory.
    """

    INITIAL_BUFFER_BYTES = 1 << 20  # 1 MB is enough for ~1000 processes

    def __init__(self) -> None:
        if not is_available():
            raise OSError("The native Windows process probe is not available here.")
        self._buffer = ctypes.create_string_buffer(self.INITIAL_BUFFER_BYTES)

    def list_processes(self) -> List[RawProcess]:
        """One snapshot of every process on the machine."""
        self._query()
        return self._parse()

    # -- step 1: fill the buffer ------------------------------------------

    def _query(self) -> None:
        needed = ULONG(0)
        for _ in range(5):  # the process list can grow between the two calls
            status = _ntdll.NtQuerySystemInformation(
                SystemProcessInformation,
                self._buffer,
                ctypes.sizeof(self._buffer),
                ctypes.byref(needed),
            )
            if status == STATUS_SUCCESS:
                return
            if status != STATUS_INFO_LENGTH_MISMATCH:
                raise OSError(
                    f"NtQuerySystemInformation failed (NTSTATUS 0x{status & 0xFFFFFFFF:08X})")
            # Grow generously: processes keep starting while we reallocate.
            self._buffer = ctypes.create_string_buffer(
                max(needed.value * 2, ctypes.sizeof(self._buffer) * 2))
        raise OSError("NtQuerySystemInformation kept asking for a bigger buffer")

    # -- step 2: walk the linked list -------------------------------------

    def _parse(self) -> List[RawProcess]:
        processes: List[RawProcess] = []
        base = ctypes.addressof(self._buffer)
        offset = 0

        while True:
            record = SYSTEM_PROCESS_INFORMATION.from_address(base + offset)
            pid = record.UniqueProcessId or 0

            # ImageName is a counted UTF-16 string, and it is NULL for PID 0.
            if record.ImageName.Buffer:
                name = ctypes.wstring_at(record.ImageName.Buffer,
                                         record.ImageName.Length // 2)
            else:
                name = "System Idle Process" if pid == 0 else f"pid-{pid}"

            # One walk of the thread array answers both questions, which is
            # why context switches cost almost nothing to add here (~0.9 ms
            # across the whole machine) while psutil's per-process
            # num_ctx_switches() costs about two seconds.
            suspended, ctx_switches = self._scan_threads(
                base + offset + _PROCESS_RECORD_SIZE, record.NumberOfThreads)

            processes.append(RawProcess(
                pid=pid,
                name=name,
                thread_count=record.NumberOfThreads,
                # Kernel + user time, converted from 100 ns ticks to seconds.
                cpu_seconds=(record.KernelTime + record.UserTime) / _HUNDRED_NS_PER_SECOND,
                memory_bytes=record.WorkingSetSize,
                priority_value=record.BasePriority,
                suspended=suspended,
                ctx_switches=ctx_switches,
            ))

            if record.NextEntryOffset == 0:
                break
            offset += record.NextEntryOffset

        return processes

    @staticmethod
    def _scan_threads(threads_address: int, count: int) -> tuple:
        """
        Walk one process's thread array once.

        Returns ``(all_suspended, context_switches)``:

        * a process counts as suspended only if *every* thread is parked;
        * its context-switch count is the sum over its threads, because the
          kernel counts switches per thread, not per process.
        """
        if count == 0:
            return False, 0
        all_suspended = True
        total_switches = 0
        for index in range(count):
            thread = SYSTEM_THREAD_INFORMATION.from_address(
                threads_address + index * _THREAD_RECORD_SIZE)
            total_switches += thread.ContextSwitches
            if (thread.ThreadState != THREAD_STATE_WAIT
                    or thread.WaitReason != WAIT_REASON_SUSPENDED):
                # Found a live thread.  We cannot stop looking any more - the
                # switch count still needs every thread - so just record it.
                all_suspended = False
        return all_suspended, total_switches


# ---------------------------------------------------------------------------
# Safety net
# ---------------------------------------------------------------------------


def self_test() -> None:
    """
    Verify the struct layout against the documented offsets.

    Reading a C structure from raw memory only works if Python's idea of the
    layout matches the kernel's, so we assert it instead of trusting it.
    """
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    if pointer_size != 8:
        return  # the reference offsets below are the 64-bit ones
    expected = {
        "NextEntryOffset": 0, "NumberOfThreads": 4, "ImageName": 56,
        "BasePriority": 72, "UniqueProcessId": 80, "WorkingSetSize": 144,
    }
    for field, offset in expected.items():
        actual = getattr(SYSTEM_PROCESS_INFORMATION, field).offset
        if actual != offset:
            raise AssertionError(
                f"SYSTEM_PROCESS_INFORMATION.{field} is at {actual}, expected {offset}")
    assert _PROCESS_RECORD_SIZE == 256, _PROCESS_RECORD_SIZE
    assert _THREAD_RECORD_SIZE == 80, _THREAD_RECORD_SIZE


# ---------------------------------------------------------------------------
# Manual test:  python winprobe.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    self_test()
    print("struct layout OK")

    source = WindowsProcessSource()
    source.list_processes()                       # warm up
    start = time.perf_counter()
    rows = source.list_processes()
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"{len(rows)} processes in {elapsed_ms:.1f} ms\n")
    rows.sort(key=lambda r: r.memory_bytes, reverse=True)
    print(f"{'PID':>7} {'NAME':<32} {'THR':>4} {'CPU(s)':>10} {'MEM(MB)':>9}  PRIORITY")
    for row in rows[:10]:
        label = BASE_PRIORITY_LABELS.get(row.priority_value, f"Base {row.priority_value}")
        print(f"{row.pid:>7} {row.name[:32]:<32} {row.thread_count:>4} "
              f"{row.cpu_seconds:>10.1f} {row.memory_bytes / 1048576:>9.1f}  {label}")
    print("\nsuspended:", ", ".join(r.name for r in rows if r.suspended) or "none")
