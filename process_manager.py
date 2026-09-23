"""
process_manager.py
==================
Phase 4, part 2: the only module in this project that *changes* anything.

Everything else observes.  This module lowers priorities, suspends and resumes
processes - real operating-system actions with real consequences - so it is
written to be paranoid rather than convenient.

The five safety rules
---------------------
1. **Nothing runs without being asked.**  There is no timer, no automatic
   trigger, no "apply all".  Every method here is called from a button press
   that the user has already confirmed in a dialog.
2. **The protected list is enforced here, not in the UI.**  recommender.py
   also consults it so it never *suggests* a forbidden action, but the check
   that matters is this one: a bug in the GUI, or somebody calling this module
   from a script, still cannot suspend csrss.exe.
3. **The PID must still be the process the caller meant.**  Windows recycles
   PIDs quickly.  Between the dashboard drawing a row and the user clicking
   Apply, PID 4312 may have become something else entirely, so every action
   re-checks the process name before touching it.
4. **We never touch ourselves.**  Suspending the dashboard's own process would
   freeze the window with no way to un-freeze it.
5. **Everything is written down.**  Every attempt - success, refusal or
   failure - is appended to data/action_log.csv with a timestamp, so there is
   an audit trail of what the tool did.

Failures are returned, never raised: a process that exits while the dialog is
open, or one the OS will not let us modify, produces an ActionResult saying so.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ACTION_LOG = PROJECT_DIR / "data" / "action_log.csv"

ACTION_LOG_COLUMNS = ["timestamp", "pid", "name", "action", "result", "detail"]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Action names, also used as the labels on the buttons.
ACTION_LOWER_PRIORITY = "Lower Priority"
ACTION_SUSPEND = "Suspend"
ACTION_RESUME = "Resume"


# ---------------------------------------------------------------------------
# The protected list
# ---------------------------------------------------------------------------

# Windows depends on these to stay alive and responsive.  Suspending or
# starving any of them ranges from "the desktop freezes" to "the machine
# blue-screens or logs you out", and none of them is ever the actual cause of
# a resource problem worth acting on.  Comparison is case-insensitive because
# Windows process names are.
#
# The pseudo-processes (System, Registry, ...) are not really user processes
# at all - they are kernel bookkeeping that psutil surfaces as PIDs.
PROTECTED_PROCESS_NAMES = frozenset(name.lower() for name in (
    # Kernel / pseudo-processes
    "System Idle Process", "System", "Registry", "Secure System",
    "Memory Compression",
    # Session and logon core - killing or freezing these ends the session
    "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe", "logonui.exe",
    "services.exe", "lsass.exe", "lsm.exe", "fontdrvhost.exe",
    # Service host: one svchost hosts many Windows services at once
    "svchost.exe",
    # Desktop compositor and shell - suspending these freezes the UI
    "dwm.exe", "explorer.exe",
    # Security - deliberately excluded from tampering
    "MsMpEng.exe",
))

# PIDs 0 and 4 are the Idle and System pseudo-processes on Windows.
PROTECTED_PIDS = frozenset((0, 4))


def is_protected(pid: int, name: Optional[str] = None) -> bool:
    """
    True when this process must never be modified.

    Checked by both the recommender (so nothing is suggested) and every action
    below (so nothing is performed), because the UI is not a security boundary.
    """
    if pid in PROTECTED_PIDS:
        return True
    if pid == os.getpid():          # rule 4: never act on the dashboard itself
        return True
    if name and name.lower() in PROTECTED_PROCESS_NAMES:
        return True
    return False


def protection_reason(pid: int, name: Optional[str] = None) -> str:
    """Human-readable explanation for why something is off limits."""
    if pid == os.getpid():
        return "This is the resource manager's own process"
    if pid in PROTECTED_PIDS or (name and name.lower() in PROTECTED_PROCESS_NAMES):
        return "Critical system process - protected from modification"
    return ""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """What happened when an action was attempted."""

    success: bool
    action: str
    pid: int
    name: str
    message: str

    # "done", "refused" (we would not), or "failed" (the OS would not)
    outcome: str = "done"

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


class ProcessManager:
    """
    Performs OS actions on request and records every one of them.

    Nothing here is called automatically - see the module docstring.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = Path(log_path) if log_path else DEFAULT_ACTION_LOG
        # Actions come from GUI callbacks and the log is read on the same
        # thread, but a lock costs nothing and makes the file safe if this is
        # ever driven from somewhere else.
        self._lock = threading.Lock()
        self._ensure_log()

    # ------------------------------------------------------------------
    # Public actions
    # ------------------------------------------------------------------

    def lower_priority(self, pid: int, expected_name: Optional[str] = None) -> ActionResult:
        """
        Drop the process to below-normal priority.

        The gentlest of the three actions: the process keeps running and keeps
        its state, it simply loses scheduling arguments against everything
        else.  Nothing is lost, and it can be undone from Task Manager.
        """
        def act(process: psutil.Process) -> str:
            if psutil.WINDOWS:
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                return "priority set to Below Normal"
            process.nice(10)        # POSIX: higher nice value = lower priority
            return "nice value set to 10"

        return self._perform(ACTION_LOWER_PRIORITY, pid, expected_name, act)

    def suspend_process(self, pid: int, expected_name: Optional[str] = None) -> ActionResult:
        """
        Freeze every thread in the process.

        This is the destructive one.  The process stops dead where it is: it
        cannot save, cannot respond, cannot finish what it was doing, and it
        stays that way until something resumes it.  The GUI puts an explicit
        warning in front of the user before this is ever called.
        """
        def act(process: psutil.Process) -> str:
            process.suspend()
            return "all threads suspended"

        return self._perform(ACTION_SUSPEND, pid, expected_name, act)

    def resume_process(self, pid: int, expected_name: Optional[str] = None) -> ActionResult:
        """Undo a suspend - the reason suspending is recoverable at all."""
        def act(process: psutil.Process) -> str:
            process.resume()
            return "threads resumed"

        return self._perform(ACTION_RESUME, pid, expected_name, act)

    # ------------------------------------------------------------------
    # The one code path that touches the OS
    # ------------------------------------------------------------------

    def _perform(self, action: str, pid: int, expected_name: Optional[str],
                 operation) -> ActionResult:
        """
        Every action goes through here, so every safety check is applied once
        and every outcome is logged exactly once.
        """
        name = expected_name or f"pid-{pid}"

        # --- guard 1: is this something we are allowed to touch? ----------
        if is_protected(pid, expected_name):
            return self._finish(ActionResult(
                False, action, pid, name,
                f"Refused: {protection_reason(pid, expected_name)}", "refused"))

        try:
            process = psutil.Process(pid)
            actual_name = process.name()
        except ValueError:
            # psutil rejects a negative or non-integer PID before it looks
            # anything up.  Nothing here may raise (see the module docstring),
            # so that becomes a refusal like any other.
            return self._finish(ActionResult(
                False, action, pid, name,
                f"Refused: {pid!r} is not a valid process id", "refused"))
        except psutil.NoSuchProcess:
            return self._finish(ActionResult(
                False, action, pid, name,
                "Process has already exited", "failed"))
        except psutil.AccessDenied:
            return self._finish(ActionResult(
                False, action, pid, name,
                "Access denied - this process is protected by Windows", "failed"))

        # --- guard 2: is it still the process the caller meant? -----------
        # Windows reuses PIDs aggressively.  Acting on a stale PID would hit an
        # innocent bystander, so a mismatch is refused rather than guessed at.
        if expected_name and actual_name.lower() != expected_name.lower():
            return self._finish(ActionResult(
                False, action, pid, name,
                f"Refused: PID {pid} is now '{actual_name}', not '{expected_name}'",
                "refused"))

        # --- guard 3: re-check protection against the *real* name ---------
        # The caller's name could have been wrong or missing; this checks what
        # the OS actually reports.
        if is_protected(pid, actual_name):
            return self._finish(ActionResult(
                False, action, pid, actual_name,
                f"Refused: {protection_reason(pid, actual_name)}", "refused"))

        # --- perform ------------------------------------------------------
        try:
            detail = operation(process)
        except psutil.NoSuchProcess:
            return self._finish(ActionResult(
                False, action, pid, actual_name,
                "Process exited before the action could be applied", "failed"))
        except psutil.AccessDenied:
            return self._finish(ActionResult(
                False, action, pid, actual_name,
                "Access denied - try running as administrator", "failed"))
        except (OSError, psutil.Error) as exc:
            return self._finish(ActionResult(
                False, action, pid, actual_name,
                f"Failed: {exc}", "failed"))

        return self._finish(ActionResult(
            True, action, pid, actual_name,
            f"{action} applied to {actual_name} ({pid}): {detail}", "done"))

    # ------------------------------------------------------------------
    # The audit log
    # ------------------------------------------------------------------

    def _ensure_log(self) -> None:
        """Create data/action_log.csv with a header if it is not there yet."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists() or self.log_path.stat().st_size == 0:
                with open(self.log_path, "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(ACTION_LOG_COLUMNS)
        except OSError as exc:      # a missing log must not break the feature
            log.warning("Could not create %s: %s", self.log_path, exc)

    def _finish(self, result: ActionResult) -> ActionResult:
        """Log an outcome and hand it back to the caller."""
        row = [
            datetime.now().strftime(TIMESTAMP_FORMAT),
            result.pid,
            result.name,
            result.action,
            result.outcome,
            result.message,
        ]
        try:
            with self._lock:
                with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(row)
        except OSError as exc:
            log.warning("Could not write to %s: %s", self.log_path, exc)

        log.info("%s | %s (%d) | %s", result.outcome.upper(),
                 result.name, result.pid, result.message)
        return result

    def recent_actions(self, limit: int = 10) -> List[Dict[str, str]]:
        """The most recent log entries, newest first (for the GUI panel)."""
        if not self.log_path.exists():
            return []
        try:
            with self._lock:
                with open(self.log_path, newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
        except OSError as exc:
            log.warning("Could not read %s: %s", self.log_path, exc)
            return []
        return rows[-limit:][::-1]


# ---------------------------------------------------------------------------
# Manual test:  python process_manager.py
#
# Exercises every guard using a harmless throwaway process, so nothing on the
# machine is affected.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manager = ProcessManager()
    print(f"action log: {manager.log_path}\n")

    print("--- guards (nothing is modified) ---")
    for pid, name in ((4, "System"), (0, "System Idle Process"),
                      (1234, "csrss.exe"), (5678, "svchost.exe"),
                      (os.getpid(), "python.exe")):
        print(f"  {name:<22} protected={is_protected(pid, name)!s:<5} "
              f"{protection_reason(pid, name)}")

    print("\n--- refusals ---")
    print(" ", manager.suspend_process(4, "System"))
    print(" ", manager.lower_priority(os.getpid(), "python.exe"))
    print(" ", manager.suspend_process(999_999, "ghost.exe"))

    # A real, harmless target: a sleeping child process we own.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(1.0)
    print(f"\n--- real actions on a throwaway child (PID {child.pid}) ---")
    print(" ", manager.lower_priority(child.pid, "python.exe"))
    print(" ", manager.suspend_process(child.pid, "python.exe"))
    print("    status now:", psutil.Process(child.pid).status())
    print(" ", manager.resume_process(child.pid, "python.exe"))
    print("    status now:", psutil.Process(child.pid).status())
    print(" ", manager.suspend_process(child.pid, "wrong_name.exe"))
    child.terminate()

    print("\n--- recent actions ---")
    for entry in manager.recent_actions(8):
        print(f"  {entry['timestamp']}  {entry['result']:<8} {entry['action']:<15} "
              f"{entry['name']} ({entry['pid']})")
