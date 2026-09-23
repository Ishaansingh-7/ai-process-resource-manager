"""
collector.py
============
Background data collection.

Every ``interval`` seconds the collector takes a snapshot of all running
processes and appends one CSV row per process to ``data/process_data.csv``.
That file is the training set for the machine-learning stage of the project,
so the collector is deliberately boring and robust: if a single sample fails
(a process vanished, the CSV is momentarily locked by Excel, ...) it logs the
problem and keeps going rather than killing the thread.

CSV layout (see monitor.CSV_COLUMNS):
    timestamp, pid, name, cpu_percent, memory_mb, thread_count, priority, status
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from monitor import CSV_COLUMNS, ProcessMonitor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Paths are resolved relative to this file, so the program behaves the same no
# matter which folder you launch it from.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "process_data.csv"
DEFAULT_INTERVAL = 3.0  # seconds between snapshots
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"  # sorts correctly and pandas parses it


class DataCollector(threading.Thread):
    """
    A daemon thread that appends process snapshots to a CSV file.

    Usage::

        collector = DataCollector()
        collector.start()
        ...
        collector.stop()
    """

    def __init__(
        self,
        csv_path: Path = DEFAULT_CSV_PATH,
        interval: float = DEFAULT_INTERVAL,
        normalize_cpu: bool = True,
    ) -> None:
        # daemon=True means this thread will not stop Python from exiting if
        # the GUI is closed abruptly.
        super().__init__(name="DataCollector", daemon=True)

        self.csv_path = Path(csv_path)
        self.interval = float(interval)

        # Its own monitor instance, so its CPU baseline is independent of the
        # GUI's (see the note at the top of monitor.py).
        self._monitor = ProcessMonitor(normalize_cpu=normalize_cpu)
        self._stop_event = threading.Event()

        # Simple statistics the GUI shows in its status bar.
        self.rows_written = 0
        self.samples_written = 0
        self.last_error: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        """Thread entry point -- do not call directly, use start()."""
        self._prepare_csv()
        log.info("Collector started: every %.1fs -> %s", self.interval, self.csv_path)

        # The first snapshot has no previous CPU-time baseline, so every
        # cpu_percent would be 0.0.  Take it, throw it away, and only start
        # recording from the second one: no useless all-zero rows in the
        # training data.
        self._monitor.list_processes()
        self._stop_event.wait(self.interval)

        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._write_snapshot()
            except Exception as exc:  # never let one bad sample kill the thread
                self.last_error = str(exc)
                log.warning("Snapshot failed: %s", exc)

            # Subtract the time the snapshot took so the sampling rate does
            # not slowly drift.
            delay = max(0.0, self.interval - (time.monotonic() - started))
            self._stop_event.wait(delay)

        log.info("Collector stopped after %d samples (%d rows).",
                 self.samples_written, self.rows_written)

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the thread to finish the current sleep and exit."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    # -- internals ---------------------------------------------------------

    def _prepare_csv(self) -> None:
        """Create data/, write the header if the file is new, migrate if it is old."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            self._migrate_csv()
            return  # appending to an existing dataset
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(CSV_COLUMNS)
        log.info("Created %s", self.csv_path)

    def _migrate_csv(self) -> None:
        """
        Bring an older dataset up to the current column set.

        Phases add columns (the final phase added the three context-switch
        ones).  Appending 11-field rows under an 8-field header would silently
        shift every value into the wrong column and quietly poison both the
        training data and the anomaly baselines, so the existing file is
        rewritten with the new columns left empty for the old rows.  Empty is
        the honest value: those samples were taken before the program measured
        context switches.  Readers already cope - model.load_dataset() and
        anomaly._load_baselines() both coerce and drop unparseable cells.
        """
        try:
            # Read the header in its own block: Windows will not rename a file
            # that is still open, so nothing below may run with a live handle.
            with open(self.csv_path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), None)

            if header is None or header == CSV_COLUMNS:
                return                          # empty, or already current

            if header != CSV_COLUMNS[:len(header)]:
                # Not one of our older layouts - do not guess what the columns
                # mean.  Keep the file and start a fresh one beside it.
                backup = self.csv_path.with_suffix(".unrecognised.csv")
                self.csv_path.replace(backup)
                with open(self.csv_path, "w", newline="", encoding="utf-8") as out:
                    csv.writer(out).writerow(CSV_COLUMNS)
                log.warning("%s had an unrecognised header; moved it to %s "
                            "and started a new dataset.",
                            self.csv_path.name, backup.name)
                return

            padding = [""] * (len(CSV_COLUMNS) - len(header))
            temporary = self.csv_path.with_suffix(".migrating.csv")
            rows = 0
            with open(self.csv_path, newline="", encoding="utf-8") as fh, \
                    open(temporary, "w", newline="", encoding="utf-8") as out:
                reader = csv.reader(fh)
                next(reader, None)              # drop the old header
                writer = csv.writer(out)
                writer.writerow(CSV_COLUMNS)
                for row in reader:
                    writer.writerow(row + padding)
                    rows += 1
            # Swap in only once the new file is complete, so an interrupted
            # migration cannot leave a half-written dataset behind.
            temporary.replace(self.csv_path)
            log.info("Migrated %s to the current %d-column layout (%d rows, "
                     "%d new column(s) left blank).", self.csv_path.name,
                     len(CSV_COLUMNS), rows, len(padding))
        except OSError as exc:
            log.warning("Could not migrate %s: %s", self.csv_path, exc)

    def _write_snapshot(self) -> None:
        """Take one snapshot and append a row per process."""
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        processes = self._monitor.list_processes(sort_by_cpu=False)

        # Opening the file per sample (instead of holding it open) keeps the
        # data safe if the program is killed, and lets you inspect the CSV
        # while the program is still running.
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(p.to_row(timestamp) for p in processes)

        self.rows_written += len(processes)
        self.samples_written += 1
        self.last_error = None
        log.debug("Wrote %d rows at %s", len(processes), timestamp)


# ---------------------------------------------------------------------------
# Manual test:  python collector.py    (Ctrl+C to stop)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    collector = DataCollector()
    collector.start()
    try:
        while True:
            time.sleep(1)
            print(f"\rsamples: {collector.samples_written}  "
                  f"rows: {collector.rows_written}", end="", flush=True)
    except KeyboardInterrupt:
        print()
        collector.stop()
