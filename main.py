"""
main.py
=======
Entry point for the AI-Based Process Resource Manager.

It wires the three pieces together:

    monitor.py   -> reads processes from the OS
    collector.py -> background thread, appends snapshots to data/process_data.csv
    gui.py       -> Tkinter dashboard, refreshes on its own timer

Run it with::

    python main.py                 # dashboard + background CSV collection
    python main.py --collect-only  # no window, just build the dataset
    python main.py --help          # all options
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Start-up checks
# ---------------------------------------------------------------------------


def require_dependencies() -> None:
    """Fail with a helpful message instead of a raw traceback."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        sys.exit("psutil is not installed.  Run:  pip install -r requirements.txt")
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.exit("tkinter is missing from this Python installation.\n"
                 "On Windows, re-run the official Python installer and enable "
                 "'tcl/tk and IDLE'.")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(default_csv, default_interval: float, default_refresh: float):
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AI-Based Process Resource Manager - live Windows process "
                    "monitor with background data collection.",
    )
    parser.add_argument("--interval", type=float, default=default_interval,
                        metavar="SEC",
                        help=f"seconds between CSV snapshots (default: {default_interval})")
    parser.add_argument("--refresh", type=float, default=default_refresh,
                        metavar="SEC",
                        help=f"seconds between dashboard refreshes (default: {default_refresh})")
    parser.add_argument("--csv", default=str(default_csv), metavar="PATH",
                        help=f"where to write the dataset (default: {default_csv})")
    parser.add_argument("--no-collector", action="store_true",
                        help="show the dashboard without writing to the CSV")
    parser.add_argument("--collect-only", action="store_true",
                        help="collect data in the terminal, without the GUI")
    parser.add_argument("--raw-cpu", action="store_true",
                        help="report CPU %% per core (100%% = one full core) "
                             "instead of per machine")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print debug logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    require_dependencies()

    # Imported after the dependency check so a missing package produces the
    # friendly message above rather than an ImportError traceback.
    import psutil
    from collector import DEFAULT_CSV_PATH, DEFAULT_INTERVAL, DataCollector
    from gui import DEFAULT_REFRESH_SECONDS, Dashboard

    args = parse_args(DEFAULT_CSV_PATH, DEFAULT_INTERVAL, DEFAULT_REFRESH_SECONDS)
    configure_logging(args.verbose)

    if not psutil.WINDOWS:
        log.warning("Not running on Windows - process priority and status "
                    "labels will look different.")

    normalize_cpu = not args.raw_cpu
    collector = None

    # --- background data collection ---------------------------------------
    if not args.no_collector:
        collector = DataCollector(csv_path=args.csv, interval=args.interval,
                                  normalize_cpu=normalize_cpu)
        collector.start()

    try:
        if args.collect_only:
            # Headless mode: useful for leaving the machine running to build
            # up a training set before working on the ML model.
            if collector is None:
                sys.exit("--collect-only cannot be combined with --no-collector")
            print(f"Collecting every {args.interval:g}s into {collector.csv_path}"
                  "\nPress Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
                    print(f"\r  samples: {collector.samples_written:<6} "
                          f"rows: {collector.rows_written:<10}", end="", flush=True)
            except KeyboardInterrupt:
                print()
        else:
            # mainloop() blocks here until the user closes the window.
            log.info("Launching dashboard (refresh every %.1fs)...", args.refresh)
            Dashboard(refresh_seconds=args.refresh, collector=collector,
                      normalize_cpu=normalize_cpu).mainloop()
    finally:
        # Runs on a normal exit, on Ctrl+C, and if the GUI crashes, so the CSV
        # is always closed cleanly.
        if collector is not None:
            collector.stop()
            log.info("Dataset: %d rows in %s", collector.rows_written,
                     collector.csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
