"""
gui.py
======
The Tkinter dashboard.

Layout
------
    +----------------------------------------------------------------+
    |  header : title + summary cards (CPU, RAM used, RAM free, #)    |
    +----------------------------------------------------------------+
    |  toolbar: search box, pause, refresh-now, colour legend         |
    +-----------------------------------------------+----------------+
    |  table  : one row per process, AI class,       |  ALERTS        |
    |           warning icon, recommended action     +----------------+
    |                                                |  RECENT        |
    |                                                |  ACTIONS       |
    +-----------------------------------------------+----------------+
    |  details: the selected process + [Apply] [Ignore]               |
    +----------------------------------------------------------------+
    |  status : refresh time, totals, model / detector state          |
    +----------------------------------------------------------------+

Why there is a background thread
--------------------------------
Tkinter is single threaded: whatever runs inside a widget callback freezes the
window until it returns.  One refresh reads every process, classifies them,
checks them for anomalies, fits trend lines and works out recommendations, so
a worker thread does all of it and drops the result into a queue.  The UI
polls that queue with ``after()`` and only ever touches widgets from the main
thread.

How the pipeline fits together
------------------------------
    monitor.py          -> what every process is doing right now   (Phase 1)
    predictor.py        -> what kind of process it is              (Phase 2)
    anomaly.py          -> is it behaving unlike itself?           (Phase 3)
    predictor_trend.py  -> where is it heading next?               (Phase 3)
    recommender.py      -> should a human do something?            (Phase 4)
    process_manager.py  -> does it, but only when told to          (Phase 4)

All of them return dictionaries keyed by PID, so sorting or filtering the
table can never pair a process with somebody else's verdict.

The one rule about actions
--------------------------
The worker thread computes *recommendations* and nothing else.  An OS action
only ever happens on the main thread, from a button the user pressed, after a
confirmation dialog they answered.  There is deliberately no code path that
lets a refresh change anything on the machine.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, NamedTuple, Optional, Sequence

from monitor import ProcessInfo, ProcessMonitor, Snapshot

# Phases 2-4 are optional as far as the dashboard is concerned: Phase 1 works
# on its own, so every import is guarded and every use is behind a check.
try:
    import predictor
    from predictor import Prediction
except Exception as exc:  # pragma: no cover - depends on the environment
    predictor = None
    Prediction = None
    _PREDICTOR_IMPORT_ERROR = str(exc)
else:
    _PREDICTOR_IMPORT_ERROR = None

try:
    import anomaly
    import predictor_trend
    from anomaly import AnomalyResult
    from history import ProcessHistory, SystemHistory
    from predictor_trend import Forecast
except Exception as exc:  # pragma: no cover
    anomaly = None
    predictor_trend = None
    AnomalyResult = None
    ProcessHistory = None
    SystemHistory = None
    Forecast = None
    _ANALYSIS_IMPORT_ERROR = str(exc)
else:
    _ANALYSIS_IMPORT_ERROR = None

try:
    import charts
    from charts import ProcessDetailWindow, SystemChartsPanel
except Exception as exc:  # pragma: no cover
    charts = None
    ProcessDetailWindow = None
    SystemChartsPanel = None
    _CHARTS_IMPORT_ERROR = str(exc)
else:
    _CHARTS_IMPORT_ERROR = None

try:
    # The two OS simulators.  They are self-contained windows: the dashboard
    # only ever opens them, so a problem in either cannot affect monitoring.
    from simulator_ui import DeadlockWindow, SchedulerWindow
except Exception as exc:  # pragma: no cover
    DeadlockWindow = None
    SchedulerWindow = None
    _SIMULATOR_IMPORT_ERROR = str(exc)
else:
    _SIMULATOR_IMPORT_ERROR = None

try:
    import recommender
    from process_manager import (ACTION_LOWER_PRIORITY, ACTION_RESUME,
                                 ACTION_SUSPEND, ProcessManager)
    from recommender import ACTION_MONITOR, ACTION_NONE, Recommendation
except Exception as exc:  # pragma: no cover
    recommender = None
    ProcessManager = None
    Recommendation = None
    ACTION_NONE = "No action needed"
    ACTION_MONITOR = "Monitor"
    ACTION_LOWER_PRIORITY = "Lower Priority"
    ACTION_SUSPEND = "Suspend"
    ACTION_RESUME = "Resume"
    _ACTIONS_IMPORT_ERROR = str(exc)
else:
    _ACTIONS_IMPORT_ERROR = None

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Appearance / behaviour constants
# ---------------------------------------------------------------------------

DEFAULT_REFRESH_SECONDS = 2.5   # how often the worker takes a snapshot
UI_POLL_MS = 150                # how often the UI checks the queue
MAX_ALERTS_LISTED = 12          # keep the alerts panel readable
MAX_ACTIONS_LISTED = 6          # entries in the Recent Actions panel
CHART_MINUTES = 5.0             # span of the rolling system chart

DARK = "#111827"       # header background
DARK_CARD = "#1f2937"  # summary card background
BG = "#f4f5f7"         # window background
PANEL = "#ffffff"      # table / panel background
TEXT = "#111827"
MUTED = "#9aa4b2"
ACCENT = "#2563eb"
BORDER = "#dfe3e8"
ALERT_RED = "#b3261e"
ALERT_AMBER = "#a45c00"
OK_GREEN = "#166534"

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL_BOLD = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_CARD_VALUE = ("Segoe UI", 17, "bold")
FONT_CARD_LABEL = ("Segoe UI", 8)

# Row colours driven by the model's prediction (Phase 2).  Two background
# shades per class give alternating stripes without losing the colour coding;
# the tints are pale so black text stays readable.
UNKNOWN_LABEL = "UNKNOWN"
CLASS_STYLES = {
    "NORMAL":           {"shades": ("#f4fbf6", "#e9f6ed"), "fg": "#14532d",
                         "legend": "#5cb87a", "text": "Normal"},
    "CPU_INTENSIVE":    {"shades": ("#fff6d5", "#fff1c2"), "fg": "#6b4e00",
                         "legend": "#f2c53d", "text": "CPU intensive"},
    "MEMORY_INTENSIVE": {"shades": ("#ffe8d1", "#ffdcbd"), "fg": "#7c3a0a",
                         "legend": "#f08c34", "text": "Memory intensive"},
    UNKNOWN_LABEL:      {"shades": (PANEL, "#f7f8fa"), "fg": TEXT,
                         "legend": "#c4c9d0", "text": "No model"},
}

# Phase 3 indicators.  Two glyphs rather than two colours, because a Treeview
# can only colour a whole row and the row colour already carries the Phase 2
# classification.
ICON_ANOMALY = "⚠"     # warning triangle
ICON_TREND = "↗"       # up-and-right arrow
ICON_NONE = ""

# Phase 4: how urgent each recommendation is, for sorting the table.
ACTION_RANK = {
    ACTION_SUSPEND: 3,
    ACTION_LOWER_PRIORITY: 2,
    ACTION_MONITOR: 1,
    ACTION_NONE: 0,
}

# Table columns:
#   (column id, heading, width, anchor, ProcessInfo attribute, is numeric)
# "alert", "ai" and "action" have no ProcessInfo attribute - they come from
# the per-PID dictionaries and are handled separately when sorting.
COLUMNS = (
    ("alert",    ICON_ANOMALY,         36, "center", None,           True),
    ("name",     "Process",           175, "w",      "name",         False),
    ("pid",      "PID",                68, "e",      "pid",          True),
    ("cpu",      "CPU %",              72, "e",      "cpu_percent",  True),
    ("memory",   "Memory (MB)",       100, "e",      "memory_mb",    True),
    ("threads",  "Threads",            68, "e",      "thread_count", True),
    ("ctxsw",    "Ctx Sw/s",           82, "e",      "ctx_switches_per_sec", True),
    ("priority", "Priority",           92, "w",      "priority",     False),
    ("status",   "Status",             82, "w",      "status",       False),
    # Wide enough for the longest label plus its confidence,
    # e.g. "MEMORY_INTENSIVE  (96%)".
    ("ai",       "AI Classification", 182, "w",      None,           False),
    ("action",   "Recommended",       132, "w",      None,           True),
)
_COLUMN_BY_ID = {c[0]: c for c in COLUMNS}
_TEXT_ATTRS = ("name", "priority", "status")

# Columns the user may hide from the "Columns" menu.  Everything here is
# either also shown in the details panel below the table or is secondary to
# reading the machine at a glance, so hiding one loses nothing.  "alert",
# "name", "pid", "cpu" and "memory" are deliberately not in this list.
OPTIONAL_COLUMNS = ("threads", "ctxsw", "priority", "status", "ai", "action")


def _ctx_switch_line(process: ProcessInfo) -> str:
    """
    One line describing a process's context-switch behaviour.

    The table shows only the rate, because that is the number that describes
    what is happening now.  This is where the raw counters and the
    voluntary / involuntary split belong - and where we say plainly that
    Windows does not provide the split rather than printing a fake zero.
    See the long note at the top of monitor.py.
    """
    rate = f"{process.ctx_switches_per_sec:,.0f}/s"
    if process.ctx_switches_invol is None:
        return (f"{rate}   (total {process.ctx_switches_vol:,} since it started; "
                f"Windows does not separate voluntary from involuntary)")
    return (f"{rate}   (voluntary {process.ctx_switches_vol:,}, "
            f"involuntary {process.ctx_switches_invol:,} since it started)")


class SampleResult(NamedTuple):
    """Everything one refresh produces, passed from the worker to the UI."""

    snapshot: Snapshot
    predictions: Dict[int, "Prediction"]
    anomalies: Dict[int, "AnomalyResult"]
    forecasts: Dict[int, "Forecast"]
    recommendations: Dict[int, "Recommendation"]
    # Per-process sample history, but only for the PIDs that have a detail
    # window open.  The rolling window itself belongs to the worker thread;
    # copying it for every process on every refresh would be wasteful, and
    # reading it from the UI thread while the worker appends would be a race.
    # So the UI publishes which PIDs it cares about and the worker answers.
    watched_history: Dict[int, list]


class Dashboard(tk.Tk):
    """The main window."""

    def __init__(
        self,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        collector=None,
        normalize_cpu: bool = True,
        enable_ai: bool = True,
        enable_analysis: bool = True,
        enable_actions: bool = True,
        enable_charts: bool = True,
    ) -> None:
        super().__init__()

        self.title("AI-Based Process Resource Manager")
        self.geometry("1480x950")
        self.minsize(1200, 700)
        self.configure(bg=BG)

        self.refresh_seconds = float(refresh_seconds)
        self.collector = collector          # optional DataCollector, for status
        self.enable_ai = enable_ai and predictor is not None
        self.enable_analysis = enable_analysis and anomaly is not None
        self.enable_actions = enable_actions and recommender is not None
        self.enable_charts = enable_charts and charts is not None
        self._monitor = ProcessMonitor(normalize_cpu=normalize_cpu)

        # Machine-wide CPU/RAM over time, for the rolling chart.  Kept in
        # memory rather than re-read from the 1.6 MB CSV every 2.5 seconds -
        # parsing that file would cost more than everything else in a refresh
        # put together.  Appended on the main thread as each result arrives.
        self._system_history = SystemHistory() if SystemHistory else None
        # The first snapshot carries no CPU delta to report (see
        # ProcessMonitor.system_stats), so it is counted but not plotted.
        self._first_snapshot = True
        # Detail windows currently open, keyed by PID, so each one can be fed
        # fresh data on every refresh and forgotten when it is closed.
        self._detail_windows: Dict[int, "ProcessDetailWindow"] = {}
        # The two simulator windows, at most one of each.  They hold no live
        # data and are never touched by a refresh.
        self._scheduler_window = None
        self._deadlock_window = None

        # The only object in the program that can change anything.  It is
        # created here but never called from the worker thread - see the
        # module docstring.
        self._manager = ProcessManager() if self.enable_actions else None

        # --- shared state between the worker thread and the UI -------------
        self._queue: "queue.Queue[SampleResult]" = queue.Queue()
        self._stop_event = threading.Event()
        self._latest: Optional[SampleResult] = None

        # Written by the worker thread, read by the UI.  Plain string
        # assignment is atomic in CPython, so no lock is needed.
        self._ai_status = "starting..." if self.enable_ai else (
            _PREDICTOR_IMPORT_ERROR or "disabled")
        self._analysis_status = "starting..." if self.enable_analysis else (
            _ANALYSIS_IMPORT_ERROR or "disabled")

        # --- view state ----------------------------------------------------
        self._sort_column = "cpu"
        self._sort_reverse = True     # start with the busiest process on top
        self._paused = False
        self._dirty = False           # force a repaint (filter/sort changed)
        # The SampleResult the charts were last drawn from.  Repainting the
        # table is cheap (~5 ms) and happens on every filter keystroke; a
        # chart redraw costs five times that, and the data has not changed,
        # so charts are skipped unless the snapshot itself is new.
        self._charted: Optional[SampleResult] = None
        self._poll_job = None         # pending after() id, cancelled on close
        # PIDs the user pressed Ignore on: their advice is hidden for the rest
        # of the session, but they are still monitored and still listed.
        self._ignored: set = set()
        self._pending: Optional[tuple] = None   # (pid, name, action) for Apply
        # PIDs with a detail window open; read by the worker thread.
        self._watched_pids: set = set()
        # How many system samples the chart needs to fill its time window.
        self._chart_samples = int(CHART_MINUTES * 60 / self.refresh_seconds) + 2

        self._build_styles()
        self._build_header()
        self._build_toolbar()
        self._build_body()
        self._build_charts()
        self._build_details()
        self._build_statusbar()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._refresh_action_log()
        self._start_sampler()
        self.after(UI_POLL_MS, self._poll_queue)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        # "clam" is the one built-in theme that lets us restyle the Treeview
        # headings on every platform.
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Treeview",
                        background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=24, font=FONT, borderwidth=0)
        style.configure("Treeview.Heading",
                        background="#e8eaed", foreground=TEXT,
                        font=FONT_BOLD, relief="flat", padding=(6, 6))
        style.map("Treeview.Heading", background=[("active", "#dcdfe4")])
        style.map("Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])
        # The destructive action gets a visually distinct button.
        style.configure("Danger.TButton", foreground=ALERT_RED, font=FONT_BOLD)

        self.grid_rowconfigure(2, weight=1)   # the table takes the free space
        self.grid_columnconfigure(0, weight=1)

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=DARK, padx=16, pady=12)
        header.grid(row=0, column=0, sticky="ew")

        tk.Label(header, text="AI-Based Process Resource Manager",
                 bg=DARK, fg="#ffffff", font=FONT_TITLE).pack(anchor="w")
        tk.Label(header, text="Live process monitor  ·  random-forest classification"
                             "  ·  3-sigma anomaly detection  ·  trend forecasting"
                             "  ·  recommended actions",
                 bg=DARK, fg=MUTED, font=FONT).pack(anchor="w", pady=(2, 10))

        cards = tk.Frame(header, bg=DARK)
        cards.pack(fill="x")
        for i in range(4):
            cards.grid_columnconfigure(i, weight=1, uniform="card")

        self.card_cpu = self._stat_card(cards, 0, "TOTAL CPU")
        self.card_ram = self._stat_card(cards, 1, "RAM IN USE")
        self.card_free = self._stat_card(cards, 2, "RAM AVAILABLE")
        self.card_procs = self._stat_card(cards, 3, "PROCESSES")

    def _stat_card(self, parent: tk.Frame, column: int, label: str) -> tk.Label:
        card = tk.Frame(parent, bg=DARK_CARD, padx=14, pady=10)
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
        tk.Label(card, text=label, bg=DARK_CARD, fg=MUTED,
                 font=FONT_CARD_LABEL).pack(anchor="w")
        value = tk.Label(card, text="--", bg=DARK_CARD, fg="#ffffff",
                         font=FONT_CARD_VALUE)
        value.pack(anchor="w")
        return value

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=BG, padx=16, pady=10)
        bar.grid(row=1, column=0, sticky="ew")

        tk.Label(bar, text="Filter:", bg=BG, fg=TEXT, font=FONT).pack(side="left")

        self.filter_var = tk.StringVar()
        # trace_add fires on every keystroke -> the table filters as you type.
        self.filter_var.trace_add("write", lambda *_: self._request_render())
        ttk.Entry(bar, textvariable=self.filter_var, width=24,
                  font=FONT).pack(side="left", padx=(6, 16))

        self.pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Pause updates", variable=self.pause_var,
                        command=self._toggle_pause).pack(side="left")

        ttk.Button(bar, text="Refresh now",
                   command=self._request_render).pack(side="left", padx=(16, 0))
        self._build_column_menu(bar)
        self._build_simulator_buttons(bar)

        # Colour legend, so the row colours explain themselves.
        legend = tk.Frame(bar, bg=BG)
        legend.pack(side="right")
        for label in ("NORMAL", "CPU_INTENSIVE", "MEMORY_INTENSIVE"):
            spec = CLASS_STYLES[label]
            swatch = tk.Frame(legend, bg=BG)
            swatch.pack(side="left", padx=(14, 0))
            tk.Frame(swatch, bg=spec["legend"], width=12, height=12,
                     highlightthickness=1,
                     highlightbackground="#b9bfc7").pack(side="left")
            tk.Label(swatch, text=spec["text"], bg=BG, fg="#4b5563",
                     font=FONT_SMALL).pack(side="left", padx=(5, 0))

    def _build_column_menu(self, bar: tk.Frame) -> None:
        """
        A checklist of the columns that may be hidden.

        Eleven columns is more than the table can show comfortably on a
        smaller window, and which ones matter depends on what you are looking
        at, so the choice is the user's rather than ours.  Treeview does the
        work: "displaycolumns" changes what is drawn without touching the
        data, so hiding a column costs nothing and sorting still works on it.
        """
        self._column_vars: Dict[str, tk.BooleanVar] = {}
        menubutton = ttk.Menubutton(bar, text="Columns ▾")
        menu = tk.Menu(menubutton, tearoff=False)
        for col_id in OPTIONAL_COLUMNS:
            heading = _COLUMN_BY_ID[col_id][1]
            var = tk.BooleanVar(value=True)
            self._column_vars[col_id] = var
            menu.add_checkbutton(label=heading, variable=var,
                                 command=self._apply_column_visibility)
        menubutton.configure(menu=menu)
        menubutton.pack(side="left", padx=(10, 0))

    def _apply_column_visibility(self) -> None:
        visible = [c[0] for c in COLUMNS
                   if c[0] not in self._column_vars
                   or self._column_vars[c[0]].get()]
        # Treeview rejects an empty display list; the non-optional columns
        # above mean this can never actually be empty, but be explicit.
        self.tree.configure(displaycolumns=visible or [c[0] for c in COLUMNS])

    def _build_simulator_buttons(self, bar: tk.Frame) -> None:
        """Entry points for the two OS simulators (scheduler.py, deadlock.py)."""
        if SchedulerWindow is not None:
            ttk.Button(bar, text="CPU Scheduling Simulator",
                       command=self._open_scheduler).pack(side="left", padx=(10, 0))
        if DeadlockWindow is not None:
            ttk.Button(bar, text="Deadlock Detection",
                       command=self._open_deadlock).pack(side="left", padx=(8, 0))

    def _open_scheduler(self) -> None:
        """Open (or raise) the scheduling simulator."""
        if self._scheduler_window is not None and self._scheduler_window.winfo_exists():
            self._scheduler_window.lift()
            self._scheduler_window.focus_set()
            return
        self._scheduler_window = SchedulerWindow(self)

    def _open_deadlock(self) -> None:
        """Open (or raise) the deadlock detection window."""
        if self._deadlock_window is not None and self._deadlock_window.winfo_exists():
            self._deadlock_window.lift()
            self._deadlock_window.focus_set()
            return
        self._deadlock_window = DeadlockWindow(self)

    def _build_body(self) -> None:
        """The table on the left, alerts and recent actions stacked right."""
        body = tk.Frame(self, bg=BG, padx=16)
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)     # table stretches
        body.grid_columnconfigure(1, minsize=330)  # side panels are fixed

        self._build_table(body)

        side = tk.Frame(body, bg=BG)
        side.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(0, weight=3)        # alerts get the space
        side.grid_rowconfigure(1, weight=1)
        self._build_alerts(side)
        self._build_actions_panel(side)

    def _build_table(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=BG)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            frame,
            columns=[c[0] for c in COLUMNS],
            show="headings",          # hide the empty first "tree" column
            selectmode="browse",
        )
        for col_id, heading, width, anchor, _attr, _numeric in COLUMNS:
            # Clicking a heading re-sorts the table (see _sort_by).
            # The heading uses the same alignment as its data so the two line up.
            self.tree.heading(col_id, text=heading, anchor=anchor,
                              command=lambda c=col_id: self._sort_by(c))
            self.tree.column(col_id, width=width, anchor=anchor,
                             stretch=(col_id == "name"))
        self._update_headings()

        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        # Selecting a row fills in the details panel underneath;
        # double-clicking opens that process's own charts in a new window.
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double_click)

        # One tag per (class, stripe) pair: the prediction picks the colour,
        # the row index picks the shade.
        for label, spec in CLASS_STYLES.items():
            for stripe, shade in enumerate(spec["shades"]):
                self.tree.tag_configure(f"{label}-{stripe}",
                                        background=shade, foreground=spec["fg"])

    def _panel(self, parent: tk.Frame, row: int, title: str, subtitle: str = ""):
        """A titled white panel containing a read-only Text widget."""
        panel = tk.Frame(parent, bg=PANEL, highlightthickness=1,
                         highlightbackground=BORDER)
        panel.grid(row=row, column=0, sticky="nsew", pady=(0 if row == 0 else 12, 0))
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        head = tk.Frame(panel, bg=PANEL, padx=10, pady=8)
        head.grid(row=0, column=0, sticky="ew")
        label = tk.Label(head, text=title, bg=PANEL, fg=TEXT, font=FONT_SMALL_BOLD)
        label.pack(side="left")
        if subtitle:
            tk.Label(head, text=subtitle, bg=PANEL, fg=MUTED,
                     font=FONT_SMALL).pack(side="right")

        text = tk.Text(panel, wrap="word", relief="flat", bg=PANEL,
                       font=FONT_SMALL, padx=10, pady=2, width=34,
                       highlightthickness=0, cursor="arrow", spacing3=2)
        text.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(panel, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.grid(row=1, column=1, sticky="ns")

        text.tag_configure("anomaly", foreground=ALERT_RED, font=FONT_SMALL_BOLD)
        text.tag_configure("trend", foreground=ALERT_AMBER, font=FONT_SMALL_BOLD)
        text.tag_configure("ok", foreground=OK_GREEN, font=FONT_SMALL_BOLD)
        text.tag_configure("detail", foreground="#4b5563", lmargin1=14, lmargin2=14)
        text.tag_configure("advice", foreground=ACCENT, lmargin1=14, lmargin2=14)
        text.tag_configure("quiet", foreground=MUTED)
        text.configure(state="disabled")
        return label, text

    def _build_alerts(self, parent: tk.Frame) -> None:
        self.alerts_title, self.alerts_text = self._panel(
            parent, 0, "ALERTS", f"{ICON_ANOMALY} anomaly    {ICON_TREND} trend")

    def _build_actions_panel(self, parent: tk.Frame) -> None:
        """The audit trail, read back from data/action_log.csv."""
        self.actions_title, self.actions_text = self._panel(
            parent, 1, "RECENT ACTIONS")

    def _build_charts(self) -> None:
        """The rolling system chart and top-five bars, under the table."""
        if not self.enable_charts:
            self.charts_panel = None
            return
        holder = tk.Frame(self, bg=BG, padx=16, pady=0)
        holder.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        holder.grid_columnconfigure(0, weight=1)

        self.charts_panel = SystemChartsPanel(holder, minutes=CHART_MINUTES,
                                              highlightthickness=1,
                                              highlightbackground=BORDER)
        self.charts_panel.grid(row=0, column=0, sticky="ew")

    def _build_details(self) -> None:
        """A panel describing the selected row, with the action buttons."""
        panel = tk.Frame(self, bg=PANEL, padx=14, pady=10,
                         highlightthickness=1, highlightbackground=BORDER)
        panel.grid(row=4, column=0, sticky="ew", padx=16, pady=(12, 0))
        panel.grid_columnconfigure(0, weight=1)

        left = tk.Frame(panel, bg=PANEL)
        left.grid(row=0, column=0, sticky="ew")

        self.detail_title = tk.Label(left, text="No process selected",
                                     bg=PANEL, fg=TEXT, font=FONT_BOLD, anchor="w")
        self.detail_title.pack(fill="x")

        # A fixed height keeps the window from jumping as the text changes.
        self.detail_body = tk.Label(left, text="Click a row to see its "
                                               "classification, anomaly check, "
                                               "trend forecast and recommendation.",
                                    bg=PANEL, fg="#4b5563", font=FONT_SMALL,
                                    anchor="nw", justify="left", height=5)
        self.detail_body.pack(fill="x")

        # Buttons live on the right, away from the text, and start disabled:
        # nothing is clickable until a row with an applicable action is chosen.
        buttons = tk.Frame(panel, bg=PANEL)
        buttons.grid(row=0, column=1, sticky="e", padx=(16, 0))

        self.apply_button = ttk.Button(buttons, text="Apply", state="disabled",
                                       width=20, command=self._on_apply)
        self.apply_button.pack(pady=(2, 6))
        self.ignore_button = ttk.Button(buttons, text="Ignore", state="disabled",
                                        width=20, command=self._on_ignore)
        self.ignore_button.pack()

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bg=BG, padx=16, pady=8)
        bar.grid(row=5, column=0, sticky="ew")
        self.status_label = tk.Label(bar, text="Starting up...", bg=BG,
                                     fg="#6b7280", font=FONT_SMALL, anchor="w")
        self.status_label.pack(fill="x")

    # ------------------------------------------------------------------
    # Sampling (worker thread) - computes advice, never acts
    # ------------------------------------------------------------------

    def _start_sampler(self) -> None:
        self._sampler = threading.Thread(target=self._sampler_loop,
                                         name="GuiSampler", daemon=True)
        self._sampler.start()

    def _sampler_loop(self) -> None:
        """Runs off the main thread: snapshot -> analyse -> queue -> sleep."""
        classifier = None
        detector = None
        trend = None
        history = ProcessHistory() if self.enable_analysis else None
        first = True

        while not self._stop_event.is_set():
            # These are created here, on the worker thread, because the first
            # call may train a model or read a 20 000-row CSV.  Doing that on
            # the main thread would freeze the window for a few seconds.
            if self.enable_ai and classifier is None:
                self._ai_status = "loading / training model..."
                try:
                    classifier = predictor.ProcessClassifier()
                    self._ai_status = classifier.status
                except Exception as exc:
                    log.warning("Could not start the classifier: %s", exc)
                    self._ai_status = f"unavailable ({exc})"
                    self.enable_ai = False

            if self.enable_analysis and detector is None:
                self._analysis_status = "building baselines..."
                try:
                    detector = anomaly.AnomalyDetector()
                    trend = predictor_trend.TrendPredictor()
                    self._analysis_status = detector.status
                except Exception as exc:
                    log.warning("Could not start anomaly detection: %s", exc)
                    self._analysis_status = f"unavailable ({exc})"
                    self.enable_analysis = False

            try:
                snapshot = self._monitor.snapshot()

                # The rolling window has to be updated before the analysers
                # run, so that this snapshot is part of what they look at.
                if history is not None:
                    history.update(snapshot.processes)

                predictions = (classifier.predict_many(snapshot.processes)
                               if classifier is not None and classifier.available
                               else {})
                anomalies = (detector.evaluate(snapshot.processes, history)
                             if detector is not None else {})
                forecasts = (trend.evaluate(snapshot.processes, history)
                             if trend is not None else {})
                # Pure computation: recommend_all decides what *should* happen
                # and has no way to make it happen.
                recommendations = (
                    recommender.recommend_all(snapshot.processes, predictions,
                                              anomalies, forecasts)
                    if self.enable_actions else {})

                # Copy history only for the processes the UI is watching -
                # normally none, occasionally one open detail window.
                watched = {}
                if history is not None:
                    for pid in tuple(self._watched_pids):
                        watched[pid] = history.samples(pid)

                self._queue.put(SampleResult(snapshot, predictions, anomalies,
                                             forecasts, recommendations, watched))
            except Exception:
                log.exception("Failed to sample processes")

            # The very first snapshot only establishes the CPU baseline (all
            # values are 0 %), so follow it up quickly to get real numbers on
            # screen instead of making the user wait a full interval.
            self._stop_event.wait(1.0 if first else self.refresh_seconds)
            first = False

    # ------------------------------------------------------------------
    # UI updates (main thread)
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        """Drain the queue and repaint if there is something new to show."""
        received = False
        try:
            while True:
                result = self._queue.get_nowait()
                self._latest = result
                received = True
                # Recorded for every sample, even while paused, so the chart's
                # timeline has no gaps in it.
                if self._first_snapshot:
                    self._first_snapshot = False     # its 0 % is a placeholder
                elif self._system_history is not None:
                    self._system_history.add(result.snapshot.system)
        except queue.Empty:
            pass

        if self._latest is not None and (self._dirty or (received and not self._paused)):
            self._render()
            self._dirty = False

        # Keep the id so _on_close can cancel it; otherwise Tk fires the
        # callback after the widgets are destroyed and reports a Tcl error.
        self._poll_job = self.after(UI_POLL_MS, self._poll_queue)

    def _request_render(self) -> None:
        """Repaint on the next poll, even while paused (filter/sort changes)."""
        self._dirty = True

    def _render(self) -> None:
        result = self._latest
        if result is None:
            return

        self._update_cards(result.snapshot)
        rows = self._visible_rows(result)
        self._fill_table(rows, result)
        self._fill_alerts(result)
        self._refresh_details(result)
        # Only the table needs repainting when the user types in the filter or
        # clicks a column heading - the charts show the same numbers either way.
        if result is not self._charted:
            self._update_charts(result)
            self._update_detail_windows(result)
            self._charted = result
        self._update_status(result, len(rows))

    # -- charts ---------------------------------------------------------

    def _update_charts(self, result: SampleResult) -> None:
        """Hand the chart panel new numbers (see charts.py for the cost)."""
        if self.charts_panel is None or self._system_history is None:
            return
        self.charts_panel.update_charts(
            self._system_history.samples(last=self._chart_samples),
            result.snapshot.processes)

    def _on_double_click(self, event) -> None:
        """Open (or raise) a detail window for the double-clicked process."""
        if not self.enable_charts or self._latest is None:
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        pid = int(row)

        existing = self._detail_windows.get(pid)
        if existing is not None:
            existing.lift()
            existing.focus_set()
            return

        process = next((p for p in self._latest.snapshot.processes
                        if p.pid == pid), None)
        if process is None:
            return

        window = ProcessDetailWindow(self, pid, process.name,
                                     minutes=CHART_MINUTES,
                                     on_close=self._close_detail_window)
        self._detail_windows[pid] = window
        # Tell the worker to start including this PID's history.
        self._watched_pids.add(pid)
        # Show whatever we already have instead of waiting for the next sample.
        self._update_detail_windows(self._latest)

    def _close_detail_window(self, pid: int) -> None:
        """Called by the window itself when the user closes it."""
        self._detail_windows.pop(pid, None)
        self._watched_pids.discard(pid)

    def _update_detail_windows(self, result: SampleResult) -> None:
        """Feed every open detail window the newest data for its process."""
        if not self._detail_windows:
            return
        for pid, window in list(self._detail_windows.items()):
            process = next((p for p in result.snapshot.processes
                            if p.pid == pid), None)
            window.update_view(
                process,
                result.watched_history.get(pid, []),
                result.predictions.get(pid),
                result.anomalies.get(pid),
                result.forecasts.get(pid),
                result.recommendations.get(pid),
            )

    def _update_cards(self, snapshot: Snapshot) -> None:
        sysinfo = snapshot.system
        self.card_cpu.config(text=f"{sysinfo.cpu_percent:.1f} %")
        self.card_ram.config(
            text=f"{sysinfo.memory_used_mb / 1024:.1f} GB  "
                 f"({sysinfo.memory_percent:.0f}%)")
        self.card_free.config(text=f"{sysinfo.memory_available_mb / 1024:.1f} GB")
        self.card_procs.config(text=str(sysinfo.process_count))

    # -- table ----------------------------------------------------------

    def _alert_level(self, pid: int, result: SampleResult) -> int:
        """2 = anomaly, 1 = trend warning, 0 = nothing to report."""
        found = result.anomalies.get(pid)
        if found is not None and found.is_anomaly:
            return 2
        forecast = result.forecasts.get(pid)
        if forecast is not None and forecast.has_warning:
            return 1
        return 0

    def _recommendation_for(self, pid: int, result: SampleResult):
        """The advice for a PID, or None if the user pressed Ignore on it."""
        if pid in self._ignored:
            return None
        return result.recommendations.get(pid)

    def _visible_rows(self, result: SampleResult) -> List[ProcessInfo]:
        """Apply the search filter, then the current sort order."""
        processes = result.snapshot.processes
        needle = self.filter_var.get().strip().lower()
        if needle:
            processes = [p for p in processes
                         if needle in p.name.lower() or needle in str(p.pid)]

        if self._sort_column == "alert":
            def key(process: ProcessInfo):
                return self._alert_level(process.pid, result)
        elif self._sort_column == "action":
            def key(process: ProcessInfo):
                found = self._recommendation_for(process.pid, result)
                return (ACTION_RANK.get(found.action, 0), found.confidence) \
                    if found else (0, 0.0)
        elif self._sort_column == "ai":
            # Sort by predicted label, then by how sure the model is.
            def key(process: ProcessInfo):
                prediction = result.predictions.get(process.pid)
                return ((prediction.label, prediction.confidence)
                        if prediction else (UNKNOWN_LABEL, 0.0))
        else:
            attr = _COLUMN_BY_ID[self._sort_column][4]
            if attr in _TEXT_ATTRS:
                def key(process: ProcessInfo):
                    return getattr(process, attr).lower()
            else:
                def key(process: ProcessInfo):
                    return getattr(process, attr)

        return sorted(processes, key=key, reverse=self._sort_reverse)

    def _fill_table(self, rows: Sequence[ProcessInfo], result: SampleResult) -> None:
        """
        Rebuild the table.

        Clearing and re-inserting every row is the simplest correct approach,
        but it would also throw away the user's scroll position and selection
        on every refresh -- so we save both and put them back afterwards.
        """
        tree = self.tree
        scroll_pos = tree.yview()[0]
        selected = set(tree.selection())

        tree.delete(*tree.get_children())
        for index, process in enumerate(rows):
            prediction = result.predictions.get(process.pid)
            if prediction is not None and prediction.label in CLASS_STYLES:
                label = prediction.label
                # The table font is proportional, so padding would not line the
                # percentages up; parentheses keep it reading as one phrase.
                verdict = f"{label}  ({prediction.confidence:.0f}%)"
            else:
                label = UNKNOWN_LABEL
                verdict = "—"

            level = self._alert_level(process.pid, result)
            icon = ICON_ANOMALY if level == 2 else ICON_TREND if level == 1 else ICON_NONE

            found = self._recommendation_for(process.pid, result)
            if process.pid in self._ignored:
                advice = "(ignored)"
            elif found is None or found.action == ACTION_NONE:
                advice = "—"
            else:
                advice = found.action

            tree.insert(
                "", "end",
                iid=str(process.pid),       # PIDs are unique -> usable as row id
                values=(
                    icon,
                    process.name,
                    process.pid,
                    f"{process.cpu_percent:.1f}",
                    f"{process.memory_mb:,.1f}",
                    process.thread_count,
                    f"{process.ctx_switches_per_sec:,.0f}",
                    process.priority,
                    process.status,
                    verdict,
                    advice,
                ),
                # e.g. "CPU_INTENSIVE-0": the class picks the colour, the
                # stripe index picks the light/dark shade of that colour.
                tags=(f"{label}-{index % 2}",),
            )

        still_there = [iid for iid in selected if tree.exists(iid)]
        if still_there:
            tree.selection_set(still_there)
        tree.yview_moveto(scroll_pos)

    # -- alerts panel ---------------------------------------------------

    def _collect_alerts(self, result: SampleResult) -> List[tuple]:
        """
        Everything currently flagged, worst first.

        One entry per process even when both analysers fire, so a badly
        behaved process cannot fill the panel on its own.
        """
        alerts = []
        for process in result.snapshot.processes:
            found = result.anomalies.get(process.pid)
            forecast = result.forecasts.get(process.pid)
            is_anomaly = found is not None and found.is_anomaly
            has_trend = forecast is not None and forecast.has_warning
            if not (is_anomaly or has_trend):
                continue

            reasons = []
            if is_anomaly:
                reasons.append(found.reason)
            if has_trend:
                reasons.append(forecast.message)

            # Anomalies outrank trend warnings; within each, the analyser's
            # own rank decides (see AnomalyResult.rank / Forecast.rank).
            rank = ((1, found.rank) if is_anomaly else (0, forecast.rank))
            advice = self._recommendation_for(process.pid, result)
            alerts.append((rank, process, is_anomaly, reasons, advice))

        alerts.sort(key=lambda entry: entry[0], reverse=True)
        return alerts

    def _fill_alerts(self, result: SampleResult) -> None:
        alerts = self._collect_alerts(result)

        text = self.alerts_text
        text.configure(state="normal")
        text.delete("1.0", "end")

        self.alerts_title.config(
            text=f"ALERTS  ({len(alerts)})" if alerts else "ALERTS")

        if not alerts:
            if not self.enable_analysis:
                text.insert("end", f"Detection unavailable:\n{self._analysis_status}\n",
                            "quiet")
            else:
                text.insert("end", "No anomalies or concerning trends.\n", "quiet")
                text.insert("end", f"\n{self._analysis_status}\n", "quiet")
        else:
            for _rank, process, is_anomaly, reasons, advice in alerts[:MAX_ALERTS_LISTED]:
                icon = ICON_ANOMALY if is_anomaly else ICON_TREND
                tag = "anomaly" if is_anomaly else "trend"
                text.insert("end", f"{icon} {process.name}  ({process.pid})\n", tag)
                for reason in reasons:
                    text.insert("end", f"{reason}\n", "detail")
                # Surface the suggestion here too, so an actionable process is
                # visible without hunting for it in the table.
                if advice is not None and advice.action not in (ACTION_NONE,):
                    text.insert("end", f"Suggested: {advice.action}\n", "advice")
                text.insert("end", "\n")

            hidden = len(alerts) - MAX_ALERTS_LISTED
            if hidden > 0:
                text.insert("end", f"... and {hidden} more\n", "quiet")

        text.configure(state="disabled")

    # -- recent actions panel -------------------------------------------

    def _refresh_action_log(self) -> None:
        """
        Re-read data/action_log.csv into the panel.

        Only called at start-up and after an action, rather than on every
        refresh: the log only changes when the user does something.
        """
        text = self.actions_text
        text.configure(state="normal")
        text.delete("1.0", "end")

        if self._manager is None:
            text.insert("end", f"Actions unavailable:\n"
                               f"{_ACTIONS_IMPORT_ERROR or 'disabled'}\n", "quiet")
        else:
            entries = self._manager.recent_actions(MAX_ACTIONS_LISTED)
            if not entries:
                text.insert("end", "Nothing has been applied yet.\n", "quiet")
                text.insert("end", "Actions you confirm appear here and in "
                                   "data/action_log.csv\n", "quiet")
            else:
                for entry in entries:
                    outcome = entry.get("result", "")
                    tag = {"done": "ok", "refused": "trend"}.get(outcome, "anomaly")
                    text.insert("end", f"{entry.get('action', '?')}  ·  "
                                       f"{outcome}\n", tag)
                    text.insert("end", f"{entry.get('name', '?')} "
                                       f"({entry.get('pid', '?')})  "
                                       f"{entry.get('timestamp', '')}\n", "detail")
                    text.insert("end", "\n")

        text.configure(state="disabled")

    # -- details panel --------------------------------------------------

    def _on_select(self, _event=None) -> None:
        if self._latest is not None:
            self._refresh_details(self._latest)

    def _selected_process(self, result: SampleResult) -> Optional[ProcessInfo]:
        selection = self.tree.selection()
        if not selection:
            return None
        pid = int(selection[0])
        return next((p for p in result.snapshot.processes if p.pid == pid), None)

    def _refresh_details(self, result: SampleResult) -> None:
        """Describe the selected process and decide what the buttons do."""
        process = self._selected_process(result)
        if process is None:
            self.detail_title.config(text="No process selected")
            self.detail_body.config(text="Click a row to see its classification, "
                                         "anomaly check, trend forecast and "
                                         "recommendation.")
            self._set_buttons(None, None)
            return

        pid = process.pid
        self.detail_title.config(
            text=f"{process.name}   ·   PID {pid}   ·   "
                 f"{process.cpu_percent:.1f} % CPU   ·   "
                 f"{process.memory_mb:,.1f} MB   ·   "
                 f"{process.thread_count} threads   ·   {process.status}")

        prediction = result.predictions.get(pid)
        classification = (f"{prediction.label} ({prediction.confidence:.0f}% confidence)"
                          if prediction else f"unavailable - {self._ai_status}")

        found = result.anomalies.get(pid)
        if found is None:
            anomaly_line = f"unavailable - {self._analysis_status}"
        else:
            marker = f"{ICON_ANOMALY} " if found.is_anomaly else ""
            anomaly_line = f"{marker}{found.reason}"

        forecast = result.forecasts.get(pid)
        if forecast is None:
            trend_line = f"unavailable - {self._analysis_status}"
        elif forecast.has_warning:
            trend_line = f"{ICON_TREND} {forecast.message}"
        else:
            trend_line = (f"{forecast.message} "
                          f"(CPU {forecast.cpu_rate_per_min:+.2f} %/min, "
                          f"memory {forecast.memory_rate_per_min:+,.1f} MB/min)")

        advice = result.recommendations.get(pid)
        if advice is None:
            advice_line = f"unavailable - {_ACTIONS_IMPORT_ERROR or 'disabled'}"
        elif pid in self._ignored:
            advice_line = f"{advice.action} - ignored for this session"
        else:
            advice_line = (f"{advice.action}  ({advice.confidence:.0f}% confidence)"
                           f"  -  {advice.reason}")

        self.detail_body.config(
            text=f"Classification :  {classification}\n"
                 f"Context switch :  {_ctx_switch_line(process)}\n"
                 f"Anomaly check  :  {anomaly_line}\n"
                 f"Trend forecast :  {trend_line}\n"
                 f"Recommendation :  {advice_line}")

        self._set_buttons(process, advice)

    def _set_buttons(self, process: Optional[ProcessInfo], advice) -> None:
        """
        Decide what, if anything, the two buttons do for this selection.

        A suspended process always gets a Resume button regardless of what the
        engine thinks - leaving somebody unable to undo a suspend from the tool
        that applied it would be a poor piece of design.
        """
        if process is None or self._manager is None:
            self._pending = None
            self.apply_button.config(state="disabled", text="Apply",
                                     style="TButton")
            self.ignore_button.config(state="disabled")
            return

        if process.status == "suspended":
            self._pending = (process.pid, process.name, ACTION_RESUME)
            self.apply_button.config(state="normal", text=f"{ACTION_RESUME}",
                                     style="TButton")
            self.ignore_button.config(state="disabled")
            return

        actionable = (advice is not None and advice.executable
                      and process.pid not in self._ignored)
        if actionable:
            self._pending = (process.pid, process.name, advice.action)
            self.apply_button.config(
                state="normal", text=f"Apply: {advice.action}",
                style="Danger.TButton" if advice.action == ACTION_SUSPEND
                else "TButton")
            self.ignore_button.config(state="normal")
        else:
            self._pending = None
            self.apply_button.config(state="disabled", text="Apply",
                                     style="TButton")
            self.ignore_button.config(state="disabled")

    # ------------------------------------------------------------------
    # Actions (main thread only, always confirmed)
    # ------------------------------------------------------------------

    def _on_apply(self) -> None:
        """
        Confirm, then perform, then report.

        This is the only place in the program that leads to an OS change, and
        it cannot be reached without a button press followed by a Yes in a
        dialog.
        """
        if self._pending is None or self._manager is None:
            return
        pid, name, action = self._pending

        advice = None
        if self._latest is not None:
            advice = self._latest.recommendations.get(pid)

        lines = [f"Apply \"{action}\" to {name} (PID {pid})?", ""]
        if advice is not None and action != ACTION_RESUME:
            lines += [f"Why: {advice.reason}", ""]
            if advice.warning:
                lines += [advice.warning, ""]
        lines.append("This changes a running process on your machine.")

        if not messagebox.askyesno(f"Confirm: {action}", "\n".join(lines),
                                   icon="warning", default="no", parent=self):
            return

        if action == ACTION_SUSPEND:
            result = self._manager.suspend_process(pid, name)
        elif action == ACTION_LOWER_PRIORITY:
            result = self._manager.lower_priority(pid, name)
        elif action == ACTION_RESUME:
            result = self._manager.resume_process(pid, name)
        else:                       # advisory actions have no implementation
            return

        if result.success:
            messagebox.showinfo("Action applied", str(result), parent=self)
        else:
            messagebox.showerror("Action not applied", str(result), parent=self)

        # Show the outcome immediately rather than waiting for the next sample.
        self._refresh_action_log()
        self._request_render()

    def _on_ignore(self) -> None:
        """Hide this process's advice for the rest of the session."""
        if self._pending is None:
            return
        pid, name, _action = self._pending
        self._ignored.add(pid)
        log.info("Ignoring recommendations for %s (%d) this session", name, pid)
        self._request_render()

    # -- status bar -----------------------------------------------------

    def _update_status(self, result: SampleResult, shown: int) -> None:
        snapshot = result.snapshot
        threads = sum(p.thread_count for p in snapshot.processes)
        parts = [
            f"Updated {snapshot.timestamp.strftime('%H:%M:%S')}",
            f"{shown} of {len(snapshot.processes)} processes shown",
            f"{threads:,} threads",
            f"source: {self._monitor.source}",
        ]
        if self._paused:
            parts.append("PAUSED")

        # Summarise what the model decided, e.g. "AI: 340 NORMAL, 2 CPU_INTENSIVE".
        if result.predictions:
            counts: Dict[str, int] = {}
            for prediction in result.predictions.values():
                counts[prediction.label] = counts.get(prediction.label, 0) + 1
            breakdown = ", ".join(f"{count} {label}" for label, count
                                  in sorted(counts.items(), key=lambda kv: -kv[1]))
            parts.append(f"AI: {breakdown}")
        else:
            parts.append(f"AI: {self._ai_status}")

        if result.anomalies or result.forecasts:
            anomalies = sum(1 for a in result.anomalies.values() if a.is_anomaly)
            trends = sum(1 for f in result.forecasts.values() if f.has_warning)
            parts.append(f"alerts: {anomalies} anomaly / {trends} trend")
        else:
            parts.append(f"detection: {self._analysis_status}")

        if result.recommendations:
            actionable = sum(1 for pid, r in result.recommendations.items()
                             if r.executable and pid not in self._ignored)
            parts.append(f"suggested actions: {actionable}")

        if self.collector is None:
            parts.append("collector: off")
        elif self.collector.last_error:
            parts.append(f"collector error: {self.collector.last_error}")
        else:
            parts.append(f"collector: {self.collector.samples_written} samples / "
                         f"{self.collector.rows_written:,} rows -> "
                         f"{self.collector.csv_path.name}")

        self.status_label.config(text="   ·   ".join(parts))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _sort_by(self, column: str) -> None:
        """Sort by the clicked column; click the same one again to reverse."""
        if column == self._sort_column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            # Numbers (and alert / action levels) read best biggest-first.
            self._sort_reverse = _COLUMN_BY_ID[column][5]
        self._update_headings()
        self._request_render()

    def _update_headings(self) -> None:
        """Show a small arrow on whichever column we are sorting by."""
        for col_id, heading, *_ in COLUMNS:
            arrow = ""
            if col_id == self._sort_column:
                arrow = "  ▼" if self._sort_reverse else "  ▲"
            self.tree.heading(col_id, text=heading + arrow)

    def _toggle_pause(self) -> None:
        # Sampling continues while paused (so CPU % stays accurate); we simply
        # stop repainting, which lets you read or select a row in peace.
        self._paused = self.pause_var.get()
        self._request_render()

    def _on_close(self) -> None:
        """Shut the worker thread and any detail windows down, then exit."""
        log.info("Closing dashboard...")
        self._stop_event.set()
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass            # already fired
            self._poll_job = None
        for window in list(self._detail_windows.values()) + [
                self._scheduler_window, self._deadlock_window]:
            if window is None:
                continue
            try:
                window.destroy()
            except tk.TclError:
                pass            # already gone
        self._detail_windows.clear()
        self._watched_pids.clear()
        self.destroy()


# ---------------------------------------------------------------------------
# Manual test:  python gui.py     (dashboard only, no CSV collection)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    Dashboard().mainloop()
