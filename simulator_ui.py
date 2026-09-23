"""
simulator_ui.py
===============
The windows for the two OS simulators: CPU scheduling and deadlock detection.

This file is deliberately *only* user interface.  Every algorithm lives in
scheduler.py and deadlock.py, which import nothing from here and can be run,
tested and explained on their own.  That separation is the point: the logic
being demonstrated should not be tangled up with the widgets demonstrating it.

Both windows are plain Toplevels owned by the dashboard.  They hold no live
data, are never touched by a refresh, and a failure in either cannot affect
monitoring - the dashboard imports them defensively and simply hides the
buttons if this module will not load.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

import deadlock
import scheduler
from deadlock import detect_deadlock, is_safe, need_matrix, request_resources
from scheduler import (ALGORITHM_NAMES, DEFAULT_QUANTUM, IDLE_LABEL,
                       ROUND_ROBIN, SimProcess, compare_all, run)

log = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    CHARTS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the environment
    FigureCanvasTkAgg = None
    Figure = None
    CHARTS_AVAILABLE = False
    _CHARTS_ERROR = str(exc)

# The same palette as charts.py, so the simulators look like the rest of the
# application rather than like two bolted-on dialogs.
WINDOW_BG = "#111827"
PANEL_BG = "#1f2937"
FG = "#e5e7eb"
FG_MUTED = "#9aa4b2"
GRID = "#374151"
FIELD_BG = "#0f172a"
OK_GREEN = "#4ade80"
BAD_RED = "#f87171"
WARN_AMBER = "#fbbf24"

FONT = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 12, "bold")
FONT_HUGE = ("Segoe UI", 15, "bold")
FONT_MONO = ("Consolas", 9)

# One colour per process in the Gantt chart, reused cyclically.  Chosen to stay
# distinguishable next to each other and to keep dark text readable on top.
GANTT_COLORS = ["#f2c53d", "#f08c34", "#60a5fa", "#a78bfa", "#4ade80",
                "#f472b6", "#38bdf8", "#fb923c", "#34d399", "#e879f9"]
IDLE_COLOR = "#374151"


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------


ACCENT = "#2563eb"
ACCENT_HOVER = "#3b82f6"
ACCENT_PRESSED = "#1d4ed8"
BUTTON_HOVER = "#4b5563"


def _style_dark(widget_root) -> None:
    """
    Teach ttk the dark palette, once per window.

    ttk widgets do not take bg/fg like plain Tk ones - they are drawn by a
    theme engine, so every colour has to go through a named style.  Left
    alone they render in the light system colours, which looks like two
    different applications stapled together.  "clam" is used because it is
    the one built-in theme that actually honours these settings.
    """
    style = ttk.Style(widget_root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # Secondary buttons: flat, the colour of the panel borders.
    style.configure("Sim.TButton", font=FONT, padding=(10, 5), relief="flat",
                    background=GRID, foreground=FG, bordercolor=GRID,
                    lightcolor=GRID, darkcolor=GRID, focuscolor=PANEL_BG)
    style.map("Sim.TButton",
              background=[("pressed", WINDOW_BG), ("active", BUTTON_HOVER)],
              foreground=[("disabled", FG_MUTED)])

    # The primary action (Run / Check) gets the accent colour, so the one
    # button you are meant to press is obvious.
    style.configure("SimGo.TButton", font=FONT_BOLD, padding=(10, 5),
                    relief="flat", background=ACCENT, foreground="#ffffff",
                    bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                    focuscolor=ACCENT)
    style.map("SimGo.TButton",
              background=[("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER)])

    style.configure("Sim.TCombobox", fieldbackground=FIELD_BG, background=GRID,
                    foreground=FG, arrowcolor=FG, bordercolor=GRID,
                    lightcolor=GRID, darkcolor=GRID, selectbackground=FIELD_BG,
                    selectforeground=FG, padding=(6, 4))
    # A readonly combobox draws itself from the "readonly" state, so the plain
    # configure() above is not enough on its own.
    style.map("Sim.TCombobox",
              fieldbackground=[("readonly", FIELD_BG)],
              foreground=[("readonly", FG)],
              selectbackground=[("readonly", FIELD_BG)],
              selectforeground=[("readonly", FG)],
              background=[("readonly", GRID), ("active", BUTTON_HOVER)],
              arrowcolor=[("active", FG)])
    # The dropdown itself is a Tk listbox, reachable only through the option
    # database rather than through ttk styles.
    for option, value in (("background", FIELD_BG), ("foreground", FG),
                          ("selectBackground", ACCENT),
                          ("selectForeground", "#ffffff")):
        widget_root.option_add(f"*TCombobox*Listbox.{option}", value)

    style.configure("Sim.Vertical.TScrollbar", background=GRID,
                    troughcolor=WINDOW_BG, bordercolor=WINDOW_BG,
                    arrowcolor=FG, lightcolor=GRID, darkcolor=GRID)
    style.map("Sim.Vertical.TScrollbar",
              background=[("active", BUTTON_HOVER)])

    style.configure("Sim.TRadiobutton", background=PANEL_BG, foreground=FG,
                    font=FONT, focuscolor=PANEL_BG, indicatorcolor=FIELD_BG,
                    bordercolor=GRID, lightcolor=GRID, darkcolor=GRID)
    style.map("Sim.TRadiobutton",
              background=[("active", PANEL_BG)],
              indicatorcolor=[("selected", ACCENT)],
              foreground=[("active", FG)])
    style.configure("Sim.Treeview", background=PANEL_BG, fieldbackground=PANEL_BG,
                    foreground=FG, rowheight=22, font=FONT, borderwidth=0)
    style.configure("Sim.Treeview.Heading", background=GRID, foreground=FG,
                    font=FONT_BOLD, relief="flat")
    style.map("Sim.Treeview", background=[("selected", "#2563eb")],
              foreground=[("selected", "#ffffff")])


def _entry(parent, width: int = 6, justify: str = "center") -> tk.Entry:
    """A dark, compact text box - the building block of every editable grid."""
    return tk.Entry(parent, width=width, justify=justify, font=FONT,
                    bg=FIELD_BG, fg=FG, insertbackground=FG,
                    relief="flat", highlightthickness=1,
                    highlightbackground=GRID, highlightcolor="#2563eb")


def _label(parent, text: str, **kwargs) -> tk.Label:
    options = dict(bg=kwargs.pop("bg", PANEL_BG), fg=kwargs.pop("fg", FG),
                   font=kwargs.pop("font", FONT))
    options.update(kwargs)
    return tk.Label(parent, text=text, **options)


def _section(parent, title: str) -> tk.Frame:
    """A titled panel."""
    holder = tk.Frame(parent, bg=PANEL_BG, highlightthickness=1,
                      highlightbackground=GRID)
    _label(holder, title, font=FONT_BOLD).pack(anchor="w", padx=10, pady=(8, 4))
    body = tk.Frame(holder, bg=PANEL_BG)
    body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    holder.body = body                      # type: ignore[attr-defined]
    return holder


def _read_number(entry: tk.Entry, what: str, integer: bool = False,
                 minimum: Optional[float] = None) -> float:
    """
    Read one box, with an error message naming the field rather than the type.

    Every grid cell goes through here so a typo produces "Burst time for P2:
    'x' is not a number" instead of a ValueError traceback.
    """
    text = entry.get().strip()
    if not text:
        raise ValueError(f"{what} is empty")
    try:
        value = int(text) if integer else float(text)
    except ValueError:
        raise ValueError(f"{what}: {text!r} is not a "
                         f"{'whole number' if integer else 'number'}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what}: must be at least {minimum:g}")
    return value


class ScrollableFrame(tk.Frame):
    """A frame whose contents scroll vertically once they outgrow the window."""

    def __init__(self, parent, height: int = 220, **kwargs):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self._canvas = tk.Canvas(self, bg=PANEL_BG, highlightthickness=0,
                                 height=height)
        bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview,
                            style="Sim.Vertical.TScrollbar")
        self.inner = tk.Frame(self._canvas, bg=PANEL_BG)
        self._window = self._canvas.create_window((0, 0), window=self.inner,
                                                  anchor="nw")
        self._canvas.configure(yscrollcommand=bar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self._canvas.bind("<Configure>", self._on_canvas)

    def _on_inner(self, _event=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas(self, event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)


# ===========================================================================
# CPU SCHEDULING SIMULATOR
# ===========================================================================


class SchedulerWindow(tk.Toplevel):
    """
    Enter processes, pick an algorithm, see the numbers and the Gantt chart.

    The interesting button is "Load from real data": it fills the table from
    processes actually measured on this machine, so the algorithms run on real
    workloads instead of textbook numbers.  How the burst times are derived -
    and what that derivation assumes - is documented at length in scheduler.py
    and summarised on screen when the button is used.
    """

    COLUMNS = ("Process", "Arrival", "Burst", "Priority")

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.title("CPU Scheduling Simulator")
        self.geometry("1220x900")
        self.minsize(1000, 700)
        self.configure(bg=WINDOW_BG)
        _style_dark(self)

        self._rows: List[dict] = []
        self._last_result = None

        self._build_header()
        body = tk.Frame(self, bg=WINDOW_BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        body.grid_columnconfigure(0, minsize=430)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_output_panel(body)

        # A small default workload, so the window is useful the moment it opens.
        for row in (("P1", 0, 5, 2), ("P2", 1, 3, 1),
                    ("P3", 2, 8, 4), ("P4", 3, 2, 3)):
            self._add_row(*row)

    # -- construction ---------------------------------------------------

    def _build_header(self) -> None:
        head = tk.Frame(self, bg=WINDOW_BG, padx=12, pady=10)
        head.pack(fill="x")
        _label(head, "CPU Scheduling Simulator", bg=WINDOW_BG,
               font=FONT_TITLE).pack(anchor="w")
        _label(head, "Four classic algorithms, the four standard metrics, and a "
                     "Gantt chart  ·  lower priority number = higher priority",
               bg=WINDOW_BG, fg=FG_MUTED).pack(anchor="w", pady=(2, 0))

    def _build_input_panel(self, parent) -> None:
        column = tk.Frame(parent, bg=WINDOW_BG)
        column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        table = _section(column, "PROCESSES")
        table.pack(fill="both", expand=True)

        heading = tk.Frame(table.body, bg=PANEL_BG)
        heading.pack(fill="x")
        for index, name in enumerate(self.COLUMNS):
            _label(heading, name, fg=FG_MUTED, font=FONT_BOLD, width=9,
                   anchor="center").grid(row=0, column=index, padx=2)
        _label(heading, "", width=3).grid(row=0, column=len(self.COLUMNS))

        self._scroller = ScrollableFrame(table.body, height=230)
        self._scroller.pack(fill="both", expand=True, pady=(4, 6))

        buttons = tk.Frame(table.body, bg=PANEL_BG)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Add row", style="Sim.TButton",
                   command=lambda: self._add_row()).pack(side="left")
        ttk.Button(buttons, text="Clear all", style="Sim.TButton",
                   command=self._clear_rows).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Load from real data", style="Sim.TButton",
                   command=self._load_real).pack(side="left", padx=(6, 0))

        controls = _section(column, "ALGORITHM")
        controls.pack(fill="x", pady=(10, 0))
        line = tk.Frame(controls.body, bg=PANEL_BG)
        line.pack(fill="x")

        self.algorithm_var = tk.StringVar(value=ALGORITHM_NAMES[0])
        picker = ttk.Combobox(line, textvariable=self.algorithm_var, width=16,
                              state="readonly", values=list(ALGORITHM_NAMES),
                              style="Sim.TCombobox", font=FONT)
        picker.pack(side="left")
        picker.bind("<<ComboboxSelected>>", lambda _e: self._sync_quantum())

        # The quantum only means anything for Round Robin, so it only appears
        # for Round Robin rather than sitting there greyed out and confusing.
        self._quantum_box = tk.Frame(line, bg=PANEL_BG)
        _label(self._quantum_box, "Time quantum:").pack(side="left", padx=(12, 4))
        self.quantum_entry = _entry(self._quantum_box, width=6)
        self.quantum_entry.insert(0, f"{DEFAULT_QUANTUM:g}")
        self.quantum_entry.pack(side="left")

        actions = tk.Frame(controls.body, bg=PANEL_BG)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Run", style="SimGo.TButton",
                   command=self._run).pack(side="left")
        ttk.Button(actions, text="Compare all four", style="Sim.TButton",
                   command=self._compare).pack(side="left", padx=(8, 0))

        self.note_label = _label(controls.body, "", fg=FG_MUTED,
                                 wraplength=380, justify="left")
        self.note_label.pack(fill="x", pady=(10, 0))
        self._sync_quantum()

    def _build_output_panel(self, parent) -> None:
        column = tk.Frame(parent, bg=WINDOW_BG)
        column.grid(row=0, column=1, sticky="nsew")
        column.grid_rowconfigure(1, weight=1)
        column.grid_columnconfigure(0, weight=1)

        summary = _section(column, "RESULTS")
        summary.grid(row=0, column=0, sticky="ew")
        self.headline = _label(summary.body, "Enter processes and press Run.",
                               font=FONT_HUGE)
        self.headline.pack(anchor="w")
        self.averages = _label(summary.body, "", fg=FG_MUTED, justify="left")
        self.averages.pack(anchor="w", pady=(4, 6))

        self.results = ttk.Treeview(
            summary.body, style="Sim.Treeview", height=7, selectmode="none",
            columns=("pid", "arrival", "burst", "priority", "start",
                     "completion", "turnaround", "waiting", "response"),
            show="headings")
        for key, title, width in (
                ("pid", "Process", 110), ("arrival", "Arrival", 66),
                ("burst", "Burst", 62), ("priority", "Priority", 62),
                ("start", "Start", 62), ("completion", "Completion", 82),
                ("turnaround", "Turnaround", 84), ("waiting", "Waiting", 70),
                ("response", "Response", 76)):
            self.results.heading(key, text=title)
            self.results.column(key, width=width, anchor="center",
                                stretch=(key == "pid"))
        self.results.pack(fill="x")

        chart = _section(column, "GANTT CHART")
        chart.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self._chart_holder = chart.body
        if CHARTS_AVAILABLE:
            self.figure = Figure(figsize=(7.4, 2.9), dpi=100, facecolor=PANEL_BG)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self._chart_holder)
            widget = self.canvas.get_tk_widget()
            widget.configure(bg=PANEL_BG, highlightthickness=0)
            widget.pack(fill="both", expand=True)
        else:
            self.figure = None
            self.canvas = None
            _label(self._chart_holder,
                   f"Charts unavailable: {_CHARTS_ERROR}", fg=FG_MUTED).pack()

    # -- the editable table ---------------------------------------------

    def _add_row(self, pid: str = "", arrival: float = 0,
                 burst: float = 1, priority: int = 1) -> None:
        index = len(self._rows)
        frame = tk.Frame(self._scroller.inner, bg=PANEL_BG)
        frame.pack(fill="x", pady=1)

        entries = {}
        for column, (key, value) in enumerate(
                (("pid", pid or f"P{index + 1}"), ("arrival", arrival),
                 ("burst", burst), ("priority", priority))):
            box = _entry(frame, width=9,
                         justify="left" if key == "pid" else "center")
            box.insert(0, str(value))
            box.grid(row=0, column=column, padx=2, sticky="ew")
            entries[key] = box
        frame.grid_columnconfigure(0, weight=1)

        row = {"frame": frame, **entries}
        remove = tk.Button(frame, text="×", font=FONT_BOLD, width=2,
                           bg=PANEL_BG, fg=BAD_RED, relief="flat",
                           activebackground=GRID, activeforeground=BAD_RED,
                           command=lambda r=row: self._remove_row(r))
        remove.grid(row=0, column=len(self.COLUMNS), padx=(4, 0))
        self._rows.append(row)

    def _remove_row(self, row: dict) -> None:
        row["frame"].destroy()
        if row in self._rows:
            self._rows.remove(row)

    def _clear_rows(self) -> None:
        for row in list(self._rows):
            self._remove_row(row)

    def _read_processes(self) -> List[SimProcess]:
        """Turn the grid into SimProcess objects, complaining usefully."""
        processes: List[SimProcess] = []
        seen = set()
        for index, row in enumerate(self._rows, start=1):
            name = row["pid"].get().strip() or f"P{index}"
            if name in seen:
                raise ValueError(f"Two processes are both called {name!r}")
            seen.add(name)
            arrival = _read_number(row["arrival"], f"Arrival time for {name}",
                                   minimum=0)
            burst = _read_number(row["burst"], f"Burst time for {name}")
            priority = int(_read_number(row["priority"], f"Priority for {name}",
                                        integer=True))
            if burst <= 0:
                raise ValueError(f"Burst time for {name} must be greater than 0")
            processes.append(SimProcess(pid=name, arrival=arrival,
                                        burst=burst, priority=priority))
        if not processes:
            raise ValueError("Add at least one process first")
        return processes

    def _sync_quantum(self) -> None:
        if self.algorithm_var.get() == ROUND_ROBIN:
            self._quantum_box.pack(side="left")
        else:
            self._quantum_box.pack_forget()

    # -- running ---------------------------------------------------------

    def _run(self) -> None:
        try:
            processes = self._read_processes()
            algorithm = self.algorithm_var.get()
            quantum = DEFAULT_QUANTUM
            if algorithm == ROUND_ROBIN:
                quantum = _read_number(self.quantum_entry, "Time quantum")
                if quantum <= 0:
                    raise ValueError("The time quantum must be greater than 0")
            result = run(algorithm, processes, quantum)
        except ValueError as exc:
            messagebox.showerror("Cannot run", str(exc), parent=self)
            return

        self._last_result = result
        self._show_result(result)

    def _show_result(self, result) -> None:
        title = result.algorithm
        if result.quantum:
            title += f"   (quantum {result.quantum:g})"
        self.headline.config(text=title, fg=FG)
        self.averages.config(text=(
            f"Average waiting time  {result.avg_waiting:.2f}        "
            f"Average turnaround  {result.avg_turnaround:.2f}        "
            f"Average response  {result.avg_response:.2f}\n"
            f"CPU busy {result.cpu_utilisation:.0f} % of the schedule   ·   "
            f"{result.context_switches} context switch"
            f"{'' if result.context_switches == 1 else 'es'}   ·   "
            f"finished at t = {result.makespan:g}"))

        self.results.delete(*self.results.get_children())
        for m in sorted(result.metrics, key=lambda m: m.start):
            self.results.insert("", "end", values=(
                m.pid, f"{m.arrival:g}", f"{m.burst:g}", m.priority,
                f"{m.start:g}", f"{m.completion:g}", f"{m.turnaround:g}",
                f"{m.waiting:g}", f"{m.response:g}"))
        self._draw_gantt(result)

    def _compare(self) -> None:
        try:
            processes = self._read_processes()
            quantum = _read_number(self.quantum_entry, "Time quantum")
            if quantum <= 0:
                raise ValueError("The time quantum must be greater than 0")
            results = compare_all(processes, quantum)
        except ValueError as exc:
            messagebox.showerror("Cannot compare", str(exc), parent=self)
            return

        best = min(results.values(), key=lambda r: r.avg_waiting)
        self.headline.config(text=f"Best average waiting time: {best.algorithm}",
                             fg=OK_GREEN)
        self.averages.config(text=(
            "All four algorithms on the same workload.  Average waiting time is "
            "the usual yardstick, but note how Round Robin trades a worse "
            "waiting time for a much better response time."))

        self.results.delete(*self.results.get_children())
        for name, result in results.items():
            self.results.insert("", "end", values=(
                name, "-", "-",
                f"q={result.quantum:g}" if result.quantum else "-",
                "-", f"{result.makespan:g}",
                f"{result.avg_turnaround:.2f}",
                f"{result.avg_waiting:.2f}",
                f"{result.avg_response:.2f}"))
        self._draw_comparison(results)

    # -- charts ----------------------------------------------------------

    def _prepare_axes(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=FG_MUTED, labelsize=8)
        return ax

    def _colour_for(self, pid: str, order: List[str]) -> str:
        if pid == IDLE_LABEL:
            return IDLE_COLOR
        return GANTT_COLORS[order.index(pid) % len(GANTT_COLORS)]

    def _draw_gantt(self, result) -> None:
        """
        The classic single-row Gantt chart: time runs left to right and each
        block is a stretch during which one process held the CPU.
        """
        if not CHARTS_AVAILABLE:
            return
        ax = self._prepare_axes()
        order: List[str] = []
        for s in result.timeline:
            if s.pid != IDLE_LABEL and s.pid not in order:
                order.append(s.pid)

        for s in result.timeline:
            colour = self._colour_for(s.pid, order)
            ax.barh(0, s.duration, left=s.start, height=0.78,
                    color=colour, edgecolor=PANEL_BG, linewidth=1.5)
            # Only label a block wide enough to hold the text.
            span = max(result.makespan, 1e-9)
            if s.duration / span > 0.05:
                ax.text(s.start + s.duration / 2, 0,
                        "idle" if s.is_idle else s.pid,
                        ha="center", va="center", fontsize=8,
                        color=FG if s.is_idle else "#111827",
                        fontweight="bold")

        # Tick at every boundary: this is how a Gantt chart is read.
        edges = sorted({0.0} | {s.start for s in result.timeline}
                       | {s.end for s in result.timeline})
        ax.set_xticks(edges)
        ax.set_xticklabels([f"{e:g}" for e in edges], fontsize=7.5)
        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, max(result.makespan, 1))
        ax.set_xlabel("time", color=FG_MUTED, fontsize=8)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{result.algorithm} — execution order", color=FG,
                     fontsize=9, loc="left", pad=8)
        self.figure.subplots_adjust(left=0.04, right=0.985, top=0.82, bottom=0.28)
        self.canvas.draw_idle()

    def _draw_comparison(self, results) -> None:
        """Average waiting time per algorithm, side by side."""
        if not CHARTS_AVAILABLE:
            return
        ax = self._prepare_axes()
        names = list(results)
        waiting = [results[n].avg_waiting for n in names]
        turnaround = [results[n].avg_turnaround for n in names]
        response = [results[n].avg_response for n in names]

        positions = range(len(names))
        width = 0.26
        for offset, values, colour, label in (
                (-width, waiting, GANTT_COLORS[0], "Avg waiting"),
                (0.0, turnaround, GANTT_COLORS[1], "Avg turnaround"),
                (width, response, GANTT_COLORS[2], "Avg response")):
            ax.bar([p + offset for p in positions], values, width,
                   color=colour, label=label)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("time units", color=FG_MUTED, fontsize=8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        legend = ax.legend(fontsize=7.5, facecolor=WINDOW_BG, edgecolor=GRID,
                           ncol=3, loc="upper right")
        for text in legend.get_texts():
            text.set_color(FG)
        best = min(range(len(names)), key=lambda i: waiting[i])
        ax.set_title(f"Lowest average waiting time: {names[best]} "
                     f"({waiting[best]:.2f})", color=FG, fontsize=9,
                     loc="left", pad=8)
        self.figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.14)
        self.canvas.draw_idle()

    # -- real data -------------------------------------------------------

    def _load_real(self) -> None:
        """
        Fill the table from processes actually measured on this machine.

        The on-screen note states the derivation and its main caveat, because
        somebody reading the numbers deserves to know they are scaled CPU-time
        totals rather than textbook bursts.
        """
        try:
            derived = scheduler.load_from_csv(limit=6)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Cannot load real data", str(exc), parent=self)
            return

        self._clear_rows()
        for item in derived:
            p = item.process
            self._add_row(p.pid[:18], p.arrival, p.burst, p.priority)

        total = sum(d.cpu_seconds for d in derived)
        self.note_label.config(
            text=(f"Loaded {len(derived)} real processes from "
                  f"data/process_data.csv.\n"
                  f"Burst = measured CPU time (cpu% ÷ 100 × cores × interval, "
                  f"summed over every sample), scaled so the busiest is "
                  f"{scheduler.SCALE_TARGET:g} units. "
                  f"{total:,.0f} CPU-seconds of real work in total.\n"
                  f"Caveat: a real process is many short CPU bursts, not one "
                  f"long one, and arrival here means 'first seen', not "
                  f"'created'. See scheduler.py for the full list."),
            fg=WARN_AMBER)


# ===========================================================================
# DEADLOCK DETECTION
# ===========================================================================


class MatrixGrid(tk.Frame):
    """An editable grid of numbers with row and column headings."""

    def __init__(self, parent, rows: int, columns: int,
                 row_names: List[str], column_names: List[str]) -> None:
        super().__init__(parent, bg=PANEL_BG)
        self.cells: List[List[tk.Entry]] = []
        _label(self, "", width=5).grid(row=0, column=0)
        for j, name in enumerate(column_names):
            _label(self, name, fg=FG_MUTED, font=FONT_BOLD,
                   width=4).grid(row=0, column=j + 1, padx=1)
        for i in range(rows):
            _label(self, row_names[i], fg=FG_MUTED, font=FONT_BOLD,
                   width=5, anchor="w").grid(row=i + 1, column=0, sticky="w")
            row_cells = []
            for j in range(columns):
                box = _entry(self, width=4)
                box.insert(0, "0")
                box.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                row_cells.append(box)
            self.cells.append(row_cells)

    def read(self, what: str) -> List[List[int]]:
        return [[int(_read_number(cell, f"{what} [{i}][{j}]", integer=True,
                                  minimum=0))
                 for j, cell in enumerate(row)]
                for i, row in enumerate(self.cells)]

    def fill(self, matrix: List[List[int]]) -> None:
        for row_cells, values in zip(self.cells, matrix):
            for cell, value in zip(row_cells, values):
                cell.delete(0, "end")
                cell.insert(0, str(value))


class VectorGrid(tk.Frame):
    """A single editable row of numbers - the Available vector."""

    def __init__(self, parent, columns: int, column_names: List[str]) -> None:
        super().__init__(parent, bg=PANEL_BG)
        self.cells: List[tk.Entry] = []
        for j, name in enumerate(column_names):
            _label(self, name, fg=FG_MUTED, font=FONT_BOLD,
                   width=4).grid(row=0, column=j, padx=1)
            box = _entry(self, width=4)
            box.insert(0, "0")
            box.grid(row=1, column=j, padx=1)
            self.cells.append(box)

    def read(self, what: str) -> List[int]:
        return [int(_read_number(cell, f"{what} [{j}]", integer=True, minimum=0))
                for j, cell in enumerate(self.cells)]

    def fill(self, vector: List[int]) -> None:
        for cell, value in zip(self.cells, vector):
            cell.delete(0, "end")
            cell.insert(0, str(value))


class DeadlockWindow(tk.Toplevel):
    """
    Banker's algorithm and deadlock detection over an editable set of matrices.

    Two modes, because they answer genuinely different questions (see
    deadlock.py): *avoidance* asks "if I grant this, can I still promise
    everybody finishes?" and needs the Max matrix; *detection* asks "who is
    stuck right now?" and needs the current Request matrix instead.

    The trace panel is the point of the window.  A verdict on its own explains
    nothing; the step list shows which process was chosen at each stage, what
    it needed, and what was free at the time.
    """

    MODE_BANKER = "Banker's algorithm (avoidance)"
    MODE_DETECT = "Deadlock detection"

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.title("Deadlock Detection")
        self.geometry("1180x860")
        self.minsize(980, 700)
        self.configure(bg=WINDOW_BG)
        _style_dark(self)

        self.mode_var = tk.StringVar(value=self.MODE_BANKER)
        self.process_count = tk.IntVar(value=5)
        self.resource_count = tk.IntVar(value=3)
        self._names: List[str] = []
        self._resources: List[str] = []

        self._build_header()
        body = tk.Frame(self, bg=WINDOW_BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        body.grid_columnconfigure(0, minsize=470)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_output_panel(body)

        self._rebuild_grids()
        self.load_example()

    def _build_header(self) -> None:
        head = tk.Frame(self, bg=WINDOW_BG, padx=12, pady=10)
        head.pack(fill="x")
        _label(head, "Deadlock Detection", bg=WINDOW_BG,
               font=FONT_TITLE).pack(anchor="w")
        _label(head, "Banker's algorithm decides whether granting a request "
                     "keeps the system safe  ·  detection finds processes that "
                     "are already stuck",
               bg=WINDOW_BG, fg=FG_MUTED).pack(anchor="w", pady=(2, 0))

    def _build_input_panel(self, parent) -> None:
        column = tk.Frame(parent, bg=WINDOW_BG)
        column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        setup = _section(column, "SETUP")
        setup.pack(fill="x")
        line = tk.Frame(setup.body, bg=PANEL_BG)
        line.pack(fill="x")
        _label(line, "Processes:").pack(side="left")
        tk.Spinbox(line, from_=1, to=10, width=4, textvariable=self.process_count,
                   font=FONT, bg=FIELD_BG, fg=FG, buttonbackground=GRID,
                   relief="flat", justify="center").pack(side="left", padx=(6, 14))
        _label(line, "Resource types:").pack(side="left")
        tk.Spinbox(line, from_=1, to=6, width=4, textvariable=self.resource_count,
                   font=FONT, bg=FIELD_BG, fg=FG, buttonbackground=GRID,
                   relief="flat", justify="center").pack(side="left", padx=(6, 14))
        ttk.Button(line, text="Resize grids", style="Sim.TButton",
                   command=self._rebuild_grids).pack(side="left")

        modes = tk.Frame(setup.body, bg=PANEL_BG)
        modes.pack(fill="x", pady=(10, 0))
        for mode in (self.MODE_BANKER, self.MODE_DETECT):
            ttk.Radiobutton(modes, text=mode, value=mode, variable=self.mode_var,
                            style="Sim.TRadiobutton",
                            command=self._on_mode_change).pack(anchor="w")

        self._grids_section = _section(column, "MATRICES")
        self._grids_section.pack(fill="both", expand=True, pady=(10, 0))

        actions = tk.Frame(column, bg=WINDOW_BG)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Check", style="SimGo.TButton",
                   command=self._check).pack(side="left")
        ttk.Button(actions, text="Load example", style="Sim.TButton",
                   command=self.load_example).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Show Need matrix", style="Sim.TButton",
                   command=self._show_need).pack(side="left", padx=(8, 0))

    def _build_output_panel(self, parent) -> None:
        column = tk.Frame(parent, bg=WINDOW_BG)
        column.grid(row=0, column=1, sticky="nsew")
        column.grid_rowconfigure(1, weight=1)
        column.grid_columnconfigure(0, weight=1)

        verdict = _section(column, "VERDICT")
        verdict.grid(row=0, column=0, sticky="ew")
        self.verdict_label = _label(verdict.body, "Press Check.", font=FONT_HUGE)
        self.verdict_label.pack(anchor="w")
        self.sequence_label = _label(verdict.body, "", fg=FG_MUTED,
                                     wraplength=620, justify="left")
        self.sequence_label.pack(anchor="w", pady=(4, 0))

        trace = _section(column, "STEP-BY-STEP TRACE")
        trace.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.trace_text = tk.Text(trace.body, wrap="word", relief="flat",
                                  bg=FIELD_BG, fg=FG, font=FONT_MONO,
                                  padx=10, pady=8, highlightthickness=0,
                                  insertbackground=FG, spacing3=4)
        bar = ttk.Scrollbar(trace.body, orient="vertical",
                            command=self.trace_text.yview,
                            style="Sim.Vertical.TScrollbar")
        self.trace_text.configure(yscrollcommand=bar.set, state="disabled")
        self.trace_text.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.trace_text.tag_configure("good", foreground=OK_GREEN)
        self.trace_text.tag_configure("bad", foreground=BAD_RED)
        self.trace_text.tag_configure("head", foreground=FG, font=("Consolas", 9, "bold"))
        self.trace_text.tag_configure("muted", foreground=FG_MUTED)

    # -- grids -----------------------------------------------------------

    def _second_matrix_title(self) -> str:
        return "Max" if self.mode_var.get() == self.MODE_BANKER else "Request"

    def _rebuild_grids(self) -> None:
        for child in self._grids_section.body.winfo_children():
            child.destroy()
        rows = max(1, int(self.process_count.get()))
        columns = max(1, int(self.resource_count.get()))
        self._names = deadlock.default_names(rows)
        self._resources = [chr(ord("A") + j) for j in range(columns)]

        body = self._grids_section.body
        _label(body, "Allocation — what each process holds now",
               fg=FG_MUTED).pack(anchor="w")
        self.allocation_grid = MatrixGrid(body, rows, columns,
                                          self._names, self._resources)
        self.allocation_grid.pack(anchor="w", pady=(2, 10))

        self._second_label = _label(body, "", fg=FG_MUTED)
        self._second_label.pack(anchor="w")
        self.second_grid = MatrixGrid(body, rows, columns,
                                      self._names, self._resources)
        self.second_grid.pack(anchor="w", pady=(2, 10))

        _label(body, "Available — free right now", fg=FG_MUTED).pack(anchor="w")
        self.available_grid = VectorGrid(body, columns, self._resources)
        self.available_grid.pack(anchor="w", pady=(2, 0))
        self._on_mode_change()

    def _on_mode_change(self) -> None:
        if not hasattr(self, "_second_label"):
            return
        if self.mode_var.get() == self.MODE_BANKER:
            self._second_label.config(
                text="Max — the most each process will ever need")
        else:
            self._second_label.config(
                text="Request — what each process is asking for right now")

    def load_example(self) -> None:
        """
        Fill in the worked example from the textbook.

        A reliable demo case matters: this one is published, it is safe, and
        its answer can be checked against the book.
        """
        banker = self.mode_var.get() == self.MODE_BANKER
        self.process_count.set(5)
        self.resource_count.set(3)
        self._rebuild_grids()
        if banker:
            self.allocation_grid.fill(deadlock.TEXTBOOK_ALLOCATION)
            self.second_grid.fill(deadlock.TEXTBOOK_MAX)
            self.available_grid.fill(deadlock.TEXTBOOK_AVAILABLE)
            note = ("Silberschatz's Banker's example. Expected: SAFE, with a "
                    "sequence such as P1 -> P3 -> P4 -> P2 -> P0.")
        else:
            self.allocation_grid.fill(deadlock.DETECTION_ALLOCATION)
            request = [list(r) for r in deadlock.DETECTION_REQUEST]
            request[2] = [0, 0, 1]      # the change that causes the deadlock
            self.second_grid.fill(request)
            self.available_grid.fill(deadlock.DETECTION_AVAILABLE)
            note = ("Silberschatz's detection example with P2 asking for one "
                    "more C. Expected: P1, P2, P3 and P4 deadlocked.")
        self.verdict_label.config(text="Example loaded — press Check.", fg=FG)
        self.sequence_label.config(text=note)
        self._write_trace([("Example loaded. Press Check to run the algorithm.\n",
                            "muted")])

    # -- running ---------------------------------------------------------

    def _read_state(self):
        allocation = self.allocation_grid.read("Allocation")
        second = self.second_grid.read(self._second_matrix_title())
        available = self.available_grid.read("Available")
        return allocation, second, available

    def _check(self) -> None:
        try:
            allocation, second, available = self._read_state()
            if self.mode_var.get() == self.MODE_BANKER:
                self._run_banker(allocation, second, available)
            else:
                self._run_detection(allocation, second, available)
        except ValueError as exc:
            messagebox.showerror("Cannot check", str(exc), parent=self)

    def _run_banker(self, allocation, maximum, available) -> None:
        result = is_safe(allocation, maximum, available, self._names)
        if result.safe:
            self.verdict_label.config(text="SAFE", fg=OK_GREEN)
            self.sequence_label.config(
                text=f"Safe sequence:  {result.sequence_text}\n"
                     f"Every process can finish in this order, so deadlock is "
                     f"impossible from this state.")
        else:
            self.verdict_label.config(text="UNSAFE", fg=BAD_RED)
            self.sequence_label.config(text=result.message)

        lines = [(f"Need = Max - Allocation\n", "head")]
        lines += self._matrix_lines(result.need)
        lines.append(("\nSafety algorithm\n", "head"))
        lines.append((f"  Work starts as Available = "
                      f"{self._fmt(available)}\n\n", "muted"))
        for step in result.steps:
            lines.append((f"  {step.describe()}\n\n",
                          "good" if step.granted else "bad"))
        if result.safe:
            lines.append(("  Every process finished, so the state is SAFE.\n",
                          "good"))
        self._write_trace(lines)

    def _run_detection(self, allocation, request, available) -> None:
        result = detect_deadlock(allocation, request, available, self._names)
        if result.has_deadlock:
            self.verdict_label.config(
                text=f"DEADLOCK — {', '.join(result.deadlocked)}", fg=BAD_RED)
        else:
            self.verdict_label.config(text="NO DEADLOCK", fg=OK_GREEN)
        self.sequence_label.config(text=result.message)

        lines = [("Detection algorithm\n", "head"),
                 ("  Works from Request (what is being asked for now), not "
                  "Need (the worst case) — that is what makes this detection "
                  "rather than avoidance.\n", "muted"),
                 (f"  Work starts as Available = {self._fmt(available)}\n\n",
                  "muted")]
        for step in result.steps:
            lines.append((f"  {step.describe()}\n\n",
                          "good" if step.granted else "bad"))
        self._write_trace(lines)

    def _show_need(self) -> None:
        """Need = Max - Allocation, on its own - it is what Banker's works from."""
        if self.mode_var.get() != self.MODE_BANKER:
            messagebox.showinfo(
                "Need matrix",
                "The Need matrix only applies to Banker's algorithm. Detection "
                "works from the Request matrix, which you are already editing.",
                parent=self)
            return
        try:
            allocation, maximum, _available = self._read_state()
            need = need_matrix(allocation, maximum)
            deadlock.validate(allocation, maximum, _available)
        except ValueError as exc:
            messagebox.showerror("Cannot compute Need", str(exc), parent=self)
            return
        self.verdict_label.config(text="Need = Max - Allocation", fg=FG)
        self.sequence_label.config(
            text="What each process could still ask for. Banker's works from "
                 "this rather than from Max, because what matters is how much "
                 "more a process might want given what it already holds.")
        self._write_trace(self._matrix_lines(need))

    # -- output helpers ---------------------------------------------------

    @staticmethod
    def _fmt(vector) -> str:
        return "(" + ", ".join(str(v) for v in vector) + ")"

    def _matrix_lines(self, matrix) -> List[tuple]:
        header = "       " + "".join(f"{r:>4}" for r in self._resources) + "\n"
        lines = [(header, "muted")]
        for name, row in zip(self._names, matrix):
            lines.append((f"  {name:<5}" + "".join(f"{v:>4}" for v in row) + "\n",
                          None))
        return lines

    def _write_trace(self, lines: List[tuple]) -> None:
        self.trace_text.configure(state="normal")
        self.trace_text.delete("1.0", "end")
        for text, tag in lines:
            if tag:
                self.trace_text.insert("end", text, tag)
            else:
                self.trace_text.insert("end", text)
        self.trace_text.configure(state="disabled")


# ---------------------------------------------------------------------------
# Manual test:  python simulator_ui.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = tk.Tk()
    root.title("simulator windows")
    root.configure(bg=WINDOW_BG)
    tk.Label(root, text="Both simulator windows are open.", bg=WINDOW_BG,
             fg=FG, font=FONT, padx=30, pady=20).pack()
    SchedulerWindow(root)
    DeadlockWindow(root)
    root.mainloop()
