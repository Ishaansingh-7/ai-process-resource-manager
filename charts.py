"""
charts.py
=========
Phase 5: the matplotlib visualisations embedded in the Tkinter window.

Two views live here:

* :class:`SystemChartsPanel` - the strip under the table.  A rolling line
  chart of total CPU and RAM over the last few minutes, next to two bar charts
  showing the top five processes by CPU and by memory.
* :class:`ProcessDetailWindow` - opened by double-clicking a row.  CPU, memory
  and thread count over time for that one process, plus everything Phases 2-4
  concluded about it.

Making it fast: what the profiling showed
----------------------------------------
The brief asked for these charts not to slow the table down, profiled the way
Phase 1 profiled psutil.  Measured on this machine (~340 processes):

    handing new numbers to the artists ............  0.3 ms
    full matplotlib repaint .......................  61 ms
      of which: Agg rendering ..................... 48 ms
                transferring the image to Tk ...... 12 ms
    for scale, a whole process snapshot ...........  6 ms

So the redraw cost ten times a complete scan of every process on the machine,
and it was almost entirely fixed overhead: repainting with *no data change at
all* still cost 61 ms.  Removing the grid, the legend and half the resolution
only got it to 53 ms, because the cost is re-rendering the axes, spines, ticks
and labels from scratch every time.

The fix is **blitting**:

1. Every artist that changes - the two lines, ten bars, ten labels - is marked
   ``animated=True``, which excludes it from a normal draw.
2. The figure is drawn once and the result kept as a bitmap (the
   "background"): axes, grid, ticks, legend, all of the expensive furniture.
3. Each refresh restores that bitmap, draws only the ~22 animated artists on
   top, and pushes the result to Tk.

That turns a 61 ms repaint into roughly 5 ms.  The background is only rebuilt
when something structural changes - the window is resized, or an axis limit
moves - and the bar charts round their limits to "nice" numbers precisely so
that small fluctuations do not keep invalidating it.

``draw_idle``/``blit`` also avoid the flicker that comes from repainting
synchronously inside an event handler.

Matplotlib is an optional dependency: if it is missing, ``CHARTS_AVAILABLE``
is False and the dashboard runs exactly as it did in Phase 4.
"""

from __future__ import annotations

import logging
import math
import tkinter as tk
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)

try:
    # Note: no pyplot.  Embedding uses the Figure class and the Tk canvas
    # directly, which avoids pyplot's global figure registry - that registry
    # is what causes stray windows and leaks in Tk applications.
    import matplotlib.patheffects as path_effects
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    CHARTS_AVAILABLE = True
    CHARTS_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - depends on the environment
    path_effects = None  # type: ignore[assignment]
    FigureCanvasTkAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    CHARTS_AVAILABLE = False
    CHARTS_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
# These mirror the dark panels in gui.py (the header and its summary cards) so
# the charts read as part of the same design rather than a matplotlib default
# pasted into the window.

CHART_BG = "#1f2937"        # same as the header's summary cards
WINDOW_BG = "#111827"       # same as the header bar
FG = "#e5e7eb"              # primary text on dark
FG_MUTED = "#9aa4b2"        # ticks and secondary text
GRID = "#374151"

# Metric colours are shared with the classification legend in the table, so
# "yellow means CPU" and "orange means memory" hold everywhere in the app.
CPU_COLOR = "#f2c53d"
MEM_COLOR = "#f08c34"
THREAD_COLOR = "#60a5fa"
CTX_COLOR = "#a78bfa"       # context switches - distinct from the other three

TITLE_SIZE = 8
TICK_SIZE = 7
LABEL_SIZE = 7.5

TOP_N = 5                   # bars per chart
NAME_CHARS = 18             # truncate process names on the bar labels
LABEL_GAP = 0.02            # bar-tip to label, as a fraction of the axis width


def _style_axes(ax, grid: bool = True) -> None:
    """Apply the dark theme to one axes."""
    ax.set_facecolor(CHART_BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG_MUTED, labelsize=TICK_SIZE, length=3, width=0.6)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)      # grid behind the data, never on top of it


def _title(ax, text: str) -> None:
    ax.set_title(text, color=FG, fontsize=TITLE_SIZE, pad=6, loc="left")


def _shorten(name: str) -> str:
    return name if len(name) <= NAME_CHARS else name[:NAME_CHARS - 1] + "…"


def _nice_limit(value: float) -> float:
    """
    Round an axis limit up to a "nice" number (1, 1.5, 2, 3, 5, 7.5, 10 x 10^n).

    This exists for blitting, not for looks: if the bar charts rescaled to the
    exact maximum on every refresh, the axis would change constantly and the
    cached background would have to be rebuilt every time, undoing the whole
    optimisation.  Rounding to coarse steps means the limit usually stays put.
    """
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    for multiple in (1.0, 1.5, 2.0, 3.0, 5.0, 7.5):
        if value <= multiple * base:
            return multiple * base
    return 10.0 * base


def _label_effects(foreground: str = CHART_BG):
    """
    An outline in `foreground` so a bar label survives whatever it lands on.

    Labels beside a bar are light text outlined in the chart background;
    labels inside a bar are dark text outlined in the bar's own colour.
    """
    if path_effects is None:
        return []
    return [path_effects.withStroke(linewidth=2.4, foreground=foreground)]


# ---------------------------------------------------------------------------
# The strip under the table
# ---------------------------------------------------------------------------


class SystemChartsPanel(tk.Frame):
    """
    Rolling system chart + top-five bar charts, in a single figure.

    One figure means one canvas and therefore one repaint per refresh instead
    of three.  See the module docstring for how the repaint is kept cheap.
    """

    def __init__(self, parent, minutes: float = 5.0, **kwargs) -> None:
        super().__init__(parent, bg=CHART_BG, **kwargs)
        self.minutes = minutes
        self.canvas = None

        if not CHARTS_AVAILABLE:
            tk.Label(self, text=f"Charts unavailable: {CHARTS_IMPORT_ERROR}\n"
                                f"pip install -r requirements.txt",
                     bg=CHART_BG, fg=FG_MUTED,
                     font=("Segoe UI", 9)).pack(expand=True)
            return

        self.figure = Figure(figsize=(12, 2.6), dpi=100, facecolor=CHART_BG)
        # Left column: the timeline, full height.  Right column: two stacked
        # bar charts.  Margins are set explicitly instead of using
        # tight_layout(), which re-solves the whole layout on every draw.
        grid = self.figure.add_gridspec(
            2, 2, width_ratios=[1.85, 1],
            left=0.042, right=0.945, top=0.86, bottom=0.16,
            hspace=0.85, wspace=0.10)

        self.ax_time = self.figure.add_subplot(grid[:, 0])
        self.ax_cpu_bars = self.figure.add_subplot(grid[0, 1])
        self.ax_mem_bars = self.figure.add_subplot(grid[1, 1])

        self._build_timeline()
        self._build_bars()

        # Everything that changes between refreshes.  Marked animated so a
        # normal draw() skips them, leaving a clean background to cache.
        self._animated: List = [self.cpu_line, self.ram_line]
        self._animated += list(self.cpu_bars) + list(self.mem_bars)
        self._animated += self.cpu_labels + self.mem_labels
        for artist in self._animated:
            artist.set_animated(True)

        self._background = None          # the cached bitmap; None = rebuild
        self._cpu_limit: Optional[float] = None
        self._mem_limit: Optional[float] = None
        self._ram_top: Optional[float] = None

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=CHART_BG, highlightthickness=0)
        widget.pack(fill="both", expand=True)
        # Resizing changes every pixel of the background, so throw it away and
        # rebuild on the next update.
        widget.bind("<Configure>", lambda _e: self.invalidate())
        self.canvas.draw()

    # -- construction ---------------------------------------------------

    def _build_timeline(self) -> None:
        ax = self.ax_time
        _style_axes(ax)
        _title(ax, f"System usage · last {self.minutes:g} minutes")
        # Fixed limits: the x window always spans the same number of minutes
        # and CPU is always a percentage, so neither ever invalidates the
        # cached background.
        ax.set_xlim(-self.minutes, 0)
        ax.set_ylim(0, 100)
        ax.set_xlabel("minutes ago", color=FG_MUTED, fontsize=LABEL_SIZE)
        ax.set_ylabel("CPU %", color=CPU_COLOR, fontsize=LABEL_SIZE)

        # RAM shares the x axis but needs its own scale, so it gets a twin
        # axis on the right rather than being squashed onto a 0-100 axis.
        self.ax_ram = ax.twinx()
        _style_axes(self.ax_ram, grid=False)   # one grid is enough
        self.ax_ram.set_ylabel("RAM (GB)", color=MEM_COLOR, fontsize=LABEL_SIZE)

        # Created empty now, fed with numbers on every refresh.
        self.cpu_line, = ax.plot([], [], color=CPU_COLOR, linewidth=1.6,
                                 label="CPU %")
        self.ram_line, = self.ax_ram.plot([], [], color=MEM_COLOR, linewidth=1.6,
                                          label="RAM (GB)")

        # One legend covering both axes - matplotlib will not do this by
        # itself because the artists belong to different axes objects.
        handles = [self.cpu_line, self.ram_line]
        legend = ax.legend(handles, [h.get_label() for h in handles],
                           loc="upper left", fontsize=TICK_SIZE, framealpha=0.85,
                           facecolor=WINDOW_BG, edgecolor=GRID, ncol=2)
        for text in legend.get_texts():
            text.set_color(FG)

    def _build_bars(self) -> None:
        positions = list(range(TOP_N))
        for ax, title in ((self.ax_cpu_bars, f"Top {TOP_N} by CPU %"),
                          (self.ax_mem_bars, f"Top {TOP_N} by memory (MB)")):
            _style_axes(ax)
            ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.55)
            ax.grid(False, axis="y")
            _title(ax, title)
            # Process names are drawn *inside* the chart as animated labels
            # rather than as y tick labels.  Tick labels are part of the
            # background, so changing them every refresh would invalidate the
            # cache; this way the y axis never changes at all.
            ax.set_yticks(positions)
            ax.set_yticklabels([""] * TOP_N)
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(TOP_N - 0.5, -0.5)     # biggest at the top
            ax.set_xlim(0, 1)

        # Horizontal bars: the labels are process names, which need the
        # horizontal room a vertical bar chart cannot give them.
        self.cpu_bars = self.ax_cpu_bars.barh(positions, [0] * TOP_N,
                                              color=CPU_COLOR, height=0.62)
        self.mem_bars = self.ax_mem_bars.barh(positions, [0] * TOP_N,
                                              color=MEM_COLOR, height=0.62)

        # Built once and reused: restyling a label every refresh must not
        # allocate a new path-effect object each time.
        self._fx_beside = _label_effects(CHART_BG)
        self._fx_in_cpu = _label_effects(CPU_COLOR)
        self._fx_in_mem = _label_effects(MEM_COLOR)

        self.cpu_labels = [self.ax_cpu_bars.text(0, i, "", va="center",
                                                 fontsize=TICK_SIZE, color=FG,
                                                 path_effects=self._fx_beside)
                           for i in positions]
        self.mem_labels = [self.ax_mem_bars.text(0, i, "", va="center",
                                                 fontsize=TICK_SIZE, color=FG,
                                                 path_effects=self._fx_beside)
                           for i in positions]

    # -- updating -------------------------------------------------------

    def invalidate(self) -> None:
        """Force a full redraw on the next update (resize, theme change...)."""
        self._background = None

    def update_charts(self, system_samples: Sequence, processes: Sequence) -> None:
        """Feed new numbers to the existing artists and repaint the cheap way."""
        if not CHARTS_AVAILABLE or self.canvas is None:
            return
        try:
            # Each updater reports whether it changed something structural
            # (an axis limit), which is the only reason to rebuild the cache.
            structural = self._update_timeline(system_samples)
            structural |= self._update_bars(processes)

            if self._background is None or structural:
                self._draw_full()
            else:
                self._blit()
        except Exception:
            log.exception("Chart update failed")
            self._background = None      # recover on the next refresh

    def _update_timeline(self, samples: Sequence) -> bool:
        if not samples:
            return False
        now = samples[-1].t
        # x is "minutes ago", so the newest sample sits at 0 and history
        # scrolls off to the left.  Using elapsed time rather than sample
        # index keeps the chart honest if a refresh is slow or skipped.
        xs = [(s.t - now) / 60.0 for s in samples]
        self.cpu_line.set_data(xs, [s.cpu_percent for s in samples])
        self.ram_line.set_data(xs, [s.memory_used_mb / 1024.0 for s in samples])

        # Total RAM does not change while the program runs, so this fires once.
        total_gb = max(samples[-1].memory_total_mb / 1024.0, 1.0)
        if self._ram_top != total_gb:
            self.ax_ram.set_ylim(0, total_gb)
            self._ram_top = total_gb
            return True
        return False

    def _update_bars(self, processes: Sequence) -> bool:
        if not processes:
            return False
        top_cpu = sorted(processes, key=lambda p: p.cpu_percent, reverse=True)[:TOP_N]
        top_mem = sorted(processes, key=lambda p: p.memory_mb, reverse=True)[:TOP_N]

        structural = self._apply_bars(
            self.ax_cpu_bars, self.cpu_bars, self.cpu_labels, top_cpu,
            lambda p: p.cpu_percent, "{:.1f} %", self._cpu_limit,
            self._fx_in_cpu)
        if structural is not None:
            self._cpu_limit = structural

        structural_mem = self._apply_bars(
            self.ax_mem_bars, self.mem_bars, self.mem_labels, top_mem,
            lambda p: p.memory_mb, "{:,.0f} MB", self._mem_limit,
            self._fx_in_mem)
        if structural_mem is not None:
            self._mem_limit = structural_mem

        return structural is not None or structural_mem is not None

    def _text_width(self, ax, label) -> float:
        """
        How wide a label is in data units, so we can tell whether it fits
        beside its bar.

        Measured with the real renderer once there is one.  Before the first
        draw there is not, so we fall back to an estimate of ~0.6 em per
        character (a fair average for DejaVu Sans) - that only has to be good
        enough for the opening frame, which the next refresh corrects.
        """
        text = label.get_text()
        if not text:
            return 0.0
        x0, x1 = ax.get_xlim()
        span = x1 - x0
        axes_px = max(ax.bbox.width, 1.0)
        try:
            px = label.get_window_extent(self.canvas.get_renderer()).width
        except Exception:
            px = len(text) * TICK_SIZE * 0.6 * self.figure.dpi / 72.0
        return span * px / axes_px

    def _apply_bars(self, ax, bars, labels, rows, value_of, number_format,
                    current_limit, inside_fx) -> Optional[float]:
        """
        Update one bar chart.  Returns the new axis limit if it had to change
        (which means the background needs rebuilding), otherwise None.
        """
        values = [value_of(p) for p in rows]
        names = [_shorten(p.name) for p in rows]
        # Pad if the machine somehow has fewer than TOP_N processes.
        while len(values) < TOP_N:
            values.append(0.0)
            names.append("")

        limit = _nice_limit(max(values) * 1.05)
        changed = None
        # Only rescale when the data has genuinely outgrown or fallen well
        # inside the current limit - otherwise the axis (and the cached
        # background) would churn on every tiny fluctuation.
        if (current_limit is None or limit > current_limit
                or limit < current_limit * 0.45):
            ax.set_xlim(0, limit)
            changed = limit
        else:
            limit = current_limit

        gap = limit * LABEL_GAP

        for bar, label, value, name in zip(bars, labels, values, names):
            bar.set_width(value)
            if not name:
                label.set_text("")
                continue

            label.set_text(f"{name}   {number_format.format(value)}")
            width = self._text_width(ax, label)
            if value + gap + width <= limit:
                # Room to the right of the bar - the easiest thing to read.
                label.set_x(value + gap)
                label.set_ha("left")
                label.set_color(FG)
                label.set_path_effects(self._fx_beside)
            elif width + 2 * gap <= value:
                # The bar runs too close to the axis limit to label beside it,
                # so the label goes inside: dark text on the bright fill,
                # tucked against the tip.  The longest bar always lands here.
                label.set_x(value - gap)
                label.set_ha("right")
                label.set_color(CHART_BG)
                label.set_path_effects(inside_fx)
            else:
                # Neither side fits (a very narrow window).  Start at the axis
                # and let the outline carry it, which is the old behaviour.
                label.set_x(gap)
                label.set_ha("left")
                label.set_color(FG)
                label.set_path_effects(self._fx_beside)
        return changed

    # -- the two ways of painting ---------------------------------------

    def _draw_full(self) -> None:
        """Repaint everything and cache the static part for later blits."""
        self.canvas.draw()      # animated artists are skipped by design
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)
        self._blit()

    def _blit(self) -> None:
        """Restore the cached background and draw only what moved."""
        self.canvas.restore_region(self._background)
        for artist in self._animated:
            artist.axes.draw_artist(artist)
        self.canvas.blit(self.figure.bbox)


# ---------------------------------------------------------------------------
# The per-process window
# ---------------------------------------------------------------------------


class ProcessDetailWindow(tk.Toplevel):
    """
    Everything known about one process, opened by double-clicking its row.

    It keeps updating while it is open: the dashboard hands it a new snapshot
    on every refresh, exactly like the panels in the main window.  This one
    uses a plain ``draw_idle`` rather than blitting - its y axes rescale to
    each process's own range, so there is no stable background to cache, and
    it only exists while somebody is looking at it.
    """

    def __init__(self, parent, pid: int, name: str, minutes: float = 5.0,
                 on_close=None) -> None:
        super().__init__(parent)
        self.pid = pid
        self.process_name = name
        self.minutes = minutes
        self._on_close = on_close

        self.title(f"{name}  ·  PID {pid}")
        self.geometry("740x760")
        self.minsize(560, 560)
        self.configure(bg=WINDOW_BG)
        self.protocol("WM_DELETE_WINDOW", self.close)

        header = tk.Frame(self, bg=WINDOW_BG, padx=14, pady=10)
        header.pack(fill="x")
        tk.Label(header, text=f"{name}  ·  PID {pid}", bg=WINDOW_BG, fg=FG,
                 font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x")
        self.summary_label = tk.Label(header, text="waiting for data...",
                                      bg=WINDOW_BG, fg=FG_MUTED,
                                      font=("Segoe UI", 9), anchor="w",
                                      justify="left")
        self.summary_label.pack(fill="x", pady=(4, 0))

        if not CHARTS_AVAILABLE:
            tk.Label(self, text=f"Charts unavailable: {CHARTS_IMPORT_ERROR}",
                     bg=WINDOW_BG, fg=FG_MUTED,
                     font=("Segoe UI", 9)).pack(expand=True)
            self.canvas = None
            return

        self.figure = Figure(figsize=(7, 5.6), dpi=100, facecolor=WINDOW_BG)
        grid = self.figure.add_gridspec(4, 1, left=0.13, right=0.97,
                                        top=0.96, bottom=0.075, hspace=0.30)
        # sharex: the four charts describe the same span of time, so they
        # must always show the same window.
        self.ax_cpu = self.figure.add_subplot(grid[0])
        self.ax_mem = self.figure.add_subplot(grid[1], sharex=self.ax_cpu)
        self.ax_threads = self.figure.add_subplot(grid[2], sharex=self.ax_cpu)
        self.ax_ctx = self.figure.add_subplot(grid[3], sharex=self.ax_cpu)

        specs = (
            (self.ax_cpu, "CPU %", CPU_COLOR),
            (self.ax_mem, "Memory (MB)", MEM_COLOR),
            (self.ax_threads, "Threads", THREAD_COLOR),
            # Context switches per second: how often the scheduler took this
            # process off a core, or it gave the core up.  Spikes here next to
            # flat CPU mean a process that is waiting on I/O rather than
            # computing.  See monitor.py for the full explanation.
            (self.ax_ctx, "Ctx Sw/s", CTX_COLOR),
        )
        self.lines = []
        for ax, label, color in specs:
            _style_axes(ax)
            ax.set_ylabel(label, color=color, fontsize=LABEL_SIZE)
            ax.set_xlim(-minutes, 0)
            line, = ax.plot([], [], color=color, linewidth=1.6)
            self.lines.append(line)
        self.ax_ctx.set_xlabel("minutes ago", color=FG_MUTED,
                               fontsize=LABEL_SIZE)
        # Only the bottom chart needs x tick labels; the other three share them.
        for ax in (self.ax_cpu, self.ax_mem, self.ax_threads):
            ax.tick_params(labelbottom=False)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=WINDOW_BG, highlightthickness=0)
        widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.canvas.draw()

    # ------------------------------------------------------------------

    def update_view(self, process, samples: Sequence,
                    prediction=None, anomaly_result=None,
                    forecast=None, recommendation=None) -> None:
        """Called by the dashboard on every refresh while this window is open."""
        if process is None:
            self.summary_label.config(
                text="This process has exited.\n"
                     "The charts show its last recorded history.")
            return

        lines = [
            f"{process.cpu_percent:.1f} % CPU   ·   {process.memory_mb:,.1f} MB   ·   "
            f"{process.thread_count} threads   ·   "
            f"{getattr(process, 'ctx_switches_per_sec', 0.0):,.0f} ctx sw/s   ·   "
            f"{process.priority}   ·   {process.status}",
        ]
        if prediction is not None:
            lines.append(f"Classification :  {prediction.label} "
                         f"({prediction.confidence:.0f}% confidence)")
        if anomaly_result is not None:
            mark = "⚠ " if anomaly_result.is_anomaly else ""
            lines.append(f"Anomaly check  :  {mark}{anomaly_result.reason}")
        if forecast is not None:
            mark = "↗ " if forecast.has_warning else ""
            lines.append(f"Trend forecast :  {mark}{forecast.message}")
        if recommendation is not None:
            lines.append(f"Recommendation :  {recommendation.action} "
                         f"({recommendation.confidence:.0f}% confidence)")
        self.summary_label.config(text="\n".join(lines))

        if not CHARTS_AVAILABLE or self.canvas is None or not samples:
            return

        try:
            now = samples[-1].t
            xs = [(s.t - now) / 60.0 for s in samples]
            series = (
                [s.cpu_percent for s in samples],
                [s.memory_mb for s in samples],
                [float(s.thread_count) for s in samples],
                [getattr(s, "ctx_per_sec", 0.0) for s in samples],
            )
            for line, values, ax in zip(self.lines, series,
                                        (self.ax_cpu, self.ax_mem,
                                         self.ax_threads, self.ax_ctx)):
                line.set_data(xs, values)
                # Autoscale y to this process's own range - a process using
                # 40 MB and one using 4 GB both have to be readable.
                high = max(values) if values else 1.0
                low = min(values) if values else 0.0
                pad = max((high - low) * 0.15, high * 0.05, 1.0)
                ax.set_ylim(max(0.0, low - pad), high + pad)
            self.canvas.draw_idle()
        except Exception:
            log.exception("Detail chart update failed")

    def close(self) -> None:
        """Tell the dashboard to stop updating us, then go away."""
        if self._on_close is not None:
            self._on_close(self.pid)
        self.destroy()


# ---------------------------------------------------------------------------
# Profile:  python charts.py
#
# Answers the question the phase brief asks - does adding charts slow the
# refresh down?  Compares the naive full repaint against the blitting path.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from history import SystemHistory
    from monitor import ProcessMonitor

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not CHARTS_AVAILABLE:
        raise SystemExit(f"matplotlib is not installed: {CHARTS_IMPORT_ERROR}")

    root = tk.Tk()
    root.title("charts.py profile")
    root.configure(bg=WINDOW_BG)
    panel = SystemChartsPanel(root, minutes=5.0)
    panel.pack(fill="both", expand=True)
    root.update()

    monitor = ProcessMonitor()
    system_history = SystemHistory()
    monitor.list_processes()

    print("collecting samples", end="", flush=True)
    snapshot = None
    for _ in range(6):
        time.sleep(0.4)
        snapshot = monitor.snapshot()
        system_history.add(snapshot.system)
        print(".", end="", flush=True)
    samples = system_history.samples()
    print(f"\n{len(snapshot.processes)} processes, {len(samples)} system samples\n")

    def timeit(label, function, repeats=15):
        function()
        root.update()
        best, total = 1e9, 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            root.update()
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
            total += elapsed
        print(f"  {label:<44} best {best * 1000:6.1f} ms   "
              f"mean {total / repeats * 1000:6.1f} ms")

    print("the naive way vs the way this module does it:")
    timeit("full repaint (canvas.draw)", lambda: panel._draw_full())
    timeit("update_charts() - blitted",
           lambda: panel.update_charts(samples, snapshot.processes))

    print("\nfor scale, work a refresh already did:")
    timeit("monitor.snapshot() (Phase 1)", monitor.snapshot, repeats=6)

    root.destroy()
