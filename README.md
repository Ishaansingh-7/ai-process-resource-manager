# AI-Based Process Resource Manager

Live Windows process monitor. It classifies running processes with a random
forest, flags ones behaving unlike their own history, projects short-term
resource trends, and suggests what to do about it. Suggestions only; nothing
is applied without confirmation.

Python, Tkinter, scikit-learn, matplotlib. CPU scheduling and Banker's
algorithm simulators are included, written from scratch.

```
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
```

## Quick start

```bash
git clone https://github.com/Ishaansingh-7/ai-process-resource-manager.git
cd ai-process-resource-manager

python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python main.py
```

The dashboard opens and starts collecting in the background. A trained model
(`models/resource_model.pkl`) is committed, so classification works on the
first run.

### Other ways to run it

```bash
python main.py --collect-only   # no window, just build the dataset
python main.py --no-collector   # dashboard only, don't write to the CSV
python main.py --interval 1     # snapshot every second
python main.py --raw-cpu        # CPU % per core instead of per machine
python main.py --help           # everything else

python model.py                 # retrain the classifier
python test_simulators.py       # unit tests for the two simulators
```

## The dataset

The collector appends one CSV row per process per snapshot to
`data/process_data.csv`. The real file grew to about 44 MB, so it's gitignored.
A 50,000-row sample is committed instead:

```
data/process_data_sample.csv
```

To train on the sample instead of collecting your own:

```bash
python model.py --csv data/process_data_sample.csv
```

Columns: `timestamp, pid, name, cpu_percent, memory_mb, thread_count,
priority, status, ctx_switches_vol, ctx_switches_invol, ctx_switches_per_sec`.

The three context-switch columns are empty on Windows. The OS exposes a single
total with no voluntary/involuntary split, and psutil's per-process call for it
costs seconds per refresh.

## How it works

Built in five phases, and the module layout still follows them.

### Phase 1 - monitoring

| Module | Role |
| --- | --- |
| `winprobe.py` | Windows process probe. One `NtQuerySystemInformation` call returns the whole process table in ~6 ms. |
| `monitor.py` | The only module that talks to the OS. Returns plain `ProcessInfo` / `SystemInfo` / `Snapshot` objects. |
| `collector.py` | Background thread appending snapshots to the CSV. A failed sample is logged, never fatal. |
| `history.py` | Short rolling windows (`deque`) of recent readings, per process and machine-wide. |

Why `winprobe.py` exists: asking psutil for memory, CPU time, thread count and
status triggers a per-process native call, and each one walks the entire
process table. Cost scales quadratically. Measured on ~330 processes:

```
psutil, all fields ................ ~4200 ms per refresh
winprobe.py ........................... ~6 ms per refresh
```

A 2.5-second refresh isn't possible with the first number. psutil still handles
system-wide totals and acts as the non-Windows fallback.

### Phase 2 - classification

`model.py` trains a `RandomForestClassifier` mapping
`(cpu_percent, memory_mb, thread_count)` to `NORMAL` / `CPU_INTENSIVE` /
`MEMORY_INTENSIVE`.

Labels come from a threshold rule rather than manual annotation, since nobody
was going to hand-label 500,000 rows. This is weak supervision. A random forest
fits the problem because decision trees split on thresholds like
`cpu_percent > 80`, and no feature scaling is needed.

`predictor.py` loads the saved model and classifies live processes. Missing
scikit-learn, a missing model file or an empty dataset all degrade to
"classifier unavailable" rather than raising, and the dashboard keeps showing
raw metrics.

### Phase 3 - anomalies and trends

`anomaly.py` runs two checks.

**3-sigma spike.** Per-program-name baselines (mean and standard deviation of
CPU and memory) are built from the CSV. A process is flagged when a reading
sits more than three standard deviations above that program's own mean.
Grouped by name rather than PID, since PIDs change on every restart while
`chrome.exe` should behave like `chrome.exe` did yesterday.

**Monotonic memory growth.** Five consecutive samples with no drop is the shape
of a leak. This reads the live rolling window rather than the CSV, since the
question is what's leaking now.

`predictor_trend.py` handles the forward-looking half. It fits a line through
each process's last 5-10 readings with `numpy.polyfit` and extrapolates to a
time-to-threshold:

```
seconds_to_limit = (limit - current_value) / slope      (slope > 0)
```

### Phase 4 - recommend, then ask

`recommender.py` reads all three analyses and produces a single
`Recommendation(action, reason, confidence)`.

```
classification (Phase 2) -+
anomaly check  (Phase 3) -+--> Recommendation
trend forecast (Phase 3) -+
```

It doesn't execute anything. The classifier's labels are statements about
numbers, not intent, and a video encoder pinned at 100% CPU is doing its job.

`process_manager.py` is the only module that changes system state (lower
priority, suspend, resume):

- Nothing runs without a button press the user already confirmed in a dialog.
- The protected-process list is enforced here, not in the UI. A GUI bug or a
  script importing this module still can't suspend `csrss.exe`.
- Every action re-checks the process name first. Windows recycles PIDs, and
  PID 4312 may be something else by the time Apply is clicked.
- Every attempt, refusals included, is appended to `data/action_log.csv`.

### Phase 5 - charts

`charts.py` embeds matplotlib in Tkinter: a rolling line chart of total CPU and
RAM, bar charts of the top five processes by CPU and memory, and a per-process
detail window on double-click.

A full matplotlib repaint costs ~61 ms against ~0.3 ms to hand the artists new
numbers, so the charts repaint on their own slower timer instead of on every
table refresh.

## The OS simulators

Both are written from scratch, import nothing from the UI, and run and test on
their own. `simulator_ui.py` holds their windows and nothing else.

**`scheduler.py`** - FCFS, SJF, Priority and Round Robin, measured with
completion, turnaround, waiting and response time.

**`deadlock.py`** - the Banker's algorithm. Avoidance (`is_safe`,
`request_resources`) and detection are separate functions, since the two get
confused often. Avoidance asks "if I grant this, can everybody still finish?"
before deadlock happens. Detection looks for a cycle that already exists.

`test_simulators.py` checks both against worked examples from Silberschatz,
Galvin and Gagne's *Operating System Concepts*, where the correct answer is
printed in the book. Alongside those are property tests: `waiting ==
turnaround - burst` under every algorithm, and the timeline accounting for
every unit of CPU time with no gaps or overlaps.

## Requirements

Python 3.10+ on Windows. It runs elsewhere through the psutil fallback, but
priority and status labels differ.

```
psutil        process and system metrics
pandas, numpy load and clean the dataset
scikit-learn  RandomForestClassifier
joblib        save / load the model
matplotlib    the embedded charts
```

Tkinter ships with the official Python Windows installer. Enable
"tcl/tk and IDLE" if `import tkinter` fails.

## Licence

MIT - see [LICENSE](LICENSE).
