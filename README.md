# AI-Based Process Resource Manager

A live Windows process monitor that classifies what every running process is
doing, flags the ones behaving abnormally, forecasts where they are heading,
and recommends an action — without ever taking that action on its own.

Built in Python with Tkinter, scikit-learn and matplotlib. The two classic OS
simulators (CPU scheduling and the Banker's algorithm) are implemented from
scratch and shipped alongside it.

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
git clone https://github.com/<you>/AI-resource-manager.git
cd AI-resource-manager

python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python main.py
```

The dashboard opens and starts collecting data in the background. A trained
model (`models/resource_model.pkl`) is committed, so classification works on
the first run.

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
`data/process_data.csv`. The real file grew to ~44 MB, so it is gitignored;
a 50,000-row sample is committed instead:

```
data/process_data_sample.csv
```

To train on the sample rather than waiting to collect your own:

```bash
python model.py --csv data/process_data_sample.csv
```

Columns are `timestamp, pid, name, cpu_percent, memory_mb, thread_count,
priority, status, ctx_switches_vol, ctx_switches_invol, ctx_switches_per_sec`.
The three context-switch columns are empty on Windows — the OS exposes a
single total with no voluntary/involuntary split, and psutil's per-process
call for it costs seconds per refresh.

## How it works

The project was built in five phases, and the modules still map onto them.

### Phase 1 — monitoring

| Module | Role |
| --- | --- |
| `winprobe.py` | Fast Windows process probe: **one** `NtQuerySystemInformation` call returns the whole process table in ~6 ms. |
| `monitor.py` | The only module that talks to the OS. Returns plain `ProcessInfo` / `SystemInfo` / `Snapshot` objects. |
| `collector.py` | Background thread appending snapshots to the CSV. Deliberately boring: a failed sample is logged, never fatal. |
| `history.py` | Short rolling windows (`deque`) of recent readings, per process and machine-wide. |

The reason `winprobe.py` exists: asking psutil for memory, CPU time, thread
count and status forces a *per-process* native call, and each one walks the
entire process table — so the cost is quadratic. Measured on ~330 processes:

```
psutil, all fields ................ ~4200 ms per refresh
winprobe.py ........................... ~6 ms per refresh
```

A 2.5-second refresh cannot afford the first number. psutil is still used for
system-wide totals and as the portable non-Windows fallback.

### Phase 2 — classification

`model.py` trains a `RandomForestClassifier` to map
`(cpu_percent, memory_mb, thread_count)` onto `NORMAL` / `CPU_INTENSIVE` /
`MEMORY_INTENSIVE`.

Nobody hand-labelled 500,000 rows, so the labels come from a threshold rule —
*weak supervision*, where a cheap rule stands in for a human annotator. A
random forest suits the problem because a decision tree splits on thresholds
like `cpu_percent > 80`, which is exactly the shape of the data; it also needs
no feature scaling.

`predictor.py` loads the saved model and classifies live processes. It never
raises: missing scikit-learn, missing model file or empty dataset all degrade
to "classifier unavailable" and the dashboard carries on showing raw metrics.

### Phase 3 — anomalies and trends

`anomaly.py` runs two independent checks:

1. **3-sigma spike.** A per-*program-name* baseline (mean and standard
   deviation of CPU and memory) is built from the CSV; a process is flagged
   when it sits more than three standard deviations above its own historical
   mean. Grouping by name, not PID, is deliberate — PIDs change on every
   restart, but `chrome.exe` should behave like `chrome.exe` did yesterday.
2. **Monotonic memory growth.** Five consecutive live samples with no drop at
   all is the shape of a leak. This reads the live rolling window, not the
   CSV, because the question is what is leaking *now*.

`predictor_trend.py` answers the forward-looking question. It fits a straight
line through each process's last 5–10 readings with `numpy.polyfit`, and
extrapolates to a time-to-threshold:

```
seconds_to_limit = (limit - current_value) / slope      (slope > 0)
```

### Phase 4 — recommend, then ask

`recommender.py` reads all three analyses and produces a single
`Recommendation(action, reason, confidence)`.

```
classification (Phase 2) ─┐
anomaly check  (Phase 3) ─┼─► Recommendation
trend forecast (Phase 3) ─┘
```

It deliberately never acts. The classifier's labels are statements about
numbers, not intent — a video encoder pinned at 100% CPU is doing its job.

`process_manager.py` is the only module that changes anything (lower priority,
suspend, resume), and it is written to be paranoid:

- Nothing runs without a button press the user already confirmed in a dialog.
- The protected-process list is enforced *here*, not in the UI — a GUI bug, or
  a script importing this module, still cannot suspend `csrss.exe`.
- Every action re-checks the process name first, because Windows recycles PIDs
  and PID 4312 may be something else by the time Apply is clicked.
- Every attempt, including refusals, is appended to `data/action_log.csv`.

### Phase 5 — charts

`charts.py` embeds matplotlib in Tkinter: a rolling line chart of total CPU and
RAM, bar charts of the top five processes by CPU and memory, and a per-process
detail window on double-click.

A full matplotlib repaint costs ~61 ms against ~0.3 ms to hand the artists new
numbers, so the charts repaint on their own slower timer rather than on every
table refresh.

## The OS simulators

Both are implemented from scratch, import nothing from the UI, and can be run
and tested on their own. `simulator_ui.py` holds their windows and nothing else.

- **`scheduler.py`** — FCFS, SJF, Priority and Round Robin, measured with
  completion / turnaround / waiting / response time.
- **`deadlock.py`** — the Banker's algorithm. Avoidance (`is_safe`,
  `request_resources`) and detection are kept as separate functions, because
  confusing the two is the classic mistake: avoidance asks "if I grant this,
  can everybody still finish?" *before* deadlock happens; detection looks for
  a cycle that already exists.

`test_simulators.py` checks both against worked examples from Silberschatz,
Galvin & Gagne's *Operating System Concepts*, where the correct answer is
printed in the book — an outside authority rather than a snapshot of whatever
the code happens to do today. Alongside those are property tests, e.g.
`waiting == turnaround - burst` under every algorithm, and that the timeline
accounts for every unit of CPU time with no gaps or overlaps.

## Requirements

Python 3.10+ on Windows (it runs elsewhere via the psutil fallback, but
priority and status labels differ).

```
psutil        process and system metrics
pandas, numpy load and clean the dataset
scikit-learn  RandomForestClassifier
joblib        save / load the model
matplotlib    the embedded charts
```

Tkinter ships with the official Python Windows installer — enable
"tcl/tk and IDLE" if `import tkinter` fails.

## Licence

MIT — see [LICENSE](LICENSE).
