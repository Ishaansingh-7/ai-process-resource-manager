"""
model.py
========
Phase 2: training the process classifier.

The machine-learning approach, in short
--------------------------------------
This is **supervised classification with rule-based (weak) labelling**:

1. Phase 1 logged real process measurements to ``data/process_data.csv``.
   Nobody hand-labelled those 20 000+ rows, so we generate the labels with a
   simple rule (the thresholds below).  This technique is called *weak
   supervision*: a cheap rule stands in for a human annotator.
2. A ``RandomForestClassifier`` then learns to map three features
   ``(cpu_percent, memory_mb, thread_count)`` onto one of three labels
   ``NORMAL / CPU_INTENSIVE / MEMORY_INTENSIVE``.
3. predictor.py loads the saved model and classifies live processes in the
   dashboard.

Why a random forest?  It is a bagged ensemble of decision trees, and a
decision tree splits on thresholds like ``cpu_percent > 80`` - exactly the
shape of this problem.  It needs no feature scaling, it is fast to train on
20 000 rows, and it gives us two things the dashboard uses directly: a class
probability (shown as the confidence %) and feature importances.

An honest note for the demo
---------------------------
Because the labels are produced by a rule that uses two of the three features,
the model is essentially *re-learning that rule*, so accuracy will come out at
or near 100 %.  That is expected, and it is a demonstration that the pipeline
works - not evidence that the model discovered anything new.  If you are asked
"why is your accuracy 100 %?", that is the answer.  The classifier becomes
genuinely useful once the labels come from something the features do not
already contain - for example labelling a process by what it did *next*
(future CPU usage), which turns this into prediction rather than description.

Run it directly to train:

    python model.py                      # thresholds from the project spec
    python model.py --auto-thresholds    # thresholds from this machine's data
    python model.py --cpu-threshold 5 --memory-threshold 5
"""

from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_DIR / "data" / "process_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "resource_model.pkl"

# The three input columns, in the order the model expects them.  This order is
# saved with the model so predictor.py can never feed the columns in the wrong
# sequence.
FEATURES: List[str] = ["cpu_percent", "memory_mb", "thread_count"]

LABEL_NORMAL = "NORMAL"
LABEL_CPU = "CPU_INTENSIVE"
LABEL_MEMORY = "MEMORY_INTENSIVE"
LABELS: List[str] = [LABEL_NORMAL, LABEL_CPU, LABEL_MEMORY]

# Labelling rule from the project specification.
#   cpu_percent > 80          -> CPU_INTENSIVE
#   memory_mb   > 80 % of RAM -> MEMORY_INTENSIVE
#   otherwise                 -> NORMAL
CPU_THRESHOLD_PERCENT = 80.0
MEMORY_THRESHOLD_FRACTION = 0.80

# Used by --auto-thresholds: a process counts as "intensive" when it lands in
# the top 0.5 % of everything this machine was observed doing.
AUTO_QUANTILE = 0.995

RANDOM_STATE = 42       # fixed so training is reproducible for the demo
TEST_SIZE = 0.2         # 80 / 20 train-test split
N_ESTIMATORS = 100      # number of trees in the forest


# ---------------------------------------------------------------------------
# Loading the dataset
# ---------------------------------------------------------------------------


def total_ram_mb() -> float:
    """Total physical RAM, used to turn the memory rule into an absolute MB value."""
    return psutil.virtual_memory().total / (1024 * 1024)


def load_dataset(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Read data/process_data.csv into a DataFrame.

    Raises FileNotFoundError with an actionable message if Phase 1 has not
    collected anything yet.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No dataset at {csv_path}.\n"
            "Collect some data first:  python main.py --collect-only")

    frame = pd.read_csv(csv_path)

    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing column(s): {missing}")

    # A row is only usable if all three features are present and numeric.
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=FEATURES).reset_index(drop=True)
    if len(frame) < before:
        log.warning("Dropped %d row(s) with unreadable values.", before - len(frame))

    if frame.empty:
        raise ValueError(f"{csv_path} contains no usable rows.")
    return frame


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def make_labels(
    frame: pd.DataFrame,
    cpu_threshold: float = CPU_THRESHOLD_PERCENT,
    memory_threshold_mb: Optional[float] = None,
) -> pd.Series:
    """
    Apply the rule-based labelling.

    ``np.select`` evaluates the conditions in order, so a process that is over
    *both* thresholds is labelled CPU_INTENSIVE - the CPU rule is listed first
    in the specification and therefore wins.
    """
    if memory_threshold_mb is None:
        memory_threshold_mb = MEMORY_THRESHOLD_FRACTION * total_ram_mb()

    conditions = [
        frame["cpu_percent"] > cpu_threshold,
        frame["memory_mb"] > memory_threshold_mb,
    ]
    return pd.Series(
        np.select(conditions, [LABEL_CPU, LABEL_MEMORY], default=LABEL_NORMAL),
        index=frame.index,
        name="label",
    )


def auto_thresholds(frame: pd.DataFrame,
                    quantile: float = AUTO_QUANTILE) -> Tuple[float, float]:
    """
    Derive thresholds from the data instead of using fixed numbers.

    The specification's absolute thresholds (80 % CPU, 80 % of RAM) describe a
    machine under extreme load.  On an ordinary desktop no process ever gets
    there, so every row would be labelled NORMAL and the classifier would have
    nothing to learn.  This picks a high percentile of what *this* machine was
    actually observed doing, which keeps the same idea - "unusually hungry
    process" - while producing all three classes.
    """
    cpu = float(frame["cpu_percent"].quantile(quantile))
    memory = float(frame["memory_mb"].quantile(quantile))
    # Never return 0: with a mostly idle machine the percentile can be 0.0,
    # which would label every non-zero row as CPU_INTENSIVE.
    return max(cpu, 0.01), max(memory, 1.0)


def describe_dataset(frame: pd.DataFrame, labels: pd.Series,
                     cpu_threshold: float, memory_threshold_mb: float) -> None:
    """Print what the data looks like - useful when the labels come out lopsided."""
    ram = total_ram_mb()
    print(f"Dataset      : {len(frame):,} rows, "
          f"{frame['timestamp'].nunique() if 'timestamp' in frame else '?'} snapshots")
    print(f"Thresholds   : cpu_percent > {cpu_threshold:g} %   "
          f"memory_mb > {memory_threshold_mb:,.0f} MB "
          f"({memory_threshold_mb / ram * 100:.1f} % of {ram:,.0f} MB RAM)")
    print("Feature range:")
    for column in FEATURES:
        series = frame[column]
        print(f"    {column:<13} min={series.min():>10,.2f}  "
              f"median={series.median():>10,.2f}  "
              f"p99={series.quantile(0.99):>10,.2f}  max={series.max():>10,.2f}")
    print("Labels       :")
    counts = labels.value_counts()
    for name in LABELS:
        count = int(counts.get(name, 0))
        print(f"    {name:<17} {count:>8,}  ({count / len(labels) * 100:5.2f} %)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(frame: pd.DataFrame, labels: pd.Series,
                test_size: float = TEST_SIZE,
                random_state: int = RANDOM_STATE) -> Tuple[RandomForestClassifier, Dict]:
    """
    Fit the random forest and measure it on data it has never seen.

    Returns the fitted model plus a dict of metrics for printing / saving.
    """
    # .to_numpy() gives a plain array, so the model stores no pandas column
    # names.  The feature *order* is what matters, and it is saved alongside
    # the model in the bundle below.
    features = frame[FEATURES].to_numpy(dtype=float)
    targets = labels.to_numpy()

    classes = sorted(set(targets))
    counts = {name: int((targets == name).sum()) for name in classes}

    # Stratifying keeps the class proportions identical in the train and test
    # halves, but it needs at least two examples of every class.
    stratify = targets if len(classes) > 1 and min(counts.values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size,
        random_state=random_state, stratify=stratify)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,              # use every core; training is embarrassingly parallel
    )
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    # With a single class scikit-learn warns that the confusion matrix will be
    # 1x1.  That situation is already reported far more clearly by main(), so
    # the duplicate warning is suppressed here and *only* here.
    with warnings.catch_warnings():
        if len(classes) < 2:
            warnings.simplefilter("ignore", UserWarning)
        report = classification_report(y_test, test_predictions, zero_division=0)
        confusion = confusion_matrix(y_test, test_predictions, labels=classes)

    metrics = {
        "n_samples": len(frame),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "classes": classes,
        "class_counts": counts,
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "report": report,
        "confusion": confusion,
        "importances": dict(zip(FEATURES, model.feature_importances_)),
    }
    return model, metrics


def save_bundle(model: RandomForestClassifier, path: Path,
                cpu_threshold: float, memory_threshold_mb: float,
                metrics: Dict) -> Path:
    """
    Save the model *and everything needed to use it correctly*.

    Pickling the bare classifier would lose the feature order and the
    thresholds it was trained with, so we store a small dictionary instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "labels": LABELS,
        "cpu_threshold": cpu_threshold,
        "memory_threshold_mb": memory_threshold_mb,
        "total_ram_mb": total_ram_mb(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "metrics": {k: v for k, v in metrics.items() if k != "confusion"},
    }
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path = DEFAULT_MODEL_PATH) -> Dict:
    """Load a bundle saved by :func:`save_bundle`, checking it is usable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not a model bundle from model.py")

    # A model pickled by a different scikit-learn build usually still works,
    # but a warning makes a confusing failure much easier to diagnose.
    saved_version = bundle.get("sklearn_version")
    if saved_version and saved_version != sklearn.__version__:
        log.warning("Model was trained with scikit-learn %s, running %s.",
                    saved_version, sklearn.__version__)
    return bundle


def train_and_save(csv_path: Path = DEFAULT_CSV_PATH,
                   model_path: Path = DEFAULT_MODEL_PATH,
                   cpu_threshold: float = CPU_THRESHOLD_PERCENT,
                   memory_threshold_mb: Optional[float] = None,
                   fallback_to_auto: bool = True,
                   quiet: bool = False) -> Dict:
    """
    The whole pipeline in one call: load -> label -> train -> save.

    predictor.py calls this when models/resource_model.pkl does not exist yet,
    which is what makes the "Training model..." path work automatically.

    ``fallback_to_auto`` covers the common case where the specification's
    absolute thresholds put every row in one class on an ordinary machine.
    Rather than saving a model that can only ever answer NORMAL, this switches
    to data-derived thresholds and says so.  (``python model.py`` does *not*
    do this - run directly, it honours the thresholds you asked for and prints
    a warning instead.)
    """
    frame = load_dataset(csv_path)
    if memory_threshold_mb is None:
        memory_threshold_mb = MEMORY_THRESHOLD_FRACTION * total_ram_mb()

    labels = make_labels(frame, cpu_threshold, memory_threshold_mb)

    if labels.nunique() < 2 and fallback_to_auto:
        cpu_threshold, memory_threshold_mb = auto_thresholds(frame)
        labels = make_labels(frame, cpu_threshold, memory_threshold_mb)
        log.warning(
            "Every row labelled '%s' at the specified thresholds; using "
            "data-derived ones instead (cpu > %.2f %%, memory > %.0f MB).",
            LABEL_NORMAL, cpu_threshold, memory_threshold_mb)

    model, metrics = train_model(frame, labels)
    saved_to = save_bundle(model, model_path, cpu_threshold, memory_threshold_mb, metrics)

    if not quiet:
        log.info("Trained on %d rows (test accuracy %.4f) -> %s",
                 metrics["n_samples"], metrics["test_accuracy"], saved_to)
    return metrics


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model.py",
        description="Train the process classifier on data/process_data.csv.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), metavar="PATH",
                        help="dataset to train on")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), metavar="PATH",
                        help="where to save the trained model")
    parser.add_argument("--cpu-threshold", type=float, default=CPU_THRESHOLD_PERCENT,
                        metavar="PCT",
                        help=f"cpu_percent above this is CPU_INTENSIVE "
                             f"(default: {CPU_THRESHOLD_PERCENT:g})")
    parser.add_argument("--memory-threshold", type=float,
                        default=MEMORY_THRESHOLD_FRACTION * 100, metavar="PCT",
                        help=f"memory above this %% of total RAM is MEMORY_INTENSIVE "
                             f"(default: {MEMORY_THRESHOLD_FRACTION * 100:g})")
    parser.add_argument("--auto-thresholds", action="store_true",
                        help="derive both thresholds from the data instead "
                             f"(top {(1 - AUTO_QUANTILE) * 100:g}%% of observed usage)")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    frame = load_dataset(args.csv)

    if args.auto_thresholds:
        cpu_threshold, memory_threshold_mb = auto_thresholds(frame)
        print(f"Auto thresholds from the top {(1 - AUTO_QUANTILE) * 100:g} % "
              f"of observed usage.\n")
    else:
        cpu_threshold = args.cpu_threshold
        memory_threshold_mb = args.memory_threshold / 100 * total_ram_mb()

    labels = make_labels(frame, cpu_threshold, memory_threshold_mb)

    print("=" * 68)
    describe_dataset(frame, labels, cpu_threshold, memory_threshold_mb)
    print("=" * 68)

    # A classifier needs at least two classes to learn anything.  Rather than
    # silently saving a model that can only ever answer NORMAL, say so.
    present = labels.nunique()
    if present < 2:
        cpu_suggestion, memory_suggestion = auto_thresholds(frame)
        print("\n*** WARNING: every row got the same label "
              f"({labels.iloc[0]}). ***\n"
              "The thresholds are higher than anything in this dataset, so the\n"
              "model can only ever predict one class.  Either collect data while\n"
              "the machine is genuinely busy, or lower the thresholds:\n\n"
              f"    python model.py --auto-thresholds\n"
              f"    python model.py --cpu-threshold {cpu_suggestion:.2f} "
              f"--memory-threshold {memory_suggestion / total_ram_mb() * 100:.2f}\n")

    model, metrics = train_model(frame, labels)

    print(f"\nTrained a random forest of {N_ESTIMATORS} trees on "
          f"{metrics['n_train']:,} rows, tested on {metrics['n_test']:,}.\n")
    print(f"Training accuracy : {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy     : {metrics['test_accuracy']:.4f}")

    print("\nClassification report (held-out test set)")
    print("-" * 68)
    print(metrics["report"])

    print("Confusion matrix (rows = actual, columns = predicted)")
    print("-" * 68)
    classes = metrics["classes"]
    print(f"{'':<18}" + "".join(f"{c:>18}" for c in classes))
    for name, row in zip(classes, metrics["confusion"]):
        print(f"{name:<18}" + "".join(f"{int(v):>18,}" for v in row))

    print("\nFeature importance (how much each column drove the decisions)")
    print("-" * 68)
    for name, importance in sorted(metrics["importances"].items(),
                                   key=lambda kv: kv[1], reverse=True):
        bar = "#" * int(round(importance * 40))
        print(f"    {name:<15} {importance:6.3f}  {bar}")

    saved_to = save_bundle(model, args.model, cpu_threshold,
                           memory_threshold_mb, metrics)
    print(f"\nSaved model -> {saved_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
