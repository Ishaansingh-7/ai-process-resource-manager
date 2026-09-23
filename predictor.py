"""
predictor.py
============
Uses the trained model from model.py to classify *live* processes.

This is the bridge between the machine-learning code and the dashboard.  It
deliberately never raises: if scikit-learn is missing, if the model file has
not been created yet, or if the dataset is empty, the classifier simply
reports that it is unavailable and the dashboard carries on showing raw
metrics.  A monitoring tool that crashes because a model file is missing would
be a bad monitoring tool.

Typical use (this is what gui.py does)::

    classifier = ProcessClassifier()          # trains automatically if needed
    predictions = classifier.predict_many(snapshot.processes)
    predictions[pid].label        # 'NORMAL' / 'CPU_INTENSIVE' / 'MEMORY_INTENSIVE'
    predictions[pid].confidence   # 0-100
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

log = logging.getLogger(__name__)

# The ML stack is an optional dependency as far as the dashboard is concerned:
# Phase 1 works perfectly well without it.  Importing it defensively keeps
# `python main.py` working on a machine where only psutil is installed.
try:
    import numpy as np

    import model

    ML_AVAILABLE = True
    ML_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    np = None            # type: ignore[assignment]
    model = None         # type: ignore[assignment]
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(exc)


UNKNOWN_LABEL = "UNKNOWN"


class Prediction(NamedTuple):
    """What the model thinks about one process."""

    label: str          # NORMAL / CPU_INTENSIVE / MEMORY_INTENSIVE
    confidence: float   # 0-100, the probability the forest gave the winning class


class ProcessClassifier:
    """
    Loads models/resource_model.pkl and classifies live processes.

    Check :attr:`available` before trusting the output; :attr:`status` holds a
    short human-readable explanation that the dashboard shows in its status
    bar.
    """

    def __init__(self,
                 model_path: Optional[Path] = None,
                 csv_path: Optional[Path] = None,
                 auto_train: bool = True) -> None:
        self.available = False
        self.status = "not loaded"
        self.model_path = Path(model_path) if model_path else (
            model.DEFAULT_MODEL_PATH if ML_AVAILABLE else None)
        self.csv_path = Path(csv_path) if csv_path else (
            model.DEFAULT_CSV_PATH if ML_AVAILABLE else None)

        self._model = None
        self._features: Sequence[str] = ()
        self._classes: Sequence[str] = ()
        self._bundle: Optional[dict] = None

        self._load(auto_train=auto_train)

    # ------------------------------------------------------------------
    # Start-up
    # ------------------------------------------------------------------

    def _load(self, auto_train: bool) -> None:
        """Load the saved model, training one first if there is none yet."""
        if not ML_AVAILABLE:
            self.status = "scikit-learn not installed (pip install -r requirements.txt)"
            log.warning("ML libraries unavailable: %s", ML_IMPORT_ERROR)
            return

        bundle = self._read_bundle()
        if bundle is None and auto_train:
            # This is the "graceful first run" path: no model on disk yet, so
            # build one from whatever Phase 1 has collected.
            print("Training model...")
            try:
                model.train_and_save(self.csv_path, self.model_path, quiet=True)
            except FileNotFoundError:
                self.status = "no model yet - run: python main.py --collect-only"
                log.warning("Cannot train: %s does not exist.", self.csv_path)
                return
            except Exception as exc:
                self.status = f"training failed: {exc}"
                log.warning("Training failed: %s", exc)
                return
            bundle = self._read_bundle()

        if bundle is None:
            self.status = "no model - run: python model.py"
            return

        self._bundle = bundle
        self._model = bundle["model"]
        self._features = bundle.get("features", model.FEATURES)
        self._classes = list(self._model.classes_)
        self.available = True

        # Training used every core (n_jobs=-1), which is right for 18 000 rows.
        # Predicting ~350 rows is the opposite case: handing work to a thread
        # pool costs more than the work itself.  Measured here, 32 ms parallel
        # vs 4 ms on one thread, so predictions run single-threaded.
        self._model.n_jobs = 1

        trained_at = bundle.get("trained_at", "unknown date")
        self.status = f"{len(self._classes)} classes, trained {trained_at}"
        log.info("Loaded classifier from %s (%s)", self.model_path, self.status)

        # A model that only knows one label will label everything the same
        # way.  Say so rather than letting the demo look convincing.
        if len(self._classes) < 2:
            self.status = (f"only knows '{self._classes[0]}' - "
                           "retrain with: python model.py --auto-thresholds")
            log.warning("The model was trained on a single class (%s).", self._classes[0])

    def _read_bundle(self) -> Optional[dict]:
        """Try to load the pickle; return None if it is missing or unusable."""
        try:
            return model.load_bundle(self.model_path)
        except FileNotFoundError:
            return None
        except Exception as exc:
            log.warning("Could not read %s (%s); it will be retrained.",
                        self.model_path, exc)
            return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_many(self, processes: Sequence) -> Dict[int, Prediction]:
        """
        Classify a whole snapshot at once, keyed by PID.

        One batched call to the forest for ~350 processes takes a couple of
        milliseconds; calling predict() in a loop would be far slower, and the
        dashboard does this on every refresh.
        """
        if not self.available or not processes:
            return {}

        try:
            # Columns are pulled by name in the order stored with the model,
            # so the features can never end up shuffled.
            matrix = np.array(
                [[getattr(p, name) for name in self._features] for p in processes],
                dtype=float)

            # predict_proba gives one probability per class; the largest is
            # both the predicted label and the confidence we display.
            probabilities = self._model.predict_proba(matrix)
            winners = probabilities.argmax(axis=1)

            return {
                process.pid: Prediction(
                    label=str(self._classes[winner]),
                    confidence=float(probabilities[row, winner] * 100.0),
                )
                for row, (process, winner) in enumerate(zip(processes, winners))
            }
        except Exception as exc:  # never take the dashboard down
            log.warning("Prediction failed: %s", exc)
            return {}

    def predict(self, process) -> Prediction:
        """Classify a single ProcessInfo."""
        result = self.predict_many([process])
        return result.get(process.pid, Prediction(UNKNOWN_LABEL, 0.0))

    # ------------------------------------------------------------------
    # Introspection (used by the demo output below)
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """One-line summary of the loaded model."""
        if not self.available:
            return f"classifier unavailable: {self.status}"
        bundle = self._bundle or {}
        metrics = bundle.get("metrics", {})
        return (f"random forest, classes={list(self._classes)}, "
                f"features={list(self._features)}, "
                f"trained on {metrics.get('n_samples', '?')} rows, "
                f"test accuracy {metrics.get('test_accuracy', float('nan')):.4f}")


# ---------------------------------------------------------------------------
# Manual test:  python predictor.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from monitor import ProcessMonitor

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    classifier = ProcessClassifier()
    print(classifier.describe(), "\n")

    monitor = ProcessMonitor()
    monitor.list_processes()      # first pass establishes the CPU baseline
    time.sleep(1.0)
    processes = monitor.list_processes()

    started = time.perf_counter()
    predictions = classifier.predict_many(processes)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"classified {len(processes)} processes in {elapsed_ms:.1f} ms\n")

    print(f"{'PID':>7}  {'NAME':<28} {'CPU%':>6} {'MEM(MB)':>10} {'THR':>4}  "
          f"{'PREDICTION':<17} CONF")
    for process in processes[:15]:
        prediction = predictions.get(process.pid, Prediction(UNKNOWN_LABEL, 0.0))
        print(f"{process.pid:>7}  {process.name[:28]:<28} {process.cpu_percent:>6.1f} "
              f"{process.memory_mb:>10,.1f} {process.thread_count:>4}  "
              f"{prediction.label:<17} {prediction.confidence:5.1f} %")

    counts: Dict[str, int] = {}
    for prediction in predictions.values():
        counts[prediction.label] = counts.get(prediction.label, 0) + 1
    print("\nacross all processes:", counts or "no predictions")
