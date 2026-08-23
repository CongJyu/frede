"""Server-side two-phase Human-in-the-Loop session + metrics.

Port of notebook Phase 4 Parts 2–3. Evaluators first review a balanced sample
pool WITHOUT any explanation, then review the SAME pool WITH XAI highlights and
reason codes. Judgments are persisted to CSV; results aggregate human accuracy
(vs ground truth) and Cohen's κ (human vs model) per mode.

The sample pool is loaded lazily so the rest of the app (Analyzer, Examples)
works even before `make train` has produced it.
"""
from __future__ import annotations

import csv
import json
import threading
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from .. import config

_CSV_COLUMNS = [
    "sample_idx", "original_idx", "true_label", "model_pred", "human_judgment",
    "human_confidence", "mode", "time_seconds", "feedback", "fake_prob",
]


class HitlUnavailableError(RuntimeError):
    """Raised when the HITL sample pool has not been built yet."""


class HitlService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pool: list[dict] | None = None
        self.mode = "no_xai"          # "no_xai" -> "with_xai"
        self.current_idx = 0
        self.start_time = time.time()
        self.judgments: list[dict] = []
        self._ensure_csv_header()

    # ------------------------------------------------------------------ setup

    def _ensure_pool(self) -> None:
        if self.pool is not None:
            return
        if not config.HITL_POOL_PATH.exists():
            raise HitlUnavailableError(
                f"HITL pool missing at {config.HITL_POOL_PATH}. Run `make train` first."
            )
        with open(config.HITL_POOL_PATH, encoding="utf-8") as f:
            self.pool = json.load(f)["samples"]

    def _ensure_csv_header(self) -> None:
        if not config.HITL_RESULTS_PATH.exists():
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.HITL_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_COLUMNS)

    # --------------------------------------------------------------- sessions

    def session(self) -> dict:
        with self._lock:
            self._ensure_pool()
            return self._session_locked()

    def _session_locked(self) -> dict:
        total = len(self.pool)
        done = self.mode == "with_xai" and self.current_idx >= total
        sample = None

        if not done:
            s = self.pool[self.current_idx]
            sample = {
                "idx": s["idx"],
                "text": s["text"],
                # true_label / model_pred intentionally omitted — evaluators
                # must not see the answer; metrics are computed server-side.
            }
            if self.mode == "with_xai":
                sample.update(
                    {
                        "highlighted_html": s["highlighted_html"],
                        "reason_codes": s["reason_codes"],
                        "summary": s["summary"],
                    }
                )

        status = "Ready"
        if done:
            status = "All evaluations complete! View the results."
        return {
            "mode": self.mode,
            "pass_label": "WITHOUT XAI (baseline)" if self.mode == "no_xai" else "WITH XAI",
            "pass_number": 1 if self.mode == "no_xai" else 2,
            "in_pass_idx": self.current_idx,
            "total": total,
            "done": done,
            "status": status,
            "sample": sample,
        }

    def submit_judgment(self, judgment: str, confidence: int, feedback: str) -> dict:
        with self._lock:
            self._ensure_pool()
            s = self.pool[self.current_idx]
            elapsed = time.time() - self.start_time
            record = {
                "sample_idx": self.current_idx,
                "original_idx": s["idx"],
                "true_label": s["true_label"],
                "model_pred": s["model_pred"],
                "human_judgment": 1 if judgment == "Fake" else 0,
                "human_confidence": confidence,
                "mode": self.mode,
                "time_seconds": round(elapsed, 1),
                "feedback": feedback,
                "fake_prob": s["fake_prob"],
            }
            self.judgments.append(record)
            self._append_csv(record)

            self.current_idx += 1
            self.start_time = time.time()

            if self.current_idx >= len(self.pool) and self.mode == "no_xai":
                self.mode = "with_xai"
                self.current_idx = 0
                self.start_time = time.time()
                status = "Baseline phase complete! Now reviewing WITH XAI explanations."
            elif self.current_idx >= len(self.pool):
                status = "All evaluations complete! Check the results below."
            else:
                status = f"Recorded. Moving to sample {self.current_idx + 1}."

            sess = self._session_locked()
            sess["status"] = status
            return sess

    # ----------------------------------------------------------------- results

    def results(self) -> dict:
        with self._lock:
            self._ensure_pool()
            return self._results_locked()

    def _results_locked(self) -> dict:
        if not self.judgments:
            return {"mode_metrics": {}, "records": [], "summary": "No results yet."}

        df = pd.DataFrame(self.judgments)
        mode_metrics: dict[str, dict] = {}
        for mode in ("no_xai", "with_xai"):
            mode_df = df[df["mode"] == mode]
            if mode_df.empty:
                continue
            human = mode_df["human_judgment"].values
            truth = mode_df["true_label"].values
            model = mode_df["model_pred"].values
            mode_metrics[mode] = {
                "label": "Without XAI" if mode == "no_xai" else "With XAI",
                "count": int(len(mode_df)),
                "human_accuracy": float(accuracy_score(truth, human)),
                "human_precision": float(precision_score(truth, human, zero_division=0)),
                "human_recall": float(recall_score(truth, human, zero_division=0)),
                "human_f1": float(f1_score(truth, human, zero_division=0)),
                "model_accuracy": float(accuracy_score(truth, model)),
                "kappa": float(cohen_kappa_score(model, human)),
                "avg_time": float(mode_df["time_seconds"].mean()),
                "avg_confidence": float(mode_df["human_confidence"].mean()),
            }

        summary = self._summarize(mode_metrics)
        return {"mode_metrics": mode_metrics, "records": self.judgments, "summary": summary}

    @staticmethod
    def _summarize(metrics: dict[str, dict]) -> str:
        if len(metrics) < 2:
            return "Complete both phases to see the XAI comparison."
        without = metrics["no_xai"]
        with_xai = metrics["with_xai"]
        delta_acc = with_xai["human_accuracy"] - without["human_accuracy"]
        delta_f1 = with_xai["human_f1"] - without["human_f1"]
        acc_dir = "improved" if delta_acc > 0.001 else ("declined" if delta_acc < -0.001 else "was unchanged")
        f1_dir = "improved" if delta_f1 > 0.001 else ("declined" if delta_f1 < -0.001 else "was unchanged")
        return (
            f"Human accuracy {acc_dir} by **{abs(delta_acc):.1%}** and F1 {f1_dir} "
            f"by **{abs(delta_f1):.1%}** when XAI explanations were shown "
            f"(with-XAI decision time {with_xai['avg_time']:.1f}s vs "
            f"{without['avg_time']:.1f}s)."
        )

    # -------------------------------------------------------------- persistence

    def _append_csv(self, record: dict) -> None:
        with open(config.HITL_RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([record.get(c, "") for c in _CSV_COLUMNS])

    def reset(self) -> None:
        with self._lock:
            self.mode = "no_xai"
            self.current_idx = 0
            self.start_time = time.time()
            self.judgments = []
            if config.HITL_RESULTS_PATH.exists():
                config.HITL_RESULTS_PATH.unlink()
            self._ensure_csv_header()


_service = HitlService()


def get_hitl_service() -> HitlService:
    return _service
