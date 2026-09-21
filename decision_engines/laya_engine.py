"""Adapter for Convai's Laya, run locally through the laya-mlx MLX port.

No API key, no network call at inference time -- everything runs on-device
(Apple Silicon). https://github.com/mizorewww/laya-mlx
"""

from __future__ import annotations

import time
from typing import Any, Dict

from .base import DecisionEngine, DecisionResult

DEFAULT_CHECKPOINT = "convaiinnovations/laya"


def _normalize_answer(raw: Any) -> Dict[str, Any]:
    """laya-mlx answers mirror the same {type, choice/score/noul, probabilities,
    confidence} shape Jev uses. Accept a couple of plausible key spellings so a
    minor version difference in the library doesn't hard-crash the benchmark.
    """
    if not isinstance(raw, dict):
        return {"type": None}
    qtype = raw.get("type")
    out: Dict[str, Any] = {"type": qtype, "probabilities": raw.get("probabilities")}
    if "confidence" in raw:
        out["confidence"] = raw["confidence"]
    if qtype == "choice":
        out["choice"] = raw.get("choice", raw.get("value", raw.get("answer")))
    elif qtype == "score":
        out["score"] = raw.get("score", raw.get("value"))
        if "legend" in raw:
            out["legend"] = raw["legend"]
    elif qtype == "noul":
        out["noul"] = raw.get("noul", raw.get("probability", raw.get("value")))
    return out


class LayaEngine(DecisionEngine):
    name = "laya"

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT):
        import laya_mlx as laya  # imported lazily: mlx is Apple-Silicon-only

        t0 = time.perf_counter()
        self._agent = laya.load(checkpoint)
        self.load_time_s = time.perf_counter() - t0
        self.checkpoint = checkpoint

    def decide(self, state: str, questions: Dict[str, Any]) -> DecisionResult:
        t0 = time.perf_counter()
        try:
            raw = self._agent.predict(state, questions)
        except Exception as exc:  # local inference: surface and keep the run going
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return DecisionResult(
                engine=self.name, answers={}, latency_ms=latency_ms, error=str(exc)
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_answers = raw.get("answers", raw) if isinstance(raw, dict) else {}
        answers = {qid: _normalize_answer(a) for qid, a in raw_answers.items()}

        return DecisionResult(
            engine=self.name,
            answers=answers,
            latency_ms=latency_ms,
            usage={"cost_usd": 0.0},  # local inference, no per-call API cost
        )
