"""Shared interface both decision engines (Jev and Laya) implement.

Both TypeSafe's Jev and Convai's Laya speak the same typed-decision
vocabulary -- a question is one of `choice` / `score` / `noul` -- which is
what makes a head-to-head comparison meaningful: the exact same
`questions` dict (see dataset/schema.py) is sent to both engines unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DecisionResult:
    engine: str
    answers: Dict[str, Any]  # question_id -> raw answer dict {"type", "choice"/"score"/"noul", "confidence", "probabilities"}
    latency_ms: float
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def value(self, question_id: str) -> Any:
        """Normalize one answer to a plain comparable value.

        choice -> the option string
        score  -> nearest integer level (rounded)
        noul   -> bool, thresholded at p >= 0.5
        """
        answer = self.answers.get(question_id)
        if not answer:
            return None
        qtype = answer.get("type")
        if qtype == "choice":
            return answer.get("choice")
        if qtype == "score":
            score = answer.get("score")
            return None if score is None else int(round(score))
        if qtype == "noul":
            prob = answer.get("noul")
            return None if prob is None else prob >= 0.5
        return None


class DecisionEngine(ABC):
    name: str

    @abstractmethod
    def decide(self, state: str, questions: Dict[str, Any]) -> DecisionResult:
        """Run one typed-decision call and return a normalized DecisionResult."""
        raise NotImplementedError
