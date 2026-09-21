"""The actual business decision the agent takes, downstream of the typed
answers a decision engine (Jev or Laya) returns.

This is the real-life implication of the comparison: which engine's typed
answers, plugged into the exact same routing rules, send a ticket to the
right queue with the right priority.
"""

from __future__ import annotations

from dataclasses import dataclass

QUEUE_BY_INTENT = {
    "refund": "billing_team",
    "billing_question": "billing_team",
    "technical_help": "support_technical",
    "cancellation": "retention_team",
    "information": "general_support",
    "other": "general_support",
}


@dataclass
class RoutingDecision:
    queue: str
    priority: str  # "P1" | "P2" | "P3"
    escalate_to_human: bool


def route_ticket(answers: dict) -> RoutingDecision:
    intent = answers.get("intent")
    is_urgent = bool(answers.get("is_urgent"))
    frustration = answers.get("frustration") or 0
    refund_requested = bool(answers.get("refund_requested"))
    churn_risk = bool(answers.get("churn_risk"))

    queue = QUEUE_BY_INTENT.get(intent, "general_support")
    if intent == "technical_help" and is_urgent:
        queue = "engineering_oncall"

    if frustration >= 3 or (is_urgent and frustration >= 2):
        priority = "P1"
    elif is_urgent or frustration >= 2:
        priority = "P2"
    else:
        priority = "P3"

    escalate_to_human = churn_risk or frustration >= 3 or (refund_requested and is_urgent)

    return RoutingDecision(queue=queue, priority=priority, escalate_to_human=escalate_to_human)
