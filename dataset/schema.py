"""The one typed-decision schema fed to both engines, unchanged.

Reused verbatim from laya_mlx.triage_questions() -- it's a ready-made,
production-shaped preset for customer support ticket triage, and its
{type, instructions, criteria} shape is exactly the request format Jev's
/v1/systemone endpoint expects too. Using the same dict for both engines is
what keeps the comparison fair.
"""

from __future__ import annotations

from typing import Any, Dict


def triage_questions() -> Dict[str, Any]:
    try:
        import laya_mlx as laya

        return laya.triage_questions()
    except ImportError:
        pass

    # Fallback copy (kept in sync with laya_mlx.triage_questions) so the
    # benchmark can still run against Jev alone on a machine without mlx.
    return {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want in `message`?",
            "criteria": {
                "refund": "money returned or a duplicate charge reversed",
                "technical_help": "a bug, outage or integration problem",
                "billing_question": "a question about an invoice, plan or payment method",
                "information": "general information, pricing or how-to",
                "cancellation": "wants to cancel or downgrade",
                "other": "none of the other options fits",
            },
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does `message` communicate time pressure or a deadline?",
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer sound in `message`?",
            "criteria": [
                "calm and neutral",
                "concerned but civil",
                "clearly annoyed",
                "very angry or using strong language",
            ],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?",
        },
    }


GROUND_TRUTH_FIELDS = ["intent", "is_urgent", "frustration", "refund_requested", "churn_risk"]
