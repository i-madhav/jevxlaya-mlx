"""A Google ADK agent whose "decision" step is swappable between Jev and Laya.

TriageAgent is a plain (non-LLM) google-adk BaseAgent: it receives a ticket
through the normal ADK Runner/Session/Event pipeline, calls a
DecisionEngine to answer the typed triage questions, applies routing_logic,
and writes the outcome into session state -- the same agent shell,
swapping only which brain makes the decision.

Kept LLM-free on purpose: the variable under test is Jev vs. Laya, not an
orchestrating Gemini model, so no extra API key is required to run this.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Dict

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import ConfigDict

from dataset.schema import triage_questions
from decision_engines.base import DecisionEngine, DecisionResult
from .routing_logic import route_ticket


class TriageAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    engine: Any = None  # DecisionEngine, kept as Any so pydantic doesn't need to validate it

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        message = ""
        if ctx.user_content and ctx.user_content.parts:
            message = ctx.user_content.parts[0].text or ""

        questions = triage_questions()
        t0 = time.perf_counter()
        decision: DecisionResult = await asyncio.to_thread(self.engine.decide, message, questions)
        agent_latency_ms = (time.perf_counter() - t0) * 1000.0

        answers = {qid: decision.value(qid) for qid in questions}
        routing = route_ticket(answers)

        summary = (
            f"[{self.engine.name}] intent={answers.get('intent')} "
            f"queue={routing.queue} priority={routing.priority} "
            f"escalate={routing.escalate_to_human} ({decision.latency_ms:.1f} ms)"
        )

        state_delta = {
            "last_ticket": message,
            "last_answers": answers,
            "last_routing": {
                "queue": routing.queue,
                "priority": routing.priority,
                "escalate_to_human": routing.escalate_to_human,
            },
            "last_engine_latency_ms": decision.latency_ms,
            "last_agent_latency_ms": agent_latency_ms,
            "last_engine_error": decision.error,
        }

        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part.from_text(text=summary)]),
            actions=EventActions(state_delta=state_delta),
        )


async def run_triage(engine: DecisionEngine, message: str) -> Dict[str, Any]:
    """Drive TriageAgent through a real ADK Runner/Session for one ticket."""
    agent = TriageAgent(name=f"triage_agent_{engine.name}", engine=engine)
    runner = InMemoryRunner(agent=agent, app_name="poc_jev_vs_laya")
    session = await runner.session_service.create_session(
        app_name="poc_jev_vs_laya", user_id="benchmark"
    )

    async for _event in runner.run_async(
        user_id="benchmark",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=message)]),
    ):
        pass

    final = await runner.session_service.get_session(
        app_name="poc_jev_vs_laya", user_id="benchmark", session_id=session.id
    )
    return dict(final.state)
