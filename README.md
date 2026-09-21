# Jev vs Laya: a ticket-triage decision benchmark

A head-to-head comparison of two "System One" typed-decision models on the
same real-life job: **routing an incoming customer support ticket**.

- **Jev** ([TypeSafe](https://console.typesafe.ai/keys)) -- cloud API, `POST /v1/systemone`.
- **Laya** ([Convai Innovations](https://github.com/mizorewww/laya-mlx)) -- runs locally on Apple Silicon via MLX, no network call.

Both were independently built for the exact same niche -- fast, bounded,
non-generative decisions (`choice` / `score` / `noul`) for routing,
classification, moderation and guardrails inside an app or agent -- and
both speak the same typed-question vocabulary. That's what makes them
directly comparable: this POC sends the **identical** question schema
(`dataset/schema.py`, itself `laya_mlx.triage_questions()`) to both engines,
unchanged, and scores their answers against the same 30 hand-labeled
tickets in `dataset/tickets.json`.

## Why this scenario

Ticket triage is a real production decision point: given a customer
message, decide (a) which team queue it goes to, (b) how urgent it is, and
(c) whether a human needs to step in right now (angry, threatening to
churn, asking for money back). Convai frames this exact workload
("ticket routing... the classification layer around an LLM stack") as
Laya's target use case, and TypeSafe frames Jev the same way ("routing,
classification, and other decision points... where a fast, predictable
answer matters more than generated prose"). So instead of inventing an
artificial task, the benchmark uses the shared target use case directly.

The agent shell is a real [Google ADK](https://github.com/google/adk-python)
agent (`agent/triage_agent.py`): `TriageAgent` is a plain ADK `BaseAgent`
that receives a ticket through a real `Runner`/`Session`/`Event` pipeline,
asks its configured decision engine (Jev or Laya) the typed questions, and
applies `agent/routing_logic.py` to produce a queue + priority +
escalate-to-human decision. The same agent code runs unchanged with either
engine plugged in -- only the "brain" swaps. It's intentionally LLM-free
(no Gemini call, no extra API key) so the only variable under test is Jev
vs. Laya, not an orchestrating model.

## Setup

Requires macOS on Apple Silicon (Laya's MLX runtime needs it) and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Get a Jev API key at <https://console.typesafe.ai/keys>, then:

```bash
cp .env.example .env
# edit .env and paste your key into TYPESAFE_API_KEY=
```

Laya needs no key -- the first run downloads the `convaiinnovations/laya`
checkpoint from Hugging Face (~330 MB) and caches it locally.

## Run it

```bash
python -m benchmark.run_benchmark
```

This runs all 30 tickets through each available engine directly (accuracy
+ latency + cost), then routes a 5-ticket sample through the real ADK
`TriageAgent` for both engines to confirm end-to-end agent integration and
report ADK's orchestration overhead. Results print as tables and are
written to `results.json` (per-ticket predictions included, for auditing
individual disagreements).

If `TYPESAFE_API_KEY` isn't set, the benchmark runs Laya alone and says so
-- useful for validating the pipeline before you have a Jev key.

## What's measured

- **Accuracy**: per-field accuracy across `intent` (6-way routing target),
  `is_urgent`, `frustration` (0-3 ordinal), `refund_requested`,
  `churn_risk`, plus strict exact-match-on-all-5-fields accuracy and
  frustration's mean absolute error (a softer metric for the ordinal
  field).
- **Speed**: mean / median / p95 latency per raw `decide()` call, and the
  extra latency the ADK agent shell adds on top.
- **Cost**: Jev is billed per input token (TypeSafe's published rate);
  Laya runs locally so its marginal cost per call is $0.

## Layout

```
dataset/
  tickets.json       30 labeled support tickets (ground truth)
  schema.py           the shared typed-question schema
decision_engines/
  base.py              DecisionEngine interface + DecisionResult
  jev_engine.py        TypeSafe Jev HTTP adapter
  laya_engine.py       local laya-mlx adapter
agent/
  routing_logic.py     ticket -> {queue, priority, escalate_to_human}
  triage_agent.py       the Google ADK agent (BaseAgent, no LLM)
benchmark/
  run_benchmark.py      runs everything, prints tables, writes results.json
```
