"""Head-to-head benchmark: Jev (TypeSafe, cloud) vs Laya (Convai, local MLX)
on the same real-life decision -- customer support ticket triage & routing.

Two things are measured for each engine, against the same 30-ticket labeled
dataset and the exact same typed-decision schema:
  1. Direct accuracy/latency/cost of the raw decide() call.
  2. A small end-to-end pass through a real Google ADK agent (TriageAgent),
     to confirm the engine works as an agent's decision-making tool and to
     report the orchestration overhead ADK adds on top of the raw call.

Usage:
    python -m benchmark.run_benchmark [--adk-sample N]

Jev requires TYPESAFE_API_KEY (see https://console.typesafe.ai/keys); if it
is not set, the benchmark runs Laya alone and says so.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from agent.triage_agent import run_triage
from dataset.schema import GROUND_TRUTH_FIELDS, triage_questions
from decision_engines.base import DecisionEngine
from decision_engines.jev_engine import JevEngine
from decision_engines.laya_engine import LayaEngine

console = Console(width=160)
DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "tickets.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results.json"


def load_dataset() -> List[Dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text())


def build_engines() -> Dict[str, DecisionEngine]:
    engines: Dict[str, DecisionEngine] = {}

    if os.environ.get("TYPESAFE_API_KEY"):
        engines["jev"] = JevEngine()
    else:
        console.print(
            "[yellow]Skipping Jev: TYPESAFE_API_KEY not set "
            "(get one at https://console.typesafe.ai/keys).[/yellow]"
        )

    try:
        engines["laya"] = LayaEngine()
    except Exception as exc:  # e.g. not on Apple Silicon, or mlx missing
        console.print(f"[yellow]Skipping Laya: {exc}[/yellow]")

    return engines


def score_ticket(answers: Dict[str, Any], labels: Dict[str, Any]) -> Dict[str, bool]:
    correctness = {}
    for field in GROUND_TRUTH_FIELDS:
        predicted = answers.get(field)
        expected = labels.get(field)
        if field == "frustration":
            correctness[field] = predicted is not None and abs(predicted - expected) <= 0
        else:
            correctness[field] = predicted == expected
    return correctness


def run_direct_pass(engine: DecisionEngine, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    questions = triage_questions()
    per_field_correct = {f: 0 for f in GROUND_TRUTH_FIELDS}
    frustration_abs_errors = []
    exact_matches = 0
    latencies_ms = []
    total_cost_usd = 0.0
    errors = 0
    rows = []

    for ticket in tickets:
        result = engine.decide(ticket["message"], questions)
        if result.error:
            errors += 1
            rows.append({"id": ticket["id"], "error": result.error})
            continue

        answers = {qid: result.value(qid) for qid in questions}
        correctness = score_ticket(answers, ticket["labels"])
        for field, ok in correctness.items():
            per_field_correct[field] += int(ok)
        if all(correctness.values()):
            exact_matches += 1

        pred_frust = answers.get("frustration")
        if pred_frust is not None:
            frustration_abs_errors.append(abs(pred_frust - ticket["labels"]["frustration"]))

        latencies_ms.append(result.latency_ms)
        total_cost_usd += result.usage.get("cost_usd", 0.0)
        rows.append({"id": ticket["id"], "predicted": answers, "expected": ticket["labels"]})

    n_scored = len(tickets) - errors
    return {
        "engine": engine.name,
        "n_tickets": len(tickets),
        "n_errors": errors,
        "per_field_accuracy": {
            f: (per_field_correct[f] / n_scored if n_scored else None) for f in GROUND_TRUTH_FIELDS
        },
        "exact_match_accuracy": exact_matches / n_scored if n_scored else None,
        "frustration_mean_abs_error": (
            statistics.mean(frustration_abs_errors) if frustration_abs_errors else None
        ),
        "latency_ms": {
            "mean": statistics.mean(latencies_ms) if latencies_ms else None,
            "median": statistics.median(latencies_ms) if latencies_ms else None,
            "p95": (
                statistics.quantiles(latencies_ms, n=20)[18]
                if len(latencies_ms) >= 20
                else (max(latencies_ms) if latencies_ms else None)
            ),
        },
        "total_cost_usd": total_cost_usd,
        "rows": rows,
    }


async def run_adk_sample(engine: DecisionEngine, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    agent_latencies = []
    engine_latencies = []
    for ticket in tickets:
        state = await run_triage(engine, ticket["message"])
        if state.get("last_engine_error"):
            continue
        agent_latencies.append(state["last_agent_latency_ms"])
        engine_latencies.append(state["last_engine_latency_ms"])

    overhead = [a - e for a, e in zip(agent_latencies, engine_latencies)]
    return {
        "engine": engine.name,
        "n_sampled": len(tickets),
        "adk_overhead_ms_mean": statistics.mean(overhead) if overhead else None,
    }


def print_summary(direct_results: Dict[str, Dict[str, Any]], adk_results: Dict[str, Dict[str, Any]]):
    accuracy_table = Table(title="Accuracy -- ticket triage (30 labeled tickets)")
    accuracy_table.add_column("Engine")
    accuracy_table.add_column("Exact match (all 5 fields)")
    for field in GROUND_TRUTH_FIELDS:
        accuracy_table.add_column(field)
    accuracy_table.add_column("Frustration MAE")
    accuracy_table.add_column("Errors")

    for name, r in direct_results.items():
        accuracy_table.add_row(
            name,
            f"{r['exact_match_accuracy']:.0%}" if r["exact_match_accuracy"] is not None else "-",
            *[
                f"{r['per_field_accuracy'][f]:.0%}" if r["per_field_accuracy"][f] is not None else "-"
                for f in GROUND_TRUTH_FIELDS
            ],
            f"{r['frustration_mean_abs_error']:.2f}" if r["frustration_mean_abs_error"] is not None else "-",
            str(r["n_errors"]),
        )
    console.print(accuracy_table)

    perf_table = Table(title="Speed & cost")
    perf_table.add_column("Engine")
    perf_table.add_column("Mean latency (ms)")
    perf_table.add_column("Median latency (ms)")
    perf_table.add_column("p95 latency (ms)")
    perf_table.add_column("Cost / 30 tickets (USD)")
    perf_table.add_column("ADK overhead (ms)")

    for name, r in direct_results.items():
        adk_overhead = adk_results.get(name, {}).get("adk_overhead_ms_mean")
        lat = r["latency_ms"]
        perf_table.add_row(
            name,
            f"{lat['mean']:.1f}" if lat["mean"] is not None else "-",
            f"{lat['median']:.1f}" if lat["median"] is not None else "-",
            f"{lat['p95']:.1f}" if lat["p95"] is not None else "-",
            f"${r['total_cost_usd']:.5f}",
            f"{adk_overhead:.1f}" if adk_overhead is not None else "-",
        )
    console.print(perf_table)


async def main_async(adk_sample: int):
    tickets = load_dataset()
    engines = build_engines()
    if not engines:
        console.print("[red]No engines available -- nothing to benchmark.[/red]")
        return

    direct_results = {}
    for name, engine in engines.items():
        console.print(f"Running direct pass for [bold]{name}[/bold] on {len(tickets)} tickets...")
        direct_results[name] = run_direct_pass(engine, tickets)

    adk_results = {}
    sample = tickets[: adk_sample] if adk_sample > 0 else []
    if sample:
        for name, engine in engines.items():
            console.print(f"Running ADK agent pass for [bold]{name}[/bold] on {len(sample)} tickets...")
            adk_results[name] = await run_adk_sample(engine, sample)

    print_summary(direct_results, adk_results)

    RESULTS_PATH.write_text(
        json.dumps({"direct": direct_results, "adk": adk_results}, indent=2, default=str)
    )
    console.print(f"\nFull results written to [bold]{RESULTS_PATH}[/bold]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adk-sample",
        type=int,
        default=5,
        help="How many tickets to also run through the real ADK agent (0 to skip).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.adk_sample))


if __name__ == "__main__":
    main()
