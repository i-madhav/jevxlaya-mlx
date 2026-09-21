# Benchmark results

Jev (TypeSafe) vs Laya (Convai, via `laya-mlx`) on the same task: customer
support ticket triage and routing, run through the real `TriageAgent`
(Google ADK) in this repo. Same 30 labeled tickets, same typed-decision
schema, sent unchanged to both engines. See [README.md](README.md) for the
full setup and [results.json](results.json) for per-ticket predictions.

## Accuracy (30 labeled tickets)

| Engine | Exact match (all 5 fields) | intent | is_urgent | frustration | refund_requested | churn_risk | Frustration MAE | Errors |
|---|---|---|---|---|---|---|---|---|
| jev  | 53% | 83% | 83% | 77% | 97% | 80% | 0.23 | 0 |
| laya | 10% | 60% | 67% | 17% | 90% | 83% | 0.87 | 0 |

## Speed & cost

| Engine | Mean latency (ms) | Median latency (ms) | p95 latency (ms) | Cost / 30 tickets (USD) | ADK overhead (ms) |
|---|---|---|---|---|---|
| jev  | 422.3 | 391.1 | 840.0 | $0.00068 | 0.5 |
| laya | 152.2 | 148.4 | 195.6 | $0.00000 | 0.1 |

## Takeaway

Laya is ~2.8x faster and free to run locally, matching its speed claim.
Jev is slower and has a (small) per-call cost, but is over 5x more
accurate on strict exact-match across all 5 decision fields, with the
widest gap on the ordinal `frustration` score (MAE 0.23 vs 0.87). For a
decision that routes real tickets and triggers escalations, the accuracy
gap outweighs the latency difference here.

Reproduce with:

```bash
source .venv/bin/activate
python -m benchmark.run_benchmark
```
