"""Metrics: per-cycle CSV logging and aggregate summaries.

Consumes the CycleOutcome list produced by MultiCycleGraspRunner.
Writes one row per cycle, keyed by run_id + cycle_index so multi-run
sweeps land in a single file that pandas can query.

CSV schema (columns in order):
    run_id           — free-form string identifying the sweep row
    cycle_index      — 0..N-1 within a run
    path             — light-local | heavy-local | offload
    battery_low      — bool, B(t) at decision time
    network_good     — bool, C(t) at decision time
    soc_at_decision  — 0..1, battery SoC when decision was made
    data_rate_mbps   — Mbps, C(t) sample at decision time
    latency_s        — total wall time of the chosen path
    compute_latency_s — path's local/edge compute portion
    tx_latency_s     — offload's transmission portion (0 for local)
    energy_j         — energy drawn from battery this cycle
    predicted_success — bool, Bernoulli draw from the path's success rate
    physical_delivery — bool, was object actually placed in bin B
    started_at_s     — sim-time when the cycle began
    finished_at_s    — sim-time when the cycle ended (DONE or FAILED)
    final_state      — grasp FSM's terminal state (done | failed)
    decision_variant — name of the Decision (rule-based | always-*)
    initial_soc      — starting battery SoC for this run
    basestation_x, basestation_y — BS position for this run

The extra "run parameter" columns (decision_variant, initial_soc,
basestation_x/y) are optional constants — supplied by the caller so
downstream analysis can filter/group without a separate metadata table.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from typing import Any, Iterable

from sim.outcome import CycleOutcome


# Fixed column order so multi-run appends keep the same schema.
_CSV_FIELDS = [
    "run_id",
    "cycle_index",
    "path",
    "battery_low",
    "network_good",
    "soc_at_decision",
    "data_rate_mbps",
    "latency_s",
    "compute_latency_s",
    "tx_latency_s",
    "energy_j",
    "predicted_success",
    "physical_delivery",
    "started_at_s",
    "finished_at_s",
    "final_state",
    "decision_variant",
    "initial_soc",
    "basestation_x",
    "basestation_y",
]


def _outcome_to_row(
    outcome: CycleOutcome,
    run_id: str,
    run_params: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one CycleOutcome + run params into a CSV row dict."""
    pr = outcome.path_result
    return {
        "run_id": run_id,
        "cycle_index": outcome.cycle_index,
        "path": outcome.path_name,
        "battery_low": outcome.battery_low,
        "network_good": outcome.network_good,
        "soc_at_decision": round(outcome.battery_soc_at_decision, 6),
        "data_rate_mbps": round(outcome.data_rate_bps / 1e6, 4),
        "latency_s": round(pr.latency_s, 6) if pr else 0.0,
        "compute_latency_s": round(pr.compute_latency_s, 6) if pr else 0.0,
        "tx_latency_s": round(pr.tx_latency_s, 6) if pr else 0.0,
        "energy_j": round(pr.energy_j, 6) if pr else 0.0,
        "predicted_success": pr.success if pr else False,
        "physical_delivery": outcome.physically_delivered,
        "started_at_s": round(outcome.started_at_s, 3),
        "finished_at_s": round(outcome.finished_at_s, 3),
        "final_state": outcome.final_state,
        # Injected run params (constant within a run).
        "decision_variant": run_params.get("decision_variant", ""),
        "initial_soc": run_params.get("initial_soc", 0.0),
        "basestation_x": run_params.get("basestation_x", 0.0),
        "basestation_y": run_params.get("basestation_y", 0.0),
    }


def write_cycles_csv(
    outcomes: Iterable[CycleOutcome],
    filepath: str,
    run_id: str,
    run_params: dict[str, Any] | None = None,
    append: bool = True,
) -> int:
    """Write outcomes as CSV rows. Creates the file with a header if
    it doesn't exist. If append=False, truncates the file first.
    Returns the number of rows written."""
    run_params = run_params or {}
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    write_header = (not append) or (not os.path.exists(filepath))
    mode = "w" if not append else "a"

    n_written = 0
    with open(filepath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for outcome in outcomes:
            writer.writerow(_outcome_to_row(outcome, run_id, run_params))
            n_written += 1
    return n_written


def summarize(outcomes: list[CycleOutcome]) -> dict[str, Any]:
    """Aggregate stats for a single run's outcomes. Handy for logging
    a one-liner per sweep row."""
    if not outcomes:
        return {
            "n_cycles": 0,
            "path_counts": {},
            "mean_latency_s": 0.0,
            "total_energy_j": 0.0,
            "predicted_success_rate": 0.0,
            "physical_delivery_rate": 0.0,
        }

    n = len(outcomes)
    path_counts = dict(Counter(o.path_name for o in outcomes))
    latencies = [o.path_result.latency_s for o in outcomes if o.path_result]
    energies = [o.path_result.energy_j for o in outcomes if o.path_result]
    predicted = [o.path_result.success for o in outcomes if o.path_result]
    delivered = [o.physically_delivered for o in outcomes]

    return {
        "n_cycles": n,
        "path_counts": path_counts,
        "mean_latency_s": sum(latencies) / max(len(latencies), 1),
        "total_energy_j": sum(energies),
        "predicted_success_rate": sum(predicted) / max(len(predicted), 1),
        "physical_delivery_rate": sum(delivered) / max(len(delivered), 1),
    }
