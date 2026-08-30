"""CycleOutcome data type — decoupled from the physics runtime.

Extracted from sim/grasp.py so headless sweeps (scripts/s9_sweep.py)
can construct and consume outcomes without importing pybullet. The
grasp FSM still owns the CycleOutcome instances it produces; this
module just holds the definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sim.paths import PathResult


@dataclass
class CycleOutcome:
    """Per-cycle metric bundle produced when a GraspCycle completes.

    Aggregated by MultiCycleGraspRunner and consumed by the CSV logger.
    `path_result` is the raw PathResult from the chosen execution path;
    other fields are FSM-level bookkeeping (which object, timing,
    final state).
    """

    cycle_index: int
    target_object_id: int
    final_state: str                # GraspState.value at end
    # D(t) inputs at the moment of the decision.
    battery_low: bool = False
    network_good: bool = False
    battery_soc_at_decision: float = 0.0
    data_rate_bps: float = 0.0
    # Path chosen and what it produced.
    path_name: str = ""
    path_result: Optional[PathResult] = None
    # Wall-clock timing (sim seconds).
    started_at_s: float = 0.0
    finished_at_s: float = 0.0
    # Was the object physically placed in bin B?
    physically_delivered: bool = False
