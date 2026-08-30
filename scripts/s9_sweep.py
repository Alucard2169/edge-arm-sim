"""Step 9 — phase-5 D(t) evaluation sweep, physics-free.

The physics-in-the-loop version of this sweep (previous s9) pegged the
CPU because headless PyBullet steps as fast as the machine allows.
This version drops PyBullet entirely: since the sweep evaluates D(t)
decision behavior, path selection, and battery drain — all pure Python —
the physics is control-loop context, not the object of study.

What we simulate:
    - Battery draining across cycles
    - Network channel varying with BS distance and shadow noise
    - Decision looking up the rule table (or a fixed baseline)
    - Path stubs returning latency, energy, and Bernoulli success
    - Runner-level termination (battery depleted, max cycles)

What we do NOT simulate:
    - Actual arm/AGV motion (already verified by s7 and s8)
    - Contact physics or grasp failure modes independent of the model's
      predicted success. In this sweep, physical_delivery == predicted
      success. If the model predicted the grasp would work, we count
      the object as delivered. Honest simplification, defensible in
      the paper: the D(t) contribution is decision quality, not
      compensation for physical mishaps.

    - Actual sim time per cycle. We track cumulative sim time as
      (fixed_motion_time + path.latency_s) per cycle so timing
      columns in the CSV remain meaningful for downstream plots.

Seeding (multi-seed sweeps):
    Each seed_idx k induces a fully-specified "world":
    a network shadow-fading trace and per-path Bernoulli streams.
    Every variant at (soc, bs, seed_idx=k) samples from the same
    streams, so between-variant comparisons are paired and have
    tighter CIs than independent seeds would give. Pairing on path
    streams is per-invocation, not per-cycle — i.e., the N-th call
    to LightLocal is deterministic across variants regardless of
    which cycle number invokes it.

Runtime: full 4x4x4 = 64-run sweep at 10 cycles each = 640 cycles
completes in about 1-2 seconds. --seeds 30 scales linearly (~30-60s).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import time
from typing import Callable

from sim.battery import Battery, BatteryConfig
from sim.decision import (
    AlwaysLocalHeavy,
    AlwaysLocalLight,
    AlwaysOffload,
    Decision,
)
from sim.metrics import summarize, write_cycles_csv
from sim.network import NetworkChannel, NetworkConfig
from sim.outcome import CycleOutcome
from sim.paths import (
    HeavyLocal, HeavyLocalConfig,
    LightLocal, LightLocalConfig,
    Offload, OffloadConfig,
    Path,
)


# The AGV position when the D(t) decision is made. In the physics FSM,
# INFER runs after APPROACH — the AGV is at the pick waypoint. Network
# variation across the sweep comes from the sweep of BS positions, not
# from AGV motion within a cycle.
DECISION_AGV_XY: tuple[float, float] = (1.0, 0.0)

# Fixed non-decision time per cycle (approach, descend, close, lift,
# transit to bin B, drop, return home). Measured from s7 patrol grasp
# at ~12s. Constant across all paths, so cycle_duration - fixed = the
# D(t)-influenced portion.
FIXED_MOTION_TIME_S: float = 12.0


DECISION_FACTORIES: dict[str, Callable[[], Decision]] = {
    "rule-based":     Decision,
    "always-light":   AlwaysLocalLight,
    "always-heavy":   AlwaysLocalHeavy,
    "always-offload": AlwaysOffload,
}


# ------------------------------------------------------------- one cycle

def simulate_cycle(
    cycle_index: int,
    sim_time_s: float,
    agv_xy: tuple[float, float],
    battery: Battery,
    network: NetworkChannel,
    decision: Decision,
    paths: dict[str, Path],
) -> CycleOutcome:
    """One D(t) cycle without physics. Returns a CycleOutcome for the CSV."""
    # Sample B(t) and C(t) at decision time.
    battery_low = battery.is_low()
    soc_at_decision = battery.state_of_charge()
    data_rate, network_good = network.sample(agv_xy)

    # Consult decision, execute chosen path.
    path_name = decision.decide(
        cycle=cycle_index,
        battery_low=battery_low,
        network_good=network_good,
    )
    path = paths[path_name]
    result = path.execute(agv_xy, network)

    # Drain battery.
    battery.consume(result.energy_j)

    # Fake sim-time bookkeeping.
    started = sim_time_s
    finished = started + FIXED_MOTION_TIME_S + result.latency_s

    return CycleOutcome(
        cycle_index=cycle_index,
        target_object_id=cycle_index,       # synthetic id per cycle
        final_state="done",
        battery_low=battery_low,
        network_good=network_good,
        battery_soc_at_decision=soc_at_decision,
        data_rate_bps=data_rate,
        path_name=path_name,
        path_result=result,
        started_at_s=started,
        finished_at_s=finished,
        # Without physics, we count a cycle as delivered iff the model
        # predicted a good grasp. See module docstring for rationale.
        physically_delivered=result.success,
    )


# ------------------------------------------------------- one full run

def _stable_seed(text: str) -> int:
    """Deterministic seed from a string, 32-bit range for numpy."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def run_one(
    decision_variant: str,
    initial_soc: float,
    basestation_xy: tuple[float, float],
    n_cycles: int,
    seed_idx: int,                                          # NEW
) -> list[CycleOutcome]:
    """Set up modules for one sweep row and run up to n_cycles."""
    # CHANGED: seed derivation excludes variant so that all variants at
    # the same (soc, bs, seed_idx) see identical network/path streams.
    # This makes between-variant comparisons paired and low-variance.
    seed = _stable_seed(f"{initial_soc}|{basestation_xy}|{seed_idx}")

    battery = Battery(BatteryConfig(
        capacity_j=15.0,
        soc_initial=initial_soc,
        soc_low_threshold=0.30,
    ))
    network = NetworkChannel(NetworkConfig(
        basestation_xy=basestation_xy,
        seed=seed,
    ))
    decision = DECISION_FACTORIES[decision_variant]()
    paths = {
        "light-local": LightLocal(LightLocalConfig(seed=seed + 1)),
        "heavy-local": HeavyLocal(HeavyLocalConfig(seed=seed + 2)),
        "offload":     Offload(OffloadConfig(seed=seed + 3)),
    }

    outcomes: list[CycleOutcome] = []
    sim_time_s = 0.0
    for cycle_index in range(n_cycles):
        if battery.is_depleted():
            break
        oc = simulate_cycle(
            cycle_index=cycle_index,
            sim_time_s=sim_time_s,
            agv_xy=DECISION_AGV_XY,
            battery=battery,
            network=network,
            decision=decision,
            paths=paths,
        )
        outcomes.append(oc)
        sim_time_s = oc.finished_at_s
    return outcomes


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sweep_results.csv",
                        help="output CSV filepath")
    parser.add_argument("--cycles", type=int, default=10,
                        help="max cycles per run (may end early on battery depletion)")
    parser.add_argument("--seeds", type=int, default=1,                 # NEW
                        help="number of seeds per (variant, scenario) combination "
                             "(default 1 preserves single-seed behavior)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny sweep for a smoke test (8 runs per seed)")
    args = parser.parse_args()

    if args.quick:
        soc_values = [0.5, 0.2]
        bs_positions = [(0.0, 5.0), (0.0, 20.0)]
        decisions = ["rule-based", "always-heavy"]
    else:
        soc_values = [1.0, 0.5, 0.35, 0.20]
        bs_positions = [(0.0, 2.0), (0.0, 5.0), (0.0, 10.0), (0.0, 20.0)]
        decisions = ["rule-based", "always-light", "always-heavy", "always-offload"]

    # CHANGED: seed_idx is the OUTER loop so partial runs still cover
    # every scenario for the seeds that completed.
    combos = list(itertools.product(
        range(args.seeds), decisions, soc_values, bs_positions,
    ))

    # Truncate the output at start (fresh sweep).
    if os.path.exists(args.output):
        os.remove(args.output)
    print(f"Sweeping {len(combos)} combinations "
          f"({args.seeds} seed(s) × {len(decisions)} variants × "
          f"{len(soc_values)} SoCs × {len(bs_positions)} BS positions), "
          f"up to {args.cycles} cycles each")
    print(f"Output: {args.output}")
    print("=" * 78)

    t0 = time.perf_counter()
    for i, (seed_idx, decision_variant, soc, bs_xy) in enumerate(combos, 1):
        run_id = (f"{decision_variant}_soc{soc:.2f}"
                  f"_bs{bs_xy[0]:.0f},{bs_xy[1]:.0f}"
                  f"_seed{seed_idx}")                       # CHANGED
        outcomes = run_one(
            decision_variant=decision_variant,
            initial_soc=soc,
            basestation_xy=bs_xy,
            n_cycles=args.cycles,
            seed_idx=seed_idx,                              # NEW
        )
        n_written = write_cycles_csv(
            outcomes,
            filepath=args.output,
            run_id=run_id,
            run_params={
                "decision_variant": decision_variant,
                "initial_soc": soc,
                "basestation_x": bs_xy[0],
                "basestation_y": bs_xy[1],
                "seed": seed_idx,                           # NEW
            },
        )
        s = summarize(outcomes)
        pc = s["path_counts"]
        print(f"[{i:4d}/{len(combos)}] {run_id:<55s}  "
              f"cyc={n_written:2d}  "
              f"L={pc.get('light-local', 0):2d} "
              f"H={pc.get('heavy-local', 0):2d} "
              f"O={pc.get('offload', 0):2d}  "
              f"E={s['total_energy_j']:6.3f}J  "
              f"deliv={s['physical_delivery_rate'] * 100:3.0f}%")
    t1 = time.perf_counter()
    print("=" * 78)
    print(f"Sweep complete: {len(combos)} runs in {t1 - t0:.2f}s wall-clock")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()