"""Step 9 — phase-5 evaluation sweep.

Headless sweep across (initial SoC × basestation distance × decision
variant). Each combination runs N cycles of the phase-5 adaptive grasp
pipeline and writes one CSV row per cycle to data/sweep_results.csv.

Sweep dimensions (defaults, 4×4×4 = 64 runs):
    initial SoC:      [1.0, 0.5, 0.35, 0.20]
    basestation:      [(0,2), (0,5), (0,10), (0,20)]  # near..very-far
    decision variant: [rule-based, always-light, always-heavy, always-offload]

Runtime: ~15-30 minutes wall-clock for the full sweep (headless mode
runs at max physics speed, no rendering).

Output: data/sweep_results.csv — one row per grasp cycle, with all
sweep parameters denormalized into columns for easy pandas queries.

Analyze in a notebook:
    df = pd.read_csv("data/sweep_results.csv")
    df.groupby(["decision_variant", "initial_soc"])["energy_j"].sum()
    df.groupby(["decision_variant"])["latency_s"].mean()
    # etc.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import time
from typing import Callable

from sim.agv import AGVConfig, Waypoint
from sim.arm import ArmConfig
from sim.battery import Battery, BatteryConfig
from sim.decision import (
    AlwaysLocalHeavy,
    AlwaysLocalLight,
    AlwaysOffload,
    Decision,
)
from sim.grasp import GraspConfig, GraspCycle
from sim.metrics import summarize, write_cycles_csv
from sim.multi_cycle import MultiCycleGraspRunner, RunnerConfig
from sim.network import NetworkChannel, NetworkConfig
from sim.paths import (
    HeavyLocal, HeavyLocalConfig,
    LightLocal, LightLocalConfig,
    Offload, OffloadConfig,
)
from sim.robot import Robot, RobotConfig
from sim.scene import ObjectSpec, Scene, SceneConfig
from sim.world import World, WorldConfig


# ------------------------------------------------------- decision factory

DECISION_FACTORIES: dict[str, Callable[[], Decision]] = {
    "rule-based":     Decision,
    "always-light":   AlwaysLocalLight,
    "always-heavy":   AlwaysLocalHeavy,
    "always-offload": AlwaysOffload,
}


# --------------------------------------------------------- scene builder

def make_scene_with_n_objects(n: int) -> SceneConfig:
    """Return a SceneConfig with n varied primitive objects by cycling
    through a small prototype list. Needed so each sweep run has enough
    objects for max_cycles cycles without repeating IDs."""
    prototypes = SceneConfig().objects  # default 5 varied objects
    objs = tuple(prototypes[i % len(prototypes)] for i in range(n))
    return SceneConfig(objects=objs)


# ------------------------------------------------------------- run helper

def _stable_seed(text: str) -> int:
    """Deterministic seed from a string. 32-bit range for numpy."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def run_one(
    decision_variant: str,
    initial_soc: float,
    basestation_xy: tuple[float, float],
    n_cycles: int,
) -> tuple[list, dict]:
    """Set up and run one sweep combination. Returns (outcomes, summary)."""
    seed = _stable_seed(f"{decision_variant}|{initial_soc}|{basestation_xy}")

    world = World(WorldConfig(gui=False))
    world.reset()

    scene_cfg = make_scene_with_n_objects(n_cycles)
    bin_a_xy = scene_cfg.bin_a.center_xy
    bin_b_xy = scene_cfg.bin_b.center_xy

    robot_cfg = RobotConfig(
        agv=AGVConfig(
            cruise_speed=0.5,
            waypoints=(
                Waypoint(xy=(1.0, 0.0),  dwell_s=1e9, face_xy=bin_a_xy),
                Waypoint(xy=(-1.0, 0.0), dwell_s=1e9, face_xy=bin_b_xy),
            ),
        ),
        arm=ArmConfig(with_gripper=True),
    )
    robot = Robot(world, robot_cfg)
    robot.reset()

    scene = Scene(world, scene_cfg)
    scene.reset()

    # Demo-scale battery so drain shows up in n_cycles.
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

    # Path stubs — seed each so a run's success/latency samples are
    # reproducible for the (variant, soc, bs) triple.
    paths = {
        "light-local": LightLocal(LightLocalConfig(seed=seed + 1)),
        "heavy-local": HeavyLocal(HeavyLocalConfig(seed=seed + 2)),
        "offload":     Offload(OffloadConfig(seed=seed + 3)),
    }

    grasp = GraspCycle(world, robot, scene, GraspConfig(),
                       battery=battery, network=network,
                       decision=decision, paths=paths)
    runner = MultiCycleGraspRunner(world, grasp, RunnerConfig(
        max_cycles=n_cycles,
        inter_cycle_pause_s=0.1,   # tighter than the GUI demo
    ))

    world.register_callback(rate_hz=30.0, fn=robot.step_callback)
    world.register_callback(rate_hz=30.0, fn=grasp.tick)
    world.register_callback(rate_hz=30.0, fn=runner.tick)

    runner.start()
    # Generous ceiling: 60 s per cycle should cover any offload-far
    # latency spike. Actual runtime is bounded by should_stop.
    world.run(duration_s=60.0 * n_cycles, should_stop=runner.is_stopped)
    world.close()

    return runner.outcomes, summarize(runner.outcomes)


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sweep_results.csv",
                        help="output CSV filepath")
    parser.add_argument("--cycles", type=int, default=10,
                        help="cycles per run")
    parser.add_argument("--quick", action="store_true",
                        help="tiny sweep for a smoke test (8 runs)")
    args = parser.parse_args()

    if args.quick:
        soc_values = [0.5, 0.2]
        bs_positions = [(0.0, 5.0), (0.0, 20.0)]
        decisions = ["rule-based", "always-heavy"]
    else:
        soc_values = [1.0, 0.5, 0.35, 0.20]
        bs_positions = [(0.0, 2.0), (0.0, 5.0), (0.0, 10.0), (0.0, 20.0)]
        decisions = ["rule-based", "always-light", "always-heavy", "always-offload"]

    combos = list(itertools.product(decisions, soc_values, bs_positions))

    # Truncate the output file at start (fresh sweep).
    if os.path.exists(args.output):
        os.remove(args.output)
    print(f"Sweeping {len(combos)} combinations, {args.cycles} cycles each")
    print(f"Output: {args.output}")
    print("=" * 78)

    t0 = time.perf_counter()
    for i, (decision_variant, soc, bs_xy) in enumerate(combos, 1):
        run_id = f"{decision_variant}_soc{soc:.2f}_bs{bs_xy[0]:.0f},{bs_xy[1]:.0f}"
        print(f"[{i:3d}/{len(combos)}] {run_id}")

        outcomes, summary = run_one(
            decision_variant=decision_variant,
            initial_soc=soc,
            basestation_xy=bs_xy,
            n_cycles=args.cycles,
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
            },
        )
        pc = summary["path_counts"]
        print(f"          -> {n_written:2d} cycles  "
              f"paths: L={pc.get('light-local', 0):2d} "
              f"H={pc.get('heavy-local', 0):2d} "
              f"O={pc.get('offload', 0):2d}  "
              f"E={summary['total_energy_j']:6.3f}J  "
              f"delivery={summary['physical_delivery_rate'] * 100:.0f}%")

    t1 = time.perf_counter()
    print("=" * 78)
    print(f"Sweep complete: {len(combos)} runs in {t1 - t0:.1f}s wall-clock")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
