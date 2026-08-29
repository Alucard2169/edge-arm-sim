"""Step 8 checkpoint: adaptive grasp cycle driven by D(t).

Same physical setup as s7 (patrolling AGV, bin A -> bin B), but each
cycle now consults the D(t) rule table on the current battery + network
state to pick light-local / heavy-local / offload, incurs simulated
latency for the chosen path, drains the battery, and moves on to the
next object in bin A. Repeats until bin A is empty or battery dies.

What you'll see:
    - AGV at bin A. Multiple objects in bin.
    - For each cycle: APPROACH -> INFER (paths get logged) -> DESCEND ->
      CLOSE -> LIFT -> TRANSIT -> APPROACH_DROP -> DESCEND_DROP -> OPEN ->
      LIFT_AWAY -> RETURN_HOME -> DONE.
    - Between cycles: 0.5 s pause, then next object.
    - Runner prints a summary line per cycle: which path was chosen,
      whether the grasp physically succeeded, current SoC.
    - Bin A empties one object at a time; bin B fills.

The initial battery is deliberately set to just above the 30% threshold
(SoC = 0.35) so a few heavy cycles push it into the LOW regime and the
rule table flips over to light-local / offload — makes the D(t) logic
visible without needing 100 cycles.
"""

import numpy as np
import pybullet as p

from sim.agv import AGVConfig, Waypoint
from sim.arm import ArmConfig
from sim.battery import Battery, BatteryConfig
from sim.decision import Decision
from sim.grasp import GraspCycle, GraspConfig
from sim.multi_cycle import MultiCycleGraspRunner, RunnerConfig
from sim.network import NetworkChannel, NetworkConfig
from sim.paths import HeavyLocal, LightLocal, Offload
from sim.robot import Robot, RobotConfig
from sim.scene import Scene, SceneConfig
from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(
        gui=True,
        cam_distance=3.5,
        cam_yaw=90.0,
        cam_pitch=-35.0,
        cam_target=(0.0, 0.0, 0.2),
    ))
    world.reset()

    scene_cfg = SceneConfig()
    bin_a_xy = scene_cfg.bin_a.center_xy
    bin_b_xy = scene_cfg.bin_b.center_xy

    cfg = RobotConfig(
        agv=AGVConfig(
            cruise_speed=0.5,
            waypoints=(
                Waypoint(xy=(1.0, 0.0),  dwell_s=1e9, face_xy=bin_a_xy),
                Waypoint(xy=(-1.0, 0.0), dwell_s=1e9, face_xy=bin_b_xy),
            ),
        ),
        arm=ArmConfig(with_gripper=True),
    )
    robot = Robot(world, cfg)
    robot.reset()

    scene = Scene(world, scene_cfg)
    scene.reset()

    # Phase-5 modules. Battery is deliberately SMALL for this demo
    # (~15 J capacity, starting at 50%) so drain across 5 cycles is
    # visible and pushes SoC past the 30% LOW threshold, letting the
    # rule table flip mid-run. Realistic AGV capacity would be
    # 100-1000 kJ but per-cycle inference energy is milli-joules, so
    # visible drain requires either a small battery or thousands of
    # cycles. The metrics sweep in commit 5 will use realistic scale
    # over many cycles.
    battery = Battery(BatteryConfig(
        capacity_j=15.0,
        soc_initial=0.5,            # 7.5 J starting energy
        soc_low_threshold=0.30,     # 4.5 J is LOW
    ))
    network = NetworkChannel(NetworkConfig(seed=42))
    decision = Decision()
    paths = {
        "light-local": LightLocal(),
        "heavy-local": HeavyLocal(),
        "offload":     Offload(),
    }

    grasp = GraspCycle(world, robot, scene, GraspConfig(),
                       battery=battery, network=network,
                       decision=decision, paths=paths)

    runner = MultiCycleGraspRunner(world, grasp, RunnerConfig(
        max_cycles=len(scene.object_ids),  # one per object
        inter_cycle_pause_s=0.5,
    ))

    cid = world.client_id

    # Callback ordering: robot first, then grasp.tick, then runner.tick
    # so runner sees the terminal DONE state on the same tick it happens.
    world.register_callback(rate_hz=30.0, fn=robot.step_callback)
    world.register_callback(rate_hz=30.0, fn=grasp.tick)
    world.register_callback(rate_hz=30.0, fn=runner.tick)

    # Debug marker for the current IK target.
    ik_marker_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[1.0, 0.2, 0.9, 0.8],
        physicsClientId=cid,
    )
    ik_marker_id = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=ik_marker_vis,
        basePosition=[0.0, 0.0, -1.0],
        physicsClientId=cid,
    )

    def update_ik_marker(sim_time_s: float) -> None:
        if grasp._ik_target_pos is not None:
            p.resetBasePositionAndOrientation(
                ik_marker_id, grasp._ik_target_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=cid,
            )

    world.register_callback(rate_hz=15.0, fn=update_ik_marker)

    # Kick off the first cycle after the initial pause.
    runner.start()

    # End-of-run summary.
    summary_done = {"flag": False}

    def summary(sim_time_s: float) -> None:
        if runner.is_stopped() and not summary_done["flag"]:
            print("\n" + "=" * 66)
            print(f"RUN FINISHED at t={sim_time_s:.2f}s")
            print("=" * 66)
            print(f"  cycles run:        {len(runner.outcomes)}")
            print(f"  final SoC:         {battery.state_of_charge() * 100:5.1f}%")
            print(f"  total energy used: {battery.total_consumed_j():7.3f} J "
                  f"of {battery.cfg.capacity_j:.1f} J capacity")

            # Per-path summary.
            from collections import Counter
            path_counter = Counter(o.path_name for o in runner.outcomes)
            print(f"  path selections:")
            for name in ("light-local", "heavy-local", "offload"):
                cnt = path_counter.get(name, 0)
                print(f"    {name:>12s}: {cnt:3d}")

            delivered = sum(1 for o in runner.outcomes if o.physically_delivered)
            predicted = sum(1 for o in runner.outcomes
                            if o.path_result and o.path_result.success)
            n = max(len(runner.outcomes), 1)
            print(f"  physical delivery: {delivered}/{len(runner.outcomes)} "
                  f"({delivered / n * 100:.0f}%)")
            print(f"  predicted success: {predicted}/{len(runner.outcomes)} "
                  f"({predicted / n * 100:.0f}%)")
            print("=" * 66)
            summary_done["flag"] = True

    world.register_callback(rate_hz=5.0, fn=summary)

    print("Adaptive grasp run. Close window to exit.")
    world.run(duration_s=None)
    world.close()


if __name__ == "__main__":
    main()
