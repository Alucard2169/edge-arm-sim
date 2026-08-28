"""Step 6 checkpoint: full pick-and-place cycle with AGV stationary.

The FSM runs once: pick first object from bin A, teleport AGV to bin B,
drop, retract, done.

What you'll see:
    - AGV at bin A. First object (red cube) sitting inside.
    - A magenta sphere shows the current IK fingertip target (helpful
      for spotting when a state commands an unreachable pose).
    - Arm rises 20 cm above the cube, descends, closes gripper on it.
    - Arm lifts the cube. AGV teleports to bin B and rotates 180°.
    - Arm moves over bin B, descends, opens gripper. Cube falls in.
    - Arm retracts. Console prints "DONE" and success verdict.

Console prints state transitions as they happen. A successful cycle
takes ~10-15 s of sim time.

Waypoints use face_xy so the AGV points at the bin it needs to reach
into (not at the next patrol waypoint). Without this, the arm would
be facing the wrong way at each stop and IK would fail.
"""

import numpy as np
import pybullet as p

from sim.agv import AGVConfig, Waypoint
from sim.arm import ArmConfig
from sim.grasp import GraspCycle, GraspConfig
from sim.robot import Robot, RobotConfig
from sim.scene import Scene, SceneConfig
from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(
        gui=True,
        cam_distance=2.5,
        cam_yaw=90.0,
        cam_pitch=-30.0,
        cam_target=(0.0, 0.0, 0.2),
    ))
    world.reset()

    # Bin positions come from SceneConfig defaults: bin A at (1.5, 0),
    # bin B at (-1.5, 0). We park the AGV at (1.0, 0) with its arm
    # pointing at bin A, and later at (-1.0, 0) pointing at bin B.
    scene_cfg = SceneConfig()
    bin_a_xy = scene_cfg.bin_a.center_xy
    bin_b_xy = scene_cfg.bin_b.center_xy

    cfg = RobotConfig(
        agv=AGVConfig(
            cruise_speed=0.5,
            waypoints=(
                # wp0: pick point, face bin A (+x)
                Waypoint(xy=(1.0, 0.0),  dwell_s=1e9, face_xy=bin_a_xy),
                # wp1: drop point, face bin B (-x)
                Waypoint(xy=(-1.0, 0.0), dwell_s=1e9, face_xy=bin_b_xy),
            ),
        ),
        arm=ArmConfig(with_gripper=True),
    )
    robot = Robot(world, cfg)
    robot.reset()

    scene = Scene(world, scene_cfg)
    scene.reset()

    cid = world.client_id

    # ---- Debug: magenta sphere at current IK fingertip target ----
    ik_marker_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[1.0, 0.2, 0.9, 0.8],
        physicsClientId=cid,
    )
    ik_marker_id = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=ik_marker_vis,
        basePosition=[0.0, 0.0, -1.0],  # off-screen until first update
        physicsClientId=cid,
    )

    # ---- Callbacks ----
    world.register_callback(rate_hz=30.0, fn=robot.step_callback)

    grasp = GraspCycle(world, robot, scene, GraspConfig())

    started = {"flag": False}

    def maybe_start(sim_time_s: float) -> None:
        if not started["flag"] and sim_time_s > 0.5:
            target = scene.object_ids[0]
            print(f"[t={sim_time_s:5.2f}s] starting cycle, target obj id={target}")
            grasp.start(target)
            started["flag"] = True

    world.register_callback(rate_hz=10.0, fn=maybe_start)
    world.register_callback(rate_hz=30.0, fn=grasp.tick)

    def update_ik_marker(sim_time_s: float) -> None:
        if grasp._ik_target_pos is not None:
            p.resetBasePositionAndOrientation(
                ik_marker_id, grasp._ik_target_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=cid,
            )

    world.register_callback(rate_hz=15.0, fn=update_ik_marker)

    def report(sim_time_s: float) -> None:
        tip_pos, _ = robot.arm.gripper.get_fingertip_pose()
        obj_pos, _ = p.getBasePositionAndOrientation(
            scene.object_ids[0], physicsClientId=cid,
        )
        tgt = grasp._ik_target_pos
        tgt_str = ("none" if tgt is None
                   else f"({tgt[0]:+.2f}, {tgt[1]:+.2f}, {tgt[2]:+.2f})")
        # Joint-space error: how far the arm is from its commanded pose.
        if grasp._q_target is not None:
            q_actual = robot.arm.get_joint_positions()
            q_err = float(np.max(np.abs(q_actual - grasp._q_target)))
            q_err_str = f"q_err={np.degrees(q_err):5.1f}deg"
        else:
            q_err_str = "q_err=  n/a"
        print(f"[t={sim_time_s:5.1f}s] state={grasp.state.value:>14s}  "
              f"tip=({tip_pos[0]:+.2f}, {tip_pos[1]:+.2f}, {tip_pos[2]:+.2f})  "
              f"tgt={tgt_str}  {q_err_str}  "
              f"obj0=({obj_pos[0]:+.2f}, {obj_pos[1]:+.2f}, {obj_pos[2]:+.2f})")

    world.register_callback(rate_hz=1.0, fn=report)

    summary_done = {"flag": False}

    def summary(sim_time_s: float) -> None:
        if grasp.is_done() and not summary_done["flag"]:
            print("=" * 60)
            print(f"CYCLE FINISHED at t={sim_time_s:.2f}s")
            print(f"  final state: {grasp.state.value}")
            print(f"  target in bin B: {grasp.object_in_bin_b()}")
            print("=" * 60)
            summary_done["flag"] = True

    world.register_callback(rate_hz=5.0, fn=summary)

    print("Running pick-and-place cycle. Close window to exit.")
    world.run(duration_s=None)
    world.close()


if __name__ == "__main__":
    main()
