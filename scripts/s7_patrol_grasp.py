"""Step 7 checkpoint — PHASE 4 DELIVERABLE.

End-to-end grasp cycle on a patrolling AGV. Unlike s6, the AGV
actually drives between bins (no teleport). The FSM commands motion:
    - Start at bin A (parked, arm reaches into bin)
    - APPROACH -> DESCEND -> CLOSE (pick up cube)
    - LIFT (raise object above bin)
    - TRANSIT (drive AGV from wp0 to wp1 at cruise speed)
    - APPROACH_DROP -> DESCEND_DROP -> OPEN (place in bin B)
    - LIFT_AWAY
    - RETURN_HOME (drive AGV back to wp0)
    - DONE

What you'll see:
    - Blue AGV at bin A. First object (red cube) sitting inside.
    - Arm smoothly reaches in, grasps, lifts.
    - AGV cruises across the workspace to bin B (~4 seconds).
    - Arm descends into bin B, opens, cube drops.
    - AGV cruises back to bin A.
    - Cycle DONE.

Total cycle time ~15-20 seconds (dominated by two 4s drives).

Waypoints have huge dwells (1e9 s) so the FSM controls all AGV motion.
Setting dwells to shorter values (e.g. 2 s) would make the AGV patrol
autonomously between grasps — that's the pattern phase 5 will use when
D(t) triggers offloading decisions per grasp cycle.
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
        cam_distance=3.5,
        cam_yaw=90.0,
        cam_pitch=-35.0,
        cam_target=(0.0, 0.0, 0.2),
    ))
    world.reset()

    scene_cfg = SceneConfig()
    bin_a_xy = scene_cfg.bin_a.center_xy   # (1.5, 0)
    bin_b_xy = scene_cfg.bin_b.center_xy   # (-1.5, 0)

    cfg = RobotConfig(
        agv=AGVConfig(
            cruise_speed=0.5,   # 2 m gap => 4 s traversal
            waypoints=(
                # wp0: pick point beside bin A, arm facing bin A
                Waypoint(xy=(1.0, 0.0),  dwell_s=1e9, face_xy=bin_a_xy),
                # wp1: drop point beside bin B, arm facing bin B
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

    # Magenta debug sphere at the current IK fingertip target.
    ik_marker_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[1.0, 0.2, 0.9, 0.8],
        physicsClientId=cid,
    )
    ik_marker_id = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=ik_marker_vis,
        basePosition=[0.0, 0.0, -1.0],
        physicsClientId=cid,
    )

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
        agv_pos, _ = robot.agv.pose()
        tip_pos, _ = robot.arm.gripper.get_fingertip_pose()
        obj_pos, _ = p.getBasePositionAndOrientation(
            scene.object_ids[0], physicsClientId=cid,
        )
        if grasp._q_target is not None:
            q_actual = robot.arm.get_joint_positions()
            q_err = float(np.max(np.abs(q_actual - grasp._q_target)))
            q_err_str = f"q_err={np.degrees(q_err):5.1f}deg"
        else:
            q_err_str = "q_err=  n/a"
        print(f"[t={sim_time_s:5.1f}s] state={grasp.state.value:>14s}  "
              f"agv=({agv_pos[0]:+.2f}, {agv_pos[1]:+.2f})  "
              f"tip=({tip_pos[0]:+.2f}, {tip_pos[1]:+.2f}, {tip_pos[2]:+.2f})  "
              f"{q_err_str}  "
              f"obj0=({obj_pos[0]:+.2f}, {obj_pos[1]:+.2f}, {obj_pos[2]:+.2f})")

    world.register_callback(rate_hz=1.0, fn=report)

    summary_done = {"flag": False}

    def summary(sim_time_s: float) -> None:
        if grasp.is_done() and not summary_done["flag"]:
            print("=" * 60)
            print(f"CYCLE FINISHED at t={sim_time_s:.2f}s")
            print(f"  final state: {grasp.state.value}")
            print(f"  target in bin B: {grasp.object_in_bin_b()}")
            print(f"  AGV back home: {robot.agv.is_at_waypoint(0)}")
            print("=" * 60)
            summary_done["flag"] = True

    world.register_callback(rate_hz=5.0, fn=summary)

    print("Patrol grasp cycle. Close window to exit.")
    world.run(duration_s=None)
    world.close()


if __name__ == "__main__":
    main()
