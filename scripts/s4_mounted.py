"""Step 4 checkpoint: xArm 6 mounted on AGV, holds home pose during patrol.

What you'll see:
    - Blue AGV patrols the rectangular loop from checkpoint 3
    - xArm 6 sits on top of the AGV, in home pose (elbow bent, wrist down)
    - The arm rides along cleanly — it never separates from the AGV,
      never flops, never jitters as the AGV rotates through waypoints
    - Green waypoint markers visible
    - Console prints once per second: AGV waypoint, arm ee position

Sanity check:
    - Arm base sits on the AGV top surface (z = 0.30 m for default AGV)
    - Arm ee is above the AGV, moving with it, still tracking home pose
    - When AGV rotates 90° at a waypoint, arm rotates with it (ee traces
      an arc around the AGV centre, not stays fixed in world frame)

If the arm floats off the AGV or lags behind, the teleport is broken.
If the arm's joints spaz out during AGV rotation, the base yaw handoff
is broken.
"""

import numpy as np
import pybullet as p

from sim.agv import AGVConfig, Waypoint
from sim.arm import ArmConfig
from sim.robot import Robot, RobotConfig
from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(gui=True, cam_distance=4.5, cam_pitch=-40.0))
    world.reset()

    cfg = RobotConfig(
        agv=AGVConfig(
            cruise_speed=0.4,
            waypoints=(
                Waypoint(xy=(1.0, 0.0),  dwell_s=1.5),
                Waypoint(xy=(1.0, 1.0),  dwell_s=1.5),
                Waypoint(xy=(-1.0, 1.0), dwell_s=1.5),
                Waypoint(xy=(-1.0, 0.0), dwell_s=1.5),
            ),
        ),
        arm=ArmConfig(),  # defaults — home pose, mounted centred
        arm_mount_offset=(0.0, 0.0, 0.0),
    )
    robot = Robot(world, cfg)
    robot.reset()

    cid = world.client_id

    # ---- Waypoint markers ----
    wp_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.2, 0.8, 0.2, 0.7],
        physicsClientId=cid,
    )
    for wp in cfg.agv.waypoints:
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=wp_vis,
            basePosition=[wp.xy[0], wp.xy[1], 0.05],
            physicsClientId=cid,
        )

    # ---- Single callback drives the whole robot ----
    world.register_callback(rate_hz=30.0, fn=robot.step_callback)

    # ---- 1 Hz report ----
    def report(sim_time_s: float) -> None:
        agv_pos, agv_orn = robot.agv.pose()
        yaw = p.getEulerFromQuaternion(agv_orn.tolist())[2]
        ee_pos, _ = robot.arm.get_ee_pose()
        status = "dwell" if robot.agv.is_dwelling() else "cruise"
        print(f"[t={sim_time_s:5.1f}s] "
              f"agv=({agv_pos[0]:+.2f}, {agv_pos[1]:+.2f}) {status:>6}  "
              f"yaw={np.degrees(yaw):+6.1f}°  "
              f"ee=({ee_pos[0]:+.2f}, {ee_pos[1]:+.2f}, {ee_pos[2]:+.2f})")

    world.register_callback(rate_hz=1.0, fn=report)

    print("Robot patrolling. Ctrl+drag rotates camera. Close window to exit.")
    world.run(duration_s=None)
    world.close()


if __name__ == "__main__":
    main()
