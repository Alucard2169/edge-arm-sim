"""Step 3 checkpoint: AGV patrols a closed loop of waypoints.

What you'll see:
    - GUI with ground plane and a blue box (the AGV)
    - Green spheres marking the four waypoints
    - AGV moves at 0.5 m/s from waypoint to waypoint, pausing 1s at each
    - Yellow line traces the current segment (helps see the direction)
    - Console prints once per second: current waypoint, dwelling status,
      AGV pose

Sanity check:
    - Total loop length ~= 6 m (2m + 1m + 2m + 1m)
    - At 0.5 m/s cruise + 4 s dwelling total, one loop should take ~16 s
    - AGV yaw should rotate to match travel direction on each segment
    - AGV should sit flat on the ground (never sink or float)

If the AGV floats or sinks, the z offset in _set_pose is wrong.
If yaw jumps discontinuously between segments, the yaw calc is wrong.
"""

import numpy as np
import pybullet as p

from sim.agv import AGV, AGVConfig, Waypoint
from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(gui=True, cam_distance=4.0, cam_pitch=-45.0))
    world.reset()

    agv_cfg = AGVConfig(
        cruise_speed=0.5,
        waypoints=(
            Waypoint(xy=(1.0, 0.0),  dwell_s=1.0),
            Waypoint(xy=(1.0, 1.0),  dwell_s=1.0),
            Waypoint(xy=(-1.0, 1.0), dwell_s=1.0),
            Waypoint(xy=(-1.0, 0.0), dwell_s=1.0),
        ),
    )
    agv = AGV(world, agv_cfg)
    agv.reset()

    cid = world.client_id

    # ---- Waypoint markers (visual only, no collision) ----
    wp_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.2, 0.8, 0.2, 0.7],
        physicsClientId=cid,
    )
    for wp in agv_cfg.waypoints:
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=wp_vis,
            basePosition=[wp.xy[0], wp.xy[1], 0.05],
            physicsClientId=cid,
        )

    # ---- Register AGV control at 30 Hz ----
    world.register_callback(rate_hz=30.0, fn=agv.control_step)

    # ---- Direction line: update in place each tick ----
    pos0, _ = agv.pose()
    line_state = {
        "id": p.addUserDebugLine(
            lineFromXYZ=pos0.tolist(),
            lineToXYZ=pos0.tolist(),
            lineColorRGB=[1.0, 1.0, 0.0],
            lineWidth=2.0,
            physicsClientId=cid,
        )
    }

    def draw_segment(sim_time_s: float) -> None:
        pos, _ = agv.pose()
        next_wp = agv_cfg.waypoints[
            (agv.current_waypoint_index() + 1) % len(agv_cfg.waypoints)
        ]
        line_state["id"] = p.addUserDebugLine(
            lineFromXYZ=[pos[0], pos[1], 0.35],
            lineToXYZ=[next_wp.xy[0], next_wp.xy[1], 0.35],
            lineColorRGB=[1.0, 1.0, 0.0],
            lineWidth=2.0,
            replaceItemUniqueId=line_state["id"],
            physicsClientId=cid,
        )

    world.register_callback(rate_hz=15.0, fn=draw_segment)

    # ---- 1 Hz report ----
    def report(sim_time_s: float) -> None:
        pos, orn = agv.pose()
        yaw = p.getEulerFromQuaternion(orn.tolist())[2]
        status = "dwell" if agv.is_dwelling() else "cruise"
        print(f"[t={sim_time_s:5.1f}s] "
              f"wp_idx={agv.current_waypoint_index()} {status:>6}  "
              f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  "
              f"yaw={np.degrees(yaw):+6.1f}°")

    world.register_callback(rate_hz=1.0, fn=report)

    print("AGV patrolling. Ctrl+drag rotates camera. Close window to exit.")
    world.run(duration_s=None)
    world.close()


if __name__ == "__main__":
    main()1
