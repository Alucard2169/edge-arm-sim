"""Step 2 checkpoint: IK works — arm tool tip tracks a draggable target.

What you'll see:
    - GUI with three sliders in the top-right panel: target_x, target_y, target_z
    - A red translucent sphere shows the current target position
    - A yellow line from the tool tip to the target shows the residual
    - As you drag the sliders, the arm reaches to the new target within
      a few tenths of a second
    - Console prints tracking error once per second

Controls:
    - Drag the sliders to move the target
    - Right-click-drag or Ctrl+drag in the viewport to rotate the camera
    - Close the window (or Ctrl-C in terminal) to exit — no timeout

Sanity check:
    - Targets inside the workspace (roughly a hemisphere of radius 0.7 m
      centered above the base) should be reached with < 1 cm error
    - Targets outside the workspace (too far, or below the ground) will
      show a large residual — that's the IK saying "closest I can get"

If the arm jumps around wildly or doesn't move at all, IK is broken.
"""

import numpy as np
import pybullet as p

from sim.arm import Arm, ArmConfig
from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(gui=True))
    world.reset()

    arm = Arm(world, ArmConfig())
    arm.reset()

    cid = world.client_id

    # ---- Debug UI: three sliders for target position ----
    # Ranges cover the reachable workspace with some slack. Default
    # values put the target roughly where the tool tip already is at
    # home pose, so nothing jumps on startup.
    ee_pos_home, _ = arm.get_ee_pose()
    slider_x = p.addUserDebugParameter("target_x", -0.8, 0.8, ee_pos_home[0],
                                       physicsClientId=cid)
    slider_y = p.addUserDebugParameter("target_y", -0.8, 0.8, ee_pos_home[1],
                                       physicsClientId=cid)
    slider_z = p.addUserDebugParameter("target_z",  0.05, 1.2, ee_pos_home[2],
                                       physicsClientId=cid)

    # ---- Target marker (visual-only sphere, no collision, no mass) ----
    marker_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.02, rgbaColor=[1.0, 0.2, 0.2, 0.7],
        physicsClientId=cid,
    )
    marker_id = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=marker_vis,
        basePosition=ee_pos_home.tolist(),
        physicsClientId=cid,
    )

    # Residual line: allocate once, update in place each tick.
    line_id = p.addUserDebugLine(
        lineFromXYZ=ee_pos_home.tolist(),
        lineToXYZ=ee_pos_home.tolist(),
        lineColorRGB=[1.0, 1.0, 0.0],
        lineWidth=2.0,
        physicsClientId=cid,
    )

    # ---- Control callback: read sliders, solve IK, command arm ----
    # Track the line_id so we can replace it in place (avoids leaking
    # debug items every tick).
    state = {"line_id": line_id}

    def control_tick(sim_time_s: float) -> None:
        target = np.array([
            p.readUserDebugParameter(slider_x, physicsClientId=cid),
            p.readUserDebugParameter(slider_y, physicsClientId=cid),
            p.readUserDebugParameter(slider_z, physicsClientId=cid),
        ])
        p.resetBasePositionAndOrientation(
            marker_id, target.tolist(), [0, 0, 0, 1],
            physicsClientId=cid,
        )
        q = arm.solve_ik(target_pos=target)
        arm.set_joint_positions(q)

        ee, _ = arm.get_ee_pose()
        state["line_id"] = p.addUserDebugLine(
            lineFromXYZ=ee.tolist(),
            lineToXYZ=target.tolist(),
            lineColorRGB=[1.0, 1.0, 0.0],
            lineWidth=2.0,
            replaceItemUniqueId=state["line_id"],
            physicsClientId=cid,
        )

    world.register_callback(rate_hz=30.0, fn=control_tick)

    # ---- 1 Hz reporting ----
    def report(sim_time_s: float) -> None:
        target = np.array([
            p.readUserDebugParameter(slider_x, physicsClientId=cid),
            p.readUserDebugParameter(slider_y, physicsClientId=cid),
            p.readUserDebugParameter(slider_z, physicsClientId=cid),
        ])
        ee, _ = arm.get_ee_pose()
        err = np.linalg.norm(ee - target)
        print(f"[t={sim_time_s:5.1f}s] "
              f"target={np.round(target, 3)}  "
              f"ee={np.round(ee, 3)}  "
              f"err={err*1000:6.1f} mm")

    world.register_callback(rate_hz=1.0, fn=report)

    print("Drag the sliders to move the target. Close the window to exit.")
    world.run(duration_s=None)  # run until user closes the window
    world.close()


if __name__ == "__main__":
    main()
