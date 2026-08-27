"""Step 5 checkpoint: scene works — bins, objects, and gripper.

What you'll see:
    - Blue AGV parked at waypoint 0 (next to bin A)
    - xArm 6 on top, holding home pose, gripper on its ee link
    - Bin A (brown) to the right of the AGV, with 5 primitive objects
      settled inside it (red cube, blue cube, yellow cylinder, purple
      cylinder, green sphere)
    - Bin B (green) far to the left, empty
    - Gripper opens and closes on a 3-second cycle so you can see the
      fingers actuating
    - Console prints once per second: gripper opening + first object pos

Sanity check:
    - Objects fall a few cm and settle in bin A within the first second
    - Objects stay inside bin A (walls hold them, none escape)
    - Fingers travel smoothly between 0 and 0.03 m per finger
    - Gripper body stays glued to arm ee (no drift, no separation)
    - When you Ctrl+drag the camera around, everything stays in place

If objects escape the bin, walls are thin or gapped.
If gripper detaches from arm, teleport in Robot.step_callback is broken.
"""

import numpy as np
import pybullet as p

from sim.agv import AGVConfig, Waypoint
from sim.arm import ArmConfig
from sim.robot import Robot, RobotConfig
from sim.scene import Scene, SceneConfig
from sim.world import World, WorldConfig


def main() -> None:
    # Camera positioned to see bin A + robot from a helpful angle.
    world = World(WorldConfig(
        gui=True,
        cam_distance=2.0,
        cam_yaw=90.0,
        cam_pitch=-30.0,
        cam_target=(1.2, 0.0, 0.2),
    ))
    world.reset()

    # AGV parked at wp0 = (1.0, 0.0), next to bin A at (1.5, 0.0).
    # Big dwells so nothing moves — we're testing scene, not patrol.
    cfg = RobotConfig(
        agv=AGVConfig(
            cruise_speed=0.4,
            waypoints=(
                Waypoint(xy=(1.0, 0.0),  dwell_s=1e9),
                Waypoint(xy=(-1.0, 0.0), dwell_s=1e9),
            ),
        ),
        arm=ArmConfig(with_gripper=True),
    )
    robot = Robot(world, cfg)
    robot.reset()

    scene = Scene(world, SceneConfig())
    scene.reset()

    # Register the robot's single tick callback.
    world.register_callback(rate_hz=30.0, fn=robot.step_callback)

    # Gripper open/close cycle: 3 s period, square wave.
    def gripper_cycle(sim_time_s: float) -> None:
        if int(sim_time_s / 1.5) % 2 == 0:
            robot.arm.gripper.open()
        else:
            robot.arm.gripper.close()

    world.register_callback(rate_hz=2.0, fn=gripper_cycle)

    # 1 Hz report.
    def report(sim_time_s: float) -> None:
        opening = robot.arm.gripper.get_opening()
        target = robot.arm.gripper.get_target_opening()
        pos0 = scene.object_positions()[0]
        print(f"[t={sim_time_s:5.1f}s] "
              f"gripper: target={target*1000:5.1f} mm  actual={opening*1000:5.1f} mm  "
              f"obj0=({pos0[0]:+.2f}, {pos0[1]:+.2f}, {pos0[2]:+.2f})")

    world.register_callback(rate_hz=1.0, fn=report)

    print("Scene up. Gripper cycles every 1.5s. Close window to exit.")
    world.run(duration_s=None)
    world.close()


if __name__ == "__main__":
    main()
