"""Smoke test for sim.arm: xArm 6 loads, holds home pose, joint info prints.

What you should see:
    - GUI opens, ground plane visible
    - xArm 6 loads at the origin in the "ready to reach forward" pose
    - Arm stays put (holds pose against gravity) for 5 seconds
    - Console prints: joint names, joint limits, home pose, end-effector pose

If any of that fails, sim/arm.py has a bug we fix before writing IK.
"""

import numpy as np

from sim.arm import Arm, ArmConfig
from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(gui=True))
    world.reset()

    arm = Arm(world, ArmConfig())
    arm.reset()

    # Print what we discovered — one-shot diagnostic.
    print("=" * 60)
    print("xArm 6 loaded.")
    print(f"  body_id           = {arm.body_id}")
    print(f"  joint_indices     = {arm.joint_indices}")
    print(f"  joint_names       = {arm.joint_names}")
    print(f"  ee_link_index     = {arm.ee_link_index}")
    print(f"  limits (lower)    = {np.round(arm.joint_limits_lower, 3)}")
    print(f"  limits (upper)    = {np.round(arm.joint_limits_upper, 3)}")
    print(f"  target home pose  = {np.round(arm.get_target_positions(), 3)}")

    ee_pos, ee_orn = arm.get_ee_pose()
    print(f"  ee_pos (world)    = {np.round(ee_pos, 3)}")
    print(f"  ee_orn (quat)     = {np.round(ee_orn, 3)}")
    print("=" * 60)

    # Register the arm's control step so POSITION_CONTROL re-applies
    # every 30 Hz (control tick). Physics still steps at 240 Hz.
    world.register_callback(rate_hz=30.0, fn=arm.control_step)

    # Also print the ee pose once a second so we can eyeball drift.
    def report(sim_time_s: float) -> None:
        q = arm.get_joint_positions()
        ee, _ = arm.get_ee_pose()
        print(f"[t={sim_time_s:4.1f}s] "
              f"q_err_max={np.max(np.abs(q - arm.get_target_positions())):.4f} rad  "
              f"ee={np.round(ee, 3)}")

    world.register_callback(rate_hz=1.0, fn=report)

    print("Running for 5 simulated seconds — arm should hold pose.")
    world.run(duration_s=5.0)
    world.close()


if __name__ == "__main__":
    main()
