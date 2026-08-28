"""Diagnostic: does kinematic arm control actually hold a commanded pose?

Strips away everything: no Robot, no AGV, no gripper, no scene, no FSM.
Just an Arm attached to a World. We command a non-trivial joint config
after 1 second and watch whether the arm reaches it and holds.

Expected if kinematic control works:
    - t=0-1s: arm at home
    - t=1s: target commanded
    - t=1-3s: arm smoothly moves toward target (max ~4°/tick at 30 Hz)
    - t=3s+: arm holds target, max_err stays < 0.1°

If the arm doesn't hold or oscillates or drifts, the bug is inside
sim/arm.py itself, isolated from every other subsystem.
"""

import numpy as np

from sim.arm import Arm, ArmConfig
from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(gui=True, cam_distance=1.5, cam_pitch=-20.0))
    world.reset()

    # Arm at origin, base fixed to the world plane. No gripper, no
    # anything else touching it. Kinematic control mode is the default.
    arm = Arm(world, ArmConfig(with_gripper=False))
    arm.reset()

    # Arm's own control step (kinematic advance) at 30 Hz.
    world.register_callback(rate_hz=30.0, fn=arm.control_step)

    # A non-trivial target: rotate base, tilt shoulder, bend elbow.
    # All values well within joint limits.
    target_q = np.array([0.5, 1.0, -1.5, 0.0, 1.0, 0.0])

    commanded = {"flag": False}

    def maybe_command(sim_time_s: float) -> None:
        if not commanded["flag"] and sim_time_s > 1.0:
            print(f"[t={sim_time_s:5.2f}s] Commanding target q={target_q}")
            arm.set_joint_positions(target_q)
            commanded["flag"] = True

    world.register_callback(rate_hz=10.0, fn=maybe_command)

    # 1 Hz report of q_actual vs q_target so we can see convergence.
    def report(sim_time_s: float) -> None:
        q_actual = arm.get_joint_positions()
        q_target = arm.get_target_positions()
        err = np.abs(q_actual - q_target)
        print(f"[t={sim_time_s:5.1f}s] "
              f"q_actual={np.round(q_actual, 3)}")
        print(f"           "
              f"q_target={np.round(q_target, 3)}")
        print(f"           "
              f"err(deg)={np.round(np.degrees(err), 1)}  "
              f"max={np.degrees(err.max()):5.1f}")

    world.register_callback(rate_hz=1.0, fn=report)

    print("Kinematic hold test: arm should reach the commanded target and stay.")
    world.run(duration_s=8.0)
    world.close()


if __name__ == "__main__":
    main()
