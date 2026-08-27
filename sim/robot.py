u"""Composite of AGV + Arm: the mobile manipulator.

The AGV is kinematic (pose set directly each tick), so the arm mounted
on top gets the same treatment — we read the AGV's top-centre pose and
teleport the arm's base to match. This is legal for the xArm 6 URDF
because it was loaded with useFixedBase=True; `resetBasePositionAndOrientation`
on a fixed-base body is PyBullet's canonical way to move it.

Consequence: the arm's inertia is not preserved across teleports.
This matches our modeling choice — the AGV is kinematic, so treating
the arm as kinematically carried by it is consistent. When the AGV
dwells and the arm reaches, joint dynamics still apply normally
(POSITION_CONTROL through the joint motors).

The `robot.py` composite exists so external code registers ONE callback
per tick — Robot handles the internal ordering (AGV -> mount -> arm).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from sim.agv import AGV, AGVConfig
from sim.arm import Arm, ArmConfig
from sim.world import World


# ------------------------------------------------------------------ config

@dataclass
class RobotConfig:
    """Config for the composite. Nested AGV/Arm configs are held here so
    a caller can tune them together (e.g. arm base offset relative to
    the AGV top)."""

    agv: AGVConfig = field(default_factory=AGVConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)

    # Offset (dx, dy, dz) of the arm base from the AGV top centre, in
    # the AGV's local frame. Default: arm mounted dead centre on top.
    # A real deployment might offset it forward so the workspace covers
    # what the camera sees; we keep it centred for now.
    arm_mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ------------------------------------------------------------------- robot

class Robot:
    """AGV + Arm as a single unit.

    Attributes:
        agv: the AGV underneath
        arm: the arm on top

    Both are fully accessible for direct use (robot.arm.solve_ik(...),
    robot.agv.pose(), etc.). Robot just handles the mount kinematics
    and gives you one callback to register.
    """

    def __init__(self, world: World, cfg: RobotConfig | None = None) -> None:
        self.world = world
        self.cfg = cfg or RobotConfig()

        self.agv = AGV(world, self.cfg.agv)
        self.arm = Arm(world, self.cfg.arm)

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Reset AGV first (so its pose is defined), then arm, then
        immediately mount the arm on top so it starts on the AGV rather
        than at the origin."""
        self.agv.reset()

        # Override the arm's base position to sit on the AGV. Note we
        # mutate the config before Arm.reset() so the URDF loads in
        # the right place (avoids a visible one-frame teleport).
        base_xyz, base_yaw = self._compute_mount_pose()
        self.arm.cfg.base_position = tuple(base_xyz)
        self.arm.cfg.base_orientation_rpy = (0.0, 0.0, base_yaw)

        self.arm.reset()

    # ------------------------------------------------------------- callback

    def step_callback(self, sim_time_s: float) -> None:
        """One callback to register with World for the whole robot.

        Order per tick:
            1. AGV control step (advance patrol)
            2. Teleport arm base to new AGV top-centre
            3. Teleport gripper to new arm ee pose (if gripper exists)
            4. Arm control step (re-apply POSITION_CONTROL target)
            5. Gripper control step (re-apply finger POSITION_CONTROL)
        """
        self.agv.control_step(sim_time_s)
        self._teleport_arm_to_agv()
        if self.arm.gripper is not None:
            self.arm.gripper.teleport_to_arm_ee()
        self.arm.control_step(sim_time_s)
        if self.arm.gripper is not None:
            self.arm.gripper.control_step(sim_time_s)

    # -------------------------------------------------------------- mount

    def _compute_mount_pose(self) -> tuple[np.ndarray, float]:
        """World-frame (position, yaw) where the arm base should sit
        this tick, given the current AGV pose and mount offset."""
        agv_pos, agv_orn = self.agv.pose()
        yaw = p.getEulerFromQuaternion(agv_orn.tolist())[2]

        # Rotate the local mount offset into world frame using AGV yaw.
        dx_l, dy_l, dz_l = self.cfg.arm_mount_offset
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        dx_w = cos_y * dx_l - sin_y * dy_l
        dy_w = sin_y * dx_l + cos_y * dy_l

        # AGV top-centre plus the rotated offset. top_center already
        # includes body_height/2 in z.
        top = self.agv.top_center()
        base_xyz = np.array([
            top[0] + dx_w,
            top[1] + dy_w,
            top[2] + dz_l,
        ])
        return base_xyz, yaw

    def _teleport_arm_to_agv(self) -> None:
        base_xyz, yaw = self._compute_mount_pose()
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(
            self.arm.body_id, base_xyz.tolist(), orn,
            physicsClientId=self.world.client_id,
        )
