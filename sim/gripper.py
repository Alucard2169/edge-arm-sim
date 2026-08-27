"""Simple parallel-jaw gripper attached to the arm's ee link.

Follows the same "kinematic-everywhere" pattern as the AGV mount: each
control tick, we compute the arm ee pose (via forward kinematics from
the arm's current joint state) and teleport the gripper's base to
match. Finger positions are controlled via POSITION_CONTROL on the
prismatic joints.

Not using a PyBullet constraint because:
1. The arm base is teleported each tick (AGV motion), which causes
   constraint solver stress and visible jitter on low-mass children.
2. Kinematic teleport keeps the gripper rigidly stuck to the arm ee
   at all times — no lag, no oscillation.

Trade-off: the gripper doesn't feel gripping forces through joint
dynamics (contacts on the fingers still work, but the gripper as a
whole moves rigidly with the arm). Fine for our purposes — the thesis
argument doesn't rest on force feedback at the palm.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p

from sim.world import World

if TYPE_CHECKING:
    from sim.arm import Arm

# Path to our design URDF (committed under sim/assets/).
GRIPPER_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "simple_gripper.urdf",
)

# The two finger joint names in the URDF, in the order we want to
# control them. Both open the same way (higher value = more open).
FINGER_JOINT_NAMES = ("finger_l_joint", "finger_r_joint")

# Physical constants from the URDF.
GRIPPER_MAX_OPENING = 0.03   # per-finger travel (m); total opening = 2x this
GRIPPER_MIN_OPENING = 0.0
# The gripper base's origin sits on the mount face. The fingertips are
# at +z ≈ 0.07 (finger visual origin z=0.03 + half-length 0.03).
GRIPPER_FINGERTIP_Z_OFFSET = 0.07


# ------------------------------------------------------------------ config

@dataclass
class GripperConfig:
    """All tunable knobs for the gripper."""

    # Extra offset (dx, dy, dz) of the gripper base from the arm's ee link,
    # in ee-local frame. Default: base sits at the ee origin, fingers
    # pointing along ee +z (which for xArm 6 link6 is the tool axis).
    mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Rotation of the gripper relative to the ee link (rpy, radians).
    # xArm 6's link6 has its z pointing "outward" from the wrist; the
    # gripper URDF is designed with +z as tool axis, so no rotation
    # needed by default.
    mount_orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # POSITION_CONTROL parameters for the finger joints.
    finger_max_force: float = 20.0
    finger_position_gain: float = 0.1
    finger_velocity_gain: float = 1.0

    # Initial opening (per-finger travel, 0 = closed, 0.03 = fully open).
    initial_opening: float = GRIPPER_MAX_OPENING


# ----------------------------------------------------------------- gripper

class Gripper:
    """Parallel-jaw gripper riding on the arm's ee link.

    Owns:
        - the loaded gripper URDF body
        - finger joint indices
        - the current opening target (applied every control tick)

    Attaches to an Arm — needs a reference to read its ee pose each tick.
    """

    def __init__(self, world: World, arm: "Arm",
                 cfg: GripperConfig | None = None) -> None:
        self.world = world
        self.arm = arm
        self.cfg = cfg or GripperConfig()

        self.body_id: int = -1
        self.finger_joint_indices: list[int] = []
        self._target_opening: float = self.cfg.initial_opening

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Load the gripper URDF at the current arm ee pose. Requires
        the arm to already be loaded and homed."""
        if self.arm.body_id < 0:
            raise RuntimeError("Arm.reset() must be called before Gripper.reset()")

        if not os.path.isfile(GRIPPER_URDF_PATH):
            raise FileNotFoundError(
                f"Gripper URDF not found at {GRIPPER_URDF_PATH}"
            )

        # Load at the arm ee pose so we don't need a snap frame later.
        ee_pos, ee_orn = self.arm.get_ee_pose()
        base_pos, base_orn = self._compute_mount_pose(ee_pos, ee_orn)

        self.body_id = p.loadURDF(
            GRIPPER_URDF_PATH,
            basePosition=base_pos.tolist(),
            baseOrientation=base_orn,
            useFixedBase=False,  # we drive it kinematically; not gravity-affected
            physicsClientId=self.world.client_id,
        )

        # Discover finger joints by name.
        self._discover_finger_joints()

        # Zero mass on the base link would be ideal but URDF specifies
        # nonzero. Instead, we disable gravity for this body so it
        # doesn't fall between teleports.
        p.changeDynamics(
            self.body_id, -1, mass=0.0,   # kinematic base
            physicsClientId=self.world.client_id,
        )

        # Apply initial opening.
        self._target_opening = self.cfg.initial_opening
        self._apply_target()

    def _discover_finger_joints(self) -> None:
        cid = self.world.client_id
        n = p.getNumJoints(self.body_id, physicsClientId=cid)
        name_to_index: dict[str, int] = {}
        for i in range(n):
            info = p.getJointInfo(self.body_id, i, physicsClientId=cid)
            name_to_index[info[1].decode("utf-8")] = i

        self.finger_joint_indices = []
        for name in FINGER_JOINT_NAMES:
            if name not in name_to_index:
                raise RuntimeError(
                    f"Gripper URDF is missing joint '{name}'. "
                    f"Found joints: {list(name_to_index.keys())}"
                )
            self.finger_joint_indices.append(name_to_index[name])

    # ------------------------------------------------------------- control

    def open(self) -> None:
        """Command fully open."""
        self.set_opening(GRIPPER_MAX_OPENING)

    def close(self) -> None:
        """Command fully closed."""
        self.set_opening(GRIPPER_MIN_OPENING)

    def set_opening(self, per_finger_travel: float) -> None:
        """Command a specific opening. Argument is per-finger travel
        (0 = closed, GRIPPER_MAX_OPENING = fully open). Clipped to limits."""
        self._target_opening = float(np.clip(
            per_finger_travel, GRIPPER_MIN_OPENING, GRIPPER_MAX_OPENING,
        ))
        self._apply_target()

    def control_step(self, sim_time_s: float) -> None:
        """Re-apply the current opening target. Register with the World
        or call from Robot.step_callback()."""
        self._apply_target()

    def _apply_target(self) -> None:
        p.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=self.finger_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[self._target_opening] * len(self.finger_joint_indices),
            forces=[self.cfg.finger_max_force] * len(self.finger_joint_indices),
            positionGains=[self.cfg.finger_position_gain] * len(self.finger_joint_indices),
            velocityGains=[self.cfg.finger_velocity_gain] * len(self.finger_joint_indices),
            physicsClientId=self.world.client_id,
        )

    # ------------------------------------------------------------- mount

    def teleport_to_arm_ee(self) -> None:
        """Snap the gripper base to the arm's current ee pose. Call
        this whenever the arm base is teleported (AGV motion) or when
        arm joints change enough that the fingers might drift."""
        ee_pos, ee_orn = self.arm.get_ee_pose()
        base_pos, base_orn = self._compute_mount_pose(ee_pos, ee_orn)
        p.resetBasePositionAndOrientation(
            self.body_id, base_pos.tolist(), base_orn,
            physicsClientId=self.world.client_id,
        )

    def _compute_mount_pose(
        self, ee_pos: np.ndarray, ee_orn: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        """Compose (ee pose) * (mount offset in ee-local frame)."""
        offset_pos = list(self.cfg.mount_offset)
        offset_orn = p.getQuaternionFromEuler(
            list(self.cfg.mount_orientation_rpy)
        )
        base_pos, base_orn = p.multiplyTransforms(
            ee_pos.tolist(), ee_orn.tolist(),
            offset_pos, offset_orn,
        )
        return np.array(base_pos), list(base_orn)

    # ------------------------------------------------------------- introspection

    def get_opening(self) -> float:
        """Average current per-finger travel (actual, not target)."""
        states = p.getJointStates(
            self.body_id, self.finger_joint_indices,
            physicsClientId=self.world.client_id,
        )
        return float(np.mean([s[0] for s in states]))

    def get_target_opening(self) -> float:
        return self._target_opening

    def get_fingertip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame (position, quaternion) of the midpoint between the
        two fingertips. This is where you actually want to place things
        for grasping — IK targets the arm's ee link, but the pinch point
        is offset by GRIPPER_FINGERTIP_Z_OFFSET along the tool axis."""
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.world.client_id,
        )
        tip_pos, tip_orn = p.multiplyTransforms(
            base_pos, base_orn,
            [0.0, 0.0, GRIPPER_FINGERTIP_Z_OFFSET], [0, 0, 0, 1],
        )
        return np.array(tip_pos), np.array(tip_orn)
