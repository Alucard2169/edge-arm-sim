"""Kinematic AGV base with waypoint patrol.

The AGV is modeled as a rigid box whose pose is set directly each control
tick — no wheels, no tire dynamics, no motor model. This is deliberate:
the thesis's D(t) decisions depend only on *where* the AGV is (position
drives distance-dependent channel quality), not on how it got there.

Path model: a closed loop of waypoints in world coordinates. The AGV
moves between waypoints at a fixed cruise speed and can pause at each
one for a configurable dwell time (needed so the arm can grasp).

The `robot.py` composite (commit 6) will read `pose()` each tick and
teleport the mounted arm to match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pybullet as p

from sim.world import World


# ------------------------------------------------------------------ config

@dataclass
class Waypoint:
    """A stop along the patrol path."""

    xy: tuple[float, float]      # world-frame ground position
    dwell_s: float = 0.0         # seconds to hold still after arriving
    # If set, AGV rotates to face this world-frame xy point on arrival
    # (and also on reset() if this is the start waypoint). Overrides the
    # default "face the next patrol waypoint" behaviour. Use this so the
    # arm ends up pointing at whatever it's supposed to interact with.
    face_xy: tuple[float, float] | None = None


@dataclass
class AGVConfig:
    """Body geometry, motion, and path."""

    # Body dimensions (metres). Loosely sized like a Fetch Freight base.
    body_length: float = 0.60  # x extent
    body_width: float = 0.40   # y extent
    body_height: float = 0.30  # z extent

    # Colour (RGBA). Pick something not-red so it doesn't clash with
    # the IK target marker in earlier scripts.
    body_rgba: tuple[float, float, float, float] = (0.15, 0.35, 0.65, 1.0)

    # Motion.
    cruise_speed: float = 0.5   # m/s along segments

    # Patrol path. Must have at least 2 waypoints. The AGV cycles:
    # wp[0] -> wp[1] -> ... -> wp[-1] -> wp[0] -> ...
    waypoints: tuple[Waypoint, ...] = (
        Waypoint(xy=(1.0, 0.0), dwell_s=1.0),
        Waypoint(xy=(1.0, 1.0), dwell_s=1.0),
        Waypoint(xy=(-1.0, 1.0), dwell_s=1.0),
        Waypoint(xy=(-1.0, 0.0), dwell_s=1.0),
    )

    # Starting waypoint index (AGV begins here, at rest).
    start_wp: int = 0

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError(
                f"Need at least 2 waypoints for a patrol; "
                f"got {len(self.waypoints)}"
            )
        if not (0 <= self.start_wp < len(self.waypoints)):
            raise ValueError(
                f"start_wp={self.start_wp} out of range "
                f"[0, {len(self.waypoints)})"
            )
        if self.cruise_speed <= 0:
            raise ValueError(
                f"cruise_speed must be positive, got {self.cruise_speed}"
            )


# --------------------------------------------------------------------- agv

class AGV:
    """Kinematic AGV base attached to a World.

    Owns:
        - the loaded box body (visual + collision)
        - the patrol schedule (current segment, distance travelled, dwell timer)

    Does not own:
        - the physics tick (World)
        - the arm mounted on top (Robot composite, commit 6)
    """

    def __init__(self, world: World, cfg: AGVConfig | None = None) -> None:
        self.world = world
        self.cfg = cfg or AGVConfig()

        self.body_id: int = -1

        # Patrol state.
        self._current_wp: int = self.cfg.start_wp
        self._dwell_remaining: float = 0.0  # seconds left at current waypoint
        # Wall/sim time of the last control_step call — used to advance
        # the patrol by dt without needing the physics timestep.
        self._last_step_time: float | None = None

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Create the body at the starting waypoint, oriented toward the
        next one. Resets patrol state."""
        if self.world.client_id < 0:
            raise RuntimeError("World.reset() must be called before AGV.reset()")

        cid = self.world.client_id

        # Half-extents for the box collision + visual.
        hx = self.cfg.body_length / 2
        hy = self.cfg.body_width / 2
        hz = self.cfg.body_height / 2

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                  rgbaColor=self.cfg.body_rgba,
                                  physicsClientId=cid)

        # baseMass=0 makes the body kinematic — we control its pose
        # directly with resetBasePositionAndOrientation each tick and
        # PyBullet won't apply gravity or momentum to it.
        wp0 = self.cfg.waypoints[self.cfg.start_wp]
        if wp0.face_xy is not None:
            start_yaw = math.atan2(
                wp0.face_xy[1] - wp0.xy[1], wp0.face_xy[0] - wp0.xy[0],
            )
        else:
            wp1 = self.cfg.waypoints[
                (self.cfg.start_wp + 1) % len(self.cfg.waypoints)
            ]
            start_yaw = math.atan2(
                wp1.xy[1] - wp0.xy[1], wp1.xy[0] - wp0.xy[0],
            )
        start_orn = p.getQuaternionFromEuler([0.0, 0.0, start_yaw])

        self.body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[wp0.xy[0], wp0.xy[1], hz],  # sit on the ground
            baseOrientation=start_orn,
            physicsClientId=cid,
        )

        # Reset patrol state.
        self._current_wp = self.cfg.start_wp
        self._dwell_remaining = wp0.dwell_s
        self._last_step_time = None

    # --------------------------------------------------------------- control

    def control_step(self, sim_time_s: float) -> None:
        """Advance the patrol by (sim_time_s - last_step_time) seconds.

        Register this via World.register_callback(rate_hz, agv.control_step).
        Rate isn't critical — anything from 10 Hz to physics rate works.
        30 Hz is a good default (same as the arm's control tick)."""
        if self._last_step_time is None:
            self._last_step_time = sim_time_s
            return
        dt = sim_time_s - self._last_step_time
        self._last_step_time = sim_time_s
        if dt <= 0:
            return

        # If we're dwelling at a waypoint, count down and don't move.
        if self._dwell_remaining > 0:
            self._dwell_remaining = max(0.0, self._dwell_remaining - dt)
            return

        # Otherwise, advance along the segment from current -> next.
        distance_to_cover = self.cfg.cruise_speed * dt
        self._advance(distance_to_cover)

    def _advance(self, distance: float) -> None:
        """Move `distance` metres along the patrol, handling waypoint
        crossings and dwells. Recursive-tail style: if we overshoot a
        waypoint by any amount, we settle at it, start its dwell, and
        the *remaining* distance is discarded (we'll pick up motion
        after the dwell expires).

        Discarding overshoot distance is a simplification. At 30 Hz
        control and 0.5 m/s cruise, each tick is ~16 mm — the position
        error from discarding is at most that. Fine for our purposes.
        """
        wps = self.cfg.waypoints
        n = len(wps)

        cur_xy = np.array(self._get_xy())
        next_wp = wps[(self._current_wp + 1) % n]
        target_xy = np.array(next_wp.xy)

        to_target = target_xy - cur_xy
        remaining = float(np.linalg.norm(to_target))

        if remaining <= distance:
            # Arrive at the next waypoint.
            if next_wp.face_xy is not None:
                arrival_yaw = math.atan2(
                    next_wp.face_xy[1] - target_xy[1],
                    next_wp.face_xy[0] - target_xy[0],
                )
            else:
                arrival_yaw = self._yaw_toward(target_xy, cur_xy)
            self._set_pose(target_xy, arrival_yaw)
            self._current_wp = (self._current_wp + 1) % n
            self._dwell_remaining = next_wp.dwell_s
            return

        # Step along the segment.
        direction = to_target / remaining
        new_xy = cur_xy + direction * distance
        yaw = math.atan2(direction[1], direction[0])
        self._set_pose(new_xy, yaw)

    # ------------------------------------------------------------ pose I/O

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current world-frame (position xyz, quaternion xyzw)."""
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.world.client_id,
        )
        return np.array(pos, dtype=np.float64), np.array(orn, dtype=np.float64)

    def top_center(self) -> np.ndarray:
        """World-frame xyz of the centre of the AGV's top surface. This
        is where the arm base will mount in commit 6."""
        pos, _ = self.pose()
        return np.array([pos[0], pos[1], pos[2] + self.cfg.body_height / 2])

    def is_dwelling(self) -> bool:
        return self._dwell_remaining > 0

    def current_waypoint_index(self) -> int:
        return self._current_wp

    def resume(self) -> None:
        """End the current dwell so the AGV starts moving toward the
        next waypoint on its next control tick. No-op if not dwelling.
        Used by external controllers (grasp FSM) to drive the AGV on
        command, when waypoints are configured with very large dwells."""
        self._dwell_remaining = 0.0

    def is_at_waypoint(self, idx: int) -> bool:
        """True if the AGV has arrived at waypoint `idx` and is currently
        dwelling there. If waypoint dwell was set to a large value, this
        stays true until resume() is called."""
        return self._current_wp == idx and self.is_dwelling()

    # ---------------------------------------------------------------- private

    def _get_xy(self) -> tuple[float, float]:
        pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.world.client_id,
        )
        return (pos[0], pos[1])

    def _set_pose(self, xy: Sequence[float], yaw: float) -> None:
        z = self.cfg.body_height / 2
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(
            self.body_id, [xy[0], xy[1], z], orn,
            physicsClientId=self.world.client_id,
        )

    def _yaw_toward(self, target_xy: np.ndarray,
                    from_xy: np.ndarray) -> float:
        d = target_xy - from_xy
        if np.linalg.norm(d) < 1e-6:
            # Degenerate — keep current yaw.
            _, orn = self.pose()
            return p.getEulerFromQuaternion(orn.tolist())[2]
        return math.atan2(d[1], d[0])
