"""PyBullet world wrapper with a rate-scheduled tick loop.

The `World` owns the physics client, timestep, ground plane, and debug camera.
Everything else in the sim (arm, AGV, scene, later D(t)) attaches to it via
`register_callback(rate_hz, fn)` and runs at its own rate on top of the fixed
physics step.

Why the callback design: phase 5 needs to add a decision loop (~1 Hz) without
touching physics code. Building the scheduler now means that hook already
exists when we need it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pybullet as p
import pybullet_data


@dataclass
class WorldConfig:
    """All tunable knobs for the physics world. Keep defaults sensible."""

    gui: bool = True
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    physics_hz: int = 240  # PyBullet's recommended default
    realtime: bool = True  # sleep to match wall clock; ignored in headless

    # Debug camera pose (used only in GUI mode)
    cam_distance: float = 2.5
    cam_yaw: float = 45.0
    cam_pitch: float = -30.0
    cam_target: tuple[float, float, float] = (0.0, 0.0, 0.3)


@dataclass
class _Callback:
    """A user-registered callback plus its scheduling state."""

    rate_hz: float
    fn: Callable[[float], None]  # receives sim_time in seconds
    _interval_steps: int = field(init=False)
    _next_step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # Set later once we know physics_hz. Placeholder so dataclass is happy.
        self._interval_steps = 1


class World:
    """PyBullet client + fixed-timestep tick loop with rate-scheduled callbacks.

    Usage:
        world = World(WorldConfig())
        world.reset()
        arm = Arm(world, ArmConfig())          # attaches via world.client_id
        world.register_callback(30.0, arm.control_step)
        world.run(duration_s=10.0)
    """

    def __init__(self, cfg: WorldConfig | None = None) -> None:
        self.cfg = cfg or WorldConfig()
        self.client_id: int = -1
        self.plane_id: int = -1
        self._callbacks: list[_Callback] = []
        self._step_count: int = 0
        self._dt: float = 1.0 / self.cfg.physics_hz

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        """Connect (or reconnect) the client and rebuild the base scene."""
        if self.client_id >= 0:
            p.disconnect(self.client_id)

        mode = p.GUI if self.cfg.gui else p.DIRECT
        self.client_id = p.connect(mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client_id)
        p.setGravity(*self.cfg.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self._dt, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(fixedTimeStep=self._dt,
                                    physicsClientId=self.client_id)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        if self.cfg.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self.cfg.cam_distance,
                cameraYaw=self.cfg.cam_yaw,
                cameraPitch=self.cfg.cam_pitch,
                cameraTargetPosition=self.cfg.cam_target,
                physicsClientId=self.client_id,
            )
            # Turn off the GUI panels that clutter the view; keep the render.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self.client_id)

        self._step_count = 0
        # Reset callback schedule so a re-run starts clean.
        for cb in self._callbacks:
            cb._next_step = 0

    def close(self) -> None:
        if self.client_id >= 0:
            p.disconnect(self.client_id)
            self.client_id = -1

    # ------------------------------------------------------------- callbacks

    def register_callback(self, rate_hz: float,
                          fn: Callable[[float], None]) -> None:
        """Register `fn(sim_time_s)` to run at approximately `rate_hz`.

        Rates are quantised to whole physics steps: the actual rate is
        physics_hz / round(physics_hz / rate_hz). At 240 Hz physics, a
        request of 30 Hz becomes exactly 30 Hz; 1 Hz becomes exactly 1 Hz.
        Non-divisor rates get rounded to the nearest achievable value.
        """
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        if rate_hz > self.cfg.physics_hz:
            raise ValueError(
                f"rate_hz={rate_hz} exceeds physics_hz={self.cfg.physics_hz}"
            )

        cb = _Callback(rate_hz=rate_hz, fn=fn)
        cb._interval_steps = max(1, round(self.cfg.physics_hz / rate_hz))
        self._callbacks.append(cb)

    # ------------------------------------------------------------------ run

    def step(self) -> None:
        """Advance physics by one tick and fire any due callbacks."""
        # Fire callbacks scheduled for this step BEFORE stepping physics,
        # so control commands set this tick take effect this tick.
        sim_time = self._step_count * self._dt
        for cb in self._callbacks:
            if self._step_count >= cb._next_step:
                cb.fn(sim_time)
                cb._next_step = self._step_count + cb._interval_steps

        p.stepSimulation(physicsClientId=self.client_id)
        self._step_count += 1

    def run(self, duration_s: float | None = None) -> None:
        """Run the tick loop. If `duration_s` is None, run until the GUI
        window is closed (or forever in headless mode -- don't do that).
        """
        if duration_s is None and not self.cfg.gui:
            raise ValueError(
                "duration_s must be set when running headless "
                "(cfg.gui=False), otherwise this would loop forever"
            )

        end_step = (int(duration_s * self.cfg.physics_hz)
                    if duration_s is not None else None)
        # Wall-clock pacing only matters in GUI mode; headless runs flat-out.
        pace = self.cfg.realtime and self.cfg.gui

        t_wall_start = time.perf_counter()

        try:
            while True:
                if end_step is not None and self._step_count >= end_step:
                    break
                if self.cfg.gui and not p.isConnected(self.client_id):
                    break  # user closed the window

                self.step()

                if pace:
                    target_wall = t_wall_start + self._step_count * self._dt
                    lag = target_wall - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
        except KeyboardInterrupt:
            pass  # clean Ctrl-C exit

    # ---------------------------------------------------------------- introspection

    @property
    def sim_time(self) -> float:
        """Seconds of simulated time since the last reset()."""
        return self._step_count * self._dt

    @property
    def step_count(self) -> int:
        return self._step_count
