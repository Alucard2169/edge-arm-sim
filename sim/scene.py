"""Two bins and objects the arm will pick from.

Each bin is 5 static bodies: a floor and 4 walls. Objects are simple
primitives (cubes, cylinders, spheres) in varied sizes and colours,
spawned inside bin A with small random rotations. Reset re-spawns.

YCB meshes could plug in here as a drop-in replacement for the primitive
factory — see `_spawn_object`. Left as a TODO because primitives are
enough to prove the physics work and don't add a package dependency.

Bin coordinates are in world frame — they're static furniture. AGV
waypoints in the checkpoint script are placed so the arm's workspace
covers the bin interior when the AGV is parked beside it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pybullet as p

from sim.world import World


# ------------------------------------------------------------------ config

@dataclass
class BinSpec:
    """Static bin at a fixed world-frame position."""

    center_xy: tuple[float, float]     # xy of the bin's inner floor centre
    inner_size: tuple[float, float, float] = (0.30, 0.30, 0.15)  # L, W, H (inner)
    wall_thickness: float = 0.01
    rgba: tuple[float, float, float, float] = (0.55, 0.35, 0.18, 1.0)  # wood brown


@dataclass
class ObjectSpec:
    """A single object to spawn in bin A."""

    shape: str                          # 'box' | 'cylinder' | 'sphere'
    size: tuple[float, ...]             # box: (hx, hy, hz); cyl: (r, h); sph: (r,)
    mass: float
    rgba: tuple[float, float, float, float]


@dataclass
class SceneConfig:
    """All tunable knobs for the scene."""

    bin_a: BinSpec = field(default_factory=lambda: BinSpec(
        center_xy=(1.5, 0.0),
        rgba=(0.55, 0.35, 0.18, 1.0),
    ))
    bin_b: BinSpec = field(default_factory=lambda: BinSpec(
        center_xy=(-1.5, 0.0),
        rgba=(0.30, 0.45, 0.30, 1.0),
    ))

    # Objects to spawn in bin A. Defaults give a small assortment.
    objects: tuple[ObjectSpec, ...] = (
        ObjectSpec(shape="box",      size=(0.025, 0.025, 0.025), mass=0.05,
                   rgba=(0.90, 0.15, 0.15, 1.0)),
        ObjectSpec(shape="box",      size=(0.030, 0.020, 0.020), mass=0.06,
                   rgba=(0.15, 0.65, 0.90, 1.0)),
        ObjectSpec(shape="cylinder", size=(0.020, 0.060),         mass=0.05,
                   rgba=(0.95, 0.85, 0.20, 1.0)),
        ObjectSpec(shape="cylinder", size=(0.018, 0.050),         mass=0.04,
                   rgba=(0.60, 0.20, 0.80, 1.0)),
        ObjectSpec(shape="sphere",   size=(0.022,),               mass=0.05,
                   rgba=(0.30, 0.85, 0.30, 1.0)),
    )

    # Random seed for object placement — same seed => same layout, so
    # sim runs are reproducible.
    placement_seed: int = 42

    # Height above the bin floor at which objects are dropped in.
    # Small drop lets them settle naturally without launching them.
    drop_height: float = 0.05


# ------------------------------------------------------------------- scene

class Scene:
    """Two bins and the objects in bin A.

    Owns:
        - the static bin bodies (10 total: 5 per bin)
        - the dynamic object bodies (one per ObjectSpec)

    Does not own:
        - the robot; scene knows nothing about it
        - the ground plane; that's World's business
    """

    def __init__(self, world: World, cfg: SceneConfig | None = None) -> None:
        self.world = world
        self.cfg = cfg or SceneConfig()

        # World-frame body ids.
        self.bin_a_ids: list[int] = []   # 5 bodies making up bin A
        self.bin_b_ids: list[int] = []   # 5 bodies making up bin B
        self.object_ids: list[int] = []  # one per ObjectSpec

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Build both bins and drop objects into bin A."""
        if self.world.client_id < 0:
            raise RuntimeError("World.reset() must be called before Scene.reset()")

        # Remove anything from a previous reset.
        self._despawn_all()

        self.bin_a_ids = self._build_bin(self.cfg.bin_a)
        self.bin_b_ids = self._build_bin(self.cfg.bin_b)
        self._spawn_objects_in_bin_a()

    def _despawn_all(self) -> None:
        cid = self.world.client_id
        for bid in self.bin_a_ids + self.bin_b_ids + self.object_ids:
            try:
                p.removeBody(bid, physicsClientId=cid)
            except p.error:
                pass
        self.bin_a_ids = []
        self.bin_b_ids = []
        self.object_ids = []

    # ---------------------------------------------------------------- bins

    def _build_bin(self, spec: BinSpec) -> list[int]:
        """Return the 5 body ids (floor + 4 walls) for one bin.

        Coordinate convention:
            - `center_xy` is the centre of the *inner* floor
            - inner_size is the usable interior (walls sit outside this)
            - the whole bin rests on z=0 (bottom of the floor slab)
        """
        cid = self.world.client_id
        cx, cy = spec.center_xy
        L, W, H = spec.inner_size
        t = spec.wall_thickness

        bodies: list[int] = []

        # Floor: sits at z in [0, t]; extent covers inner + wall footprints
        # so walls don't hang in the air.
        floor_L = L + 2 * t
        floor_W = W + 2 * t
        bodies.append(self._make_static_box(
            half_ext=(floor_L / 2, floor_W / 2, t / 2),
            pos=(cx, cy, t / 2),
            rgba=spec.rgba,
        ))

        # Walls: 4 slabs, each running the full inner length (or width) plus
        # the two wall thicknesses so the corners meet cleanly. Height H.
        # Sit on top of the floor.
        wall_z = t + H / 2

        # +y wall (front)
        bodies.append(self._make_static_box(
            half_ext=(floor_L / 2, t / 2, H / 2),
            pos=(cx, cy + W / 2 + t / 2, wall_z),
            rgba=spec.rgba,
        ))
        # -y wall (back)
        bodies.append(self._make_static_box(
            half_ext=(floor_L / 2, t / 2, H / 2),
            pos=(cx, cy - W / 2 - t / 2, wall_z),
            rgba=spec.rgba,
        ))
        # +x wall (right)
        bodies.append(self._make_static_box(
            half_ext=(t / 2, W / 2, H / 2),
            pos=(cx + L / 2 + t / 2, cy, wall_z),
            rgba=spec.rgba,
        ))
        # -x wall (left)
        bodies.append(self._make_static_box(
            half_ext=(t / 2, W / 2, H / 2),
            pos=(cx - L / 2 - t / 2, cy, wall_z),
            rgba=spec.rgba,
        ))

        return bodies

    def _make_static_box(
        self, half_ext: tuple[float, float, float],
        pos: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
    ) -> int:
        cid = self.world.client_id
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_ext),
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_ext),
                                  rgbaColor=list(rgba),
                                  physicsClientId=cid)
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pos),
            physicsClientId=cid,
        )

    # ------------------------------------------------------------- objects

    def _spawn_objects_in_bin_a(self) -> None:
        """Drop each object at a random xy inside bin A's inner footprint,
        with a small random yaw. z is drop_height above the inner floor."""
        rng = random.Random(self.cfg.placement_seed)
        spec = self.cfg.bin_a
        cx, cy = spec.center_xy
        L, W, _ = spec.inner_size

        # Keep objects inset from the walls by a margin so they don't
        # spawn intersecting a wall.
        margin = 0.04
        x_range = (cx - L / 2 + margin, cx + L / 2 - margin)
        y_range = (cy - W / 2 + margin, cy + W / 2 - margin)
        # Objects rest on the bin's inner floor (top face of the floor slab).
        floor_top_z = spec.wall_thickness
        drop_z = floor_top_z + self.cfg.drop_height

        for obj in self.cfg.objects:
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            yaw = rng.uniform(-np.pi, np.pi)
            body_id = self._spawn_object(obj, (x, y, drop_z), yaw)
            self.object_ids.append(body_id)

    def _spawn_object(
        self, obj: ObjectSpec, pos: tuple[float, float, float], yaw: float,
    ) -> int:
        """Create one dynamic object body. Swap this factory for a YCB
        loader later (load model.urdf from pybullet_object_models) — the
        caller doesn't care."""
        cid = self.world.client_id
        if obj.shape == "box":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(obj.size),
                                         physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(obj.size),
                                      rgbaColor=list(obj.rgba),
                                      physicsClientId=cid)
        elif obj.shape == "cylinder":
            radius, height = obj.size
            col = p.createCollisionShape(p.GEOM_CYLINDER,
                                         radius=radius, height=height,
                                         physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_CYLINDER,
                                      radius=radius, length=height,
                                      rgbaColor=list(obj.rgba),
                                      physicsClientId=cid)
        elif obj.shape == "sphere":
            (radius,) = obj.size
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                         physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                      rgbaColor=list(obj.rgba),
                                      physicsClientId=cid)
        else:
            raise ValueError(f"Unknown object shape: {obj.shape!r}")

        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        return p.createMultiBody(
            baseMass=obj.mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pos),
            baseOrientation=orn,
            physicsClientId=cid,
        )

    # ---------------------------------------------------------- introspection

    def object_positions(self) -> list[np.ndarray]:
        """World-frame position of every spawned object, in spawn order."""
        cid = self.world.client_id
        out: list[np.ndarray] = []
        for bid in self.object_ids:
            pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=cid)
            out.append(np.array(pos, dtype=np.float64))
        return out
