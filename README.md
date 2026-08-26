# edge-arm-sim

Simulation environment for the M.Tech thesis:
**Rule-Based Adaptive Offloading for Energy- and Network-Aware Robotic Arm Manipulation at the Edge.**

An AGV-mounted xArm 6 patrols between bins and performs vision-guided pick-and-place.
A rule table `D(t) = f(B(t), C(t))` selects among light-local / heavy-local / edge-offload
execution paths per grasp cycle, based on battery state `B(t)` and channel quality `C(t)`.

## Phase 4 scope (this repo, current)

Build the physical simulation only. No research logic yet.

Deliverable: **end-to-end grasp demo on a patrolling AGV.**

Build order (one checkpoint per script under `scripts/`):

1. World: PyBullet client, ground, camera
2. Arm + IK to world-frame target — `scripts/s2_ik_check.py`
3. AGV patrol along waypoints — `scripts/s3_agv_patrol.py`
4. Arm mounted on AGV, holds home pose while moving — `scripts/s4_mounted.py`
5. Scene: two bins + YCB objects — `scripts/s5_scene.py`
6. Static grasp cycle (AGV parked) — `scripts/s6_static_grasp.py`
7. Patrol grasp cycle (AGV moves between bins) — `scripts/s7_patrol_grasp.py`

## Out of scope for phase 4

Battery model, network model, D(t) selector, offloading paths, perception model,
grip-force estimation. All of that is phase 5 and plugs into the tick loop
defined in `sim/world.py` without touching physics code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a checkpoint

```bash
python -m scripts.s2_ik_check
```

Each `sX_*` script is self-contained and demonstrates that checkpoint works.

## Layout

```
sim/        core modules (one per block on the roadmap)
configs/    YAML config per module
scripts/    checkpoint demos, one per build step
tests/      unit tests
```
