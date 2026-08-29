"""Pick-and-place state machine for the mobile manipulator.

The FSM drives one grasp cycle: pick a target object from bin A, move
the arm through approach/descend/close/lift, transit (teleport AGV to
bin B for now — patrolling motion is s7), descend/open/lift-away over
bin B. Each state's entry sets an arm joint target (via one IK call)
and/or gripper command; each state's tick tests exit conditions.

Design rules:
    - IK is called EXACTLY ONCE per state entry. The resulting joint
      target is stored; the kinematic Arm.control_step advances the
      arm toward it at bounded velocity, so convergence is deterministic
      and doesn't depend on gain tuning.
    - Arm joints are controlled kinematically (see sim/arm.py). This
      is consistent with the AGV and gripper mount, which are also
      kinematic. Fingers stay under POSITION_CONTROL so grip contacts
      are physical.
    - Object hold across LIFT/TRANSIT/DESCEND_DROP is enforced by a
      JOINT_FIXED constraint (created on CLOSE, removed before OPEN).
      Kinematic arm motion can't rely on finger friction across sim
      ticks. Standard PyBullet pick-and-place pattern.
    - IK targets the arm's ee link (link6). The fingertip is 7 cm
      further along the tool axis. When we ask "put the fingertip at
      world pos X" and the tool is pointing down, the ee link target
      is X + 7 cm in world +z.
    - Convergence check is JOINT-SPACE: arm actual joints within
      tolerance of commanded joints.
    - Position-only IK (no target_orn). The rest-pose bias (home pose
      is already tool-down) keeps the tool close to vertical without
      forcing the solver into workspace-edge oscillation.

The FSM does NOT own perception, target selection heuristics, or the
D(t) offloading decision. It gets told what object to grasp and executes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p

from sim.battery import Battery
from sim.decision import Decision
from sim.gripper import GRIPPER_FINGERTIP_Z_OFFSET
from sim.network import NetworkChannel
from sim.paths import HeavyLocal, LightLocal, Offload, Path, PathResult
from sim.robot import Robot
from sim.scene import Scene
from sim.world import World


# ------------------------------------------------------------------ states

class GraspState(enum.Enum):
    IDLE = "idle"
    APPROACH = "approach"           # move fingertip above target
    INFER = "infer"                 # D(t) selects a path; execute + drain
    DESCEND = "descend"             # descend to grasp height
    CLOSE = "close"                 # close gripper
    LIFT = "lift"                   # lift back to approach height
    TRANSIT = "transit"             # drive AGV from pick to drop waypoint
    APPROACH_DROP = "approach_drop" # move over bin B drop point
    DESCEND_DROP = "descend_drop"   # descend to release height
    OPEN = "open"                   # release
    LIFT_AWAY = "lift_away"         # retract
    RETURN_HOME = "return_home"     # drive AGV back to pick waypoint
    DONE = "done"
    FAILED = "failed"


# ------------------------------------------------------------------ config

@dataclass
class GraspConfig:
    """Tunable parameters for one pick-and-place cycle."""

    # Vertical clearances (all in metres, world frame).
    approach_height: float = 0.20     # fingertip Z above bin floor for APPROACH
    grasp_clearance: float = 0.005    # fingertip Z above object top at CLOSE
    lift_height: float = 0.25         # fingertip Z above bin floor for LIFT
    drop_height: float = 0.10         # fingertip Z above bin B floor at OPEN

    # Joint-space convergence tolerance (radians). Arm considered "at
    # target" when max abs joint error is below this. ~1 degree.
    joint_tol: float = 0.02
    # Consecutive ticks of tolerance-met required to advance.
    stability_ticks: int = 3

    # Per-state safety timeout (seconds).
    state_timeout_s: float = 10.0
    # Timeout for driving states (TRANSIT, RETURN_HOME). AGV crossing
    # a 2 m gap at 0.5 m/s takes 4 s, so this needs enough headroom.
    drive_timeout_s: float = 20.0

    # Time to hold after commanding gripper open/close before advancing.
    gripper_settle_s: float = 0.6

    # Waypoint indices for pick and drop.
    pick_wp_index: int = 0
    drop_wp_index: int = 1


# --------------------------------------------------------------- outcome

@dataclass
class CycleOutcome:
    """Per-cycle metric bundle produced when a GraspCycle completes.

    Aggregated by MultiCycleGraspRunner and consumed by the CSV logger
    in commit 5. `path_result` is the raw PathResult from the chosen
    execution path; the other fields are the FSM-level bookkeeping
    (which object, wall-clock start/end, final state)."""

    cycle_index: int
    target_object_id: int
    final_state: str                # GraspState.value at end
    # D(t) inputs at the moment of the decision.
    battery_low: bool = False
    network_good: bool = False
    battery_soc_at_decision: float = 0.0
    data_rate_bps: float = 0.0
    # Path chosen and what it produced.
    path_name: str = ""
    path_result: Optional[PathResult] = None
    # Wall-clock timing (sim seconds).
    started_at_s: float = 0.0
    finished_at_s: float = 0.0
    # Was the object physically placed in bin B?
    physically_delivered: bool = False


# --------------------------------------------------------------- FSM class

class GraspCycle:
    """One-shot pick-and-place FSM.

    Usage:
        cycle = GraspCycle(world, robot, scene, GraspConfig())
        cycle.start(target_object_id=scene.object_ids[0])
        world.register_callback(30.0, cycle.tick)
        world.run(...)
        print(cycle.state)  # DONE or FAILED
    """

    def __init__(
        self,
        world: World,
        robot: Robot,
        scene: Scene,
        cfg: GraspConfig | None = None,
        # Phase-5 modules. All None = phase-4 behavior: INFER state
        # skipped, no battery drain, no D(t) consultation. Pass all
        # four for full phase-5 semantics.
        battery: Optional[Battery] = None,
        network: Optional[NetworkChannel] = None,
        decision: Optional[Decision] = None,
        paths: Optional[dict[str, Path]] = None,
    ) -> None:
        self.world = world
        self.robot = robot
        self.scene = scene
        self.cfg = cfg or GraspConfig()

        # Phase-5 wiring. If any of these are None, we run in phase-4
        # mode (skip INFER entirely). If all are present, INFER
        # consults Decision and executes the chosen Path.
        self.battery = battery
        self.network = network
        self.decision = decision
        # Default paths registry — one instance per path name. Caller can
        # override to inject configured Path subclasses (e.g. ablations).
        if paths is None and decision is not None:
            paths = {
                "light-local": LightLocal(),
                "heavy-local": HeavyLocal(),
                "offload":     Offload(),
            }
        self.paths = paths or {}

        self.state: GraspState = GraspState.IDLE
        self._target_object_id: int = -1
        self._pick_xy: tuple[float, float] = (0.0, 0.0)
        self._drop_xy: tuple[float, float] = (0.0, 0.0)
        self._cycle_index: int = 0

        # Kept for reporting only (the magenta debug marker reads this).
        self._ik_target_pos: Optional[np.ndarray] = None

        # Joint target commanded on state entry. Convergence is checked
        # against this in tick(). None when the current state isn't
        # driving arm motion (CLOSE, TRANSIT, OPEN, INFER).
        self._q_target: Optional[np.ndarray] = None

        # Fixed constraint between gripper base and target object, active
        # from CLOSE through OPEN. Kinematic arm motion can't rely on
        # finger friction to hold objects across sim ticks, so we glue
        # the object to the gripper during the hold phase. Standard
        # PyBullet pick-and-place pattern.
        self._grasp_constraint: Optional[int] = None

        # State bookkeeping.
        self._state_entry_time: float = 0.0
        self._stability_counter: int = 0

        # INFER-specific state.
        self._infer_wait_until_s: float = 0.0
        self._infer_result: Optional[PathResult] = None
        self._infer_battery_low: bool = False
        self._infer_network_good: bool = False
        self._infer_battery_soc: float = 0.0
        self._infer_data_rate_bps: float = 0.0

    def _phase5_active(self) -> bool:
        """True if all the phase-5 modules are wired in."""
        return (self.battery is not None
                and self.network is not None
                and self.decision is not None
                and bool(self.paths))

    # ------------------------------------------------------------------ API

    def start(self, target_object_id: int, cycle_index: int = 0) -> None:
        """Kick off a cycle targeting the given object."""
        cid = self.world.client_id
        if target_object_id not in self.scene.object_ids:
            raise ValueError(
                f"Target object {target_object_id} not in scene.object_ids"
            )
        self._target_object_id = target_object_id
        self._cycle_index = cycle_index

        obj_pos, _ = p.getBasePositionAndOrientation(
            target_object_id, physicsClientId=cid,
        )
        self._pick_xy = (obj_pos[0], obj_pos[1])
        self._drop_xy = self.scene.cfg.bin_b.center_xy

        # Reset per-cycle INFER state.
        self._infer_result = None
        self._infer_battery_low = False
        self._infer_network_good = False
        self._infer_battery_soc = 0.0
        self._infer_data_rate_bps = 0.0

        self._enter_state(GraspState.APPROACH, self.world.sim_time)

    def tick(self, sim_time_s: float) -> None:
        """Advance the FSM. Register as a World callback at 30 Hz.

        Note: does NOT call IK. IK runs once per state entry. This tick
        just watches for the state's exit condition and safety timeout.
        """
        if self.state in (GraspState.IDLE, GraspState.DONE, GraspState.FAILED):
            return

        # Per-state timeout selection.
        if self.state in (GraspState.TRANSIT, GraspState.RETURN_HOME):
            # Driving states get a longer timeout because AGV cruise
            # across the workspace takes several seconds.
            timeout = self.cfg.drive_timeout_s
        elif self.state == GraspState.INFER:
            # INFER's wait duration equals the executed path's latency,
            # which for offload-far can be seconds. Give it its own
            # latency + a fixed margin, rather than the fixed default.
            if self._infer_result is not None:
                timeout = self._infer_result.latency_s + self.cfg.state_timeout_s
            else:
                timeout = self.cfg.state_timeout_s
        else:
            timeout = self.cfg.state_timeout_s

        if sim_time_s - self._state_entry_time > timeout:
            print(f"[grasp] TIMEOUT in state {self.state.value} "
                  f"after {timeout:.1f}s")
            self._enter_state(GraspState.FAILED, sim_time_s)
            return

        self._check_transition(sim_time_s)

    # -------------------------------------------------------- state entry

    def _enter_state(self, new_state: GraspState, now: float) -> None:
        prev = self.state
        self.state = new_state
        self._state_entry_time = now
        self._stability_counter = 0
        self._q_target = None
        print(f"[grasp t={now:6.2f}s] {prev.value:>14s} -> {new_state.value}")

        pick = self._pick_xy
        drop = self._drop_xy
        bin_a_floor_z = self.scene.cfg.bin_a.wall_thickness
        bin_b_floor_z = self.scene.cfg.bin_b.wall_thickness

        # States that drive arm motion — compute IK and command joints.
        motion_targets: dict[GraspState, np.ndarray] = {}

        if new_state == GraspState.APPROACH:
            motion_targets[new_state] = np.array([
                pick[0], pick[1], bin_a_floor_z + self.cfg.approach_height,
            ])

        elif new_state == GraspState.INFER:
            # Consult D(t): sample battery and network at this exact
            # cycle, look up the rule table, execute the chosen path,
            # and schedule a wait for the path's reported latency.
            # Battery drain happens on state EXIT (physically correct:
            # energy is spent during compute, not instantly), so we
            # only *record* here what will be drained later.
            self._run_infer(now)

        elif new_state == GraspState.DESCEND:
            obj_pos, _ = p.getBasePositionAndOrientation(
                self._target_object_id,
                physicsClientId=self.world.client_id,
            )
            motion_targets[new_state] = np.array([
                pick[0], pick[1],
                obj_pos[2] + self.cfg.grasp_clearance,
            ])

        elif new_state == GraspState.CLOSE:
            # Attach the object FIRST, then close the fingers. Doing it
            # in this order means the object is rigidly glued to the
            # gripper before physics-based finger closure can push it
            # around. Visually the fingers still close on the object,
            # but no jitter from contact resolution.
            self._create_grasp_constraint()
            self.robot.arm.gripper.close()

        elif new_state == GraspState.LIFT:
            motion_targets[new_state] = np.array([
                pick[0], pick[1], bin_a_floor_z + self.cfg.lift_height,
            ])

        elif new_state == GraspState.TRANSIT:
            # Kick the AGV off its dwell so it starts moving toward the
            # next waypoint (which is the drop waypoint in a 2-waypoint
            # patrol). Convergence checked in _check_transition via
            # agv.is_at_waypoint(drop_wp_index).
            self.robot.agv.resume()

        elif new_state == GraspState.APPROACH_DROP:
            motion_targets[new_state] = np.array([
                drop[0], drop[1], bin_b_floor_z + self.cfg.approach_height,
            ])

        elif new_state == GraspState.DESCEND_DROP:
            motion_targets[new_state] = np.array([
                drop[0], drop[1], bin_b_floor_z + self.cfg.drop_height,
            ])

        elif new_state == GraspState.OPEN:
            # Release the constraint BEFORE opening the gripper so the
            # object falls naturally under gravity as the fingers open.
            self._release_grasp_constraint()
            self.robot.arm.gripper.open()

        elif new_state == GraspState.LIFT_AWAY:
            motion_targets[new_state] = np.array([
                drop[0], drop[1], bin_b_floor_z + self.cfg.lift_height,
            ])

        elif new_state == GraspState.RETURN_HOME:
            # Drive AGV back to the pick waypoint. Same pattern as
            # TRANSIT: kick off the dwell, wait for arrival.
            self.robot.agv.resume()

        # Solve IK once (if this state moves the arm) and commit the
        # joint target. Kinematic control_step advances the arm to the
        # target smoothly at bounded velocity — no snap needed.
        if new_state in motion_targets:
            fingertip_target = motion_targets[new_state]
            self._ik_target_pos = fingertip_target
            ee_target = fingertip_target + np.array([
                0.0, 0.0, GRIPPER_FINGERTIP_Z_OFFSET,
            ])
            q = self.robot.arm.solve_ik(target_pos=ee_target)
            self._q_target = q
            self.robot.arm.set_joint_positions(q)

            # Diagnostic: what does the IK solution predict the tip pose
            # will be, once the arm reaches it? Log for debugging.
            self._log_ik_prediction(fingertip_target, ee_target, q)
        else:
            self._ik_target_pos = None

    def _log_ik_prediction(
        self,
        fingertip_target: np.ndarray,
        ee_target: np.ndarray,
        q_ik: np.ndarray,
    ) -> None:
        """Temporarily snap the arm to q_ik, read the FK ee/tip pose,
        then restore the current joint state. Prints how close the IK
        solution places the fingertip to the requested target."""
        cid = self.world.client_id
        arm = self.robot.arm
        current_q = arm.get_joint_positions().copy()

        for idx, val in zip(arm.joint_indices, q_ik):
            p.resetJointState(arm.body_id, idx, float(val),
                              targetVelocity=0.0, physicsClientId=cid)
        # Need to also teleport gripper to see where fingertip lands.
        if arm.gripper is not None:
            arm.gripper.teleport_to_arm_ee()
            tip_pos, _ = arm.gripper.get_fingertip_pose()
        else:
            tip_pos, _ = arm.get_ee_pose()

        for idx, val in zip(arm.joint_indices, current_q):
            p.resetJointState(arm.body_id, idx, float(val),
                              targetVelocity=0.0, physicsClientId=cid)
        if arm.gripper is not None:
            arm.gripper.teleport_to_arm_ee()

        tip_err_mm = float(np.linalg.norm(tip_pos - fingertip_target)) * 1000.0
        print(f"           IK: ee_tgt={np.round(ee_target, 3)}  "
              f"predicted tip={np.round(tip_pos, 3)}  "
              f"tip_err={tip_err_mm:5.1f}mm")

    # -------------------------------------------------- grasp constraint

    def _create_grasp_constraint(self) -> None:
        """Fix the target object to the gripper base with a JOINT_FIXED
        constraint. The child frame position places the object at the
        current fingertip location, so it appears held between the
        fingers."""
        if self._grasp_constraint is not None:
            return  # already held
        cid = self.world.client_id
        gripper_id = self.robot.arm.gripper.body_id

        # Compute the object's pose relative to the gripper base right now.
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(
            gripper_id, physicsClientId=cid,
        )
        obj_pos, obj_orn = p.getBasePositionAndOrientation(
            self._target_object_id, physicsClientId=cid,
        )
        inv_g_pos, inv_g_orn = p.invertTransform(gripper_pos, gripper_orn)
        rel_pos, rel_orn = p.multiplyTransforms(
            inv_g_pos, inv_g_orn, obj_pos, obj_orn,
        )

        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=gripper_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self._target_object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(rel_pos),
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=list(rel_orn),
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=cid,
        )
        # Make the constraint stiff so the object holds rigidly during
        # arm motion and AGV teleport. Default maxForce is low enough
        # that the object visibly lags behind the gripper on sudden
        # motions.
        p.changeConstraint(
            self._grasp_constraint, maxForce=10000,
            physicsClientId=cid,
        )
        # Zero the object's velocity so any transient momentum from
        # finger contacts or drop-settling doesn't cause spring-back.
        p.resetBaseVelocity(
            self._target_object_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=cid,
        )

    def _release_grasp_constraint(self) -> None:
        if self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint,
                               physicsClientId=self.world.client_id)
            self._grasp_constraint = None

    def _run_infer(self, now: float) -> None:
        """INFER state entry: consult D(t), execute the chosen path,
        schedule a wait for the simulated latency. Battery drain
        happens on state exit, not here."""
        assert self._phase5_active(), "_run_infer requires phase-5 modules"

        # Sample B(t) and C(t) once each.
        battery_low = self.battery.is_low()
        agv_pos, _ = self.robot.agv.pose()
        agv_xy = (float(agv_pos[0]), float(agv_pos[1]))
        data_rate, network_good = self.network.sample(agv_xy)

        # Record for logging in the CycleOutcome.
        self._infer_battery_low = battery_low
        self._infer_network_good = network_good
        self._infer_battery_soc = self.battery.state_of_charge()
        self._infer_data_rate_bps = data_rate

        # Consult D(t) and dispatch.
        path_name = self.decision.decide(
            cycle=self._cycle_index,
            battery_low=battery_low,
            network_good=network_good,
        )
        path = self.paths.get(path_name)
        if path is None:
            print(f"[grasp] INFER: no Path registered for {path_name!r}, "
                  f"failing cycle")
            self._enter_state(GraspState.FAILED, now)
            return

        # Execute the path stub. For offload this re-samples the
        # channel — that's expected; the tx-latency calculation needs
        # its own fresh draw of channel gain. We keep our earlier
        # data_rate sample for logging as "C(t) at decision time".
        result = path.execute(agv_xy, self.network)
        self._infer_result = result

        # Schedule the wait: transition to DESCEND after latency_s elapses.
        self._infer_wait_until_s = now + result.latency_s
        print(f"           INFER: bat_low={battery_low} net_good={network_good}"
              f" -> {path_name}"
              f" (latency={result.latency_s * 1000:.0f}ms,"
              f" energy={result.energy_j * 1000:.1f}mJ,"
              f" success={result.success})")

    def _drain_infer_energy(self) -> None:
        """Called when leaving INFER — drain the battery for the
        energy the just-executed path consumed."""
        if self.battery is not None and self._infer_result is not None:
            self.battery.consume(self._infer_result.energy_j)

    # ------------------------------------------------------ state transitions

    def _check_transition(self, now: float) -> None:
        # APPROACH's next state depends on whether phase-5 D(t) is wired
        # in. Without it, we skip straight to DESCEND (phase-4 behavior).
        approach_next = (GraspState.INFER if self._phase5_active()
                         else GraspState.DESCEND)

        # States that wait for the arm to reach its joint target.
        joint_convergence_states = {
            GraspState.APPROACH: approach_next,
            GraspState.DESCEND: GraspState.CLOSE,
            GraspState.LIFT: GraspState.TRANSIT,
            GraspState.APPROACH_DROP: GraspState.DESCEND_DROP,
            GraspState.DESCEND_DROP: GraspState.OPEN,
            GraspState.LIFT_AWAY: GraspState.RETURN_HOME,
        }

        if self.state in joint_convergence_states:
            if self._is_arm_at_joint_target():
                self._stability_counter += 1
                if self._stability_counter >= self.cfg.stability_ticks:
                    self._enter_state(joint_convergence_states[self.state], now)
            else:
                self._stability_counter = 0
            return

        # INFER: hold arm pose, wait for the path's simulated latency
        # to elapse, then drain the battery and advance to DESCEND.
        if self.state == GraspState.INFER:
            if now >= self._infer_wait_until_s:
                self._drain_infer_energy()
                self._enter_state(GraspState.DESCEND, now)
            return

        if self.state == GraspState.CLOSE:
            if now - self._state_entry_time >= self.cfg.gripper_settle_s:
                self._enter_state(GraspState.LIFT, now)
            return

        if self.state == GraspState.OPEN:
            if now - self._state_entry_time >= self.cfg.gripper_settle_s:
                self._enter_state(GraspState.LIFT_AWAY, now)
            return

        # AGV-arrival-wait states: TRANSIT drives to drop, RETURN_HOME
        # drives back to pick. Both use agv.is_at_waypoint() as the
        # exit condition.
        if self.state == GraspState.TRANSIT:
            if self.robot.agv.is_at_waypoint(self.cfg.drop_wp_index):
                self._enter_state(GraspState.APPROACH_DROP, now)
            return

        if self.state == GraspState.RETURN_HOME:
            if self.robot.agv.is_at_waypoint(self.cfg.pick_wp_index):
                self._enter_state(GraspState.DONE, now)
            return

    def _is_arm_at_joint_target(self) -> bool:
        if self._q_target is None:
            return True
        q_actual = self.robot.arm.get_joint_positions()
        err = float(np.max(np.abs(q_actual - self._q_target)))
        return err < self.cfg.joint_tol

    # ----------------------------------------------------- introspection

    def is_done(self) -> bool:
        return self.state in (GraspState.DONE, GraspState.FAILED)

    def outcome(self) -> CycleOutcome:
        """Bundle of per-cycle metrics for the CSV logger. Call after
        is_done() returns True."""
        return CycleOutcome(
            cycle_index=self._cycle_index,
            target_object_id=self._target_object_id,
            final_state=self.state.value,
            battery_low=self._infer_battery_low,
            network_good=self._infer_network_good,
            battery_soc_at_decision=self._infer_battery_soc,
            data_rate_bps=self._infer_data_rate_bps,
            path_name=(self._infer_result.path_name
                       if self._infer_result is not None else ""),
            path_result=self._infer_result,
            started_at_s=0.0,  # runner fills these
            finished_at_s=self.world.sim_time,
            physically_delivered=self.object_in_bin_b(),
        )

    def object_in_bin_b(self) -> bool:
        if self._target_object_id < 0:
            return False
        pos, _ = p.getBasePositionAndOrientation(
            self._target_object_id, physicsClientId=self.world.client_id,
        )
        b = self.scene.cfg.bin_b
        cx, cy = b.center_xy
        L, W, H = b.inner_size
        return (abs(pos[0] - cx) < L / 2
                and abs(pos[1] - cy) < W / 2
                and pos[2] < H + 0.10)
