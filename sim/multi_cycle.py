"""Multi-cycle runner: chain grasp cycles until bin A is empty or
battery dies.

Wraps a single GraspCycle instance. When one cycle finishes (DONE or
FAILED), the runner picks the next unprocessed object in bin A and
starts the next cycle. Processed objects are tracked in a set so we
don't retry a delivered object.

Design: runner owns the "what to grasp next" logic; GraspCycle keeps
its single-cycle contract. Runner also carries the per-cycle
CycleOutcome list that the metrics logger consumes.

Termination conditions (any of):
    - bin A has no unprocessed objects left
    - battery is depleted (SoC = 0)
    - max_cycles reached (safety)
    - last cycle FAILED and stop_on_failure=True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pybullet as p

from sim.grasp import CycleOutcome, GraspCycle, GraspState
from sim.world import World


@dataclass
class RunnerConfig:
    """Tunable multi-cycle runner parameters."""

    # Absolute safety cap on cycle count. Real termination is bin A
    # empty or battery dead — this stops the sim in pathological cases.
    max_cycles: int = 50

    # If True, stop the run when any cycle transitions to FAILED.
    # Default False so timeouts on one cycle don't abort a long sweep.
    stop_on_failure: bool = False

    # Seconds to wait between one cycle finishing and the next starting.
    # Zero = immediate. A small delay makes the visual demo readable.
    inter_cycle_pause_s: float = 0.5


class MultiCycleGraspRunner:
    """Chains GraspCycle across many objects.

    Usage:
        runner = MultiCycleGraspRunner(world, grasp_cycle, RunnerConfig())
        world.register_callback(30.0, runner.tick)
        world.run(...)
        # After run: runner.outcomes has one CycleOutcome per cycle.
    """

    def __init__(
        self,
        world: World,
        grasp: GraspCycle,
        cfg: RunnerConfig | None = None,
    ) -> None:
        self.world = world
        self.grasp = grasp
        self.cfg = cfg or RunnerConfig()

        # State.
        self._processed_object_ids: set[int] = set()
        self._cycle_index: int = 0
        self._current_cycle_started_at_s: float = 0.0
        self._pause_until_s: float = 0.0
        self._stopped: bool = False

        # Log of all completed cycles.
        self.outcomes: list[CycleOutcome] = []

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Kick off the first cycle after the initial pause elapses."""
        # Delay the very first cycle by inter_cycle_pause so the sim
        # has a beat to render the initial scene.
        self._pause_until_s = (self.world.sim_time
                               + self.cfg.inter_cycle_pause_s)

    def tick(self, sim_time_s: float) -> None:
        """Advance runner state. Register at 30 Hz alongside grasp.tick.

        Order matters: this runs AFTER grasp.tick in the callback list,
        so if grasp.state just flipped to DONE, we see it this tick and
        can start the next cycle after the pause."""
        if self._stopped:
            return

        # Only act BETWEEN cycles. If the FSM is mid-cycle (any state
        # other than IDLE / DONE / FAILED), leave it alone.
        if self.grasp.state not in (GraspState.IDLE,
                                    GraspState.DONE,
                                    GraspState.FAILED):
            return

        # If a cycle actually ran and just finished, log it. IDLE at
        # startup won't trigger this because _current_cycle_started_at_s
        # is still 0.
        if self._current_cycle_started_at_s > 0:
            self._log_current_outcome()
            self._current_cycle_started_at_s = 0.0
            self._pause_until_s = (sim_time_s
                                   + self.cfg.inter_cycle_pause_s)
            if (self.cfg.stop_on_failure
                    and self.grasp.state == GraspState.FAILED):
                print(f"[runner] stopping: cycle {self._cycle_index - 1} "
                      "FAILED and stop_on_failure=True")
                self._stopped = True
                return

        # Honor the between-cycles pause, then start the next one if
        # termination conditions aren't met.
        if sim_time_s < self._pause_until_s:
            return
        self._maybe_start_next_cycle(sim_time_s)

    # -------------------------------------------------------- cycle control

    def _maybe_start_next_cycle(self, sim_time_s: float) -> None:
        """Termination gates + start next cycle."""
        # Termination: max cycles.
        if self._cycle_index >= self.cfg.max_cycles:
            print(f"[runner] stopping: reached max_cycles={self.cfg.max_cycles}")
            self._stopped = True
            return

        # Termination: battery depleted.
        if (self.grasp.battery is not None
                and self.grasp.battery.is_depleted()):
            print("[runner] stopping: battery depleted")
            self._stopped = True
            return

        # Termination: no more unprocessed objects.
        next_obj = self._pick_next_object()
        if next_obj is None:
            print(f"[runner] stopping: all {len(self._processed_object_ids)} "
                  "objects processed")
            self._stopped = True
            return

        # Start next cycle.
        print(f"\n[runner t={sim_time_s:6.2f}s] starting cycle "
              f"{self._cycle_index} on object id={next_obj}")
        self.grasp.start(next_obj, cycle_index=self._cycle_index)
        self._current_cycle_started_at_s = sim_time_s

    def _pick_next_object(self) -> Optional[int]:
        """Return the id of the next unprocessed object in bin A, or
        None if all have been processed. Uses the object's CURRENT
        position to filter (something we picked and dropped in B is
        no longer in A)."""
        cid = self.world.client_id
        bin_a = self.grasp.scene.cfg.bin_a
        cx, cy = bin_a.center_xy
        L, W, _ = bin_a.inner_size

        for obj_id in self.grasp.scene.object_ids:
            if obj_id in self._processed_object_ids:
                continue
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=cid)
            in_bin_a = (abs(pos[0] - cx) < L / 2
                        and abs(pos[1] - cy) < W / 2)
            if in_bin_a:
                return obj_id
        return None

    def _log_current_outcome(self) -> None:
        """Snapshot the just-finished cycle into self.outcomes."""
        outcome = self.grasp.outcome()
        # Fill in the runner-tracked fields the FSM doesn't know about.
        outcome = CycleOutcome(
            cycle_index=outcome.cycle_index,
            target_object_id=outcome.target_object_id,
            final_state=outcome.final_state,
            battery_low=outcome.battery_low,
            network_good=outcome.network_good,
            battery_soc_at_decision=outcome.battery_soc_at_decision,
            data_rate_bps=outcome.data_rate_bps,
            path_name=outcome.path_name,
            path_result=outcome.path_result,
            started_at_s=self._current_cycle_started_at_s,
            finished_at_s=self.world.sim_time,
            physically_delivered=outcome.physically_delivered,
        )
        self.outcomes.append(outcome)
        self._processed_object_ids.add(outcome.target_object_id)
        self._cycle_index += 1

        # One-liner summary.
        delivered_marker = "OK " if outcome.physically_delivered else "no "
        pred_marker = "OK " if (outcome.path_result and outcome.path_result.success) else "no "
        soc_pct = outcome.battery_soc_at_decision * 100
        print(f"[runner] cycle {outcome.cycle_index}: "
              f"path={outcome.path_name:>12s}  "
              f"delivered={delivered_marker} predicted_success={pred_marker} "
              f"soc_at_decision={soc_pct:5.1f}%")

    # --------------------------------------------------------- introspection

    def is_stopped(self) -> bool:
        return self._stopped

    def processed_count(self) -> int:
        return len(self._processed_object_ids)
