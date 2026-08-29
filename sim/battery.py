"""Battery state model for the AGV.

Drain-only for phase 5 — no charging. Real AGVs charge at docking
stations between shifts; that behaviour is out of scope for the D(t)
decision, which operates within a shift.

The battery has two roles:
    1. Provide the binary B(t) signal to D(t) (is_low() returns True/False)
    2. Accumulate energy consumed per grasp cycle (drives the "battery
       drains faster with heavy paths" thesis narrative)

State progression: start at soc_initial (fraction 0..1), each cycle
consume(joules) subtracts from remaining_j. is_low() flips True once
SoC drops below soc_low_threshold. The FSM reads is_low() at each
grasp cycle to index the D(t) rule table.
"""

from __future__ import annotations

from dataclasses import dataclass


# ------------------------------------------------------------------ config

@dataclass
class BatteryConfig:
    """Tunable battery parameters."""

    # Total battery capacity in joules. Default sized for a small mobile
    # robot: 100 Wh = 360 kJ. Fetch Freight is ~15x this, but we want
    # runs to show meaningful drain in 20-100 grasp cycles.
    capacity_j: float = 360_000.0

    # Starting state of charge, as a fraction 0..1.
    soc_initial: float = 1.0

    # Threshold below which is_low() returns True. Default 30% — common
    # "low battery" convention in robotics literature and gives clean
    # transitions between rule-table rows during a run.
    soc_low_threshold: float = 0.30

    def __post_init__(self) -> None:
        if not (0.0 < self.capacity_j):
            raise ValueError(f"capacity_j must be positive, got {self.capacity_j}")
        if not (0.0 <= self.soc_initial <= 1.0):
            raise ValueError(f"soc_initial must be in [0, 1], got {self.soc_initial}")
        if not (0.0 <= self.soc_low_threshold <= 1.0):
            raise ValueError(
                f"soc_low_threshold must be in [0, 1], got {self.soc_low_threshold}"
            )


# ---------------------------------------------------------------- battery

class Battery:
    """Simple battery state: drain via consume(), read via is_low()."""

    def __init__(self, cfg: BatteryConfig | None = None) -> None:
        self.cfg = cfg or BatteryConfig()
        self._remaining_j: float = self.cfg.capacity_j * self.cfg.soc_initial
        # Cumulative consumption for logging.
        self._total_consumed_j: float = 0.0

    def reset(self) -> None:
        """Restore initial SoC. Useful between multi-run sweeps."""
        self._remaining_j = self.cfg.capacity_j * self.cfg.soc_initial
        self._total_consumed_j = 0.0

    # ---------------------------------------------------------- drain

    def consume(self, joules: float) -> None:
        """Drain `joules` from the battery. Clamped at zero — battery
        can't go negative. The FSM logic should check is_depleted()
        before drawing large amounts if strict accounting is needed."""
        if joules < 0:
            raise ValueError(f"consume(): joules must be non-negative, got {joules}")
        self._total_consumed_j += joules
        self._remaining_j = max(0.0, self._remaining_j - joules)

    # ----------------------------------------------------- introspection

    def state_of_charge(self) -> float:
        """Fraction of capacity remaining, in [0, 1]."""
        return self._remaining_j / self.cfg.capacity_j

    def remaining_j(self) -> float:
        return self._remaining_j

    def total_consumed_j(self) -> float:
        return self._total_consumed_j

    def is_low(self) -> bool:
        """The binary B(t) signal that indexes the D(t) rule table."""
        return self.state_of_charge() < self.cfg.soc_low_threshold

    def is_depleted(self) -> bool:
        return self._remaining_j <= 0.0
