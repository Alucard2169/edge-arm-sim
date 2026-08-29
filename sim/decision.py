"""D(t) rule table: the 2x2 decision function from the paper.

Given the binary state (B(t), C(t)) at a grasp cycle, this returns
the name of the execution path to use ("light-local" | "heavy-local"
| "offload"). The FSM then looks up the path instance from the name
and calls its execute() method.

The rule table itself (paper Table IV):

    Battery      Network      Decision
    -------      -------      --------
    Low          Bad          light-local     ("degrade gracefully")
    Low          Good         offload         ("save battery, network available")
    High         Bad          heavy-local     ("network down, device has capacity")
    High         Good         heavy-local     ("both work, avoid tx energy cost")

Battery decides first; the network only breaks the tie when battery
is low. This is the paper's contribution — a specific, interpretable
mapping, not a learned policy.

Baselines from the paper's evaluation table live here too, as
AlwaysLocalLight / AlwaysLocalHeavy / AlwaysOffload subclasses. Same
decide() interface as the rule table, so the FSM code is baseline-
agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------- names

# The three legal path names. Kept as string constants so a typo in
# the rule table surfaces as a KeyError at construction time.
LIGHT_LOCAL = "light-local"
HEAVY_LOCAL = "heavy-local"
OFFLOAD = "offload"

_ALL_PATHS = frozenset({LIGHT_LOCAL, HEAVY_LOCAL, OFFLOAD})


# ------------------------------------------------------------------- record

@dataclass(frozen=True)
class DecisionRecord:
    """One D(t) evaluation, kept in Decision.history for metrics."""

    cycle: int                 # grasp cycle index
    battery_low: bool          # B(t) reading at decision time
    network_good: bool         # C(t) reading at decision time
    path_name: str             # what the decision function returned


# ------------------------------------------------------------------ config

def _default_rule_table() -> dict[tuple[bool, bool], str]:
    """The paper's Table IV, keyed by (battery_low, network_good).

    True = low battery / good network; False = high battery / bad network.
    """
    return {
        (True,  False): LIGHT_LOCAL,   # low battery, bad network
        (True,  True):  OFFLOAD,       # low battery, good network
        (False, False): HEAVY_LOCAL,   # high battery, bad network
        (False, True):  HEAVY_LOCAL,   # high battery, good network
    }


@dataclass
class DecisionConfig:
    """Configuration for the rule-based Decision.

    Ablations override `rule_table` to flip individual cells and
    measure the impact — the whole point of a rule table over a
    learned policy is that this is trivial."""

    rule_table: dict[tuple[bool, bool], str] = field(
        default_factory=_default_rule_table
    )

    def __post_init__(self) -> None:
        # Ensure the table is exhaustive over the 2x2 state space and
        # every value is a legal path name.
        expected_keys = {(False, False), (False, True),
                         (True, False), (True, True)}
        got_keys = set(self.rule_table.keys())
        if got_keys != expected_keys:
            missing = expected_keys - got_keys
            extra = got_keys - expected_keys
            raise ValueError(
                f"rule_table must cover exactly the 4 (battery_low, "
                f"network_good) cells. Missing: {missing}. Extra: {extra}"
            )
        for key, value in self.rule_table.items():
            if value not in _ALL_PATHS:
                raise ValueError(
                    f"rule_table[{key}] = {value!r} is not one of {_ALL_PATHS}"
                )


# ---------------------------------------------------------------- decision

class Decision:
    """The default rule-based D(t) — the paper's contribution.

    Usage:
        decision = Decision()
        path_name = decision.decide(cycle=t, battery_low=B_low, network_good=C_good)
        # ...FSM looks up path instance from name and calls .execute()...
    """

    name: str = "rule-based"

    def __init__(self, cfg: DecisionConfig | None = None) -> None:
        self.cfg = cfg or DecisionConfig()
        self.history: list[DecisionRecord] = []

    def decide(
        self,
        cycle: int,
        battery_low: bool,
        network_good: bool,
    ) -> str:
        """Look up the path name for the given (B, C) state and log it."""
        path_name = self.cfg.rule_table[(battery_low, network_good)]
        self.history.append(DecisionRecord(
            cycle=cycle,
            battery_low=battery_low,
            network_good=network_good,
            path_name=path_name,
        ))
        return path_name

    def reset(self) -> None:
        """Clear decision history. Called between sweeps."""
        self.history.clear()


# ------------------------------------------------------------------ baselines

class _FixedDecision(Decision):
    """Base for the three always-X baselines. Ignores B(t) and C(t)."""

    _fixed_path: str = ""  # subclasses set this

    def __init__(self) -> None:
        # Bypass rule-table validation — the fixed baselines don't use it.
        # We still call super().__init__() to get the history list, but
        # with a table that's never queried.
        super().__init__(DecisionConfig(
            rule_table={
                (False, False): self._fixed_path,
                (False, True):  self._fixed_path,
                (True, False):  self._fixed_path,
                (True, True):   self._fixed_path,
            }
        ))


class AlwaysLocalLight(_FixedDecision):
    """Baseline: always run light-local, ignore B(t) and C(t)."""
    name = "always-light"
    _fixed_path = LIGHT_LOCAL


class AlwaysLocalHeavy(_FixedDecision):
    """Baseline: always run heavy-local, ignore B(t) and C(t)."""
    name = "always-heavy"
    _fixed_path = HEAVY_LOCAL


class AlwaysOffload(_FixedDecision):
    """Baseline: always offload, ignore B(t) and C(t)."""
    name = "always-offload"
    _fixed_path = OFFLOAD
