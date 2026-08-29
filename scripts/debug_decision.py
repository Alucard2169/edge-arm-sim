"""Diagnostic: verify D(t) rule table + baselines return what we expect.

For each of the 4 (battery_low, network_good) combinations, call
decide() on the rule-based Decision and each of the three fixed
baselines. Print a table.

Also exercise the DecisionConfig validation:
    - missing cell should raise
    - illegal path name should raise
    - custom (ablation) table should be accepted
"""

from sim.decision import (
    AlwaysLocalHeavy,
    AlwaysLocalLight,
    AlwaysOffload,
    Decision,
    DecisionConfig,
    HEAVY_LOCAL,
    LIGHT_LOCAL,
    OFFLOAD,
)


def walk_rule_table() -> None:
    print("=" * 78)
    print("Rule table + baselines — decision per (battery, network) state")
    print("=" * 78)

    decisions = [
        Decision(),           # paper's Table IV
        AlwaysLocalLight(),
        AlwaysLocalHeavy(),
        AlwaysOffload(),
    ]

    # Header
    names = [d.name for d in decisions]
    print(f"  {'battery':>7s}  {'network':>7s}   "
          + "  ".join(f"{n:>15s}" for n in names))
    print("  " + "-" * 76)

    states = [(False, False), (False, True),
              (True, False),  (True, True)]

    for cycle, (bat_low, net_good) in enumerate(states):
        bat_str = "LOW " if bat_low else "HIGH"
        net_str = "GOOD" if net_good else "BAD "
        outputs = [d.decide(cycle, bat_low, net_good) for d in decisions]
        print(f"  {bat_str:>7s}  {net_str:>7s}   "
              + "  ".join(f"{o:>15s}" for o in outputs))


def test_config_validation() -> None:
    print("\n" + "=" * 78)
    print("Config validation")
    print("=" * 78)

    # 1. Missing a cell -> should raise
    try:
        DecisionConfig(rule_table={
            (True, True): OFFLOAD,
            (True, False): LIGHT_LOCAL,
            (False, True): HEAVY_LOCAL,
            # missing (False, False)
        })
        print("  FAIL: missing cell was accepted")
    except ValueError as e:
        print(f"  OK   missing cell rejected: {e}")

    # 2. Illegal path name -> should raise
    try:
        DecisionConfig(rule_table={
            (True, True):   OFFLOAD,
            (True, False):  LIGHT_LOCAL,
            (False, True):  HEAVY_LOCAL,
            (False, False): "quantum-teleport",
        })
        print("  FAIL: illegal path name was accepted")
    except ValueError as e:
        print(f"  OK   illegal path rejected: {e}")

    # 3. Ablation: flip High/Good to offload (alternative to Table IV)
    #    should be accepted since it's still a valid table.
    ablation = DecisionConfig(rule_table={
        (True, False):  LIGHT_LOCAL,
        (True, True):   OFFLOAD,
        (False, False): HEAVY_LOCAL,
        (False, True):  OFFLOAD,       # flipped from HEAVY_LOCAL
    })
    d = Decision(ablation)
    result = d.decide(cycle=0, battery_low=False, network_good=True)
    print(f"  OK   ablation table (High/Good -> offload) returns "
          f"{result!r} as expected")


def check_history_recording() -> None:
    print("\n" + "=" * 78)
    print("History recording")
    print("=" * 78)

    d = Decision()
    d.decide(cycle=0, battery_low=False, network_good=True)
    d.decide(cycle=1, battery_low=True, network_good=False)
    d.decide(cycle=2, battery_low=True, network_good=True)
    print(f"  History has {len(d.history)} entries:")
    for record in d.history:
        print(f"    cycle {record.cycle}: "
              f"bat_low={record.battery_low}, net_good={record.network_good} "
              f"-> {record.path_name}")


if __name__ == "__main__":
    walk_rule_table()
    test_config_validation()
    check_history_recording()
