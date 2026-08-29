"""Diagnostic: verify Battery and NetworkChannel behave sensibly.

No PyBullet, no FSM, no arm. Just walks the models through their
expected input ranges and prints what they return.

Two sections:

1. NetworkChannel: sweep AGV xy along the s7 patrol path (from
   (+1, 0) at bin A to (-1, 0) at bin B). At each position, print
   the data rate and whether is_good() fires. With the default
   basestation at (0, 5), the AGV is ~5m away throughout — expect
   modest rates that vary primarily due to shadow noise.

2. Battery: run 100 fake grasp cycles, each consuming a random
   amount between 100 and 5000 J. Watch SoC drop and log when
   is_low() transitions from False to True.

Expected outcomes:
    - Network rate is finite and positive everywhere; is_good() flips
      True/False across cycles due to log-normal shadowing
    - Battery drains monotonically from 100% to below 30%; is_low()
      flips True at the moment SoC crosses 0.30
"""

import numpy as np

from sim.battery import Battery, BatteryConfig
from sim.network import NetworkChannel, NetworkConfig


def sweep_network() -> None:
    print("=" * 60)
    print("NetworkChannel sweep along AGV patrol path")
    print("=" * 60)

    # Deterministic seed so the sweep is reproducible.
    net = NetworkChannel(NetworkConfig(seed=42))

    print(f"Basestation at {net.cfg.basestation_xy}, "
          f"threshold = {net.cfg.good_threshold_bps / 1e6:.1f} Mbps\n")

    # Walk along the s7 patrol path.
    xs = np.linspace(1.0, -1.0, 11)
    goods = 0
    for x in xs:
        agv_xy = (float(x), 0.0)
        d = net.distance_to_bs(agv_xy)
        rate, good = net.sample(agv_xy)
        marker = "GOOD" if good else " bad"
        if good:
            goods += 1
        print(f"  agv=({agv_xy[0]:+.2f}, 0.00)  "
              f"d={d:5.2f} m  "
              f"rate={rate / 1e6:7.2f} Mbps  {marker}")

    print(f"\n{goods}/{len(xs)} positions marked GOOD "
          f"(with sigma={net.cfg.shadow_sigma_db} dB shadowing)")


def drain_battery() -> None:
    print("\n" + "=" * 60)
    print("Battery drain over 100 fake grasp cycles")
    print("=" * 60)

    batt = Battery(BatteryConfig(capacity_j=360_000, soc_initial=1.0,
                                 soc_low_threshold=0.30))
    rng = np.random.default_rng(42)

    print(f"Capacity = {batt.cfg.capacity_j / 1000:.0f} kJ, "
          f"low threshold = {batt.cfg.soc_low_threshold * 100:.0f}%\n")

    prev_low = batt.is_low()
    for cycle in range(1, 101):
        # Each cycle: uniform draw between light-local energy (~100 J)
        # and heavy-local energy (~5000 J). Rough placeholder — real
        # per-path energy comes in commit 2.
        joules = float(rng.uniform(100, 5000))
        batt.consume(joules)
        low = batt.is_low()

        # Print every 10 cycles, plus the low-transition.
        if cycle % 10 == 0 or (low and not prev_low):
            note = "  <-- is_low() flipped True" if (low and not prev_low) else ""
            print(f"  cycle {cycle:3d}  drew {joules:6.0f} J  "
                  f"SoC = {batt.state_of_charge() * 100:5.1f}%  "
                  f"total = {batt.total_consumed_j() / 1000:6.1f} kJ{note}")
        prev_low = low

        if batt.is_depleted():
            print(f"  cycle {cycle}: battery depleted, stopping")
            break

    print(f"\nFinal SoC = {batt.state_of_charge() * 100:.1f}%  "
          f"is_low = {batt.is_low()}")


if __name__ == "__main__":
    sweep_network()
    drain_battery()
