"""Diagnostic: sanity-check the three path stubs.

Runs each path 100 times and prints the mean latency, mean energy,
and success rate. Compares against the paper's expected figures:

    light-local:  ~80 ms,  ~32 mJ,  ~85% success
    heavy-local:  ~400 ms, ~2 J,    ~95% success
    offload:      network-dependent; ~30-100 ms compute+tx,
                  ~1-100 mJ tx energy, ~95% success

Also runs Offload at two AGV positions (close-to-BS vs far-from-BS)
to verify the network dependence is visible.
"""

import numpy as np

from sim.network import NetworkChannel, NetworkConfig
from sim.paths import (
    HeavyLocal, HeavyLocalConfig,
    LightLocal, LightLocalConfig,
    Offload, OffloadConfig,
)


def summarize(name: str, results) -> None:
    latencies = np.array([r.latency_s for r in results])
    energies = np.array([r.energy_j for r in results])
    successes = np.array([r.success for r in results])
    print(f"  {name:>12s}  "
          f"latency = {latencies.mean() * 1000:6.1f} +- {latencies.std() * 1000:4.1f} ms  "
          f"energy = {energies.mean() * 1000:7.2f} +- {energies.std() * 1000:5.2f} mJ  "
          f"success = {successes.mean() * 100:5.1f}%")


def run_light_and_heavy() -> None:
    print("=" * 78)
    print("Local paths — 100 executions each")
    print("=" * 78)

    net = NetworkChannel(NetworkConfig(seed=42))  # unused by local paths

    for path in [LightLocal(LightLocalConfig(seed=42)),
                 HeavyLocal(HeavyLocalConfig(seed=42))]:
        results = [path.execute((1.0, 0.0), net) for _ in range(100)]
        summarize(path.name, results)


def run_offload_at_two_positions() -> None:
    print("\n" + "=" * 78)
    print("Offload — 100 executions at each of two AGV positions")
    print("=" * 78)

    net = NetworkChannel(NetworkConfig(basestation_xy=(0.0, 5.0), seed=42))

    for label, xy in [("close  (0, 4)", (0.0, 4.0)),
                      ("far  (10, 5)", (10.0, 5.0))]:
        # Fresh path RNG per position so success/latency samples don't
        # correlate with the network samples from the other position.
        path = Offload(OffloadConfig(seed=42))
        net.reset(seed=42)
        results = [path.execute(xy, net) for _ in range(100)]

        latencies = np.array([r.latency_s for r in results])
        energies = np.array([r.energy_j for r in results])
        successes = np.array([r.success for r in results])
        tx = np.array([r.tx_latency_s for r in results])
        rate = np.array([r.data_rate_bps for r in results]) / 1e6

        print(f"  AGV {label}")
        print(f"    data rate    = {rate.mean():7.2f} +- {rate.std():5.2f} Mbps")
        print(f"    tx latency   = {tx.mean() * 1000:7.1f} +- {tx.std() * 1000:5.1f} ms")
        print(f"    total latency= {latencies.mean() * 1000:7.1f} +- {latencies.std() * 1000:5.1f} ms")
        print(f"    tx energy    = {energies.mean() * 1000:7.2f} +- {energies.std() * 1000:5.2f} mJ")
        print(f"    success      = {successes.mean() * 100:5.1f}%")
        print()


if __name__ == "__main__":
    run_light_and_heavy()
    run_offload_at_two_positions()
