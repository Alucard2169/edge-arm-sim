"""Execution path stubs for the three D(t) branches.

Each path is a stub: it doesn't actually run inference, it just returns
a PathResult(latency_s, energy_j, success, path_name) drawn from a
calibrated distribution. This lets us exercise the D(t) rule table
across the full parameter space in minutes without needing real models,
real Docker containers, or real network round trips.

The stubs are the phase 5 deliverable. Phase 6 will swap them for
real containerized models with tc/netem-shaped network paths — the
FSM interface (path.execute()) won't change.

Calibration sources (all documented per path):
    - LightLocal:  quantized model on ARM Cortex-A per TinyML survey
                   (Schizas et al. 2022). ~80 ms latency, ~0.4 W
                   compute, ~0.85 grasp success rate.
    - HeavyLocal:  Jetson-class GPU inference per DVFO (Zhang et al.
                   2024) baseline. ~400 ms latency, ~5 W (GPU energy
                   3.1x the CPU energy per Zhang), ~0.95 success.
    - Offload:     Edge-server compute per DVFO (~30 ms), plus network
                   round-trip. Local energy = tx_power * tx_time.
                   ~0.95 success (same model as heavy-local).

The success rate gap (0.85 vs 0.95) is what makes the D(t) tradeoff
non-trivial: light-local is cheap but less accurate; heavy-local and
offload match on accuracy but differ in energy vs latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sim.network import NetworkChannel


# ------------------------------------------------------------------ result

@dataclass(frozen=True)
class PathResult:
    """One execution outcome. Consumed by the FSM (drain battery, log
    metrics) and by the eventual metrics/plots."""

    path_name: str          # "light-local" | "heavy-local" | "offload"
    latency_s: float        # end-to-end wall time this cycle
    energy_j: float         # energy drawn from the AGV battery
    success: bool           # did the grasp-detection prediction succeed
    # Diagnostic breakdown (all optional, only some paths fill some fields):
    compute_latency_s: float = 0.0
    tx_latency_s: float = 0.0
    data_rate_bps: float = 0.0


# ------------------------------------------------------------ base config

@dataclass
class PathConfig:
    """Fields common to every path stub. Individual paths subclass
    this to add path-specific fields."""

    # Latency: sampled as base * (1 + uniform(-jitter_frac, +jitter_frac))
    # so runs have realistic variation without being unpredictable.
    latency_base_s: float = 0.0
    latency_jitter_frac: float = 0.20

    # Bernoulli success rate.
    success_rate: float = 1.0

    # RNG seed for reproducible runs (None = fresh).
    seed: Optional[int] = None


# ---------------------------------------------------------- path interface

class Path:
    """Base class. Subclasses override _sample_latency() and _sample_energy()."""

    name: str = "path"

    def __init__(self, cfg: PathConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(
            seed if seed is not None else self.cfg.seed
        )

    def execute(
        self,
        agv_xy: tuple[float, float],
        network: NetworkChannel,
    ) -> PathResult:
        """Simulate one inference on this path. `agv_xy` and `network`
        are supplied so that Offload can compute transmission time from
        current channel conditions; local paths ignore them."""
        raise NotImplementedError

    # ------------------------------------------------------ helpers

    def _sample_latency(self, base: float) -> float:
        """Sample from base * uniform(1-jitter, 1+jitter)."""
        j = self.cfg.latency_jitter_frac
        multiplier = self._rng.uniform(1.0 - j, 1.0 + j)
        return float(max(0.0, base * multiplier))

    def _sample_success(self) -> bool:
        return bool(self._rng.uniform() < self.cfg.success_rate)


# --------------------------------------------------------- LightLocal

@dataclass
class LightLocalConfig(PathConfig):
    """Quantized model on ARM Cortex-A class CPU."""

    # Latency: ~80 ms per Schizas et al. 2022 TinyML deployments of
    # MobileNet-style grasp detectors. Configurable so ablations can
    # sweep the light-vs-heavy latency gap.
    latency_base_s: float = 0.080
    # Compute power draw during inference (watts). ~0.4 W is typical
    # for a mid-load ARM Cortex-A72 core, matching Cortex TDP figures.
    compute_power_w: float = 0.4
    # Success rate. 0.85 is a conservative reading of the TinyML
    # accuracy-drop-after-quantization figures in Schizas et al.
    success_rate: float = 0.85


class LightLocal(Path):
    name = "light-local"

    def __init__(self, cfg: LightLocalConfig | None = None) -> None:
        super().__init__(cfg or LightLocalConfig())
        # Narrow the type for _sample_energy without shadowing.
        self.cfg: LightLocalConfig = self.cfg  # type: ignore[assignment]

    def execute(
        self,
        agv_xy: tuple[float, float],
        network: NetworkChannel,
    ) -> PathResult:
        del agv_xy, network  # local path — ignores network
        latency = self._sample_latency(self.cfg.latency_base_s)
        energy = latency * self.cfg.compute_power_w
        return PathResult(
            path_name=self.name,
            latency_s=latency,
            energy_j=energy,
            success=self._sample_success(),
            compute_latency_s=latency,
        )


# --------------------------------------------------------- HeavyLocal

@dataclass
class HeavyLocalConfig(PathConfig):
    """Full-precision model on Jetson-class GPU."""

    # Latency: ~400 ms per DVFO Zhang et al. 2024 baseline for grasp
    # networks on Jetson-class edge devices (before their DVFS
    # optimizations).
    latency_base_s: float = 0.400
    # Compute power: ~5 W. This bakes in DVFO's finding that GPU energy
    # is 3.1-3.5x CPU energy for DNN inference — heavy-local is
    # GPU-bound where light-local is CPU-only.
    compute_power_w: float = 5.0
    # Success rate: 0.95 per typical unquantized grasp-detection
    # accuracy in the literature.
    success_rate: float = 0.95


class HeavyLocal(Path):
    name = "heavy-local"

    def __init__(self, cfg: HeavyLocalConfig | None = None) -> None:
        super().__init__(cfg or HeavyLocalConfig())
        self.cfg: HeavyLocalConfig = self.cfg  # type: ignore[assignment]

    def execute(
        self,
        agv_xy: tuple[float, float],
        network: NetworkChannel,
    ) -> PathResult:
        del agv_xy, network
        latency = self._sample_latency(self.cfg.latency_base_s)
        energy = latency * self.cfg.compute_power_w
        return PathResult(
            path_name=self.name,
            latency_s=latency,
            energy_j=energy,
            success=self._sample_success(),
            compute_latency_s=latency,
        )


# ------------------------------------------------------------- Offload

@dataclass
class OffloadConfig(PathConfig):
    """Remote inference on an edge server, with network transmission."""

    # Compute time on the edge server (assumed capable). ~30 ms per
    # DVFO edge-server figures. This is the compute part only — the
    # network round-trip is added on top from the current channel rate.
    latency_base_s: float = 0.030  # edge-server compute time
    # Frame payload size (bits). A compressed 640x480 grasp frame is
    # ~50 KB = 400 kilobits. Configurable for ablations.
    frame_size_bits: float = 400_000
    # AGV transmit power draw while uploading (watts). 0.1 W is typical
    # for a Wi-Fi endpoint at 20 dBm.
    tx_power_w: float = 0.1
    # Success rate: same as heavy-local — the edge server runs the
    # unquantized model.
    success_rate: float = 0.95
    # Round-trip overhead beyond compute + upload (server-to-client
    # response, protocol overhead). 10 ms is typical.
    rtt_overhead_s: float = 0.010


class Offload(Path):
    name = "offload"

    def __init__(self, cfg: OffloadConfig | None = None) -> None:
        super().__init__(cfg or OffloadConfig())
        self.cfg: OffloadConfig = self.cfg  # type: ignore[assignment]

    def execute(
        self,
        agv_xy: tuple[float, float],
        network: NetworkChannel,
    ) -> PathResult:
        # Sample current channel once. This IS the C(t) used for the
        # transmission calculation — matches the FSM's single-sample-
        # per-cycle discipline in NetworkChannel.sample().
        data_rate, _ = network.sample(agv_xy)
        # Guard against zero rate (e.g., AGV far from BS + bad shadow).
        # Cap tx_latency at a large but finite value so the run doesn't
        # hang; a failed offload will surface via state_timeout in the FSM.
        tx_latency = (self.cfg.frame_size_bits / max(data_rate, 1.0)
                      if data_rate > 0 else 60.0)

        compute_latency = self._sample_latency(self.cfg.latency_base_s)
        total_latency = compute_latency + tx_latency + self.cfg.rtt_overhead_s

        # Energy on the AGV side: only the transmission draws battery.
        # Edge-server compute doesn't cost the AGV anything. The
        # rtt_overhead is receive-idle, negligible power.
        energy = tx_latency * self.cfg.tx_power_w

        return PathResult(
            path_name=self.name,
            latency_s=total_latency,
            energy_j=energy,
            success=self._sample_success(),
            compute_latency_s=compute_latency,
            tx_latency_s=tx_latency,
            data_rate_bps=data_rate,
        )
