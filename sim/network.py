"""Network channel model for the AGV-to-edge-server link.

Uses the log-normal + path-loss channel calibrated during LDROA
reproduction (Eang et al. 2024): channel gain
    G(d) = G0 * d^(-gamma) * X_shadow
where d is the AGV-to-basestation distance and X_shadow is a per-cycle
log-normal shadowing sample. Data rate follows from the standard
Shannon capacity formula given a fixed bandwidth and noise floor.

Design intent: C(t) varies naturally as the AGV moves through the
patrol path — no scripted schedule needed. Distance to the (fixed)
basestation drives the variation, and log-normal shadowing adds the
per-cycle randomness that keeps identical AGV positions from producing
identical decisions.

Not modeled: multi-path fading, interference, handover between BSs.
The thesis argument only needs "signal degrades with distance and
varies over time" — this delivers both with two lines of physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ config

@dataclass
class NetworkConfig:
    """Tunable channel parameters. Defaults calibrated in the LDROA
    reproduction notebook."""

    # Basestation position in world coordinates (xy). Default offset
    # to one side of the patrol so distance varies as the AGV moves
    # rather than being roughly constant.
    basestation_xy: tuple[float, float] = (0.0, 5.0)

    # Path-loss parameters (from LDROA calibration).
    # G0: reference gain at 1 m distance.
    # gamma: path-loss exponent (urban/warehouse indoor: 3-4).
    g0: float = 5e-8
    gamma: float = 3.5

    # Log-normal shadowing standard deviation in dB. 6-8 dB is typical
    # for indoor / warehouse environments.
    shadow_sigma_db: float = 6.0

    # Transmit power (watts) and noise (watts) for Shannon capacity.
    # tx_power=0.1 W = 20 dBm is typical for Wi-Fi endpoint.
    # noise=1e-10 W = -100 dBm is typical thermal noise floor for
    # 20 MHz bandwidth at room temperature.
    tx_power_w: float = 0.1
    noise_w: float = 1e-10

    # Channel bandwidth in Hz. 20 MHz = typical Wi-Fi channel.
    bandwidth_hz: float = 20e6

    # Threshold above which is_good() returns True (bits per second).
    # 2 Mbps: roughly what a compressed 640x480 grasp frame needs at
    # ~30 Hz, matching Zhang et al. DVFO's low-end bandwidth sweep.
    good_threshold_bps: float = 2e6

    # RNG seed for shadow samples; None = fresh randomness each call.
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.g0 <= 0:
            raise ValueError(f"g0 must be positive, got {self.g0}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.bandwidth_hz <= 0:
            raise ValueError(f"bandwidth_hz must be positive, got {self.bandwidth_hz}")


# ---------------------------------------------------------------- network

class NetworkChannel:
    """Distance- and shadowing-driven data rate for the AGV-to-BS link.

    Stateless in the "no memory between cycles" sense: each call to
    data_rate_at(xy) draws a fresh shadow sample and computes a fresh
    rate. The FSM samples once per grasp cycle at INFER-state entry.
    """

    def __init__(self, cfg: NetworkConfig | None = None) -> None:
        self.cfg = cfg or NetworkConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        # Convert dB sigma to linear (natural log) sigma for lognormal draws.
        # X_shadow in dB is Normal(0, sigma_db); as a linear multiplier
        # X = 10^(X_dB / 10), so ln(X) is Normal(0, sigma_db * ln(10)/10).
        self._shadow_sigma_ln = self.cfg.shadow_sigma_db * math.log(10) / 10.0

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the shadow RNG (for reproducible sweeps)."""
        self._rng = np.random.default_rng(
            seed if seed is not None else self.cfg.seed
        )

    # -------------------------------------------------- physical model

    def distance_to_bs(self, agv_xy: tuple[float, float]) -> float:
        """2D distance from AGV to basestation (metres)."""
        dx = agv_xy[0] - self.cfg.basestation_xy[0]
        dy = agv_xy[1] - self.cfg.basestation_xy[1]
        return math.hypot(dx, dy)

    def channel_gain(self, agv_xy: tuple[float, float]) -> float:
        """Instantaneous channel gain: G0 * d^-gamma * X_shadow.
        Draws a fresh shadow sample."""
        d = max(self.distance_to_bs(agv_xy), 0.1)  # floor at 10 cm to avoid singularity
        path_loss = self.cfg.g0 * (d ** (-self.cfg.gamma))
        # X_shadow ~ LogNormal(mu=0, sigma=sigma_ln)
        x_shadow = self._rng.lognormal(mean=0.0, sigma=self._shadow_sigma_ln)
        return path_loss * x_shadow

    def data_rate_at(self, agv_xy: tuple[float, float]) -> float:
        """Shannon capacity in bits per second, given current channel
        gain and configured tx power, noise, bandwidth."""
        g = self.channel_gain(agv_xy)
        snr = (self.cfg.tx_power_w * g) / self.cfg.noise_w
        return self.cfg.bandwidth_hz * math.log2(1.0 + snr)

    # ----------------------------------------------------- FSM interface

    def is_good(self, agv_xy: tuple[float, float]) -> bool:
        """The binary C(t) signal that indexes the D(t) rule table.
        True iff the current data rate exceeds the good_threshold."""
        return self.data_rate_at(agv_xy) >= self.cfg.good_threshold_bps

    def sample(self, agv_xy: tuple[float, float]) -> tuple[float, bool]:
        """One-call sample returning (data_rate_bps, is_good). Preferred
        over calling data_rate_at and is_good separately, which would
        draw TWO shadow samples for the same cycle."""
        rate = self.data_rate_at(agv_xy)
        return rate, rate >= self.cfg.good_threshold_bps
