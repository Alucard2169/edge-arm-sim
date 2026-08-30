"""scripts/analyze_sweep.py — text analysis of the D(t) sweep CSV.

Seed-aware. Seven sections matching the phase-6 plan:
    1. Overall stats per variant (mean ± CI across seeds)
    2. Sustainability: total cycles completed per seed, per variant
    3. Rule-based path selection per (battery_low, network_good) cell
       — adherence check, should be 100%
    4. Path mix (rule-based only) vs initial SoC
    5. Path mix (rule-based only) vs basestation distance
    6. Latency distribution (p50, p95, max) per variant
    7. Offload data-rate cherry-picking: rule-based vs always-offload
       mean data rate on cycles where offload was chosen

Aggregation contract:
    - Per-run scalar metrics are computed per (variant, seed) first,
      then averaged with 95% CI across seeds. Normal approximation
      (mean ± 1.96 * SEM); with n ≥ 30 this is within ~4% of the
      t-based CI. Swap for bootstrap if a reviewer asks.
    - Distributional metrics (percentiles) are computed per
      (variant, seed) and then averaged across seeds with CIs, so
      the reported p95 is "mean p95 across seeds ± CI", not the
      p95 of the pooled distribution.
    - Rule adherence is a deterministic table lookup and is reported
      as a single fraction across all rule-based cycles.

Usage:
    python -m scripts.analyze_sweep [--input data/sweep_results.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


Z95 = 1.96

# The rule table from configs/decision_default (also in sim/decision.py).
# Key: (battery_low, network_good). Value: expected path.
EXPECTED_PATH: dict[tuple[bool, bool], str] = {
    (False, False): "heavy-local",   # HIGH battery, BAD network
    (False, True):  "heavy-local",   # HIGH battery, GOOD network
    (True,  False): "light-local",   # LOW battery, BAD network
    (True,  True):  "offload",       # LOW battery, GOOD network
}

VARIANT_ORDER = ["rule-based", "always-light", "always-heavy", "always-offload"]


# --------------------------------------------------------------- helpers

def ci95(values) -> tuple[float, float]:
    """Return (mean, half-width) of 95% normal-approx CI."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0
    sem = float(arr.std(ddof=1)) / np.sqrt(n)
    return mean, Z95 * sem


def fmt(mean: float, hw: float, precision: int = 2, unit: str = "") -> str:
    return f"{mean:.{precision}f} ± {hw:.{precision}f}{unit}"


def _order_variants(present: list[str]) -> list[str]:
    """Keep known variants in canonical order, append unknown at end."""
    ordered = [v for v in VARIANT_ORDER if v in present]
    ordered += [v for v in present if v not in ordered]
    return ordered


def _header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------- sections

def section_1_overall(df: pd.DataFrame) -> None:
    _header("1. OVERALL STATS PER VARIANT (mean ± 95% CI across seeds)")
    per_vs = df.groupby(["decision_variant", "seed"]).agg(
        cycles=("cycle_index", "count"),
        total_energy_j=("energy_j", "sum"),
        mean_latency_s=("latency_s", "mean"),
        delivery_pct=("physical_delivery", lambda s: s.mean() * 100),
        pred_success_pct=("predicted_success", lambda s: s.mean() * 100),
    ).reset_index()

    print(f"{'variant':<16} {'cycles':<16} {'energy (J)':<18} "
          f"{'mean lat (s)':<18} {'deliv %':<14}")
    print("-" * 78)
    for variant in _order_variants(per_vs.decision_variant.unique().tolist()):
        sub = per_vs[per_vs.decision_variant == variant]
        c = ci95(sub.cycles)
        e = ci95(sub.total_energy_j)
        l = ci95(sub.mean_latency_s)
        d = ci95(sub.delivery_pct)
        print(f"{variant:<16} "
              f"{fmt(*c, 1):<16} "
              f"{fmt(*e, 2):<18} "
              f"{fmt(*l, 3):<18} "
              f"{fmt(*d, 1, '%'):<14}")
    print()


def section_2_sustainability(df: pd.DataFrame) -> None:
    _header("2. SUSTAINABILITY: total cycles completed per seed, per variant")
    per_vs = df.groupby(["decision_variant", "seed"]).size().reset_index(name="cycles")
    # Compute the theoretical max: n_scenarios * cycles_per_run, from the data
    n_scenarios = df.groupby(["decision_variant", "seed"])[
        ["initial_soc", "basestation_y"]
    ].apply(lambda g: g.drop_duplicates().shape[0]).max()
    max_per_scenario = int(df.cycle_index.max()) + 1
    max_cycles = int(n_scenarios) * int(max_per_scenario)

    print(f"Theoretical max per seed: {n_scenarios} scenarios × "
          f"{max_per_scenario} cycles = {max_cycles} cycles")
    print()
    print(f"{'variant':<16} {'cycles (mean ± CI)':<24} {'as % of max':<16}")
    print("-" * 56)
    for variant in _order_variants(per_vs.decision_variant.unique().tolist()):
        sub = per_vs[per_vs.decision_variant == variant]
        m, hw = ci95(sub.cycles)
        pct_m, pct_hw = ci95(sub.cycles / max_cycles * 100)
        print(f"{variant:<16} {fmt(m, hw, 1):<24} {fmt(pct_m, pct_hw, 1, '%'):<16}")
    print()


def section_3_rule_adherence(df: pd.DataFrame) -> None:
    _header("3. RULE-BASED PATH ADHERENCE (should be 100% by construction)")
    rb = df[df.decision_variant == "rule-based"].copy()
    if rb.empty:
        print("No rule-based rows in sweep. Skipping.\n")
        return
    rb["expected"] = rb.apply(
        lambda r: EXPECTED_PATH[(bool(r.battery_low), bool(r.network_good))], axis=1
    )
    rb["adhered"] = rb["path"] == rb["expected"]

    total = len(rb)
    adhered = int(rb.adhered.sum())
    print(f"Overall: {adhered}/{total} = {adhered / total * 100:.1f}% adherence")
    print()
    print(f"{'battery_low':<14} {'network_good':<14} "
          f"{'expected':<14} {'n cycles':<10} {'adhered':<10}")
    print("-" * 66)
    for (bl, ng), sub in rb.groupby(["battery_low", "network_good"]):
        expected = EXPECTED_PATH[(bool(bl), bool(ng))]
        n = len(sub)
        a = int(sub.adhered.sum())
        print(f"{str(bl):<14} {str(ng):<14} {expected:<14} "
              f"{n:<10} {a}/{n} ({a / n * 100:.0f}%)")
    print()


def _path_mix_ci(df_rb: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """For each level of group_col, per-seed fraction of light/heavy/offload,
    then mean ± CI across seeds. Returns a display-ready DataFrame."""
    # Per (group_col, seed): counts of each path, normalized to fractions.
    per_seed = (
        df_rb.groupby([group_col, "seed", "path"]).size()
        .unstack(fill_value=0)
    )
    per_seed = per_seed.div(per_seed.sum(axis=1), axis=0).reset_index()

    rows = []
    for level, sub in per_seed.groupby(group_col):
        row = {group_col: level, "n_seeds": len(sub)}
        for path in ["light-local", "heavy-local", "offload"]:
            if path in sub.columns:
                m, hw = ci95(sub[path] * 100)
                row[path] = fmt(m, hw, 1, "%")
            else:
                row[path] = "0.0 ± 0.0%"
        rows.append(row)
    return pd.DataFrame(rows)


def section_4_pathmix_vs_soc(df: pd.DataFrame) -> None:
    _header("4. PATH MIX vs INITIAL SoC (rule-based only)")
    rb = df[df.decision_variant == "rule-based"]
    if rb.empty:
        print("No rule-based rows. Skipping.\n")
        return
    mix = _path_mix_ci(rb, "initial_soc").sort_values("initial_soc")
    print(mix.to_string(index=False))
    print()


def section_5_pathmix_vs_bs(df: pd.DataFrame) -> None:
    _header("5. PATH MIX vs BASESTATION DISTANCE (rule-based only)")
    rb = df[df.decision_variant == "rule-based"]
    if rb.empty:
        print("No rule-based rows. Skipping.\n")
        return
    mix = _path_mix_ci(rb, "basestation_y").sort_values("basestation_y")
    print(mix.to_string(index=False))
    print()


def section_6_latency(df: pd.DataFrame) -> None:
    _header("6. LATENCY DISTRIBUTION per variant (mean ± CI of per-seed stats)")
    # Per (variant, seed): p50, p95, max.
    per_vs = df.groupby(["decision_variant", "seed"]).agg(
        p50=("latency_s", lambda s: float(np.percentile(s, 50))),
        p95=("latency_s", lambda s: float(np.percentile(s, 95))),
        mx=("latency_s", "max"),
    ).reset_index()

    print(f"{'variant':<16} {'p50 (s)':<18} {'p95 (s)':<18} {'max (s)':<18}")
    print("-" * 70)
    for variant in _order_variants(per_vs.decision_variant.unique().tolist()):
        sub = per_vs[per_vs.decision_variant == variant]
        p50 = ci95(sub.p50)
        p95 = ci95(sub.p95)
        mx = ci95(sub.mx)
        print(f"{variant:<16} "
              f"{fmt(*p50, 3):<18} "
              f"{fmt(*p95, 3):<18} "
              f"{fmt(*mx, 3):<18}")
    print()


def section_7_offload_cherrypicking(df: pd.DataFrame) -> None:
    _header("7. OFFLOAD CHERRY-PICKING: mean data rate when offload was chosen")
    off = df[df.path == "offload"]
    if off.empty:
        print("No offload cycles. Skipping.\n")
        return
    per_vs = off.groupby(["decision_variant", "seed"]).agg(
        mean_rate_mbps=("data_rate_mbps", "mean"),
        n_offload_cycles=("cycle_index", "count"),
    ).reset_index()

    print(f"{'variant':<16} {'mean rate on offload (Mbps)':<32} "
          f"{'mean n offload cycles':<24}")
    print("-" * 72)
    for variant in _order_variants(per_vs.decision_variant.unique().tolist()):
        sub = per_vs[per_vs.decision_variant == variant]
        r = ci95(sub.mean_rate_mbps)
        n = ci95(sub.n_offload_cycles)
        print(f"{variant:<16} {fmt(*r, 2):<32} {fmt(*n, 1):<24}")
    print()
    print("(Rule-based should have HIGHER mean rate than always-offload — "
          "it only picks offload when C(t) is good.)")
    print()


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sweep_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "seed" not in df.columns:
        raise SystemExit(
            "CSV is missing the 'seed' column. Re-run s9_sweep after "
            "updating sim/metrics.py to include the seed field."
        )

    print(f"Loaded {len(df)} cycle rows from {args.input}")
    print(f"  variants:  {sorted(df.decision_variant.unique().tolist())}")
    print(f"  seeds:     {sorted(df.seed.unique().tolist())} "
          f"(n={df.seed.nunique()})")
    print(f"  SoCs:      {sorted(df.initial_soc.unique().tolist())}")
    print(f"  BS pos y:  {sorted(df.basestation_y.unique().tolist())}")
    print()

    section_1_overall(df)
    section_2_sustainability(df)
    section_3_rule_adherence(df)
    section_4_pathmix_vs_soc(df)
    section_5_pathmix_vs_bs(df)
    section_6_latency(df)
    section_7_offload_cherrypicking(df)


if __name__ == "__main__":
    main()
