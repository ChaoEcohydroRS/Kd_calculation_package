"""
Audit our BIC implementation against statsmodels.

`fit_beers_law_bma` and `fit_beers_law_hybrid` use the abbreviated form

    BIC = n * log(RSS/n + 1e-10) + k * log(n)

dropping the additive constant `n*log(2*pi) + n` from the full Gaussian-OLS
log-likelihood. This is equivalent for *relative* model selection (Δ-BIC and
BMA weights are unchanged), but not for the absolute number reported by
`statsmodels.RegressionResults.bic`.

This test verifies, on a synthetic two-segment histogram:
  1. The full single-line BIC matches statsmodels' to machine precision after
     adding back the dropped constant.
  2. For the piecewise model (decay-line + noise-mean, 3 free regression
     params, summed RSS over both segments), the same identity holds.
  3. ΔBIC values relative to the single-line baseline match between the two
     formulations to within the analytic +1e-10-guard residual.
  4. BMA weights, which use only ΔBIC, agree to ~1e-10 (the guard collapses
     under the exp(-0.5·ΔBIC) transform).
  5. The argmin-BIC breakpoint is the same under both formulations.

The +1e-10 guard inside `n*log(rss/n + 1e-10)` introduces an analytic
residual on ΔBIC bounded by

    |ΔBIC_ours - ΔBIC_sm|  ≲  1e-10 · n² · (1/rss_0 - 1/rss_i)

which, for the synthetic fixture (n=48, RSS_min≈4), peaks around 5e-8.
Far below any meaningful BIC threshold (Kass & Raftery 1995 cite ΔBIC>2
as "positive evidence"), and irrelevant to the BMA weights it feeds.

Run as a pytest:
    pytest tests/test_bic_vs_statsmodels.py -v

Or as a script for the printed comparison table:
    python tests/test_bic_vs_statsmodels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# BIC formulas
# ---------------------------------------------------------------------------
def bic_ours(rss: float, n: int, k: int) -> float:
    """Replicates kd_utils.Kd_analysis.fit_beers_law_bma BIC formula."""
    return n * np.log(rss / n + 1e-10) + k * np.log(n)


def bic_statsmodels_style(rss: float, n: int, k: int) -> float:
    """Canonical Gaussian-OLS BIC: -2*llf + log(n)*k.

    For OLS with Gaussian residuals,
        llf = -n/2 * log(2*pi) - n/2 * log(rss/n) - n/2
    so
        -2*llf = n*log(2*pi) + n*log(rss/n) + n.
    statsmodels' RegressionResults.bic equals this, with k = df_modelwc
    (regression params including intercept; sigma^2 is not counted).
    """
    return n * np.log(2 * np.pi) + n * np.log(rss / n) + n + k * np.log(n)


def gaussian_offset(n: int) -> float:
    """The constant our BIC drops: n*log(2*pi) + n."""
    return n * np.log(2 * np.pi) + n


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------
def make_synthetic_histogram(seed: int = 0):
    """Decay-then-flat histogram with a clean breakpoint at z = 6 m."""
    rng = np.random.default_rng(seed)
    bin_centers = np.arange(0.125, 12.125, 0.25)  # 48 bins, 0..12 m
    true_breakpoint = 6.0
    true_kd = 0.4
    true_e0 = 1000.0
    true_noise = 5.0

    expected = np.where(
        bin_centers < true_breakpoint,
        true_e0 * np.exp(-true_kd * bin_centers) + true_noise,
        true_noise,
    )
    counts = rng.poisson(expected).astype(float)
    counts = np.maximum(counts, 1)  # avoid log(0) in the noise tail
    return bin_centers, counts


# ---------------------------------------------------------------------------
# Per-candidate BIC computation (mirrors fit_beers_law_bma)
# ---------------------------------------------------------------------------
def compute_candidates(depths: np.ndarray, log_counts: np.ndarray):
    """Yield (bp_idx, depth, rss_total, k) for every viable candidate
    plus the full single-line baseline as bp_idx=None.
    """
    n = len(depths)

    # Full single-line model (baseline, no breakpoint)
    slope, intercept = np.polyfit(depths, log_counts, 1)
    rss_full = float(np.sum((slope * depths + intercept - log_counts) ** 2))
    yield (None, np.nan, rss_full, 2)

    # Piecewise: line on [0..bp), flat mean on [bp..end)
    for bp_idx in range(8, n - 1):
        z_d = depths[:bp_idx]
        lc_d = log_counts[:bp_idx]
        slope_d, intercept_d = np.polyfit(z_d, lc_d, 1)
        if slope_d >= 0:
            continue
        rss_d = float(np.sum((slope_d * z_d + intercept_d - lc_d) ** 2))
        lc_n = log_counts[bp_idx:]
        rss_n = float(np.sum((lc_n - lc_n.mean()) ** 2))
        yield (bp_idx, float(depths[bp_idx]), rss_d + rss_n, 3)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
def test_full_line_bic_matches_statsmodels_bic_attr():
    depths, counts = make_synthetic_histogram()
    log_counts = np.log(counts)
    n = len(depths)

    X_full = sm.add_constant(depths)
    res_full = sm.OLS(log_counts, X_full).fit()

    bic_manual = bic_statsmodels_style(float(res_full.ssr), n, k=2)
    assert abs(res_full.bic - bic_manual) < 1e-9, (
        f"Manual sm-style formula vs RegressionResults.bic: "
        f"{bic_manual:.6f} vs {res_full.bic:.6f}"
    )


def test_full_line_offset_is_gaussian_constant():
    depths, counts = make_synthetic_histogram()
    log_counts = np.log(counts)
    n = len(depths)

    slope, intercept = np.polyfit(depths, log_counts, 1)
    rss = float(np.sum((slope * depths + intercept - log_counts) ** 2))

    bo = bic_ours(rss, n, k=2)
    bs = bic_statsmodels_style(rss, n, k=2)

    assert abs((bs - bo) - gaussian_offset(n)) < 1e-8, (
        f"Offset {bs - bo:.6f} != n*log(2*pi)+n = {gaussian_offset(n):.6f}"
    )


def test_delta_bic_matches_across_candidates():
    depths, counts = make_synthetic_histogram()
    log_counts = np.log(counts)
    n = len(depths)

    bic_o, bic_s = [], []
    for _, _, rss, k in compute_candidates(depths, log_counts):
        bic_o.append(bic_ours(rss, n, k))
        bic_s.append(bic_statsmodels_style(rss, n, k))
    bic_o = np.array(bic_o)
    bic_s = np.array(bic_s)

    # Δ-BIC is the BIC relative to the single-line baseline (index 0).
    delta_o = bic_o - bic_o[0]
    delta_s = bic_s - bic_s[0]

    max_diff = float(np.max(np.abs(delta_o - delta_s)))
    # Analytic bound from the +1e-10 guard: 1e-10 * n^2 / RSS_min.
    # Stay an order of magnitude under any meaningful BIC threshold (>=2).
    assert max_diff < 1e-5, f"Max |ΔBIC_ours - ΔBIC_sm| = {max_diff:.3e}"


def test_bma_weights_match_across_candidates():
    depths, counts = make_synthetic_histogram()
    log_counts = np.log(counts)
    n = len(depths)

    bic_o, bic_s = [], []
    for _, _, rss, k in compute_candidates(depths, log_counts):
        bic_o.append(bic_ours(rss, n, k))
        bic_s.append(bic_statsmodels_style(rss, n, k))
    bic_o = np.array(bic_o)
    bic_s = np.array(bic_s)

    w_o = np.exp(-0.5 * (bic_o - bic_o.min()))
    w_o /= w_o.sum()
    w_s = np.exp(-0.5 * (bic_s - bic_s.min()))
    w_s /= w_s.sum()

    max_w_diff = float(np.max(np.abs(w_o - w_s)))
    assert max_w_diff < 1e-12, f"Max |w_ours - w_sm| = {max_w_diff:.3e}"


def test_argmin_breakpoint_agrees():
    """The selected (min-BIC) breakpoint must be the same under both formulas."""
    depths, counts = make_synthetic_histogram()
    log_counts = np.log(counts)
    n = len(depths)

    bp_indices = []
    bic_o, bic_s = [], []
    for bp_idx, _, rss, k in compute_candidates(depths, log_counts):
        bp_indices.append(bp_idx)
        bic_o.append(bic_ours(rss, n, k))
        bic_s.append(bic_statsmodels_style(rss, n, k))

    argmin_o = int(np.argmin(bic_o))
    argmin_s = int(np.argmin(bic_s))
    assert argmin_o == argmin_s, (
        f"min-BIC candidate disagrees: "
        f"ours bp_idx={bp_indices[argmin_o]}, sm bp_idx={bp_indices[argmin_s]}"
    )


# ---------------------------------------------------------------------------
# Standalone runner: print the comparison table
# ---------------------------------------------------------------------------
def main() -> None:
    depths, counts = make_synthetic_histogram()
    log_counts = np.log(counts)
    n = len(depths)
    offset = gaussian_offset(n)

    rows = list(compute_candidates(depths, log_counts))

    bic_o = np.array([bic_ours(rss, n, k) for _, _, rss, k in rows])
    bic_s = np.array([bic_statsmodels_style(rss, n, k) for _, _, rss, k in rows])
    delta_o = bic_o - bic_o[0]
    delta_s = bic_s - bic_s[0]

    print(f"n = {n} bins, offset n*log(2*pi)+n = {offset:.6f}")
    print()
    print(f"{'bp_idx':>6} {'depth':>6} {'k':>3} {'BIC_ours':>14} "
          f"{'BIC_sm':>14} {'ΔBIC_o':>10} {'ΔBIC_s':>10} {'Δ_diff':>10}")
    for (bp_idx, depth, _, k), bo, bs, do, ds in zip(rows, bic_o, bic_s, delta_o, delta_s):
        idx_str = f"{bp_idx:>6}" if bp_idx is not None else f"{'full':>6}"
        depth_str = f"{depth:>6.2f}" if not np.isnan(depth) else f"{'-':>6}"
        print(f"{idx_str} {depth_str} {k:>3} {bo:>14.6f} {bs:>14.6f} "
              f"{do:>10.4f} {ds:>10.4f} {abs(do-ds):>10.2e}")

    max_delta = float(np.max(np.abs(delta_o - delta_s)))
    w_o = np.exp(-0.5 * (bic_o - bic_o.min())); w_o /= w_o.sum()
    w_s = np.exp(-0.5 * (bic_s - bic_s.min())); w_s /= w_s.sum()
    max_weight = float(np.max(np.abs(w_o - w_s)))

    print()
    print(f"Max |ΔBIC_ours - ΔBIC_sm|: {max_delta:.3e}")
    print(f"Max |w_ours    - w_sm|:    {max_weight:.3e}")
    print(f"argmin_o = {int(np.argmin(bic_o))}, argmin_s = {int(np.argmin(bic_s))}")

    assert max_delta < 1e-5
    assert max_weight < 1e-12
    print()
    print("PASS: ΔBIC and BMA weights match statsmodels to floating-point precision.")


if __name__ == "__main__":
    main()
