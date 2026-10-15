"""
Characterize the model-class difference between our pipeline's
`decay-line + flat-noise` piecewise model
(`kd_utils.Kd_analysis.fit_beers_law_bma`) and Muggeo's two-free-slope
iterative piecewise regression (the `piecewise-regression` library).

This is NOT a migration test — we are not adopting `piecewise-regression`.
The fixture exists to:

  1. Document how Muggeo's free-slope fit behaves on our home-turf data
     (exponential decay over a flat noise floor): does it correctly drive
     `slope2 -> 0`, and how close does it get to the true breakpoint and Kd?
  2. Show what happens when the underlying truth violates our flat-floor
     assumption (two genuinely sloped segments): our model is misspecified;
     Muggeo's free-slope model recovers both rates.

Output is a printed comparison table plus loose guardrail assertions to
catch catastrophic regressions in either implementation. The synthetic
fixtures are seeded so the printed numbers are reproducible.

Install:
    pip install piecewise-regression

Run:
    pytest -s tests/test_bic_muggeo_comparison.py     # see the table
    python tests/test_bic_muggeo_comparison.py        # standalone
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pwlf_pkg = pytest.importorskip(
    "piecewise_regression",
    reason="install with `pip install piecewise-regression` to run this fixture",
)

from kd_utils.Kd_analysis import fit_beers_law_bma  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
TRUE_KD = 0.4
TRUE_E0 = 1000.0
TRUE_NOISE = 5.0
TRUE_BP = 6.0


def make_flat_floor(seed: int = 0):
    """Decay then flat noise floor — our model's home turf.

    Truth: log(counts) = log(E0*exp(-Kd*z) + N) for z<bp, log(N) for z>=bp.
    """
    rng = np.random.default_rng(seed)
    z = np.arange(0.125, 12.125, 0.25)  # 48 bins, 0.25 m wide
    expected = np.where(
        z < TRUE_BP,
        TRUE_E0 * np.exp(-TRUE_KD * z) + TRUE_NOISE,
        TRUE_NOISE,
    )
    counts = np.maximum(rng.poisson(expected).astype(float), 1.0)
    return z, counts


def make_two_slope(seed: int = 1):
    """Decay then *slower* decay — Muggeo's home turf, ours is misspecified.

    Two true slopes (Kd=0.4 above the breakpoint, Kd=0.05 below) make the
    'flat noise' assumption wrong: the second segment is still trending
    downward, just slowly.
    """
    rng = np.random.default_rng(seed)
    z = np.arange(0.125, 12.125, 0.25)
    kd1, kd2 = 0.4, 0.05
    A1 = TRUE_E0
    f_at_bp = A1 * np.exp(-kd1 * TRUE_BP)
    expected = np.where(
        z < TRUE_BP,
        A1 * np.exp(-kd1 * z) + TRUE_NOISE,
        f_at_bp * np.exp(-kd2 * (z - TRUE_BP)) + TRUE_NOISE,
    )
    counts = np.maximum(rng.poisson(expected).astype(float), 1.0)
    return z, counts


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------
def fit_ours(z, counts):
    """Our pipeline's BMA fit. Returns (kd, bp, e0, noise_floor, reason)."""
    df = pd.DataFrame({'zdepth': z, 'photon_counts': counts})
    return fit_beers_law_bma(df)


def fit_muggeo(z, counts):
    """Muggeo's iterative two-slope fit on log(counts). Returns dict or None
    on non-convergence.

    Parameterization (`piecewise-regression`):
        log(counts) = const + alpha1*z + beta1*max(z - bp, 0)
    so slope1 = alpha1, slope2 = alpha1 + beta1 = alpha2.
    Both `alpha2` and `beta1` are exposed in the estimates dict.
    """
    pwlf = pwlf_pkg.Fit(z.tolist(), np.log(counts).tolist(), n_breakpoints=1)
    res = pwlf.get_results()
    if not res.get('converged', False):
        return None
    e = res['estimates']
    return {
        'kd1': -e['alpha1']['estimate'],
        'kd2': -e['alpha2']['estimate'],
        'bp': e['breakpoint1']['estimate'],
        'bp_ci': tuple(e['breakpoint1']['confidence_interval']),
        'bic_muggeo': res.get('bic', np.nan),  # statsmodels-style BIC, k=4
        'rss': res.get('rss', np.nan),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(label, z, counts):
    print()
    print(f"{'='*70}")
    print(label)
    print(f"{'='*70}")
    print(f"truth:    kd = {TRUE_KD}    breakpoint = {TRUE_BP:.2f} m    "
          f"noise = {TRUE_NOISE}    n_bins = {len(z)}")
    print()

    kd_o, e0_o, bp_o, nf_o, reason = fit_ours(z, counts)
    print(f"ours (BMA, decay-line + flat-mean, k=3):")
    print(f"    kd          = {kd_o:.4f}    (truth {TRUE_KD},  err = "
          f"{(kd_o - TRUE_KD) / TRUE_KD * 100:+.1f}%)")
    print(f"    breakpoint  = {bp_o:.2f} m   (truth {TRUE_BP:.2f},  err = "
          f"{bp_o - TRUE_BP:+.2f} m)")
    print(f"    noise_floor = {nf_o:.2f}     (truth {TRUE_NOISE})")
    if reason is not None:
        print(f"    reason      = {reason}")
    print()

    m = fit_muggeo(z, counts)
    if m is None:
        print("muggeo: did not converge")
        return kd_o, bp_o, None
    print(f"muggeo (iterative, two free slopes, k=4):")
    print(f"    kd1         = {m['kd1']:.4f}    (truth {TRUE_KD},  err = "
          f"{(m['kd1'] - TRUE_KD) / TRUE_KD * 100:+.1f}%)")
    print(f"    kd2         = {m['kd2']:+.4f}   (truth 0 if flat-floor; "
          f"~0.05 if two-slope)")
    print(f"    breakpoint  = {m['bp']:.2f} m   (truth {TRUE_BP:.2f},  err = "
          f"{m['bp'] - TRUE_BP:+.2f} m)")
    print(f"    bp 95% CI   = [{m['bp_ci'][0]:.2f}, {m['bp_ci'][1]:.2f}] m")
    print(f"    bic_muggeo  = {m['bic_muggeo']:.3f}    "
          f"rss = {m['rss']:.3f}")

    return kd_o, bp_o, m


# ---------------------------------------------------------------------------
# Tests (loose guardrails — not strict equality)
# ---------------------------------------------------------------------------
def test_flat_floor_characterization():
    """Home-turf case: truth has a flat noise floor.

    Both methods should recover Kd within ~30% of truth. Muggeo's slope2
    should be small (|kd2| < 0.15) — non-zero by FP noise, but not
    indicating a real second decay regime. Muggeo's breakpoint may drift
    a meter or so from truth because its 4-param model can compensate for
    the noise-floor curvature with a nonzero slope2."""
    z, counts = make_flat_floor()
    kd_o, bp_o, m = report("Fixture A — flat noise floor (our model's class)", z, counts)

    assert 0 < kd_o, f"ours produced non-positive Kd: {kd_o}"
    assert abs(kd_o - TRUE_KD) / TRUE_KD < 0.30, \
        f"ours kd more than 30% off truth: {kd_o}"
    assert abs(bp_o - TRUE_BP) < 1.5, f"ours bp more than 1.5 m off: {bp_o}"

    if m is not None:
        # Muggeo's kd1 may be biased steeper (the free-slope model pushes
        # the breakpoint deeper to chase residual structure near the floor),
        # so the bound here is looser than for our model.
        assert abs(m['kd1'] - TRUE_KD) / TRUE_KD < 1.0, \
            f"muggeo kd1 more than 100% off: {m['kd1']}"
        # Slope2 must be small in absolute terms when truth is flat.
        assert abs(m['kd2']) < 0.15, \
            f"muggeo slope2 unexpectedly large for flat-floor: {m['kd2']}"


def test_two_slope_characterization():
    """Off-turf case: truth has two genuine slopes.

    Our flat-floor assumption is wrong. We expect:
      * Muggeo's slope2 should be clearly nonzero (~0.05 truth, but the
        finite-sample estimate may scatter; we just require kd2 > 0.005).
      * Our model's noise_floor will be biased high (it's the mean of a
        downward-trending segment, not a true noise mean).
    No equality assertions — this fixture only documents the bias."""
    z, counts = make_two_slope()
    kd_o, bp_o, m = report("Fixture B — two-slope decay (our model misspecified)", z, counts)

    if m is not None:
        assert m['kd2'] > 0.005, \
            f"muggeo failed to detect a non-flat second segment: kd2={m['kd2']}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def main() -> None:
    test_flat_floor_characterization()
    test_two_slope_characterization()
    print()
    print("=" * 70)
    print("CHARACTERIZATION COMPLETE — see tables above for model-class differences.")
    print("=" * 70)


if __name__ == "__main__":
    main()
