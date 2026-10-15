"""--min_fit_span_m rejects Kd fitted over too short a depth range."""
import numpy as np
import pandas as pd
import pytest

from kd_utils.Kd_analysis import signal_depth_span, _fit_kd_with_method


def hist(depths, counts):
    return pd.DataFrame({"zdepth": np.asarray(depths, float),
                         "photon_counts": np.asarray(counts, float)})


def test_span_measures_signal_bins_only():
    # a lone 1-photon bin at 8 m must not stretch the span
    h = hist([1.0, 1.25, 1.5, 1.75, 8.0], [80, 60, 45, 34, 1])
    assert signal_depth_span(h) == pytest.approx(0.75)


def test_span_nan_when_too_few_signal_bins():
    assert np.isnan(signal_depth_span(hist([1.0, 1.25], [40, 1])))


def test_short_span_is_rejected():
    # 1 m span: the case that produced Kd > 1.0 at Bohai
    h = hist(np.arange(1.0, 2.01, 0.25), [200, 120, 70, 42, 25])
    kd, e0, nf, reason = _fit_kd_with_method(h, method="log_linear", min_fit_span_m=2.0)
    assert np.isnan(kd)
    assert reason == "fit_span_too_short"


def test_adequate_span_still_fits():
    z = np.arange(1.0, 5.01, 0.25)
    h = hist(z, 500 * np.exp(-0.4 * z))
    kd, e0, nf, reason = _fit_kd_with_method(h, method="log_linear", min_fit_span_m=2.0)
    assert reason != "fit_span_too_short"
    assert kd == pytest.approx(0.4, abs=0.02)


def test_disabled_by_zero():
    h = hist(np.arange(1.0, 2.01, 0.25), [200, 120, 70, 42, 25])
    kd, _, _, reason = _fit_kd_with_method(h, method="log_linear", min_fit_span_m=0.0)
    assert np.isfinite(kd) and reason != "fit_span_too_short"


def test_default_is_off_when_not_passed():
    """The fitter must not change behaviour for callers that don't pass it."""
    h = hist(np.arange(1.0, 2.01, 0.25), [200, 120, 70, 42, 25])
    kd, _, _, reason = _fit_kd_with_method(h, method="log_linear")
    assert np.isfinite(kd) and reason != "fit_span_too_short"
