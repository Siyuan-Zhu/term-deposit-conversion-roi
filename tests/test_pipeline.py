"""Tests. Policy and checksum tests run offline; the rest download the UCI file once."""

import numpy as np
import pytest

from src.data import EXPECTED_ROWS, ChecksumError, load_dataset, verify_checksum
from src.model import (
    TARGET, CalibratedPropensityModel, PreCallFeatureGuard, TargetLeakageError, ece, roc_auc,
)
from src.policy import UnitEconomics, evaluate_policies, gain_curve, threshold_frontier
from src.run_pipeline import random_split, run_full_pipeline


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.fixture(scope="module")
def result():
    return run_full_pipeline()


def test_checksum_rejects_tampered_file(tmp_path):
    bad = tmp_path / "bank.csv"
    bad.write_text("not the dataset")
    with pytest.raises(ChecksumError):
        verify_checksum(bad)


def test_dataset_shape(df):
    assert len(df) == EXPECTED_ROWS == 41188
    assert int(df[TARGET].sum()) == 4640
    assert "duration" in df.columns


def test_guard_blocks_duration_and_campaign(df):
    with pytest.raises(TargetLeakageError):
        PreCallFeatureGuard.validate_columns(df.columns)
    clean = PreCallFeatureGuard.select(df, include_target=True)
    assert "duration" not in clean.columns and "campaign" not in clean.columns
    model = CalibratedPropensityModel().fit(clean)
    with pytest.raises(TargetLeakageError):
        model.predict_proba(df.drop(columns=[TARGET, "y"]))


def test_platt_scaling_reduces_ece(df):
    train, test = random_split(df)
    model = CalibratedPropensityModel().fit(PreCallFeatureGuard.select(train, include_target=True))
    te = PreCallFeatureGuard.select(test)
    y = test[TARGET].to_numpy()
    assert ece(y, model.predict_proba(te)) < 0.02 < ece(y, model.predict_uncalibrated(te))


def test_leakage_gap(result):
    m = result["random_split"]["metrics"]
    assert result["gate_blocks_duration"] is True
    assert m["roc_auc_with_duration"] - m["roc_auc_pre_call"] > 0.10
    assert m["roc_auc_pre_call"] > 0.75


def test_break_even_math():
    econ = UnitEconomics(c_call=1.0, v_convert=12.5)
    assert econ.breakeven_prob == pytest.approx(0.08)
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.5, 0.07, 0.081, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    by = {r.policy: r for r in evaluate_policies(y, p, econ, capacity_share=0.3)}
    assert by["call_everyone"].net_value == pytest.approx(2 * 12.5 - 10)
    be = by["break_even"]
    assert (be.calls, be.conversions) == (3, 2)
    assert be.net_value == pytest.approx(2 * 12.5 - 3)
    assert be.roi_pct == pytest.approx(100 * be.net_value / be.calls, abs=0.01)
    assert be.net_value_index == pytest.approx(100 * be.net_value / by["call_everyone"].net_value, abs=0.01)


def test_roi_and_net_value_index_rank_policies_differently(result):
    by = {p["policy"]: p for p in result["random_split"]["policies"]}
    assert by["naive_cutoff"]["roi_pct"] > by["break_even"]["roi_pct"]
    assert by["naive_cutoff"]["net_value_index"] < by["break_even"]["net_value_index"]
    assert by["break_even"]["net_value_index"] > 100


def test_frontier_peaks_near_break_even(result):
    best = max(result["frontier"], key=lambda r: r["net_value"])
    assert abs(best["threshold"] - 0.08) <= 0.03


def test_gain_curve_and_auc_helpers():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    share, cum = gain_curve(y, p)
    assert cum[-1] == 1.0 and share[-1] == 1.0
    assert roc_auc(y, p) == pytest.approx(0.75)
    assert threshold_frontier(y, p)[0]["calls"] == 4
