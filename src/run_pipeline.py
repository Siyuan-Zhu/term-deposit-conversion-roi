"""End-to-end run: download and verify data, leakage gate, calibrated model, call-list policies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_dataset  # noqa: E402
from src.figure import make_figure  # noqa: E402
from src.model import (  # noqa: E402
    TARGET, CalibratedPropensityModel, PreCallFeatureGuard, TargetLeakageError,
    brier, ece, leaky_auc, pr_auc, roc_auc,
)
from src.policy import UnitEconomics, evaluate_policies, threshold_frontier  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURE_PATH = REPO_ROOT / "assets" / "figure1_gain_and_nvi_curve.png"
SEED = 42
TEST_SHARE = 0.30


def random_split(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.random.default_rng(seed).permutation(len(df))
    n_test = int(round(TEST_SHARE * len(df)))
    return df.iloc[idx[n_test:]].reset_index(drop=True), df.iloc[idx[:n_test]].reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The file is in date order, so the last 30% of rows is the most recent period."""
    cut = int(round((1 - TEST_SHARE) * len(df)))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def score(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    gate_blocks_duration = False
    try:
        PreCallFeatureGuard.validate_columns(train.columns)  # full frame still holds `duration`
    except TargetLeakageError:
        gate_blocks_duration = True
    tr = PreCallFeatureGuard.select(train, include_target=True)
    te = PreCallFeatureGuard.select(test)
    model = CalibratedPropensityModel().fit(tr)
    y = test[TARGET].to_numpy()
    p_raw, p_cal = model.predict_uncalibrated(te), model.predict_proba(te)
    return {
        "gate_blocks_duration": gate_blocks_duration,
        "y": y,
        "p_cal": p_cal,
        "metrics": {
            "roc_auc_pre_call": round(roc_auc(y, p_cal), 4),
            "pr_auc_pre_call": round(pr_auc(y, p_cal), 4),
            "roc_auc_with_duration": round(leaky_auc(train, test), 4),
            "brier_uncalibrated": round(brier(y, p_raw), 4),
            "brier_calibrated": round(brier(y, p_cal), 4),
            "ece_uncalibrated_pct": round(100 * ece(y, p_raw), 2),
            "ece_calibrated_pct": round(100 * ece(y, p_cal), 2),
            "test_base_rate_pct": round(100 * y.mean(), 2),
        },
    }


def run_full_pipeline() -> dict:
    df = load_dataset()
    train, test = random_split(df)
    main = score(train, test)
    econ = UnitEconomics()
    policies = evaluate_policies(main["y"], main["p_cal"], econ)

    t_train, t_test = temporal_split(df)
    temporal = score(t_train, t_test)
    temporal_policies = evaluate_policies(temporal["y"], temporal["p_cal"], econ)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    make_figure(main["y"], main["p_cal"], FIGURE_PATH, econ)
    result = {
        "dataset": {"rows": int(len(df)), "train": int(len(train)), "test": int(len(test)),
                    "base_rate_pct": round(100 * df[TARGET].mean(), 2)},
        "unit_economics": {"c_call": econ.c_call, "v_convert": econ.v_convert, "breakeven_prob": econ.breakeven_prob},
        "gate_blocks_duration": main["gate_blocks_duration"],
        "random_split": {"metrics": main["metrics"], "policies": [p.as_dict() for p in policies]},
        "temporal_split": {"metrics": temporal["metrics"], "policies": [p.as_dict() for p in temporal_policies]},
        "frontier": threshold_frontier(main["y"], main["p_cal"], econ),
    }
    (PROCESSED_DIR / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([p.as_dict() for p in policies]).to_csv(PROCESSED_DIR / "policy_comparison.csv", index=False)
    return result


if __name__ == "__main__":
    r = run_full_pipeline()
    for name in ("random_split", "temporal_split"):
        print(f"== {name}")
        print(json.dumps(r[name]["metrics"], indent=1))
        print(pd.DataFrame(r[name]["policies"]).to_string(index=False))
