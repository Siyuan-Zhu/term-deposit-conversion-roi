"""Pre-call leakage gate and a Platt-calibrated logistic propensity model.

`PreCallFeatureGuard` rejects any column that is only known after the call.
`CalibratedPropensityModel` fits a class-weighted, L2-regularised logistic
regression by IRLS, then fits Platt scaling on 5-fold out-of-fold logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TARGET = "subscribed"

CATEGORICAL = ("job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome")
NUMERIC = ("age", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")
# pdays == 999 means "never contacted before"; it becomes an indicator plus a clipped value.
PRE_CALL_FEATURES = CATEGORICAL + NUMERIC + ("pdays",)

# Known only once the call has happened.
FORBIDDEN_POST_CALL = ("duration",)
# `campaign` counts contacts in this campaign including the current one, so it is also
# not known before dialling. Excluded rather than argued about.
EXCLUDED_AMBIGUOUS = ("campaign",)


class TargetLeakageError(ValueError):
    """Raised when a post-call column reaches the pre-call model."""


class PreCallFeatureGuard:
    @staticmethod
    def validate_columns(columns: Iterable[str]) -> list[str]:
        cols = list(columns)
        blocked = [c for c in cols if c in FORBIDDEN_POST_CALL + EXCLUDED_AMBIGUOUS]
        if blocked:
            raise TargetLeakageError(f"post-call columns reach the pre-call model: {blocked}")
        missing = [c for c in PRE_CALL_FEATURES if c not in cols]
        if missing:
            raise ValueError(f"missing pre-call features: {missing}")
        return cols

    @staticmethod
    def select(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
        cols = list(PRE_CALL_FEATURES) + ([TARGET] if include_target and TARGET in df.columns else [])
        return df[cols].copy()


# ---- metrics -------------------------------------------------------------------------------

def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float(precision[ys == 1].mean()) if ys.sum() else 0.0


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-count bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            total += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(total)


# ---- model ---------------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


def design_matrix(df: pd.DataFrame, levels: dict[str, list[str]] | None = None, scale: dict | None = None,
                  include_duration: bool = False):
    """Return (X, levels, scale). Pass levels/scale from training to encode new data identically."""
    fit = levels is None
    levels = levels or {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}
    num = df[list(NUMERIC)].to_numpy(dtype=float)
    pdays = df["pdays"].to_numpy(dtype=float)
    never = (pdays == 999).astype(float)
    num = np.column_stack([num, np.where(pdays == 999, 0.0, pdays)])
    if include_duration:
        num = np.column_stack([num, np.log1p(df["duration"].to_numpy(dtype=float))])
    if fit:
        scale = {"mu": num.mean(axis=0), "sd": num.std(axis=0) + 1e-9}
    cols = [np.ones(len(df)), never, *((num - scale["mu"]) / scale["sd"]).T]
    for c in CATEGORICAL:
        v = df[c].astype(str).to_numpy()
        cols.extend((v == lvl).astype(float) for lvl in levels[c][1:])  # first level is the baseline
    return np.column_stack(cols), levels, scale


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, pos_weight: float = 1.0, max_iter: int = 40) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    sw = np.where(y == 1, pos_weight, 1.0)
    reg = np.eye(X.shape[1]) * l2
    reg[0, 0] = 1e-6
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        w = sw * np.maximum(p * (1 - p), 1e-6)
        step = np.linalg.solve((X.T * w) @ X + reg, X.T @ (sw * (p - y)) + reg @ beta)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


@dataclass
class CalibratedPropensityModel:
    l2: float = 1.0
    pos_weight: float = 2.4
    n_folds: int = 5
    seed: int = 42

    def fit(self, df: pd.DataFrame) -> "CalibratedPropensityModel":
        PreCallFeatureGuard.validate_columns(df.columns.drop(TARGET, errors="ignore"))
        X, self._levels, self._scale = design_matrix(df)
        y = df[TARGET].to_numpy(dtype=int)
        self._beta = fit_logistic(X, y, self.l2, self.pos_weight)
        fold = np.random.default_rng(self.seed).permutation(len(y)) % self.n_folds
        oof = np.zeros(len(y))
        for k in range(self.n_folds):
            tr = fold != k
            oof[~tr] = X[~tr] @ fit_logistic(X[tr], y[tr], self.l2, self.pos_weight)
        self._platt = fit_logistic(np.column_stack([np.ones(len(oof)), oof]), y, l2=1e-4)
        return self

    def _logits(self, df: pd.DataFrame) -> np.ndarray:
        PreCallFeatureGuard.validate_columns(df.columns.drop(TARGET, errors="ignore"))
        X, _, _ = design_matrix(df, self._levels, self._scale)
        return X @ self._beta

    def predict_uncalibrated(self, df: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self._logits(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self._platt[0] + self._platt[1] * self._logits(df))


def leaky_auc(train: pd.DataFrame, test: pd.DataFrame, l2: float = 1.0) -> float:
    """Audit only: test AUC of the same logistic model with log(duration) added."""
    cols = list(PRE_CALL_FEATURES) + ["duration"]
    Xtr, lv, sc = design_matrix(train[cols], include_duration=True)
    Xte, _, _ = design_matrix(test[cols], lv, sc, include_duration=True)
    beta = fit_logistic(Xtr, train[TARGET].to_numpy(dtype=int), l2)
    return roc_auc(test[TARGET].to_numpy(dtype=int), Xte @ beta)
