"""Call-list policies and threshold/ROI optimisation in relative units.

One call costs `c_call` units and one conversion is worth `v_convert` units, so a
prospect is worth calling when p_hat >= c_call / v_convert. Only the ratio matters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

C_CALL = 1.0
V_CONVERT = 12.5  # assumption: a conversion is worth 12.5 calls


@dataclass(frozen=True)
class UnitEconomics:
    c_call: float = C_CALL
    v_convert: float = V_CONVERT

    @property
    def breakeven_prob(self) -> float:
        return self.c_call / self.v_convert


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    rule: str
    calls: int
    call_share_pct: float
    conversions: int
    capture_pct: float
    precision_pct: float
    lift: float
    net_value: float
    roi_pct: float
    net_value_index: float  # net value as a percentage of the call-everyone policy (= 100)

    def as_dict(self) -> dict:
        return asdict(self)


def _evaluate(name: str, rule: str, y: np.ndarray, called: np.ndarray, econ: UnitEconomics, base_net: float) -> PolicyResult:
    calls = int(called.sum())
    conv = int(y[called].sum())
    net = conv * econ.v_convert - calls * econ.c_call
    return PolicyResult(
        policy=name,
        rule=rule,
        calls=calls,
        call_share_pct=round(100 * calls / len(y), 2),
        conversions=conv,
        capture_pct=round(100 * conv / y.sum(), 2),
        precision_pct=round(100 * conv / calls, 2) if calls else 0.0,
        lift=round((conv / calls) / y.mean(), 2) if calls else 0.0,
        net_value=round(net, 2),
        roi_pct=round(100 * net / (calls * econ.c_call), 2) if calls else 0.0,
        net_value_index=round(100 * net / base_net, 2) if base_net else 0.0,
    )


def evaluate_policies(y, p_hat, econ: UnitEconomics = UnitEconomics(), capacity_share: float = 0.30) -> list[PolicyResult]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p_hat, dtype=float)
    n = len(y)
    base_net = y.sum() * econ.v_convert - n * econ.c_call
    cap_n = int(round(capacity_share * n))
    top = np.zeros(n, dtype=bool)
    top[np.argsort(-p, kind="mergesort")[:cap_n]] = True
    return [
        _evaluate("call_everyone", "all prospects", y, np.ones(n, dtype=bool), econ, base_net),
        _evaluate("naive_cutoff", "p_hat >= 0.50", y, p >= 0.5, econ, base_net),
        _evaluate("break_even", f"p_hat >= {econ.breakeven_prob:.3f}", y, p >= econ.breakeven_prob, econ, base_net),
        _evaluate("capacity_cap", f"top {int(capacity_share * 100)}% by p_hat", y, top, econ, base_net),
    ]


def threshold_frontier(y, p_hat, econ: UnitEconomics = UnitEconomics(), n_points: int = 101) -> list[dict]:
    """Net value and ROI as the call threshold moves from 0 to 0.5."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p_hat, dtype=float)
    out = []
    for t in np.linspace(0.0, 0.5, n_points):
        called = p >= t
        calls = int(called.sum())
        net = int(y[called].sum()) * econ.v_convert - calls * econ.c_call
        out.append({"threshold": round(float(t), 4), "calls": calls, "net_value": round(float(net), 2),
                    "roi_pct": round(100 * net / calls, 2) if calls else 0.0})
    return out


def gain_curve(y, p_hat) -> tuple[np.ndarray, np.ndarray]:
    """Share of prospects called (best first) vs share of conversions captured."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(p_hat, dtype=float), kind="mergesort")
    cum = np.cumsum(y[order]) / y.sum()
    return np.arange(1, len(y) + 1) / len(y), cum
