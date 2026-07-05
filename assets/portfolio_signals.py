"""Deterministic buy/sell/hold verdict for a held position.

Pure logic, no DB. Scores a handful of signals already produced daily in
``adj_runs`` (predicted 1-month return, MA24 vs MA55 trend, price z-score)
into a single BUY / HOLD / SELL verdict plus a human-readable reason.

Thresholds below are the calibration knobs — tune them against realised
outcomes, don't rewrite the logic.
"""

from __future__ import annotations

# ponytail: thresholds are the tuning knobs; adjust per backtest, keep the shape.
_UPSIDE = 0.03  # ±3% predicted 1-month move counts as a vote
_Z_BAND = 1.0  # |z| beyond this counts as cheap/expensive


def _f(x):
    """Best-effort float; None/blank -> None (vote skipped)."""
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def compute_verdict(*, current_price, next_month_prediction, ma24, ma55, z_score):
    """Return (verdict, reason). verdict is BUY / HOLD / SELL / UNKNOWN.

    Each available signal casts a +1 (bullish) or -1 (bearish) vote; missing
    signals are skipped. score >= +2 -> BUY, <= -2 -> SELL, else HOLD.
    """
    price = _f(current_price)
    if not price:
        return "UNKNOWN", "no current price"

    score = 0
    reasons: list[str] = []

    pred = _f(next_month_prediction)
    if pred is not None:
        ret = (pred - price) / price
        if ret > _UPSIDE:
            score += 1
            reasons.append(f"forecast +{ret:.0%}")
        elif ret < -_UPSIDE:
            score -= 1
            reasons.append(f"forecast {ret:.0%}")

    a24, a55 = _f(ma24), _f(ma55)
    if a24 is not None and a55 is not None:
        if a24 > a55:
            score += 1
            reasons.append("uptrend (MA24>MA55)")
        elif a24 < a55:
            score -= 1
            reasons.append("downtrend (MA24<MA55)")

    z = _f(z_score)
    if z is not None:
        if z < -_Z_BAND:
            score += 1
            reasons.append("cheap (low z-score)")
        elif z > _Z_BAND:
            score -= 1
            reasons.append("extended (high z-score)")

    verdict = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
    return verdict, "; ".join(reasons) or "no strong signal"


if __name__ == "__main__":
    # Strong buy: upside + uptrend + cheap
    v, _ = compute_verdict(
        current_price=100, next_month_prediction=110, ma24=50, ma55=40, z_score=-2
    )
    assert v == "BUY", v
    # Strong sell: downside + downtrend + extended
    v, _ = compute_verdict(
        current_price=100, next_month_prediction=90, ma24=40, ma55=50, z_score=2
    )
    assert v == "SELL", v
    # Mixed -> hold
    v, _ = compute_verdict(
        current_price=100, next_month_prediction=110, ma24=40, ma55=50, z_score=0
    )
    assert v == "HOLD", v
    # No price -> unknown
    v, _ = compute_verdict(
        current_price=None, next_month_prediction=1, ma24=1, ma55=1, z_score=1
    )
    assert v == "UNKNOWN", v
    print("compute_verdict self-check passed")
