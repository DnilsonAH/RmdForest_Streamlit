from typing import Dict, List


def make_decision(predicted: float, last_price: float, metrics: Dict, predicted_intraday: float = 0.0) -> Dict:
    threshold_up = 1.02
    threshold_down = 0.98
    recommendation = "MANTENER"
    reasons: List[str] = []

    if predicted > last_price * threshold_up:
        recommendation = "COMPRAR"
    elif predicted < last_price * threshold_down:
        recommendation = "VENDER"

    pred_std = float(metrics.get("pred_std", 0.0))
    r2 = float(metrics.get("R2", 0.0))
    rmse = float(metrics.get("RMSE", 0.0))

    rel_uncertainty = pred_std / max(predicted, 1e-9)
    rel_rmse = rmse / max(last_price, 1e-9)

    if rel_uncertainty > 0.05:
        reasons.append("Incertidumbre relativa alta (std/pred)")
    if r2 < 0.2:
        reasons.append("R² bajo: ajuste pobre del modelo")
    if rel_rmse > 0.03:
        reasons.append("RMSE relativo alto: volatilidad histórica elevada")

    return {
        "recommendation": recommendation,
        "predicted": float(predicted),
        "predicted_intraday": float(predicted_intraday),
        "last_price": float(last_price),
        "reasons": reasons,
    }