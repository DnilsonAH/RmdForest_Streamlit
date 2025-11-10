import os
import config
from data_loader import load_or_compute_data
from analysis import basic_stats
from visualization import plot_basic_stats, plot_indicators_sma, plot_indicators_rsi
from model import train_random_forest, predict_next, predict_intraday
from decision import make_decision


def main():
    df = load_or_compute_data(config)
    print("Filas procesadas:", len(df))
    print("Stats:", basic_stats(df))

    artifact = train_random_forest(df, config)
    print("Métricas:", artifact["metrics"])

    plot_basic_stats(df, config.PLOT_PATH)
    plot_indicators_sma(df, config.PLOT_PATH)
    plot_indicators_rsi(df, config.PLOT_PATH)

    pred, std = predict_next(df, artifact)
    pred_intraday = predict_intraday(df, config.HORIZON_HOURS)
    last = float(df["cu_close"].iloc[-1])
    dec = make_decision(
        pred,
        last,
        {**artifact["metrics"], "pred_std": std},
        predicted_intraday=pred_intraday,
    )
    print("Predicción próxima ventana:", pred)
    print(f"Predicción en {config.HORIZON_HOURS} horas:", pred_intraday)
    print("Último precio:", last)
    print("Recomendación:", dec["recommendation"])
    if dec["reasons"]:
        print("Razones:", "; ".join(dec["reasons"]))


if __name__ == "__main__":
    main()