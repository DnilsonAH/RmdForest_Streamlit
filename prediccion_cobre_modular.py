import sys
import os
import pandas as pd

import config
from data_loader import load_or_compute_data
from analysis import basic_stats, compare_recent_vs_historical
from visualization import (
    plot_basic_stats,
    plot_indicators_sma,
    plot_indicators_rsi,
)
from model import train_random_forest, load_model, predict_next, predict_intraday
from decision import make_decision


def menu():
    print("\n=== Predicción Cobre + Dólar (Modular) ===")
    print("1. Cargar y procesar datos")
    print("2. Análisis de datos")
    print("3. Entrenar modelo (RandomForest)")
    print("4. Generar visualizaciones")
    print("5. Hacer predicción y decisión")
    print("6. Ejecutar pipeline completo")
    print("7. Limpiar cache de datos")
    print("8. Salir")


def run():
    df = None
    artifact = None
    while True:
        menu()
        choice = input("Selecciona opción (1-8): ").strip()
        if choice == "1":
            df = load_or_compute_data(config)
            print(f"Datos procesados: {len(df)} filas")
        elif choice == "2":
            if df is None:
                df = load_or_compute_data(config)
            stats = basic_stats(df)
            comp = compare_recent_vs_historical(df)
            print("Estadísticas básicas:", stats)
            print("Reciente vs histórico:", comp)
        elif choice == "3":
            if df is None:
                df = load_or_compute_data(config)
            artifact = train_random_forest(df, config)
            print("Métricas de test:", artifact["metrics"])
        elif choice == "4":
            if df is None:
                df = load_or_compute_data(config)
            plot_basic_stats(df, config.PLOT_PATH)
            plot_indicators_sma(df, config.PLOT_PATH)
            plot_indicators_rsi(df, config.PLOT_PATH)
            print("Visualizaciones generadas en", config.PLOT_PATH)
        elif choice == "5":
            if df is None:
                df = load_or_compute_data(config)
            if artifact is None and os.path.exists(config.MODEL_PATH):
                artifact = load_model(config.MODEL_PATH)
            if artifact is None:
                artifact = train_random_forest(df, config)
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
        elif choice == "6":
            df = load_or_compute_data(config)
            stats = basic_stats(df)
            print("Estadísticas básicas:", stats)
            artifact = train_random_forest(df, config)
            print("Métricas de test:", artifact["metrics"])
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
        elif choice == "7":
            if os.path.exists(config.DATA_CACHE_PATH):
                os.remove(config.DATA_CACHE_PATH)
                print("Cache eliminada.")
            else:
                print("No hay cache para eliminar.")
        elif choice == "8":
            print("Saliendo...")
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    run()