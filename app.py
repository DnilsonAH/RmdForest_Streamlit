import streamlit as st
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

st.set_page_config(layout="wide")

st.title("Predicción del Precio del Cobre")

st.write("""
Esta aplicación permite visualizar y predecir el precio del cobre utilizando un modelo de Machine Learning.
""")

if st.button("Ejecutar Pipeline Completo"):
    st.write("Cargando y procesando datos...")
    df = load_or_compute_data(config)
    st.write(f"Datos procesados: {len(df)} filas")
    st.dataframe(df.head())

    st.write("Entrenando modelo...")
    artifact = train_random_forest(df, config)
    st.write("Métricas de test:", artifact["metrics"])

    st.write("Generando visualizaciones...")
    fig_basic = plot_basic_stats(df, config.PLOT_PATH)
    st.pyplot(fig_basic)

    fig_sma = plot_indicators_sma(df, config.PLOT_PATH)
    st.pyplot(fig_sma)

    fig_rsi = plot_indicators_rsi(df, config.PLOT_PATH)
    st.pyplot(fig_rsi)

    st.write("Haciendo predicción...")
    pred, std = predict_next(df, artifact)
    pred_intraday = predict_intraday(df, config.HORIZON_HOURS)
    last = float(df["cu_close"].iloc[-1])
    dec = make_decision(
        pred,
        last,
        {**artifact["metrics"], "pred_std": std},
        predicted_intraday=pred_intraday,
    )

    st.write("## Resultados de la Predicción")
    st.write(f"**Predicción a {config.HORIZON_DAYS} días:**", pred)
    st.write(f"**Predicción a {config.HORIZON_HOURS} horas:**", pred_intraday)
    st.write("**Último precio:**", last)
    st.write("**Recomendación:**", dec["recommendation"])
    if dec["reasons"]:
        st.write("**Razones:**", "; ".join(dec["reasons"]))