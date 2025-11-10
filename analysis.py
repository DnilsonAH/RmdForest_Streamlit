import numpy as np
import pandas as pd


def basic_stats(df: pd.DataFrame) -> dict:
    last_close = float(df["cu_close"].iloc[-1])
    mean_close = float(df["cu_close"].mean())
    median_vol = float(df["cu_volume"].median()) if "cu_volume" in df.columns else np.nan
    vol20 = float(df["cu_return"].rolling(20).std().iloc[-1])
    recent_mean_30 = float(df["cu_close"].tail(30).mean())
    hist_mean = mean_close
    return {
        "last_close": last_close,
        "mean_close": mean_close,
        "median_volume": median_vol,
        "volatility_20": vol20,
        "recent_mean_30": recent_mean_30,
        "historical_mean": hist_mean,
    }


def compare_recent_vs_historical(df: pd.DataFrame) -> dict:
    recent = df.tail(30)
    return {
        "recent_mean": float(recent["cu_close"].mean()),
        "recent_vol": float(recent["cu_return"].std()),
        "historical_mean": float(df["cu_close"].mean()),
        "historical_vol": float(df["cu_return"].std()),
    }