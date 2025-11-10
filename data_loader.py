import os
from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import joblib


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def download_data(config) -> pd.DataFrame:
    """
    Descarga datos de Cobre (HG=F),
    y renombra columnas con prefijo cu_.
    """
    if config.END_DATE is None:
        end = None
    else:
        end = config.END_DATE

    # Descarga cobre
    cu = yf.download(
        config.TICKER_COPPER,
        start=config.START_DATE,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    cu = _ensure_datetime_index(cu)

    if cu is None or cu.empty:
        raise RuntimeError("Descarga de Cobre sin datos válidos.")

    if isinstance(cu.columns, pd.MultiIndex):
        cu.columns = cu.columns.get_level_values(0)

    cu = cu.rename(columns={c: f"cu_{c.lower().replace(' ', '_')}" for c in cu.columns})

    df = cu.reset_index().rename(columns={"index": "Date"})

    # Limitar filas si se requiere
    if config.MAX_ROWS and len(df) > config.MAX_ROWS:
        df = df.tail(config.MAX_ROWS).copy()

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features del cobre:
    - Retornos, SMA_5/10, Volatilidad_5, Rango, Lags 1..3, RSI_14
    """
    out = df.copy()

    # Cobre
    out["cu_return"] = out["cu_close"].pct_change()
    out["cu_sma_5"] = out["cu_close"].rolling(5).mean()
    out["cu_sma_10"] = out["cu_close"].rolling(10).mean()
    out["cu_volatility_5"] = out["cu_return"].rolling(5).std()
    out["cu_range"] = (out["cu_high"] - out["cu_low"]) / (out["cu_open"] + 1e-9)
    for lag in (1, 2, 3):
        out[f"cu_lag_close_{lag}"] = out["cu_close"].shift(lag)
        out[f"cu_lag_vol_{lag}"] = out["cu_volume"].shift(lag)
    out["cu_rsi_14"] = _rsi(out["cu_close"], 14)

    out = out.dropna().reset_index(drop=True)
    return out


def load_or_compute_data(config) -> pd.DataFrame:
    """Usa cache si existe; sino descarga y procesa datos."""
    if os.path.exists(config.DATA_CACHE_PATH):
        try:
            cached = joblib.load(config.DATA_CACHE_PATH)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached
        except Exception:
            pass

    raw = download_data(config)
    processed = compute_technical_features(raw)
    joblib.dump(processed, config.DATA_CACHE_PATH)
    return processed