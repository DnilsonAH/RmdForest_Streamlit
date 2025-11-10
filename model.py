from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def _prepare_supervised(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = df.copy()
    out["target"] = out["cu_close"].shift(-horizon_days)
    out = out.dropna().reset_index(drop=True)
    return out


def _feature_columns(df: pd.DataFrame) -> list:
    exclude = {"Date", "target", "cu_close"}
    cols = [c for c in df.columns if c not in exclude]
    # Mantener sólo numéricas
    return [c for c in cols if np.issubdtype(df[c].dtype, np.number)]


def train_random_forest(df: pd.DataFrame, config) -> Dict:
    supervised = _prepare_supervised(df, config.HORIZON_DAYS)
    feature_cols = _feature_columns(supervised)

    # Split temporal 80/20
    n = len(supervised)
    split = int(n * 0.8)
    train = supervised.iloc[:split]
    test = supervised.iloc[split:]

    X_train = train[feature_cols]
    y_train = train["target"]
    X_test = test[feature_cols]
    y_test = test["target"]

    model = RandomForestRegressor(
        n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Incertidumbre: std entre árboles
    trees_preds = np.stack([estimator.predict(X_test) for estimator in model.estimators_], axis=0)
    pred_std = float(np.std(trees_preds, axis=0).mean())

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "pred_std": pred_std},
    }
    joblib.dump(artifact, config.MODEL_PATH)
    return artifact


def load_model(model_path: str) -> Dict:
    return joblib.load(model_path)


def predict_next(df: pd.DataFrame, artifact: Dict) -> Tuple[float, float]:
    """Predice usando la última fila de features; retorna (pred, std)."""
    feature_cols = artifact["feature_cols"]
    model: RandomForestRegressor = artifact["model"]

    last_row = df.iloc[[-1]][feature_cols]
    pred = float(model.predict(last_row)[0])
    # std de árboles en la última fila
    trees_preds = np.array([est.predict(last_row)[0] for est in model.estimators_])
    std = float(np.std(trees_preds))
    return pred, std

def predict_intraday(df: pd.DataFrame, horizon_hours: int) -> float:
    """Heurística simple para predicción intradía."""
    last_close = df.iloc[-1]["cu_close"]
    # Volatilidad de los últimos 2 días
    recent_vol = df.iloc[-2:]["cu_close"].std()
    # Movimiento esperado = vol normalizada a horas
    hourly_vol_factor = horizon_hours / (24 * 2)  # Asumiendo 2 días de referencia
    expected_move = recent_vol * hourly_vol_factor
    # Predicción simple: último cierre + movimiento esperado (aleatorio)
    # Se suma o resta para simular incertidumbre
    direction = np.random.choice([-1, 1])
    return last_close + direction * expected_move