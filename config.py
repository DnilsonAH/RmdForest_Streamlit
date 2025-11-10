TICKER_COPPER = "HG=F"  # Futuros de Cobre (COMEX) en Yahoo Finance

START_DATE = "2018-01-01"
END_DATE = None  # Hasta hoy si es None

HORIZON_DAYS = 7
HORIZON_HOURS = 2  # Heurística simple con RF
MAX_ROWS = 2000

RANDOM_FOREST_N_ESTIMATORS = 200
RANDOM_STATE = 42
USE_PROPHET = False

MODEL_PATH = "rf_copper_model.joblib"
MODEL_PATH_PROPHET = "rf_copper_model_prophet.joblib"
PLOT_PATH = "plots"
DATA_CACHE_PATH = "data_cache.joblib"