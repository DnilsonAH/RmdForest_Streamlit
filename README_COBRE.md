# Predicción de Cobre (Sistema Modular)

Proyecto en Python para descargar datos reales de Cobre (HG=F en yfinance), generar features técnicos, entrenar un modelo de predicción (Random Forest por defecto), visualizar resultados y emitir una recomendación simple (COMPRAR / VENDER / MANTENER) con explicación.

## Características Clave

- Datos 100% reales: la descarga falla con error si no hay datos válidos (sin datos sintéticos).
- Flujo por menú interactivo (1–8) para ejecutar por bloques y reducir carga.
- Features técnicos: retornos, medias móviles (SMA), volatilidad, rango, lags, RSI.
- Modelos: RandomForestRegressor (no lineal, robusto).
- Métricas claras: MAE, RMSE, R² sobre el tramo más reciente (split temporal 80/20).
- Visualizaciones: precio del cobre, SMAs y RSI; se guardan en `plots/`.
- Decisión transparente basada en umbrales y calidad del modelo, con razones impresas.
- Cache de datos procesados y persistencia del modelo para reproducibilidad.

## Estructura del Proyecto

```
├── analysis.py                   # Estadísticas básicas y comparación reciente vs histórico
├── config.py                     # Parámetros globales (tickers, fechas, horizontes, paths, etc.)
├── data_loader.py                # Descarga y procesamiento de datos; cache
├── decision.py                   # Regla de decisión y UI de ventana (tkinter), impresión en consola
├── model.py                      # Entrenamiento y predicción (RandomForest)
├── prediccion_cobre_modular.py   # Script principal con menú y flujo modular
├── prediccion_cobre.py           # Versión monolítica de referencia
├── visualization.py              # Gráficas de indicadores y estadísticas
├── plots/                        # Salida de imágenes PNG (se crean al graficar)
├── rf_copper_model.joblib        # Modelo Random Forest entrenado + columnas de features
└── data_cache.joblib             # Cache de datos procesados
```

## Requisitos

- Python 3.13 (recomendado)
- Paquetes:
  - yfinance, pandas, numpy, scikit-learn, matplotlib, joblib

Instalación rápida:

```
pip install yfinance pandas numpy scikit-learn matplotlib joblib
```

## Configuración

Edita `config.py` para ajustar el comportamiento:

- `TICKER_COPPER = "HG=F"`: símbolo de cobre (futuros COMEX).
- `START_DATE`, `END_DATE`: rango temporal (END_DATE=None usa hasta hoy).
- `HORIZON_DAYS = 7`: días hacia adelante a predecir (define el target `cu_Close.shift(-HORIZON_DAYS)`).
- `MAX_ROWS = 2000`: número máximo de filas para reducir carga.
- `RANDOM_FOREST_N_ESTIMATORS = 200`: número de árboles en el bosque.
- `RANDOM_STATE = 42`: semilla para reproducibilidad.
- `MODEL_PATH = "rf_copper_model.joblib"`: ruta de persistencia del modelo entrenado.
- `PLOT_PATH = "plots"`: carpeta de salida de gráficos.
- `DATA_CACHE_PATH = "data_cache.joblib"`: cache de datos procesados.

## Cómo Ejecutar

- Modo modular (recomendado):

```
py prediccion_cobre_modular.py
```

Sigue el menú interactivo para correr cada bloque.

- Modo monolítico de referencia:

```
py prediccion_cobre.py
```

## Flujo por Menú (prediccion_cobre_modular.py)

1. Cargar y procesar datos
   - Descarga cobre (HG=F) y crea features técnicos.
2. Análisis de datos
   - Estadísticas básicas (último cierre, promedio, mediana volumen, volatilidad 20 días).
   - Comparación 30 días recientes vs. histórico (medias y volatilidad).
3. Entrenar modelo
   - Entrena RandomForestRegressor.
   - Muestra métricas en el conjunto de test (último 20%).
4. Generar visualizaciones
   - Gráficas de precio del cobre, SMAs (5 y 10) y RSI 14.
   - Se guardan en `plots/basic_stats.png`, `plots/indicators_sma.png`, `plots/indicators_rsi.png`.
5. Hacer predicción y decisión
   - Predicción a `HORIZON_DAYS` y desviación estándar (incertidumbre) del modelo.
   - Emite recomendación y razones.
6. Ejecutar pipeline completo
   - Corre 1→2→3→4→5 en una secuencia.
7. Limpiar cache de datos
   - Elimina `data_cache.joblib` para recalcular datos procesados.
8. Salir

## Datos y Procesamiento

- Descarga: `data_loader.download_data()` baja cobre y dólar con columnas estándar (Date, Open, High, Low, Close, Volume) y las unifica.
- Garantía de datos reales: si la descarga falla o viene vacía, se lanza excepción con mensaje claro.
- Features (`data_loader.compute_technical_features`):
  - Cobre: `Return`, `SMA_5`, `SMA_10`, `Volatility_5`, `Range`, lags de `Close` y `Volume`, `RSI_14`.
  - Dólar: `Return`, `SMA_5`, `SMA_10`, lags, `RSI_14`.
  - Cruces: correlación móvil 20 días `corr_cu_usd_20`.

## Modelos

- Random Forest (por defecto):
  - `train_random_forest`: crea `target = cu_Close.shift(-HORIZON_DAYS)`, excluye `[Date, target, cu_Close, usd_Close]` de `feature_cols`, hace split temporal 80/20.
  - Persiste `(model, feature_cols, metrics)` en `MODEL_PATH`.
  - Predicción futura usa las últimas features y estima incertidumbre como `std` entre árboles.
- Prophet (opcional):
  - Referencia opcional; el uso con exógenas requiere variantes específicas. Por defecto `USE_PROPHET=False`.

## Métricas

- MAE, RMSE, R² en el conjunto de test (tramo más reciente).
- La decisión usa además la incertidumbre relativa (`pred_std / predicted_price`) y el error relativo (`RMSE / last_price`).

## Visualizaciones

- `plots/basic_stats.png`: precio de cierre del cobre y dólar (ejes gemelos).
- `plots/indicators_sma.png`: Close + `SMA_5` y `SMA_10` del cobre.
- `plots/indicators_rsi.png`: `RSI_14` del cobre con líneas 30/70.
- `plots/corr_cu_usd_20.png`: correlación móvil cobre↔dólar.
- En la versión monolítica también se puede graficar `price_forecast.png` (histórico + test real vs. predicho).

## Decisión

- Regla simple (ajustable):
  - Si `predicted > last * 1.02` ⇒ COMPRAR
  - Si `predicted < last * 0.98` ⇒ VENDER
  - Si no ⇒ MANTENER
- Razones añadidas según:
  - Incertidumbre relativa alta (`pred_std`) ⇒ baja confianza.
  - R² bajo ⇒ ajuste pobre del modelo.
  - RMSE relativo alto ⇒ volatilidad histórica elevada.

## Cache y Persistencia

- Cache de datos (`DATA_CACHE_PATH`): permite reusar datos procesados sin recalcular.
- Modelo (`MODEL_PATH`): guarda el modelo y columnas de features para asegurar consistencia en inferencia.
- Menú opción 7 limpia la cache de datos.

## Ejemplo de Uso Rápido

```
# 1) Ejecuta el script modular
py prediccion_cobre_modular.py

# 2) En el menú, corre en orden: 1 → 2 → 3 → 4 → 5
#    (opcional: 6 ejecuta todo el pipeline de una vez)
```

## Resolución de Problemas

- Error de descarga de datos:
  - Verifica conexión a Internet y los símbolos `TICKER_COPPER` y `TICKER_USD` (prueba `^DXY` si falla `DX-Y.NYB`).
  - Ajusta `START_DATE/MAX_ROWS` si el rango es demasiado grande.
- Predicción falla por columnas:
  - Si cambiaste `compute_technical_features`, reentrena el modelo (opción 3) para alinear `feature_cols`.
- Resultados variables:
  - Aumenta `RANDOM_FOREST_N_ESTIMATORS` (ej. 200–300) y mantén `RANDOM_STATE` fijo.
- Predicción intradía:
  - Es aproximada; requiere datos intradía reales para mayor fidelidad.

## Mejoras Sugeridas

- Mostrar importancias de features (`model.feature_importances_`) y graficarlas.
- Validación cruzada temporal (TimeSeriesSplit) y búsqueda de hiperparámetros (`max_depth`, `min_samples_leaf`).
- Intervalos más fiables: bosques cuantílicos o conformal prediction.
- Features adicionales: MACD, bandas de Bollinger, correlaciones sectoriales.
- Logging y reintentos de yfinance para robustez operativa.

---

Este proyecto es una base sólida y práctica para análisis y predicción diaria del Cobre con influencia del Dólar: rápido de ejecutar, transparente en sus decisiones y fácil de extender.