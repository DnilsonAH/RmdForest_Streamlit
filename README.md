# Proyecto de Predicción del Precio del Cobre

## Descripción

Este proyecto utiliza un modelo de Machine Learning para predecir el precio del cobre y proporciona recomendaciones de compra/venta. La aplicación está construida con Streamlit para una fácil visualización y uso.

## Características

- **Predicción de precios:** Predice el precio del cobre a corto y largo plazo.
- **Análisis de indicadores:** Calcula y visualiza indicadores técnicos como SMA y RSI.
- **Recomendaciones:** Proporciona recomendaciones de compra/venta basadas en las predicciones.
- **Interfaz web:** Una aplicación de Streamlit para interactuar con el modelo.

## Estructura del Proyecto

```
.
├── README.md
├── analysis.py
├── app.py
├── config.py
├── data_loader.py
├── decision.py
├── model.py
├── plots/
├── prediccion_cobre.py
├── prediccion_cobre_modular.py
└── visualization.py
```

## Instalación

1.  Clona este repositorio:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    ```
2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para ejecutar la aplicación Streamlit:

```bash
streamlit run app.py
```

## Configuración

El archivo `config.py` contiene los siguientes parámetros:

- `DATA_PATH`: Ruta al archivo de datos.
- `PLOT_PATH`: Directorio para guardar las visualizaciones.
- `MODEL_PATH`: Ruta para guardar el modelo entrenado.
- `HORIZON_DAYS`: Horizonte de predicción en días.
- `HORIZON_HOURS`: Horizonte de predicción en horas.
- `TEST_SIZE`: Proporción del dataset para el conjunto de test.
- `N_ESTIMATORS`: Número de estimadores para el modelo RandomForest.
- `MAX_DEPTH`: Profundidad máxima del modelo RandomForest.
- `MIN_SAMPLES_LEAF`: Número mínimo de muestras por hoja en el modelo RandomForest.
- `MIN_SAMPLES_SPLIT`: Número mínimo de muestras para dividir un nodo en el modelo RandomForest.