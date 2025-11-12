import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
import os
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Predicción del Precio del Cobre",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B35;
    }
    .recommendation-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📊 Predicción del Precio del Cobre con Machine Learning</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("---")

# Información del modelo
st.sidebar.info(f"""
**Configuración del Modelo:**
- Ticker: {config.TICKER_COPPER}
- Horizonte: {config.HORIZON_DAYS} días
- N° Estimadores: {config.RANDOM_FOREST_N_ESTIMATORS}
- Datos desde: {config.START_DATE}
""")

# Opciones de visualización
st.sidebar.markdown("### 📈 Opciones de Visualización")
show_historical = st.sidebar.checkbox("Mostrar datos históricos", value=True)
show_technical = st.sidebar.checkbox("Mostrar indicadores técnicos", value=True)
show_predictions = st.sidebar.checkbox("Mostrar predicciones", value=True)
show_features = st.sidebar.checkbox("Mostrar importancia de características", value=False)

# Refresh data
if st.sidebar.button("🔄 Actualizar Datos"):
    if os.path.exists(config.DATA_CACHE_PATH):
        os.remove(config.DATA_CACHE_PATH)
    st.sidebar.success("Cache eliminado. Recargue la página.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.markdown("""
Esta aplicación utiliza **Random Forest** para predecir 
el precio del cobre basándose en:
- Indicadores técnicos (SMA, RSI)
- Volatilidad histórica
- Patrones de precio
- Volumen de transacciones
""")

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📊 Análisis Técnico", "🤖 Predicción", "📉 Métricas del Modelo", "📚 Datos"])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.markdown('<h2 class="sub-header">Dashboard General</h2>', unsafe_allow_html=True)
    
    with st.spinner("Cargando datos..."):
        df = load_or_compute_data(config)
    
    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    last_close = df["cu_close"].iloc[-1]
    prev_close = df["cu_close"].iloc[-2]
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100
    
    with col1:
        st.metric(
            label="💰 Precio Actual",
            value=f"${last_close:.2f}",
            delta=f"{change_pct:.2f}%"
        )
    
    with col2:
        mean_30 = df["cu_close"].tail(30).mean()
        st.metric(
            label="📊 Promedio 30d",
            value=f"${mean_30:.2f}",
            delta=f"{((last_close - mean_30) / mean_30 * 100):.2f}%"
        )
    
    with col3:
        vol_20 = df["cu_return"].rolling(20).std().iloc[-1]
        st.metric(
            label="📈 Volatilidad 20d",
            value=f"{vol_20*100:.2f}%"
        )
    
    with col4:
        max_52w = df["cu_close"].tail(252).max()
        st.metric(
            label="🔼 Máximo 52s",
            value=f"${max_52w:.2f}"
        )
    
    with col5:
        min_52w = df["cu_close"].tail(252).min()
        st.metric(
            label="🔽 Mínimo 52s",
            value=f"${min_52w:.2f}"
        )
    
    st.markdown("---")
    
    # Gráfico principal interactivo
    if show_historical:
        st.markdown("### 📈 Evolución del Precio del Cobre")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Precio de Cierre', 'Volumen de Transacciones'),
            row_heights=[0.7, 0.3]
        )
        
        # Precio
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["cu_close"],
                name="Precio Cierre",
                line=dict(color="#FF6B35", width=2),
                fill='tonexty',
                fillcolor='rgba(255, 107, 53, 0.1)'
            ),
            row=1, col=1
        )
        
        # SMA
        if "cu_sma_10" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["cu_sma_10"],
                    name="SMA 10",
                    line=dict(color="#004E89", width=1, dash="dash")
                ),
                row=1, col=1
            )
        
        # Volumen
        colors = ['red' if row['cu_close'] < row['cu_open'] else 'green' 
                  for idx, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["cu_volume"],
                name="Volumen",
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volumen", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas básicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Estadísticas Descriptivas")
        stats = basic_stats(df)
        stats_df = pd.DataFrame({
            "Métrica": ["Precio Actual", "Precio Promedio", "Volatilidad 20d", "Promedio 30d"],
            "Valor": [
                f"${stats['last_close']:.2f}",
                f"${stats['mean_close']:.2f}",
                f"{stats['volatility_20']*100:.2f}%",
                f"${stats['recent_mean_30']:.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📉 Comparación Reciente vs Histórico")
        comparison = compare_recent_vs_historical(df)
        comp_df = pd.DataFrame({
            "Período": ["Reciente (30d)", "Histórico"],
            "Precio Promedio": [
                f"${comparison['recent_mean']:.2f}",
                f"${comparison['historical_mean']:.2f}"
            ],
            "Volatilidad": [
                f"{comparison['recent_vol']*100:.2f}%",
                f"{comparison['historical_vol']*100:.2f}%"
            ]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ==================== TAB 2: ANÁLISIS TÉCNICO ====================
with tab2:
    st.markdown('<h2 class="sub-header">Análisis Técnico Avanzado</h2>', unsafe_allow_html=True)
    
    if show_technical:
        # Gráfico de candlestick
        st.markdown("### 🕯️ Gráfico de Velas (Últimos 90 días)")
        
        df_recent = df.tail(90)
        
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df_recent['Date'],
            open=df_recent['cu_open'],
            high=df_recent['cu_high'],
            low=df_recent['cu_low'],
            close=df_recent['cu_close'],
            name='Precio'
        )])
        
        # Añadir SMAs
        if "cu_sma_5" in df_recent.columns:
            fig_candle.add_trace(go.Scatter(
                x=df_recent['Date'],
                y=df_recent['cu_sma_5'],
                name='SMA 5',
                line=dict(color='orange', width=1)
            ))
        
        if "cu_sma_10" in df_recent.columns:
            fig_candle.add_trace(go.Scatter(
                x=df_recent['Date'],
                y=df_recent['cu_sma_10'],
                name='SMA 10',
                line=dict(color='blue', width=1)
            ))
        
        fig_candle.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Indicadores técnicos
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 RSI (Relative Strength Index)")
            
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(go.Scatter(
                x=df_recent['Date'],
                y=df_recent['cu_rsi_14'],
                name='RSI 14',
                line=dict(color='purple', width=2)
            ))
            
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Sobrecompra (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                             annotation_text="Sobreventa (30)")
            
            fig_rsi.update_layout(
                height=300,
                yaxis_title="RSI",
                xaxis_title="Fecha",
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Interpretación del RSI
            current_rsi = df_recent['cu_rsi_14'].iloc[-1]
            if current_rsi > 70:
                st.warning(f"⚠️ RSI actual: {current_rsi:.2f} - Zona de SOBRECOMPRA")
            elif current_rsi < 30:
                st.success(f"✅ RSI actual: {current_rsi:.2f} - Zona de SOBREVENTA")
            else:
                st.info(f"ℹ️ RSI actual: {current_rsi:.2f} - Zona NEUTRAL")
        
        with col2:
            st.markdown("### 📈 Volatilidad Histórica")
            
            df_recent['volatility_30'] = df['cu_return'].rolling(30).std() * 100
            
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=df_recent['Date'],
                y=df_recent['volatility_30'],
                name='Volatilidad 30d',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
            
            fig_vol.update_layout(
                height=300,
                yaxis_title="Volatilidad (%)",
                xaxis_title="Fecha",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
            current_vol = df_recent['volatility_30'].iloc[-1]
            mean_vol = df_recent['volatility_30'].mean()
            
            if current_vol > mean_vol * 1.2:
                st.warning(f"⚠️ Volatilidad actual ({current_vol:.2f}%) está por encima del promedio")
            else:
                st.info(f"ℹ️ Volatilidad actual: {current_vol:.2f}%")
        
        # Distribución de retornos
        st.markdown("### 📊 Distribución de Retornos Diarios")
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=df['cu_return'] * 100,
            nbinsx=50,
            name='Retornos',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig_dist.update_layout(
            height=400,
            xaxis_title="Retorno Diario (%)",
            yaxis_title="Frecuencia",
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Media de Retornos", f"{df['cu_return'].mean()*100:.4f}%")
        with col2:
            st.metric("Desviación Estándar", f"{df['cu_return'].std()*100:.2f}%")
        with col3:
            st.metric("Skewness", f"{df['cu_return'].skew():.2f}")

# ==================== TAB 3: PREDICCIÓN ====================
with tab3:
    st.markdown('<h2 class="sub-header">Predicción con Random Forest</h2>', unsafe_allow_html=True)
    
    if show_predictions:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🤖 Entrenar y Predecir")
            
            if st.button("🚀 Ejecutar Predicción Completa", type="primary", use_container_width=True):
                with st.spinner("Entrenando modelo Random Forest..."):
                    # Entrenar modelo
                    artifact = train_random_forest(df, config)
                    
                    # Guardar en session state
                    st.session_state['artifact'] = artifact
                    st.session_state['df'] = df
                    
                    st.success("✅ Modelo entrenado exitosamente!")
        
        with col2:
            st.markdown("### ⚙️ Parámetros")
            st.info(f"""
            - **Horizonte**: {config.HORIZON_DAYS} días
            - **Estimadores**: {config.RANDOM_FOREST_N_ESTIMATORS}
            - **Random State**: {config.RANDOM_STATE}
            """)
        
        # Si hay un modelo entrenado
        if 'artifact' in st.session_state:
            artifact = st.session_state['artifact']
            df = st.session_state['df']
            
            st.markdown("---")
            
            # Realizar predicción
            pred, std = predict_next(df, artifact)
            pred_intraday = predict_intraday(df, config.HORIZON_HOURS)
            last = float(df["cu_close"].iloc[-1])
            
            # Decisión
            dec = make_decision(
                pred,
                last,
                {**artifact["metrics"], "pred_std": std},
                predicted_intraday=pred_intraday,
            )
            
            # Mostrar resultados
            st.markdown("### 🎯 Resultados de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"💰 Predicción ({config.HORIZON_DAYS} días)",
                    value=f"${pred:.2f}",
                    delta=f"{((pred - last) / last * 100):.2f}%"
                )
            
            with col2:
                st.metric(
                    label=f"⚡ Predicción ({config.HORIZON_HOURS} horas)",
                    value=f"${pred_intraday:.2f}",
                    delta=f"{((pred_intraday - last) / last * 100):.2f}%"
                )
            
            with col3:
                st.metric(
                    label="📊 Precio Actual",
                    value=f"${last:.2f}"
                )
            
            # Recomendación
            st.markdown("### 💡 Recomendación de Trading")
            
            if dec["recommendation"] == "COMPRAR":
                st.markdown(f'<div class="recommendation-buy">🟢 {dec["recommendation"]}</div>', 
                           unsafe_allow_html=True)
            elif dec["recommendation"] == "VENDER":
                st.markdown(f'<div class="recommendation-sell">🔴 {dec["recommendation"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="recommendation-hold">🟡 {dec["recommendation"]}</div>', 
                           unsafe_allow_html=True)
            
            if dec["reasons"]:
                st.markdown("#### ⚠️ Advertencias y Consideraciones:")
                for reason in dec["reasons"]:
                    st.warning(f"• {reason}")
            
            # Gráfico de predicción
            st.markdown("### 📈 Visualización de Predicción")
            
            # Crear fechas futuras
            last_date = df["Date"].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=config.HORIZON_DAYS,
                freq='D'
            )
            
            # Gráfico
            fig_pred = go.Figure()
            
            # Histórico reciente (últimos 30 días)
            df_recent = df.tail(30)
            fig_pred.add_trace(go.Scatter(
                x=df_recent['Date'],
                y=df_recent['cu_close'],
                name='Histórico',
                line=dict(color='blue', width=2)
            ))
            
            # Predicción
            fig_pred.add_trace(go.Scatter(
                x=[last_date, future_dates[-1]],
                y=[last, pred],
                name='Predicción',
                line=dict(color='red', width=2, dash='dash'),
                mode='lines+markers'
            ))
            
            # Banda de confianza
            fig_pred.add_trace(go.Scatter(
                x=[last_date, future_dates[-1], future_dates[-1], last_date],
                y=[last - std, pred - std, pred + std, last + std],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                name='Banda de confianza',
                showlegend=True
            ))
            
            fig_pred.update_layout(
                height=400,
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Análisis de sensibilidad
            st.markdown("### 🔍 Análisis de Sensibilidad")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Escenarios de Precio")
                
                scenarios = pd.DataFrame({
                    "Escenario": ["Pesimista", "Base", "Optimista"],
                    "Precio": [
                        f"${pred - 2*std:.2f}",
                        f"${pred:.2f}",
                        f"${pred + 2*std:.2f}"
                    ],
                    "Cambio %": [
                        f"{((pred - 2*std - last) / last * 100):.2f}%",
                        f"{((pred - last) / last * 100):.2f}%",
                        f"{((pred + 2*std - last) / last * 100):.2f}%"
                    ]
                })
                
                st.dataframe(scenarios, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Probabilidad de Movimiento")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=((pred - last) / last * 100),
                    title={'text': "Cambio Esperado (%)"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-10, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-10, -2], 'color': "lightcoral"},
                            {'range': [-2, 2], 'color': "lightyellow"},
                            {'range': [2, 10], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

# ==================== TAB 4: MÉTRICAS DEL MODELO ====================
with tab4:
    st.markdown('<h2 class="sub-header">Métricas y Rendimiento del Modelo</h2>', unsafe_allow_html=True)
    
    if 'artifact' in st.session_state:
        artifact = st.session_state['artifact']
        metrics = artifact['metrics']
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 R² Score", f"{metrics['R2']:.4f}")
        with col2:
            st.metric("📉 MAE", f"${metrics['MAE']:.2f}")
        with col3:
            st.metric("📈 RMSE", f"${metrics['RMSE']:.2f}")
        with col4:
            st.metric("🎯 Pred STD", f"${metrics['pred_std']:.2f}")
        
        st.markdown("---")
        
        # Interpretación de métricas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Interpretación de Métricas")
            
            r2 = metrics['R2']
            if r2 > 0.7:
                st.success(f"✅ R² = {r2:.4f}: Excelente ajuste del modelo")
            elif r2 > 0.4:
                st.info(f"ℹ️ R² = {r2:.4f}: Ajuste moderado")
            else:
                st.warning(f"⚠️ R² = {r2:.4f}: Ajuste bajo, considere mejorar el modelo")
            
            rmse_pct = (metrics['RMSE'] / df['cu_close'].mean()) * 100
            st.info(f"RMSE representa un {rmse_pct:.2f}% del precio promedio")
            
            mae_pct = (metrics['MAE'] / df['cu_close'].mean()) * 100
            st.info(f"MAE representa un {mae_pct:.2f}% del precio promedio")
        
        with col2:
            st.markdown("### 🎯 Calidad de la Predicción")
            
            quality_score = (r2 * 0.5 + (1 - mae_pct/100) * 0.3 + (1 - rmse_pct/100) * 0.2) * 100
            
            fig_quality = go.Figure(go.Indicator(
                mode="gauge+number",
                value=quality_score,
                title={'text': "Score de Calidad"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            
            fig_quality.update_layout(height=300)
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Importancia de características
        if show_features and 'model' in artifact:
            st.markdown("### 🔍 Importancia de Características")
            
            model = artifact['model']
            feature_cols = artifact['feature_cols']
            importances = model.feature_importances_
            
            # Crear DataFrame
            feat_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Top 15 características
            top_features = feat_importance.head(15)
            
            fig_importance = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Características Más Importantes',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig_importance.update_layout(
                height=500,
                template='plotly_white',
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Tabla de importancias
            st.markdown("#### 📋 Tabla de Importancias")
            st.dataframe(
                feat_importance.style.background_gradient(cmap='YlOrRd', subset=['Importance']),
                use_container_width=True,
                height=400
            )
    else:
        st.info("👆 Por favor, entrena el modelo primero en la pestaña de Predicción")

# ==================== TAB 5: DATOS ====================
with tab5:
    st.markdown('<h2 class="sub-header">Exploración de Datos</h2>', unsafe_allow_html=True)
    
    # Información general
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total de Registros", len(df))
    with col2:
        st.metric("📅 Fecha Inicio", df['Date'].min().strftime('%Y-%m-%d'))
    with col3:
        st.metric("📅 Fecha Fin", df['Date'].max().strftime('%Y-%m-%d'))
    
    st.markdown("---")
    
    # Filtros
    st.markdown("### 🔍 Filtrar Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Rango de fechas",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    with col2:
        num_rows = st.slider("Número de filas a mostrar", 10, 500, 50)
    
    # Filtrar datos
    if len(date_range) == 2:
        mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
        df_filtered = df.loc[mask]
    else:
        df_filtered = df
    
    # Mostrar datos
    st.markdown("### 📋 Datos Completos")
    st.dataframe(df_filtered.tail(num_rows), use_container_width=True, height=400)
    
    # Descargar datos
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos como CSV",
        data=csv,
        file_name=f"copper_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    
    # Estadísticas descriptivas
    st.markdown("### 📊 Estadísticas Descriptivas")
    st.dataframe(df_filtered.describe(), use_container_width=True)
    
    # Correlaciones
    st.markdown("### 🔗 Matriz de Correlación")
    
    # Seleccionar columnas numéricas
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calcular correlación
    corr_matrix = df_filtered[numeric_cols].corr()
    
    # Crear heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Matriz de Correlación de Variables'
    )
    
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 <b>Aplicación de Predicción del Precio del Cobre</b></p>
    <p>Desarrollado con Streamlit, Scikit-learn y Plotly</p>
    <p>© 2025 - Random Forest Copper Predictor</p>
</div>
""", unsafe_allow_html=True)