import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración inicial de la página web
st.set_page_config(page_title="AVM Vizcaya - Stacking híbrido", layout="centered")

# --- CABECERA VISUAL ---
st.title("Sistema de Tasación Automatizada (AVM) - Vizcaya")
st.write("Introduzca las características del inmueble para proyectar su valor comercial exacto a través de la arquitectura de Stacking híbrido.")
st.markdown("---")

# --- 1. FORMULARIO DE ENTRADA MANUAL DE DATOS ---
with st.form("formulario_tasacion"):
    col1, col2 = st.columns(2)
    
    with col1:
        property_type = st.selectbox("Tipología del Inmueble", ["flat", "chalet", "duplex", "penthouse", "countryHouse"])
        municipality = st.selectbox("Ubicación Geográfica", ["Bilbao", "Getxo", "Barakaldo", "Portugalete", "Santurtzi", "Basauri", "Sestao", "Leioa", "Amorebieta-Echano", "Mungia"])
        size = st.number_input("Superficie Útil (m²)", min_value=15, max_value=500, value=85, step=5)
    
    with col2:
        rooms = st.number_input("Número de Habitaciones", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Número de Baños", min_value=1, max_value=5, value=2, step=1)
        
    # Botón de ejecución interno del formulario
    botón_tasar = st.form_submit_button("Calcular Valor de Mercado")

# --- 2. MOTOR DE INFERENCIA DE DOS NIVELES (STACKING) ---
if botón_tasar:
    with st.spinner("Procesando consulta a través de los ecosistemas del Ensemble..."):
        try:
            # Cargamos de forma síncrona los componentes desde la misma carpeta
            p_lin = joblib.load('pipeline_lineal.joblib')
            p_rf = joblib.load('pipeline_rf.joblib')
            p_lgb = joblib.load('pipeline_lgb.joblib')
            p_xgb = joblib.load('pipeline_xgb.joblib')
            meta_urb = joblib.load('meta_urbano.joblib')
            meta_prem = joblib.load('meta_premium.joblib')

            # Estructuramos los datos introducidos por el usuario en un DataFrame plano
            datos_usuario = {
                'size': size,
                'rooms': rooms,
                'bathrooms': bathrooms,
                'propertyType': property_type,
                'municipality': municipality
            }
            df_entrada = pd.DataFrame([datos_usuario])

            # --- NIVEL 0: Extracción de predicciones base en escala logarítmica ---
            pred_lin = p_lin.predict(df_entrada)
            pred_rf = p_rf.predict(df_entrada)
            pred_lgb = p_lgb.predict(df_entrada)
            pred_xgb = p_xgb.predict(df_entrada)

            # --- NIVEL 1: Enrutamiento Dinámico por Ecosistemas ---
            if property_type != 'chalet':
                # Mercado Urbano General (ElasticNet, LightGBM y XGBoost)
                df_meta = pd.DataFrame({'ElasticNet': pred_lin, 'LightGBM': pred_lgb, 'XGBoost': pred_xgb})
                precio_log = meta_urb.predict(df_meta)[0]
                ecosistema_texto = "Mercado Urbano General (Pisos / Áticos / Dúplex)"
            else:
                # Mercado Residencial Premium (ElasticNet, RandomForest y LightGBM)
                df_meta = pd.DataFrame({'ElasticNet': pred_lin, 'RandomForest': pred_rf, 'LightGBM': pred_lgb})
                precio_log = meta_prem.predict(df_meta)[0]
                ecosistema_texto = "Mercado Residencial Premium (Chalets)"

            # --- INGENIERÍA INVERSA: Destransformación monetaria (exp(x) - 1) ---
            precio_euros = np.expm1(precio_log)

            # --- 3. PRESENTACIÓN DE RESULTADOS ---
            st.markdown("### Resultado del Análisis del Super Ensemble")
            
            # Formateo de moneda con estándares en castellano
            precio_formateado = f"{precio_euros:,.2f} €".replace(',', 'X').replace('.', ',').replace('X', '.')
            st.success(f"**Tarifa Estimada de Tasación Comercial:** {precio_formateado}")
            
            # Cuadro informativo de control metodológico
            st.info(f"**Ruta de Enrutamiento Activada:** {ecosistema_texto}\n\n"
                    f"**Coeficiente de Confianza del Modelo (R²):** 92,26%\n\n"
                    f"**Margen de Error Relativo (MAPE Promedio en Pisos):** 10,17%")
        
        except Exception as e:
            st.error(f"Error al procesar la predicción. Asegúrate de que los 6 archivos .joblib están en la misma carpeta que este script. Detalle: {e}")