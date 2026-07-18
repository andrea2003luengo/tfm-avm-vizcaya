import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración inicial de la página web
st.set_page_config(
    page_title="AVM Vizcaya - Stacking Híbrido Asimétrico", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CABECERA VISUAL ---
st.title("Sistema de Tasación Automatizada (AVM) - Vizcaya")
st.write("Introduzca las características del inmueble para promediar su valor comercial exacto a través de la arquitectura de Stacking Híbrido.")
st.markdown("---")

# --- CARGA DETERMINISTA DE ARTIFACTOS ---
@st.cache_resource
def cargar_pipeline_produccion():
    """Carga de forma síncrona los modelos base, meta-modelos y codificadores."""
    return {
        "pipeline_lineal": joblib.load('pipeline_lineal.joblib'),
        "pipeline_rf": joblib.load('pipeline_rf.joblib'),
        "pipeline_lgb": joblib.load('pipeline_lgb.joblib'),
        "pipeline_xgb": joblib.load('pipeline_xgb.joblib'),
        "meta_urbano": joblib.load('meta_urbano.joblib'),
        "meta_premium": joblib.load('meta_premium.joblib'),
        "ohe_encoder": joblib.load('ohe.joblib'),
        "te_encoder": joblib.load('te.joblib')
    }

try:
    avm_hub = cargar_pipeline_produccion()
except Exception as e:
    st.error(f"Error crítico al inicializar los artefactos de Machine Learning: {e}")
    st.stop()

# --- MEDIANAS DE ENTRADA OPTIMIZADAS ---
medianas_photos = {
    "Amorebieta-Echano": 15.0, "Astrabudua": 19.0, "Arteagabeitia - Retuerto - Kareaga": 23.0,
    "Bagatza - S. Vicente": 28.0, "Burtzeña": 24.0, "Centro": 22.0, "Cruces": 27.0,
    "Gorostiza - El Regato": 8.0, "Lasesarre": 27.0, "Lutxana - Llano": 28.0, "Rontegui-Pormetxeta": 21.0,
    "Centro - Ariz - Uribarri": 25.0, "Kalero - Basozelai": 22.0, "Pozokoetxe": 23.0, "San Miguel": 50.0,
    "Urbi": 0.0, "Berango": 30.0, "Ensanche-Moyua": 23.0, "Zabalburu-Diputación": 26.0,
    "Plaza Circular": 31.0, "Abandoibarra-Guggenheim": 26.0, "Albia": 33.0, "Zorrotza": 15.0,
    "Basurtu": 27.0, "Olabeaga": 13.0, "Masustegui": 22.0, "Altamira": 8.0, "Santutxu": 25.0,
    "Bolueta": 18.0, "Begoña": 34.0, "Casco Viejo": 21.0, "La Ribera-Ibarrekolanda": 22.0,
    "San Pedro de Deusto": 29.0, "Arangoiti": 22.0, "Zabala": 28.0, "Solokoetxe": 32.0,
    "Atxuri": 16.0, "San Francisco": 16.0, "Iturralde": 27.0, "Miribilla": 23.0,
    "Bilbao la Vieja": 21.0, "Sabino Arana-Jesuitas": 25.0, "Zona Indautxu": 26.0, "Alhondiga": 38.0,
    "Campuzano": 12.0, "Otxarkoaga - Txurdinaga": 26.0, "Irala": 23.0, "Artatzu-Larraskitu": 18.0,
    "Rekalde Centro": 21.0, "Ametzola": 25.0, "Uretamendi-Betolaza-Peñaskal": 15.0, "La Peña": 24.0,
    "San Adrián": 36.0, "San Ignacio": 26.0, "Arabella": 23.0, "Ciudad Jardín": 38.0,
    "Uribarri": 24.0, "Campo Volantín-Castaños": 27.0, "Mirador de Bilbao-Maurice Ravel": 16.0, "Zurbaran": 33.0,
    "Erandio": 26.0, "Galdakao": 16.0, "Polígono Rojo-Aldapa": 30.0, "Villamonte": 26.0,
    "Zona Usategui - Trinitarios": 30.0, "Sarrikobaso": 28.0, "Alango": 28.0, "Portu Zaharra": 74.0,
    "Las Arenas Centro": 30.0, "Muelle de las Arenas": 30.0, "Romo": 24.0, "Villa Plentzia": 37.0,
    "Santa Ana": 25.0, "Neguri": 24.0, "Sta. María de Getxo": 34.0, "Aldekoena-Artatzagana-Sarriena": 20.0,
    "Artatza-Pinueta-Pinosolo": 28.0, "Centro Urbano-Hirigunea": 30.0, "Lamiako-Txopoeta": 16.0,
    "Negurigane-Peruri": 46.0, "Txorierri-OOndiz-Udondo": 23.0, "Mungia": 42.0, "Muskiz": 30.0,
    "Ortuella": 19.0, "Plentzia": 36.0, "Azeta - Abatxolo": 22.0, "Buenavista": 23.0,
    "Casco Viejo - Muelle": 22.0, "La Florida": 27.0, "Repelaga": 40.0, "Capitán Mendizabal - La Sardinera": 27.0,
    "Kabiezes": 25.0, "La Magdalena": 69.0, "La Txitxarra - Murrieta - Parke Santurtzi": 10588.0,
    "Larrea - San Juan de Dios - Peñota": 10328.0, "Las Viñas": 11146.0, "Mamariga": 11603.0, "Villar - San Juan": 12178.0,
    "Zona Centro": 10931.0, "Asilo - Rebonza - Urbinaga": 7169.0, "Centro - Albiz - Markonzaga": 7707.0,
    "La Paz - El Carmen - Anunciación": 7924.0, "La Unión - Vista Alegre": 7547.0, "Sopelana": 13812.0,
    "Valle de Trapaga-Trapagaran": 9407.0, "Zalla": 16758.0
}

medianas_distance = {
    "Amorebieta-Echano": 16901.0, "Astrabudua": 7017.0, "Arteagabeitia - Retuerto - Kareaga": 5683.0,
    "Bagatza - S. Vicente": 6162.0, "Burtzeña": 3882.0, "Centro": 9152.0, "Cruces": 4477.0,
    "Gorostiza - El Regato": 6830.0, "Lasesarre": 5877.0, "Lutxana - Llano": 4652.0, "Rontegui-Pormetxeta": 5434.0,
    "Centro - Ariz - Uribarri": 5112.0, "Kalero - Basozelai": 4539.0, "Pozokoetxe": 4656.0, "San Miguel": 5929.0,
    "Urbi": 5386.0, "Berango": 11977.0, "Ensanche-Moyua": 245.0, "Zabalburu-Diputación": 561.0,
    "Plaza Circular": 578.0, "Abandoibarra-Guggenheim": 492.0, "Albia": 727.0, "Zorrotza": 3279.0,
    "Basurtu": 1291.0, "Olabeaga": 1950.0, "Masustegui": 1909.0, "Altamira": 2093.0, "Santutxu": 2111.0,
    "Bolueta": 2584.0, "Begoña": 1699.0, "Casco Viejo": 1108.0, "La Ribera-Ibarrekolanda": 2439.0,
    "San Pedro de Deusto": 1324.0, "Arangoiti": 1723.0, "Zabala": 1002.0, "Solokoetxe": 1580.0,
    "Atxuri": 1659.0, "San Francisco": 916.0, "Iturralde": 1542.0, "Miribilla": 1437.0,
    "Bilbao la Vieja": 1367.0, "Sabino Arana-Jesuitas": 801.0, "Zona Indautxu": 583.0, "Alhondiga": 452.0,
    "Campuzano": 332.0, "Otxarkoaga - Txurdinaga": 2796.0, "Irala": 1032.0, "Artatzu-Larraskitu": 1770.0,
    "Rekalde Centro": 1496.0, "Ametzola": 912.0, "Uretamendi-Betolaza-Peñaskal": 2015.0, "La Peña": 2626.0,
    "San Adrián": 1574.0, "San Ignacio": 3062.0, "Arabella": 1931.0, "Ciudad Jardín": 1094.0,
    "Uribarri": 1135.0, "Campo Volantín-Castaños": 892.0, "Mirador de Bilbao-Maurice Ravel": 1596.0, "Zurbaran": 1835.0,
    "Erandio": 5511.0, "Galdakao": 7922.0, "Polígono Rojo-Aldapa": 11848.0, "Villamonte": 11334.0,
    "Zona Usategui - Trinitarios": 11924.0, "Sarrikobaso": 11792.0, "Alango": 11104.0, "Portu Zaharra": 11617.0,
    "Las Arenas Centro": 9560.0, "Muelle de las Arenas": 9400.0, "Romo": 9290.0, "Villa Plentzia": 9786.0,
    "Santa Ana": 9040.0, "Neguri": 10400.0, "Sta. María de Getxo": 12854.0, "Aldekoena-Artatzagana-Sarriena": 8789.0,
    "Artatza-Pinueta-Pinosolo": 9203.0, "Centro Urbano-Hirigunea": 8250.0, "Lamiako-Txopoeta": 8720.0,
    "Negurigane-Peruri": 9292.0, "Txorierri-Ondiz-Udondo": 7922.0, "Mungia": 12278.0, "Muskiz": 16144.0,
    "Ortuella": 10986.0, "Plentzia": 15680.0, "Azeta - Abatxolo": 8910.0, "Buenavista": 10002.0,
    "Casco Viejo - Muelle": 9440.0, "La Florida": 9511.0, "Repelaga": 8956.0, "Capitán Mendizabal - La Sardinera": 11200.0,
    "Kabiezes": 10603.0, "La Magdalena": 12602.0, "La Txitxarra - Murrieta - Parke Santurtzi": 10588.0,
    "Larrea - San Juan de Dios - Peñota": 10328.0, "Las Viñas": 11146.0, "Mamariga": 11603.0, "Villar - San Juan": 12178.0,
    "Zona Centro": 10931.0, "Asilo - Rebonza - Urbinaga": 7169.0, "Centro - Albiz - Markonzaga": 7707.0,
    "La Paz - El Carmen - Anunciación": 7924.0, "La Unión - Vista Alegre": 7547.0, "Sopelana": 13812.0,
    "Valle de Trapaga-Trapagaran": 9407.0, "Zalla": 16758.0
}

# --- MAPEO GEOGRÁFICO DE JERARQUÍA COMPLETA ---
barrios_por_distrito = {
    "Amorebieta-Echano": ["Amorebieta-Echano"], "Astrabudua": ["Astrabudua"],
    "Arteagabeitia - Retuerto - Kareaga": ["Arteagabeitia - Retuerto - Kareaga"], "Bagatza - S. Vicente": ["Bagatza - S. Vicente"],
    "Burtzeña": ["Burtzeña"], "Centro": ["Centro"], "Cruces": ["Cruces"], "Gorostiza - El Regato": ["Gorostiza - El Regato"],
    "Lasesarre": ["Lasesarre"], "Lutxana - Llano": ["Lutxana - Llano"], "Rontegui-Pormetxeta": ["Rontegui-Pormetxeta"],
    "Centro - Ariz - Uribarri": ["Centro - Ariz - Uribarri"], "Kalero - Basozelai": ["Kalero - Basozelai"],
    "Pozokoetxe": ["Pozokoetxe"], "San Miguel": ["San Miguel"], "Urbi": ["Urbi"], "Berango": ["Berango"],
    "Abando - Albia": ["Ensanche-Moyua", "Zabalburu-Diputación", "Plaza Circular", "Abandoibarra-Guggenheim", "Albia"],
    "Basurto - Zorroza": ["Zorrotza", "Basurtu", "Olabeaga", "Masustegui", "Altamira"], "Begoña - Santutxu": ["Santutxu", "Bolueta", "Begoña"],
    "Casco Viejo": ["Casco Viejo"], "Deusto": ["La Ribera-Ibarrekolanda", "San Pedro de Deusto", "Arangoiti"],
    "Ibaiondo": ["Zabala", "Solokoetxe", "Atxuri", "San Francisco", "Iturralde", "Miribilla", "Bilbao la Vieja"],
    "Indautxu": ["Sabino Arana-Jesuitas", "Zona Indautxu", "Alhondiga", "Campuzano"], "Otxarkoaga - Txurdinaga": ["Otxarkoaga - Txurdinaga"],
    "Rekalde": ["Irala", "Artatzu-Larraskitu", "Rekalde Centro", "Ametzola", "Uretamendi-Betolaza-Peñaskal"],
    "San Adrián - La Peña": ["La Peña", "San Adrián"], "San Ignacio": ["San Ignacio"],
    "Uribarri": ["Arabella", "Ciudad Jardín", "Uribarri", "Campo Volantín-Castaños", "Mirador de Bilbao-Maurice Ravel", "Zurbaran"],
    "Erandio": ["Erandio"], "Galdakao": ["Galdakao"],
    "Algorta": ["Polígono Rojo-Aldapa", "Villamonte", "Centro", "Zona Usategui - Trinitarios", "Sarrikobaso", "Alango", "Portu Zaharra"],
    "Las Arenas": ["Las Arenas Centro", "Muelle de las Arenas", "Romo", "Villa Plentzia", "Santa Ana"],
    "Neguri": ["Neguri"], "Sta. María de Getxo": ["Sta. María de Getxo"], "Aldekoena-Artatzagana-Sarriena": ["Aldekoena-Artatzagana-Sarriena"],
    "Artatza-Pinueta-Pinosolo": ["Artatza-Pinueta-Pinosolo"], "Centro Urbano-Hirigunea": ["Centro Urbano-Hirigunea"],
    "Lamiako-Txopoeta": ["Lamiako-Txopoeta"], "Negurigane-Peruri": ["Negurigane-Peruri"], "Txorierri-Ondiz-Udondo": ["Txorierri-Ondiz-Udondo"],
    "Mungia": ["Mungia"], "Muskiz": ["Muskiz"], "Ortuella": ["Ortuella"], "Plentzia": ["Plentzia"],
    "Azeta - Abatxolo": ["Azeta - Abatxolo"], "Buenavista": ["Buenavista"], "Casco Viejo - Muelle": ["Casco Viejo - Muelle"],
    "La Florida": ["La Florida"], "Repelaga": ["Repelaga"], "Capitán Mendizabal - La Sardinera": ["Capitán Mendizabal - La Sardinera"],
    "Kabiezes": ["Kabiezes"], "La Magdalena": ["La Magdalena"], "La Txitxarra - Murrieta - Parke Santurtzi": ["La Txitxarra - Murrieta - Parke Santurtzi"],
    "Larrea - San Juan de Dios - Peñota": ["Larrea - San Juan de Dios - Peñota"], "Las Viñas": ["Las Viñas"],
    "Mamariga": ["Mamariga"], "Villar - San Juan": ["Villar - San Juan"], "Zona Centro": ["Zona Centro"],
    "Asilo - Rebonza - Urbinaga": ["Asilo - Rebonza - Urbinaga"], "Centro - Albiz - Markonzaga": ["Centro - Albiz - Markonzaga"],
    "La Paz - El Carmen - Anunciación": ["La Paz - El Carmen - Anunciación"], "La Unión - Vista Alegre": ["La Unión - Vista Alegre"],
    "Sopelana": ["Sopelana"], "Valle de Trapaga-Trapagaran": ["Valle de Trapaga-Trapagaran"], "Zalla": ["Zalla"]
}

distritos_por_muni = {
    "Amorebieta-Echano": ["Amorebieta-Echano"], "Astrabudua": ["Astrabudua"],
    "Barakaldo": ["Centro", "Rontegui-Pormetxeta", "Bagatza - S. Vicente", "Cruces", "Burtzeña", "Lasesarre", "Lutxana - Llano", "Arteagabeitia - Retuerto - Kareaga", "Gorostiza - El Regato"],
    "Basauri": ["San Miguel", "Kalero - Basozelai", "Centro - Ariz - Uribarri", "Pozokoetxe", "Urbi"], "Berango": ["Berango"],
    "Bilbao": ["Rekalde", "Ibaiondo", "Deusto", "Abando - Albia", "Indautxu", "Basurto - Zorroza", "Uribarri", "Begoña - Santutxu", "Casco Viejo", "San Ignacio", "San Adrián - La Peña", "Otxarkoaga - Txurdinaga"],
    "Erandio": ["Erandio"], "Galdakao": ["Galdakao"], "Getxo": ["Neguri", "Las Arenas", "Sta. María de Getxo", "Algorta"],
    "Leioa": ["Artatza-Pinueta-Pinosolo", "Lamiako-Txopoeta", "Centro Urbano-Hirigunea", "Txorierri-Ondiz-Udondo", "Negurigane-Peruri", "Aldekoena-Artatzagana-Sarriena"],
    "Mungia": ["Mungia"], "Muskiz": ["Muskiz"], "Ortuella": ["Ortuella"], "Plentzia": ["Plentzia"],
    "Portugalete": ["Centro", "La Florida", "Casco Viejo - Muelle", "Azeta - Abatxolo", "Buenavista", "Repelaga"],
    "Santurtzi": ["Capitán Mendizabal - La Sardinera", "Kabiezes", "La Magdalena", "La Txitxarra - Murrieta - Parke Santurtzi", "Larrea - San Juan de Dios - Peñota", "Las Viñas", "Mamariga", "Villar - San Juan", "Zona Centro"],
    "Sestao": ["La Unión - Vista Alegre", "Centro - Albiz - Markonzaga", "La Paz - El Carmen - Anunciación", "Asilo - Rebonza - Urbinaga"],
    "Sopelana": ["Sopelana"], "Valle de Trapaga-Trapagaran": ["Valle de Trapaga-Trapagaran"], "Zalla": ["Zalla"]
}

# --- 1. SECCIÓN DE INPUTS REACTIVOS (FUERA DEL FORMULARIO PARA RESPUESTA EN TIEMPO REAL) ---
col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Tipología del inmueble", ["flat", "chalet", "duplex", "countryHouse", "penthouse"])
    
    # Lógica condicional e interactiva instantánea de subtipologías[cite: 32]
    if property_type == "flat":
        sub_typology = "flat"
        st.text_input("Subtipología asignada", value="flat", disabled=True)
    elif property_type == "duplex":
        sub_typology = "duplex"
        st.text_input("Subtipología asignada", value="duplex", disabled=True)
    elif property_type == "countryHouse":
        sub_typology = "countryHouse"
        st.text_input("Subtipología asignada", value="countryHouse", disabled=True)
    elif property_type == "penthouse":
        sub_typology = "penthouse"
        st.text_input("Subtipología asignada", value="penthouse", disabled=True)
    elif property_type == "chalet":
        sub_typology = st.selectbox("Subtipología de chalet", ["terracedHouse", "independantHouse", "semidetachedHouse"])

    # Selector de ubicación jerárquico reactivo
    municipality = st.selectbox("Municipio", list(distritos_por_muni.keys()))
    district = st.selectbox("Distrito", distritos_por_muni[municipality])
    neighborhood = st.selectbox("Barrio específico", barrios_por_distrito[district])

    size = st.number_input("Superficie útil (m²)", min_value=30, max_value=600, value=85, step=5)
    
    # Lógica interactiva instantánea para orientación de chalets/casas de campo[cite: 32]
    if property_type in ["chalet", "countryHouse"]:
        exterior_input = "Sí"
        st.text_input("Orientación", value="Exterior (Fijo por tipología)", disabled=True)
    else:
        exterior_input = st.selectbox("¿Es exterior?", ["Sí", "No"])
    exterior_bool = True if exterior_input == "Sí" else False

with col2:
    rooms = st.number_input("Número de habitaciones", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Número de baños", min_value=1, max_value=7, value=2, step=1)
    
    # Lógica interactiva instantánea de plantas[cite: 32]
    if property_type in ["chalet", "countryHouse"]:
        floor_final = 0
        st.text_input("Planta / Piso", value="0 (Bajo automático)", disabled=True)
    else:
        floor_final = st.number_input("Planta / Piso", min_value=0, max_value=15, value=2, step=1)

    # Lógica interactiva instantánea de ascensor[cite: 32]
    if property_type in ["chalet", "countryHouse"]:
        lift_input = "No"
        st.text_input("¿Tiene ascensor?", value="No (Asignado automáticamente)", disabled=True)
    else:
        lift_input = st.selectbox("¿Tiene ascensor?", ["Sí", "No"])
    has_lift_bool = True if lift_input == "Sí" else False

    status = st.selectbox("Estado de conservación", ["good", "newdevelopment", "renew"])

    # Lógica interactiva instantánea de garajes[cite: 32]
    have_parking_input = st.selectbox("¿Dispone de garaje?", ["No", "Sí"])
    if have_parking_input == "No":
        have_parking_bool = False
        is_parking_included_bool = False
        st.text_input("¿Garaje incluido en el precio?", value="No aplica", disabled=True)
        parking_price = 0.0
    else:
        have_parking_bool = True
        parking_inc_input = st.selectbox("¿El garaje está incluido en el precio?", ["Sí", "No"])
        if parking_inc_input == "Sí":
            is_parking_included_bool = True
            parking_price = 0.0
        else:
            is_parking_included_bool = False
            parking_price = 30000.0

# --- BOTÓN INDEPENDIENTE DE EJECUCIÓN ---
st.markdown(" ")
botón_tasar = st.button("Calcular el precio de mercado", use_container_width=True)

# --- 2. MOTOR DE INFERENCIA DE DOS NIVELES (STACKING ASIMÉTRICO) ---
if botón_tasar:
    with st.spinner("Procesando simulación a través de los ecosistemas del Stacking..."):
        try:
            # 1. Construcción del DataFrame original en Crudo (Formato df1)[cite: 31]
            raw_entry = pd.DataFrame([{
                'floor': int(floor_final),
                'rooms': int(rooms),
                'bathrooms': int(bathrooms),
                'hasLift': bool(has_lift_bool),
                'exterior': bool(exterior_bool),
                'haveParkingSpace': bool(have_parking_bool),
                'isParkingIncluded': bool(is_parking_included_bool),
                'parkingSpacePrice': float(parking_price),
                'municipality': str(municipality),
                'district': str(district),
                'neighborhood': str(neighborhood),
                'propertyType': str(property_type),
                'subTypology': str(sub_typology),
                'status': str(status),
                'highlight': 'Standard',
                'numPhotos': int(medianas_photos.get(neighborhood, 22.0)),
                'showAddress': True,
                'distance': float(medianas_distance.get(neighborhood, 1500.0)),
                'hasVideo': False,
                'hasPlan': False,
                'has3DTour': False,
                'has360': False,
                'hasStaging': False,
                'floor_estimated': int(0),
                'hasLift_estimated': int(0),
                'exterior_estimated': int(0),
                'size': float(size)
            }])

            # 2. Conversión determinista a enteros (0/1) para variables lógicas[cite: 31]
            cols_bool_pipeline = ['hasLift', 'exterior', 'haveParkingSpace', 'isParkingIncluded', 
                                  'showAddress', 'hasVideo', 'hasPlan', 'has3DTour', 'has360', 'hasStaging']
            for c in cols_bool_pipeline:
                raw_entry[c] = raw_entry[c].astype(int)

            # 3. Transformación síncrona por encoders originales (One-Hot & Target Encoding)[cite: 31]
            encoded_entry = avm_hub["ohe_encoder"].transform(raw_entry)
            encoded_entry = avm_hub["te_encoder"].transform(encoded_entry)

            # 4. Transformación Logarítmica de Superficie (np.log1p)[cite: 31]
            encoded_entry['size_log'] = np.log1p(encoded_entry['size'])
            encoded_entry = encoded_entry.drop(columns=['size'])

            # 5. Alineación estructural exacta con las Características de Entrenamiento[cite: 31]
            features_entrenamiento = list(avm_hub["pipeline_lineal"].feature_names_in_)
            df_produccion = encoded_entry[features_entrenamiento]

            # --- Inferencia Síncrona Nivel 0 ---[cite: 31]
            oof_elastic = avm_hub["pipeline_lineal"].predict(df_produccion)
            oof_rf = avm_hub["pipeline_rf"].predict(df_produccion)
            oof_lgb = avm_hub["pipeline_lgb"].predict(df_produccion)
            oof_xgb = avm_hub["pipeline_xgb"].predict(df_produccion)

            df_meta_entrada = pd.DataFrame({
                'ElasticNet': oof_elastic,
                'RandomForest': oof_rf,
                'LightGBM': oof_lgb,
                'XGBoost': oof_xgb
            })

            # --- Enrutamiento Asimétrico Nivel 1 ---[cite: 31]
            TIPOS_PREMIUM_GANADOR = ["chalet", "countryHouse"]

            if property_type not in TIPOS_PREMIUM_GANADOR:
                prediccion_log = avm_hub["meta_urbano"].predict(df_meta_entrada)[0]
            else:
                prediccion_log = avm_hub["meta_premium"].predict(df_meta_entrada)[0]

            # --- Reversión Exponencial Inversa (np.expm1) ---[cite: 31]
            precio_final_euros = np.expm1(prediccion_log)
            precio_formateado = f"{precio_final_euros:,.2f} €".replace(',', 'X').replace('.', ',').replace('X', '.')

            # --- 3. PRESENTACIÓN DEL RESULTADO LIMPIO (MINIMALISTA Y FORMATEADO) ---
            st.markdown("### Resultado del análisis de Tasación")
            st.success(f"Valor predictivo comercial: {precio_formateado}")

        except Exception as e:
            st.error(f"Error al procesar la predicción. Detalle técnico: {e}")