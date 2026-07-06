import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración inicial de la página web
st.set_page_config(page_title="AVM Vizcaya - Stacking Híbrido", layout="centered")

# --- CABECERA VISUAL ---
st.title("Sistema de Tasación Automatizada (AVM) - Vizcaya")
st.write("Introduzca las características del inmueble para proyectar su valor comercial exacto a través de tu arquitectura de Stacking Híbrido.")
st.markdown("---")

# --- MEDIANAS DE ENTRADA OPTIMIZADAS (SÓLO BARRIOS DEL TOP 20) ---
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
    "Negurigane-Peruri": 46.0, "Txorierri-Ondiz-Udondo": 23.0, "Mungia": 42.0, "Muskiz": 30.0,
    "Ortuella": 19.0, "Plentzia": 36.0, "Azeta - Abatxolo": 22.0, "Buenavista": 23.0,
    "Casco Viejo - Muelle": 22.0, "La Florida": 27.0, "Repelaga": 40.0, "Capitán Mendizabal - La Sardinera": 27.0,
    "Kabiezes": 25.0, "La Magdalena": 69.0, "La Txitxarra - Murrieta - Parke Santurtzi": 21.0,
    "Larrea - San Juan de Dios - Peñota": 32.0, "Las Viñas": 20.0, "Mamariga": 16.0, "Villar - San Juan": 14.0,
    "Zona Centro": 17.0, "Asilo - Rebonza - Urbinaga": 18.0, "Centro - Albiz - Markonzaga": 19.0,
    "La Paz - El Carmen - Anunciación": 79.24, "La Unión - Vista Alegre": 16.0, "Sopelana": 24.0,
    "Valle de Trapaga-Trapagaran": 19.0, "Zalla": 18.0
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

# --- DICCIONARIOS DE CODIFICACIÓN EN TARGET ENCODING GEOGRÁFICO ---
target_muni = {
    "Amorebieta-Echano": 12.7575, "Astrabudua": 12.5144, "Barakaldo": 12.5300, "Basauri": 12.5741, "Berango": 13.5906,
    "Bilbao": 12.8114, "Erandio": 12.7052, "Galdakao": 12.6368, "Getxo": 13.4333, "Leioa": 13.0733,
    "Mungia": 13.2278, "Muskiz": 12.6844, "Ortuella": 12.5107, "Plentzia": 13.3567, "Portugalete": 12.4843,
    "Santurtzi": 12.4982, "Sestao": 12.1967, "Sopelana": 13.1736, "Valle de Trapaga-Trapagaran": 12.6008, "Zalla": 12.6923
}

target_dist = {
    "Amorebieta-Echano": 12.7575, "Astrabudua": 12.5144, "Arteagabeitia - Retuerto - Kareaga": 12.5262,
    "Bagatza - S. Vicente": 12.5128, "Burtzeña": 12.7624, "Centro": 12.5324, "Cruces": 12.5419,
    "Gorostiza - El Regato": 12.6497, "Lasesarre": 12.5610, "Lutxana - Llano": 12.5222, "Rontegui-Pormetxeta": 12.6522,
    "Centro - Ariz - Uribarri": 12.5744, "Kalero - Basozelai": 12.6320, "Pozokoetxe": 12.7589, "San Miguel": 12.8766,
    "Urbi": 12.7582, "Berango": 13.5906, "Abando - Albia": 13.3447, "Basurto - Zorroza": 12.6991,
    "Begoña - Santutxu": 12.4886, "Casco Viejo": 12.6563, "Deusto": 12.8880, "Ibaiondo": 12.4429,
    "Indautxu": 13.2667, "Otxarkoaga - Txurdinaga": 12.5998, "Rekalde": 12.6835, "San Adrián - La Peña": 12.6885,
    "San Ignacio": 12.8584, "Uribarri": 12.7551, "Erandio": 12.7052, "Galdakao": 12.6368, "Algorta": 13.2005,
    "Las Arenas": 13.4493, "Neguri": 13.5566, "Sta. María de Getxo": 13.6219, "Aldekoena-Artatzagana-Sarriena": 12.8565,
    "Artatza-Pinueta-Pinosolo": 13.1395, "Centro Urbano-Hirigunea": 12.8427, "Lamiako-Txopoeta": 12.8079,
    "Negurigane-Peruri": 13.0627, "Txorierri-Ondiz-Udondo": 12.8317, "Mungia": 13.2278, "Muskiz": 12.6844,
    "Ortuella": 12.5107, "Plentzia": 13.3567, "Azeta - Abatxolo": 12.4711, "Buenavista": 12.6247,
    "Casco Viejo - Muelle": 12.7163, "Centro": 12.5324, "La Florida": 12.6958, "Repelaga": 12.8433,
    "Capitán Mendizabal - La Sardinera": 12.7631, "Kabiezes": 12.7143, "La Magdalena": 12.8809,
    "La Txitxarra - Murrieta - Parke Santurtzi": 12.7914, "Larrea - San Juan de Dios - Peñota": 12.8668,
    "Las Viñas": 12.4469, "Mamariga": 12.6506, "Villar - San Juan": 12.5527, "Zona Centro": 12.7457,
    "Asilo - Rebonza - Urbinaga": 12.6622, "Centro - Albiz - Markonzaga": 12.4799, "La Paz - El Carmen - Anunciación": 12.6718,
    "La Unión - Vista Alegre": 12.1639, "Sopelana": 13.1736, "Valle de Trapaga-Trapagaran": 12.6008, "Zalla": 12.6923
}

target_neigh = {
    "Amorebieta-Echano": 12.7575, "Astrabudua": 12.5144, "Arteagabeitia - Retuerto - Kareaga": 12.5262,
    "Bagatza - S. Vicente": 12.5128, "Burtzeña": 12.7624, "Centro": 12.7050, "Cruces": 12.5419,
    "Gorostiza - El Regato": 12.6497, "Lasesarre": 12.5610, "Lutxana - Llano": 12.5222, "Rontegui-Pormetxeta": 12.6522,
    "Centro - Ariz - Uribarri": 12.5744, "Kalero - Basozelai": 12.6320, "Pozokoetxe": 12.7589, "San Miguel": 12.8766,
    "Urbi": 12.7582, "Berango": 13.5906, "Abandoibarra-Guggenheim": 13.4713, "Albia": 13.1780,
    "Ensanche-Moyua": 13.6220, "Plaza Circular": 13.1027, "Zabalburu-Diputación": 12.9848, "Altamira": 12.7787,
    "Basurtu": 13.0649, "Masustegui": 12.6580, "Olabeaga": 12.7478, "Zorrotza": 12.4962, "Begoña": 12.8294,
    "Bolueta": 12.6027, "Santutxu": 12.4563, "Casco Viejo": 12.6563, "Arangoiti": 12.7386, "La Ribera-Ibarrekolanda": 12.7976,
    "San Pedro de Deusto": 13.0273, "Atxuri": 12.5777, "Bilbao la Vieja": 12.6238, "Iturralde": 12.7229,
    "Miribilla": 12.8301, "San Francisco": 12.3544, "Solokoetxe": 12.6053, "Zabala": 12.4703, "Alhondiga": 12.9024,
    "Campuzano": 13.0100, "Sabino Arana-Jesuitas": 13.2693, "Zona Indautxu": 13.2018, "Otxarkoaga - Txurdinaga": 12.5998,
    "Ametzola": 12.8970, "Artatzu-Larraskitu": 12.7797, "Irala": 12.5944, "Rekalde Centro": 12.5708,
    "Uretamendi-Betolaza-Peñaskal": 12.6508, "La Peña": 12.6831, "San Adrián": 12.8289, "San Ignacio": 12.8584,
    "Arabella": 12.7327, "Campo Volantín-Castaños": 12.9635, "Ciudad Jardín": 12.9001, "Mirador de Bilbao-Maurice Ravel": 12.7122,
    "Uribarri": 12.6049, "Zurbaran": 12.8389, "Erandio": 12.7052, "Galdakao": 12.6368, "Alango": 12.8405,
    "Centro": 12.7050, "Polígono Rojo-Aldapa": 12.8724, "Portu Zaharra": 12.8906, "Sarrikobaso": 12.8231,
    "Villamonte": 12.8739, "Zona Usategui - Trinitarios": 13.2469, "Las Arenas Centro": 13.6182, "Muelle de las Arenas": 13.1280,
    "Romo": 12.8325, "Santa Ana": 12.9158, "Villa Plentzia": 12.9822, "Neguri": 13.5566, "Sta. María de Getxo": 13.6219,
    "Aldekoena-Artatzagana-Sarriena": 12.8565, "Artatza-Pinueta-Pinosolo": 13.1395, "Centro Urbano-Hirigunea": 12.8427,
    "Lamiako-Txopoeta": 12.8079, "Negurigane-Peruri": 13.0627, "Txorierri-Ondiz-Udondo": 12.8317, "Mungia": 13.2278,
    "Muskiz": 12.6844, "Ortuella": 12.5107, "Plentzia": 13.3567, "Azeta - Abatxolo": 12.4711, "Buenavista": 12.6247,
    "Casco Viejo - Muelle": 12.7163, "Centro": 12.7050, "La Florida": 12.6958, "Repelaga": 12.8433,
    "Capitán Mendizabal - La Sardinera": 12.7631, "Kabiezes": 12.7143, "La Magdalena": 12.8809,
    "La Txitxarra - Murrieta - Parke Santurtzi": 12.7914, "Larrea - San Juan de Dios - Peñota": 12.8668,
    "Las Viñas": 12.4469, "Mamariga": 12.6506, "Villar - San Juan": 12.5527, "Zona Centro": 12.7457,
    "Asilo - Rebonza - Urbinaga": 12.6622, "Centro - Albiz - Markonzaga": 12.4799, "La Paz - El Carmen - Anunciación": 12.6718,
    "La Unión - Vista Alegre": 12.1639, "Sopelana": 13.1736, "Valle de Trapaga-Trapagaran": 12.6008, "Zalla": 12.6923
}

# --- ELEMENTOS INTERACTIVOS (SELECTORES FUERA DE CUALQUIER FORMULARIO) ---
col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Tipología del inmueble", ["flat", "chalet", "duplex", "countryHouse", "penthouse"])
    
    # Lógica condicional estricta para 'subTypology' (actualización inmediata)
    if property_type == "flat":
        sub_typology = "flat"
        st.text_input("Subtipología asignada automáticamente", value="flat", disabled=True)
    elif property_type == "duplex":
        sub_typology = "duplex"
        st.text_input("Subtipología asignada automáticamente", value="duplex", disabled=True)
    elif property_type == "countryHouse":
        sub_typology = "countryHouse"
        st.text_input("Subtipología asignada automáticamente", value="countryHouse", disabled=True)
    elif property_type == "penthouse":
        sub_typology = "penthouse"
        st.text_input("Subtipología asignada automáticamente", value="penthouse", disabled=True)
    elif property_type == "chalet":
        sub_typology = st.selectbox("Subtipología de chalet", ["terracedHouse", "independantHouse", "semidetachedHouse"])

    # Selector de municipio
    municipio_opciones = ["Bilbao", "Barakaldo", "Getxo", "Portugalete", "Santurtzi", "Sestao", "Leioa", "Basauri", "Mungia", "Plentzia", "Berango", "Amorebieta-Echano", "Galdakao", "Ortuella", "Sopelana", "Valle de Trapaga-Trapagaran", "Astrabudua", "Erandio", "Zalla", "Muskiz"]
    municipality = st.selectbox("Municipio", municipio_opciones)

    # Mapeo condicional de distritos según el municipio (actualización inmediata)
    distritos_por_muni = {
        "Amorebieta-Echano": ["Amorebieta-Echano"], "Astrabudua": ["Astrabudua"],
        "Barakaldo": ["Centro", "Rontegui-Pormetxeta", "Bagatza - S. Vicente", "Cruces", "Burtzeña", "Lasesarre", "Lutxana - Llano", "Arteagabeitia - Retuerto - Kareaga", "Gorostiza - El Regato"],
        "Basauri": ["San Miguel", "Kalero - Basozelai", "Centro - Ariz - Uribarri", "Pozokoetxe", "Urbi"], "Berango": ["Berango"],
        "Bilbao": ["Rekalde", "Ibaiondo", "Deusto", "Abando - Albia", "Indautxu", "Basurto - Zorroza", "Uribarri", "Begoña - Santutxu", "Casco Viejo", "San Ignacio", "San Adrián - La Peña", "Otxarkoaga - Txurdinaga"],
        "Erandio": ["Erandio"], "Galdakao": ["Galdakao"], "Getxo": ["Neguri", "Las Arenas", "Sta. María de Getxo", "Algorta"],
        "Leioa": ["Artatza-Pinueta-Pinosolo", "Lamiako-Txopoeta", "Centro Urbano-Hirigunea", "Txorierri-Ondiz-Udondo", "Negurigane-Peruri", "Aldekoena-Artatzagana-Sarriena"],
        "Mungia": ["Mungia"], "Muskiz": ["Muskiz"], "Ortuella": ["Ortuella"], "Plentzia": ["Plentzia"],
        "Portugalete": ["Centro", "La Florida", "Casco Viejo - Muelle", "Azeta - Abatxolo", "Buenavista", "Repelaga"],
        "Santurtzi": ["Kabiezes", "Mamariga", "Las Viñas", "La Txitxarra - Murrieta - Parke Santurtzi", "La Magdalena", "Villar - San Juan", "Zona Centro", "Capitán Mendizabal - La Sardinera", "Larrea - San Juan de Dios - Peñota"],
        "Sestao": ["La Unión - Vista Alegre", "Centro - Albiz - Markonzaga", "La Paz - El Carmen - Anunciación", "Asilo - Rebonza - Urbinaga"],
        "Sopelana": ["Sopelana"], "Valle de Trapaga-Trapagaran": ["Valle de Trapaga-Trapagaran"], "Zalla": ["Zalla"]
    }
    district = st.selectbox("Distrito", distritos_por_muni[municipality])

    # Mapeo condicional de barrios según el distrito (actualización inmediata)
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
    neighborhood = st.selectbox("Barrio específico", barrios_por_distrito[district])

    # Rango de superficie útil
    size = st.number_input("Superficie útil (m²)", min_value=30, max_value=600, value=85, step=5)
    
    # Forzar 'Exterior' a True (1) en chalet y countryHouse
    if property_type in ["chalet", "countryHouse"]:
        exterior_int = 1
        st.text_input("Orientación", value="Exterior (Fijo por tipología)", disabled=True)
    else:
        exterior_input = st.selectbox("¿Es exterior?", ["Sí", "No"])
        exterior_int = 1 if exterior_input == "Sí" else 0

with col2:
    rooms = st.number_input("Número de habitaciones", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Número de baños", min_value=1, max_value=7, value=2, step=1)
    
    # Lógica condicional de 'floor'
    if property_type in ["chalet", "countryHouse"]:
        floor_final = 0
        st.text_input("Planta / Piso", value="0 (Bajo automático)", disabled=True)
    else:
        floor_final = st.number_input("Planta / Piso", min_value=1, max_value=15, value=2, step=1)

    # Lógica condicional de 'hasLift'
    if property_type in ["chalet", "countryHouse"]:
        has_lift_int = 0
        st.text_input("¿Tiene ascensor?", value="No (Asignado automáticamente)", disabled=True)
    else:
        lift_input = st.selectbox("¿Tiene ascensor?", ["Sí", "No"])
        has_lift_int = 1 if lift_input == "Sí" else 0

    # Estado de conservación
    status = st.selectbox("Estado de conservación", ["good", "newdevelopment", "renew"])

    # Lógica condicional de garaje y precio del garaje
    have_parking_input = st.selectbox("¿Dispone de garaje?", ["No", "Sí"])
    if have_parking_input == "No":
        have_parking_int = 0
        is_parking_included_int = 0
        st.text_input("¿Garaje incluido en el precio?", value="No aplica", disabled=True)
        parking_price = 0.0
    else:
        have_parking_int = 1
        parking_inc_input = st.selectbox("¿El garaje está incluido en el precio?", ["Sí", "No"])
        if parking_inc_input == "Sí":
            is_parking_included_int = 1
            parking_price = 0.0
        else:
            is_parking_included_int = 0
            parking_price = 30000.0

# Botón estándar fuera de formulario para ejecutar la inferencia
st.markdown("<br>", unsafe_allow_html=True)
botón_tasar = st.button("Calcular tasación comercial", use_container_width=True)

# --- 2. MOTOR DE INFERENCIA DE DOS NIVELES (STACKING) ---
if botón_tasar:
    with st.spinner("Procesando simulación a través de los ecosistemas del Stacking..."):
        try:
            # Cargamos de forma síncrona los componentes (.joblib)
            p_lin = joblib.load('pipeline_lineal.joblib')
            p_rf = joblib.load('pipeline_rf.joblib')
            p_lgb = joblib.load('pipeline_lgb.joblib')
            p_xgb = joblib.load('pipeline_xgb.joblib')
            meta_urb = joblib.load('meta_urbano.joblib')
            meta_prem = joblib.load('meta_premium.joblib')

            # Extracción de medianas con fallback de seguridad
            num_photos_default = medianas_photos.get(neighborhood, 22.0)
            distance_default = medianas_distance.get(neighborhood, 1500.0)

            # --- TRADUCCIÓN INVISIBLE DE TEXTOS A SUS RESPECTIVOS CÓDIGOS TARGET ENCODING ---
            muni_encoded = target_muni.get(municipality, 12.8114)
            dist_encoded = target_dist.get(district, 12.5324)
            neigh_encoded = target_neigh.get(neighborhood, 12.7050)

            # --- RECONSTRUCCIÓN DEL VECTOR DE COLUMNAS DEL MODELO (ONE-HOT ENCODING MANUAL) ---
            registro_modelo = {
                'numPhotos': int(num_photos_default),
                'floor': int(floor_final),
                
                # 'propertyType' encoding (Manual One-Hot)
                'propertyType_flat': 1 if property_type == "flat" else 0,
                'propertyType_chalet': 1 if property_type == "chalet" else 0,
                'propertyType_duplex': 1 if property_type == "duplex" else 0,
                'propertyType_countryHouse': 1 if property_type == "countryHouse" else 0,
                'propertyType_penthouse': 1 if property_type == "penthouse" else 0,
                
                'exterior': int(exterior_int),
                'rooms': int(rooms),
                'bathrooms': int(bathrooms),
                
                # Enviamos las codificaciones numéricas
                'municipality': float(muni_encoded),
                'district': float(dist_encoded),
                'neighborhood': float(neigh_encoded),
                
                'showAddress': 0, # Fijo en False (0)
                'distance': int(distance_default),
                'hasVideo': 0, # Fijo en False (0)
                
                # 'status' encoding (Manual One-Hot)
                'status_newdevelopment': 1 if status == "newdevelopment" else 0,
                'status_good': 1 if status == "good" else 0,
                'status_renew': 1 if status == "renew" else 0,
                
                'hasLift': int(has_lift_int),
                'hasPlan': 0,
                'has3DTour': 0,
                'has360': 0,
                'hasStaging': 0,
                
                # 'highlight' encoding (Fijo en standar)
                'highlight_Standar': 1,
                'highlight_Destacado': 0,
                'highlight_Top': 0,
                'highlight_Top+': 0,
                
                'haveParkingSpace': int(have_parking_int),
                'isParkingIncluded': int(is_parking_included_int),
                'parkingSpacePrice': float(parking_price),
                
                # 'subTypology' encoding (Manual One-Hot)
                'subTypology_flat': 1 if sub_typology == "flat" else 0,
                'subTypology_independantHouse': 1 if sub_typology == "independantHouse" else 0,
                'subTypology_duplex': 1 if sub_typology == "duplex" else 0,
                'subTypology_terracedHouse': 1 if sub_typology == "terracedHouse" else 0,
                'subTypology_countryHouse': 1 if sub_typology == "countryHouse" else 0,
                'subTypology_penthouse': 1 if sub_typology == "penthouse" else 0,
                'subTypology_semidetachedHouse': 1 if sub_typology == "semidetachedHouse" else 0,
                
                # Estimadores residuales fijos (0)
                'floor_estimated': 0,
                'hasLift_estimated': 0,
                'exterior_estimated': 0,
                
                # Ingeniería de variables exacta aplicando np.log1p
                'size_log': float(np.log1p(size))
            }

            # Asegura el orden exacto de columnas que espera el modelo
            expected_features = list(p_lin.feature_names_in_)
            df_entrada = pd.DataFrame([registro_modelo])[expected_features]

            # --- NIVEL 0: Predicciones preliminares logarítmicas ---
            pred_lin = p_lin.predict(df_entrada)
            pred_rf = p_rf.predict(df_entrada)
            pred_lgb = p_lgb.predict(df_entrada)
            pred_xgb = p_xgb.predict(df_entrada)

            # --- NIVEL 1: Enrutamiento asimétrico dinámico ---
            if property_type != 'chalet':
                # Mercado urbano general
                df_meta = pd.DataFrame({'ElasticNet': pred_lin, 'LightGBM': pred_lgb, 'XGBoost': pred_xgb})
                precio_log = meta_urb.predict(df_meta)[0]
                ecosistema_texto = "Ecosistema urbano general (Pisos / Áticos / Dúplex / Casas de campo)"
            else:
                # Mercado residencial premium
                df_meta = pd.DataFrame({'ElasticNet': pred_lin, 'RandomForest': pred_rf, 'LightGBM': pred_lgb})
                precio_log = meta_prem.predict(df_meta)[0]
                ecosistema_texto = "Ecosistema residencial premium (Chalets)"

            # --- INGENIERÍA INVERSA: Destransformación monetaria exponencial (euros) ---
            precio_euros = np.expm1(precio_log)

            # --- 3. PRESENTACIÓN DE RESULTADOS ---
            st.markdown("### Resultado del análisis del Stacking")
            precio_formateado = f"{precio_euros:,.2f} €".replace(',', 'X').replace('.', ',').replace('X', '.')
            st.success(f"**Tarifa estimada de tasación comercial:** {precio_formateado}")
            
 	    # Cuadro informativo metodológico
            st.info(f"**Ruta de enrutamiento activada:** {ecosistema_texto}\n\n"
                    f"**Coeficiente de confianza del modelo (R²):** 92,06%")
        
        except Exception as e:
            st.error(f"Error al procesar la predicción. Detalle técnico: {e}")