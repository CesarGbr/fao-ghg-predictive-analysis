import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from difflib import get_close_matches
from collections import Counter
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Para normalización geográfica (comentado por defecto, descomentar si se usa)
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter

# =====================================================================
# CONFIGURACIÓN GLOBAL
# =====================================================================

# === 1️⃣ Configurar rutas seguras ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

AÑO_MAX = 2025
ARCHIVO_ENTRADA = os.path.join(DOCS_DIR, 'Emisiones_Totales_S_Todos_los_Datos_(Normalizado).csv')
USAR_GEOPY = True  # Cambiar a True para usar normalización geográfica

# Archivo de salida del informe
INFORME_MD = os.path.join(RESULTS_DIR,'informe_analisis_emisiones.md')
informe = []  # Lista para acumular líneas del informe

def escribir_informe(texto, consola=True):
    """Escribe en consola y acumula en informe markdown"""
    if consola:
        print(texto)
    informe.append(texto)

# =====================================================================
# PASO 0: CARGA Y LIMPIEZA INICIAL
# =====================================================================

escribir_informe("=" * 80)
escribir_informe("ANÁLISIS Y NORMALIZACIÓN DE DATOS DE EMISIONES")
escribir_informe("=" * 80)
escribir_informe("")

escribir_informe(f"[0/7] Cargando datos desde '{ARCHIVO_ENTRADA}'...")
df = pd.read_csv(ARCHIVO_ENTRADA, sep=',')

escribir_informe(f"   ✓ Datos cargados: {len(df):,} registros")
escribir_informe(f"   ✓ Columnas: {', '.join(df.columns)}")
escribir_informe("")

# =====================================================================
# PASO 1: NORMALIZACIÓN Y LIMPIEZA
# =====================================================================

escribir_informe("[1/7] Normalizando y limpiando datos...")
escribir_informe("")

# 1.1 Filtrar años válidos
años_invalidos = df[df['Año'] > AÑO_MAX]
if len(años_invalidos) > 0:
    escribir_informe(f"   ⚠️  Eliminando {len(años_invalidos)} registros con años > {AÑO_MAX}")
    escribir_informe(f"       Años encontrados: {sorted(años_invalidos['Año'].unique())}")

df = df[df['Año'] <= AÑO_MAX].copy()
escribir_informe(f"   ✓ Años válidos: {df['Año'].min()} - {df['Año'].max()}")
escribir_informe("")

# 1.2 Normalizar nombres de productos (detectar duplicados por similitud)
def normalizar_texto(serie):
    """Normaliza texto: minúsculas, sin espacios extra, sin \xa0"""
    return serie.str.lower().str.strip().str.replace('\xa0', ' ').str.replace(r'\s+', ' ', regex=True)

escribir_informe("   📦 Normalizando productos...")
productos_originales = df['Producto'].unique()
df['Producto_original'] = df['Producto']
df['Producto_limpio'] = normalizar_texto(df['Producto'])

# Detectar productos similares (posibles duplicados)
# Solo agrupar si son MUY similares (cutoff alto) y tienen mismas palabras clave
productos_unicos_limpios = df['Producto_limpio'].unique()
duplicados_posibles = {}

for i, prod1 in enumerate(productos_unicos_limpios):
    # Buscar similares con cutoff más estricto
    similares = get_close_matches(prod1, productos_unicos_limpios, n=10, cutoff=0.90)
    
    if len(similares) > 1:
        # Verificar que realmente son el mismo concepto
        # (no solo similitud de caracteres)
        palabras_clave_1 = set(prod1.split())
        
        grupo_valido = [prod1]
        for similar in similares:
            if similar == prod1:
                continue
            palabras_clave_2 = set(similar.split())
            
            # Solo agrupar si comparten al menos 70% de las palabras
            overlap = len(palabras_clave_1 & palabras_clave_2)
            total = len(palabras_clave_1 | palabras_clave_2)
            
            if overlap / total >= 0.7:
                grupo_valido.append(similar)
        
        if len(grupo_valido) > 1:
            # Encontrar el nombre original más común del grupo
            prods_originales = df[df['Producto_limpio'].isin(grupo_valido)]['Producto_original'].value_counts()
            nombre_canonico = prods_originales.index[0]
            
            for prod in grupo_valido:
                if prod not in duplicados_posibles:
                    duplicados_posibles[prod] = nombre_canonico

if duplicados_posibles:
    escribir_informe(f"   ⚠️  Detectados {len(duplicados_posibles)} productos similares que se normalizarán:")
    for similar, canonico in list(duplicados_posibles.items())[:5]:
        escribir_informe(f"       '{similar}' → '{canonico}'")
    if len(duplicados_posibles) > 5:
        escribir_informe(f"       ... y {len(duplicados_posibles) - 5} más")
    
    # Crear mapeo eficiente: limpio -> original más frecuente
    producto_map = {}
    for prod_limpio in df['Producto_limpio'].unique():
        if prod_limpio in duplicados_posibles:
            producto_map[prod_limpio] = duplicados_posibles[prod_limpio]
        else:
            # Tomar el nombre original más común para este producto limpio
            modo = df[df['Producto_limpio'] == prod_limpio]['Producto_original'].mode()
            producto_map[prod_limpio] = modo[0] if len(modo) > 0 else df[df['Producto_limpio'] == prod_limpio]['Producto_original'].iloc[0]
    
    # Aplicar normalización (ahora es instantáneo)
    df['Producto_normalizado'] = df['Producto_limpio'].map(producto_map)
else:
    df['Producto_normalizado'] = df['Producto_original']
    escribir_informe(f"   ✓ No se detectaron duplicados en productos")

escribir_informe(f"   ✓ Productos únicos: {len(productos_originales)} → {df['Producto_normalizado'].nunique()}")
escribir_informe("")

# 1.3 Normalizar elementos
escribir_informe("   🧪 Normalizando elementos de emisión...")
elementos_originales = df['Elemento'].unique()
df['Elemento_original'] = df['Elemento']
df['Elemento_limpio'] = normalizar_texto(df['Elemento'])

# Detectar elementos similares con lógica MUY estricta
elementos_unicos_limpios = df['Elemento_limpio'].unique()
duplicados_elementos = {}

for elem in elementos_unicos_limpios:
    similares = get_close_matches(elem, elementos_unicos_limpios, n=10, cutoff=0.92)
    
    if len(similares) > 1:
        # Extraer gases mencionados (co2, ch4, n2o, f-gases)
        def extraer_gases(texto):
            gases = set()
            texto_lower = texto.lower()
            if 'co2eq' in texto_lower:
                gases.add('co2eq')
            elif 'co2' in texto_lower:
                gases.add('co2')
            if 'ch4' in texto_lower:
                gases.add('ch4')
            if 'n2o' in texto_lower:
                gases.add('n2o')
            if 'f-gases' in texto_lower or 'f-gas' in texto_lower:
                gases.add('f-gases')
            return gases
        
        gases_elem = extraer_gases(elem)
        
        grupo_valido = [elem]
        for similar in similares:
            if similar == elem:
                continue
            
            gases_similar = extraer_gases(similar)
            
            # REGLA CRÍTICA: Solo agrupar si mencionan EXACTAMENTE los mismos gases
            # Y al menos uno de ellos menciona un gas específico
            if gases_elem == gases_similar and len(gases_elem) > 0:
                # Además verificar que las palabras clave coincidan mucho
                palabras_1 = set(elem.split())
                palabras_2 = set(similar.split())
                
                overlap = len(palabras_1 & palabras_2)
                total = len(palabras_1 | palabras_2)
                
                if overlap / total >= 0.75:
                    grupo_valido.append(similar)
        
        if len(grupo_valido) > 1:
            elems_originales = df[df['Elemento_limpio'].isin(grupo_valido)]['Elemento_original'].value_counts()
            nombre_canonico = elems_originales.index[0]
            
            for elem_similar in grupo_valido:
                if elem_similar not in duplicados_elementos:
                    duplicados_elementos[elem_similar] = nombre_canonico

if duplicados_elementos:
    escribir_informe(f"   ⚠️  Detectados {len(duplicados_elementos)} elementos similares:")
    for similar, canonico in list(duplicados_elementos.items())[:10]:
        escribir_informe(f"       '{similar}' → '{canonico}'")
    if len(duplicados_elementos) > 10:
        escribir_informe(f"       ... y {len(duplicados_elementos) - 10} más")
else:
    escribir_informe(f"   ✓ No se detectaron duplicados en elementos")

# Crear mapeo eficiente (SIEMPRE, incluso si no hay duplicados)
elemento_map = {}
for elem_limpio in df['Elemento_limpio'].unique():
    if elem_limpio in duplicados_elementos:
        elemento_map[elem_limpio] = duplicados_elementos[elem_limpio]
    else:
        # Tomar el original más común
        modo = df[df['Elemento_limpio'] == elem_limpio]['Elemento_original'].mode()
        elemento_map[elem_limpio] = modo[0] if len(modo) > 0 else df[df['Elemento_limpio'] == elem_limpio]['Elemento_original'].iloc[0]

# Aplicar normalización (usa el diccionario pre-calculado)
df['Elemento_normalizado'] = df['Elemento_limpio'].map(elemento_map)

escribir_informe(f"   ✓ Elementos únicos: {len(elementos_originales)} → {df['Elemento_normalizado'].nunique()}")
escribir_informe("")

# 1.4 Normalizar países
escribir_informe("   🌍 Normalizando nombres de países...")

# SIEMPRE conservar el nombre original
df["Área_original"] = df["Área"]

# Ruta completa del archivo JSON
JSON_PATH = os.path.join(RESULTS_DIR, 'iso_to_continent.json')

# Leer el archivo
with open(JSON_PATH, encoding='utf-8') as f:
    iso_to_continent = json.load(f)

if USAR_GEOPY:
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter

        # Geocoder y rate limit (1s por llamada)
        geolocator = Nominatim(user_agent="emissions_analyzer_v1")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        # ⚠️ Usar la MISMA columna que luego se usa para mapear
        paises_unicos = pd.Series(df["Área_original"].unique()).dropna().tolist()

        # Diccionario {nombre_area_original: info_geo}
        pais_info_dict = {}

        escribir_informe(f"   🔄 Geocodificando {len(paises_unicos)} países con Nominatim...")
        escribir_informe(f"      (Esto puede tomar varios minutos...)")

        for i, pais in enumerate(paises_unicos):
            try:
                loc = geocode(pais, timeout=10)

                # Base con TODAS las claves por defecto
                base = {
                    'nombre_normalizado': pais,
                    'latitud': None,
                    'longitud': None,
                    'direccion_completa': 'N/A',
                    'tipo': 'N/A',
                    'importancia': 0,
                    'clase': 'N/A',
                    'pais_ISO': 'N/A',
                    'continente': 'N/A',
                    'region': 'N/A',
                    'estado': 'N/A',
                }

                if loc:
                    # Datos principales
                    base.update({
                        'nombre_normalizado': (loc.address.split(",")[0].strip()
                                               if loc.address else pais),
                        'latitud': loc.latitude,
                        'longitud': loc.longitude,
                        'direccion_completa': loc.address or 'N/A',
                        'tipo': loc.raw.get('type', 'N/A'),
                        'importancia': loc.raw.get('importance', 0),
                        'clase': loc.raw.get('class', 'N/A'),
                    })

                    # Dirección detallada (si existe)
                    addr = loc.raw.get('address', {})
                    country_code = addr.get('country_code')

                    base.update({
                        'pais_ISO': (country_code.upper() if country_code else 'N/A'),
                        'continente': addr.get('continent', 'N/A'),
                        'region': addr.get('region', addr.get('state_district', 'N/A')),
                        'estado': addr.get('state', 'N/A'),
                    })

                    # --- 🔧 COMPLEMENTOS ---
                    # Si Nominatim no devuelve el ISO, intentar resolverlo con pycountry
                    if base['pais_ISO'] == 'N/A':
                        try:
                            import pycountry
                            country_obj = pycountry.countries.lookup(pais)
                            base['pais_ISO'] = country_obj.alpha_2
                        except Exception:
                            pass

                    # Si Nominatim no devuelve continente, inferirlo del ISO
                    if base['continente'] == 'N/A':
                        if base['pais_ISO'] in iso_to_continent:
                            base['continente'] = iso_to_continent[base['pais_ISO']]


                # Guardar resultado (exitoso o no)
                pais_info_dict[pais] = base

            except Exception as e:
                # Error en geocodificación: registrar y continuar con defaults
                escribir_informe(f"      ⚠️  Error geocodificando '{pais}': {e}", consola=False)
                pais_info_dict[pais] = {
                    'nombre_normalizado': pais,
                    'latitud': None,
                    'longitud': None,
                    'direccion_completa': 'Error',
                    'tipo': 'N/A',
                    'importancia': 0,
                    'clase': 'N/A',
                    'pais_ISO': 'N/A',
                    'continente': 'N/A',
                    'region': 'N/A',
                    'estado': 'N/A',
                }

            if (i + 1) % 10 == 0 or (i + 1) == len(paises_unicos):
                escribir_informe(f"       Procesados {i + 1}/{len(paises_unicos)}...")

        # Aplicar información geográfica al DataFrame (map seguro con .get)
        df["Área_normalizada"] = df["Área_original"].map(
            lambda x: pais_info_dict.get(x, {}).get('nombre_normalizado', x)
        )
        df["Latitud"] = df["Área_original"].map(
            lambda x: pais_info_dict.get(x, {}).get('latitud', None)
        )
        df["Longitud"] = df["Área_original"].map(
            lambda x: pais_info_dict.get(x, {}).get('longitud', None)
        )
        df["País_ISO"] = df["Área_original"].map(
            lambda x: pais_info_dict.get(x, {}).get('pais_ISO', 'N/A')
        )
        df["Continente"] = df["Área_original"].map(
            lambda x: pais_info_dict.get(x, {}).get('continente', 'N/A')
        )
        df["Región"] = df["Área_original"].map(
            lambda x: pais_info_dict.get(x, {}).get('region', 'N/A')
        )

        # Exportar CSV con toda la info geográfica
        geo_info_rows = []
        for pais, info in pais_info_dict.items():
            geo_info_rows.append({
                'Nombre_original': pais,
                'Nombre_normalizado': info.get('nombre_normalizado'),
                'Latitud': info.get('latitud'),
                'Longitud': info.get('longitud'),
                'ISO_code': info.get('pais_ISO'),
                'Continente': info.get('continente'),
                'Región': info.get('region'),
                'Estado': info.get('estado'),
                'Tipo': info.get('tipo'),
                'Clase': info.get('clase'),
                'Importancia': info.get('importancia'),
                'Dirección_completa': info.get('direccion_completa'),
            })

        df_geo_info = pd.DataFrame(geo_info_rows)
        # Ruta del CSV de salida
        CSV_PATH = os.path.join(RESULTS_DIR, 'informacion_geografica_paises.csv')
        df_geo_info.to_csv(CSV_PATH, index=False)

        cambios = sum(1 for k, v in pais_info_dict.items()
                      if v.get('nombre_normalizado', k) != k)
        sin_geocodificar = sum(1 for v in pais_info_dict.values()
                               if v.get('latitud') is None)

        escribir_informe(f"   ✓ Nombres normalizados: {cambios}")
        escribir_informe(f"   ✓ Países geocodificados: {len(paises_unicos) - sin_geocodificar}/{len(paises_unicos)}")
        escribir_informe(f"   ✓ Información geográfica guardada en 'informacion_geografica_paises.csv'")

        # Mostrar algunos ejemplos
        escribir_informe(f"\n   📍 Ejemplos de normalización:")
        mostrados = 0
        for original, info in pais_info_dict.items():
            if info.get('latitud') is not None:
                escribir_informe(
                    f"      '{original}' → '{info.get('nombre_normalizado')}' "
                    f"({info.get('latitud'):.2f}, {info.get('longitud'):.2f}) "
                    f"[{info.get('continente')}]"
                )
                mostrados += 1
                if mostrados >= 5:
                    break

    except ImportError:
        escribir_informe("   ⚠️  geopy no disponible, usando nombres originales")
        df["Área_normalizada"] = df["Área_original"]
        df["Latitud"] = None
        df["Longitud"] = None
        df["País_ISO"] = None
        df["Continente"] = None
        df["Región"] = None
else:
    # Normalización básica sin geopy
    df["Área_normalizada"] = df["Área_original"].str.strip()
    df["Latitud"] = None
    df["Longitud"] = None
    df["País_ISO"] = None
    df["Continente"] = None
    df["Región"] = None
    escribir_informe(f"   ✓ Normalización básica aplicada (sin geopy)")


escribir_informe(f"   ✓ Países únicos: {df['Área_original'].nunique()}")
escribir_informe("")

# Actualizar columna Área para usar en análisis (pero conservando original)
# Usamos el original por defecto para mantener legibilidad
df['Área'] = df['Área_original']

# Limpiar columnas temporales
df = df.drop(columns=[col for col in df.columns if '_original' in col or '_limpio' in col or '_normalizado' in col or '_normalizada' in col])

# Guardar datos limpios
CSV_PATH = os.path.join(RESULTS_DIR, 'datos_normalizados.csv')
df.to_csv(CSV_PATH, index=False)
escribir_informe("   💾 Datos normalizados guardados en 'datos_normalizados.csv'")
escribir_informe("")

# =====================================================================
# PASO 2: ESTADÍSTICAS DESCRIPTIVAS
# =====================================================================

escribir_informe("[2/7] Generando estadísticas descriptivas...")
escribir_informe("")

escribir_informe("## 📊 ESTADÍSTICAS GENERALES")
escribir_informe("")
escribir_informe(f"- **Total de registros**: {len(df):,}")
escribir_informe(f"- **Países únicos**: {df['Área'].nunique()}")
escribir_informe(f"- **Productos únicos**: {df['Producto'].nunique()}")
escribir_informe(f"- **Elementos únicos**: {df['Elemento'].nunique()}")
escribir_informe(f"- **Rango temporal**: {df['Año'].min()} - {df['Año'].max()} ({df['Año'].max() - df['Año'].min() + 1} años)")
escribir_informe(f"- **Valores nulos**: {df['Valor'].isna().sum():,} ({df['Valor'].isna().sum() / len(df) * 100:.2f}%)")
escribir_informe("")

# Estadísticas de valores
escribir_informe("### 📈 Distribución de valores de emisiones")
escribir_informe("")
escribir_informe(f"- **Media**: {df['Valor'].mean():,.2f}")
escribir_informe(f"- **Mediana**: {df['Valor'].median():,.2f}")
escribir_informe(f"- **Desviación estándar**: {df['Valor'].std():,.2f}")
escribir_informe(f"- **Mínimo**: {df['Valor'].min():,.2f}")
escribir_informe(f"- **Máximo**: {df['Valor'].max():,.2f}")
escribir_informe("")

# Top países por emisiones totales
escribir_informe("### 🌍 Top 10 países por emisiones totales")
escribir_informe("")
top_paises = df.groupby('Área')['Valor'].sum().sort_values(ascending=False).head(10)
for i, (pais, valor) in enumerate(top_paises.items(), 1):
    escribir_informe(f"{i}. **{pais}**: {valor:,.2f}")
escribir_informe("")

# Top productos
escribir_informe("### 📦 Top 10 productos por emisiones totales")
escribir_informe("")
top_productos = df.groupby('Producto')['Valor'].sum().sort_values(ascending=False).head(10)
for i, (prod, valor) in enumerate(top_productos.items(), 1):
    escribir_informe(f"{i}. **{prod}**: {valor:,.2f}")
escribir_informe("")

# Top elementos
escribir_informe("### 🧪 Elementos de emisión por volumen total")
escribir_informe("")
top_elementos = df.groupby('Elemento')['Valor'].sum().sort_values(ascending=False)
for i, (elem, valor) in enumerate(top_elementos.items(), 1):
    escribir_informe(f"{i}. **{elem}**: {valor:,.2f}")
escribir_informe("")

# Emisiones por década
escribir_informe("### 📅 Evolución temporal de emisiones")
escribir_informe("")
df['Década'] = (df['Año'] // 10) * 10
emisiones_decada = df.groupby('Década')['Valor'].sum().sort_index()
escribir_informe("| Década | Emisiones Totales |")
escribir_informe("|--------|-------------------|")
for decada, valor in emisiones_decada.items():
    escribir_informe(f"| {int(decada)}s | {valor:,.2f} |")
escribir_informe("")

# Cobertura de datos por país
escribir_informe("### 🗺️ Cobertura de datos por región")
escribir_informe("")
registros_pais = df.groupby('Área').size().describe()
escribir_informe(f"- **Promedio de registros por país**: {registros_pais['mean']:.0f}")
escribir_informe(f"- **Mediana**: {registros_pais['50%']:.0f}")
escribir_informe(f"- **País con más registros**: {df.groupby('Área').size().idxmax()} ({df.groupby('Área').size().max():,})")
escribir_informe(f"- **País con menos registros**: {df.groupby('Área').size().idxmin()} ({df.groupby('Área').size().min():,})")
escribir_informe("")

# =====================================================================
# PASO 3: ANÁLISIS DE COMPLETITUD POR PAÍS
# =====================================================================

def analizar_completitud_pais(df):
    """Analiza completitud de datos para cada país"""
    resultados = []
    
    for pais in df['Área'].unique():
        df_pais = df[df['Área'] == pais].copy()
        
        año_min = df_pais['Año'].min()
        año_max = df_pais['Año'].max()
        años_span = año_max - año_min + 1
        años_unicos = df_pais['Año'].nunique()
        
        productos = df_pais['Producto'].unique()
        elementos = df_pais['Elemento'].unique()
        
        combos_disponibles = df_pais.groupby(['Producto', 'Elemento']).size()
        n_combos_disponibles = len(combos_disponibles)
        
        n_productos_total = df['Producto'].nunique()
        n_elementos_total = df['Elemento'].nunique()
        
        registros_reales = len(df_pais)
        # Densidad: registros reales vs registros esperados SI tuviera todos los años para sus combos
        registros_esperados = años_span * n_combos_disponibles
        densidad = (registros_reales / registros_esperados * 100) if registros_esperados > 0 else 0
        
        cobertura_productos = len(productos) / n_productos_total * 100
        cobertura_elementos = len(elementos) / n_elementos_total * 100
        
        registros_pre1990 = len(df_pais[df_pais['Año'] < 1990])
        registros_post1990 = len(df_pais[df_pais['Año'] >= 1990])
        
        resultados.append({
            'País': pais,
            'Año_min': año_min,
            'Año_max': año_max,
            'Años_span': años_span,
            'Años_únicos': años_unicos,
            'Continuidad_%': (años_unicos / años_span * 100) if años_span > 0 else 0,
            'N_productos': len(productos),
            'N_elementos': len(elementos),
            'N_combos_prod_elem': n_combos_disponibles,
            'Cobertura_productos_%': cobertura_productos,
            'Cobertura_elementos_%': cobertura_elementos,
            'Registros_totales': registros_reales,
            'Densidad_datos_%': densidad,
            'Registros_pre1990': registros_pre1990,
            'Registros_post1990': registros_post1990,
            'Tiene_pre1990': registros_pre1990 > 0,
            'Tiene_post1990': registros_post1990 > 0,
        })
    
    return pd.DataFrame(resultados)

escribir_informe("[3/7] Analizando completitud por país...")
escribir_informe("")

df_completitud = analizar_completitud_pais(df)
CSV_PATH = os.path.join(RESULTS_DIR, 'completitud_por_pais.csv')
df_completitud.to_csv(CSV_PATH, index=False)

escribir_informe(f"   ✓ Análisis de {len(df_completitud)} países completado")
escribir_informe("")

escribir_informe("### 📊 Estadísticas de completitud")
escribir_informe("")
escribir_informe(f"- **Países con datos pre-1990**: {df_completitud['Tiene_pre1990'].sum()} ({df_completitud['Tiene_pre1990'].sum() / len(df_completitud) * 100:.1f}%)")
escribir_informe(f"- **Países con datos post-1990**: {df_completitud['Tiene_post1990'].sum()} ({df_completitud['Tiene_post1990'].sum() / len(df_completitud) * 100:.1f}%)")
escribir_informe(f"- **Países con ambos períodos**: {(df_completitud['Tiene_pre1990'] & df_completitud['Tiene_post1990']).sum()}")
escribir_informe(f"- **Cobertura promedio de productos**: {df_completitud['Cobertura_productos_%'].mean():.1f}%")
escribir_informe(f"- **Densidad promedio de datos**: {df_completitud['Densidad_datos_%'].mean():.1f}%")
escribir_informe("")

# =====================================================================
# PASO 4: CREAR FIRMA DE COMPLETITUD
# =====================================================================

def crear_firma_completitud(df, df_completitud):
    """
    Crea vector de features para clustering basado en PATRONES de completitud,
    no en productos específicos.
    """
    firmas = []
    
    for pais in df_completitud['País']:
        df_pais = df[df['Área'] == pais].copy()
        firma = {'País': pais}
        
        # ═══════════════════════════════════════════════════════════
        # FEATURES BASADAS EN COMPLETITUD, NO EN CONTENIDO ESPECÍFICO
        # ═══════════════════════════════════════════════════════════
        
        # 1. Cobertura temporal absoluta
        firma['año_inicio'] = df_completitud[df_completitud['País'] == pais]['Año_min'].values[0]
        firma['año_fin'] = df_completitud[df_completitud['País'] == pais]['Año_max'].values[0]
        firma['años_span'] = df_completitud[df_completitud['País'] == pais]['Años_span'].values[0]
        firma['continuidad_%'] = df_completitud[df_completitud['País'] == pais]['Continuidad_%'].values[0]
        
        # 2. Cobertura de productos y elementos (porcentajes, no específicos)
        firma['cobertura_productos_%'] = df_completitud[df_completitud['País'] == pais]['Cobertura_productos_%'].values[0]
        firma['cobertura_elementos_%'] = df_completitud[df_completitud['País'] == pais]['Cobertura_elementos_%'].values[0]
        firma['n_productos'] = df_completitud[df_completitud['País'] == pais]['N_productos'].values[0]
        firma['n_elementos'] = df_completitud[df_completitud['País'] == pais]['N_elementos'].values[0]
        firma['n_combos'] = df_completitud[df_completitud['País'] == pais]['N_combos_prod_elem'].values[0]
        
        # 3. Densidad general
        firma['densidad_datos_%'] = df_completitud[df_completitud['País'] == pais]['Densidad_datos_%'].values[0]
        
        # 4. Disponibilidad por período histórico (binario y cantidad)
        firma['tiene_pre1990'] = 1 if df_completitud[df_completitud['País'] == pais]['Tiene_pre1990'].values[0] else 0
        firma['tiene_post1990'] = 1 if df_completitud[df_completitud['País'] == pais]['Tiene_post1990'].values[0] else 0
        firma['registros_pre1990'] = df_completitud[df_completitud['País'] == pais]['Registros_pre1990'].values[0]
        firma['registros_post1990'] = df_completitud[df_completitud['País'] == pais]['Registros_post1990'].values[0]
        
        # 5. Distribución temporal: registros por década (normalizado por década disponible)
        for decada in range(1960, 2030, 10):
            registros_decada = len(df_pais[
                (df_pais['Año'] >= decada) & 
                (df_pais['Año'] < decada + 10)
            ])
            # Normalizar: registros por año en esa década
            años_en_decada = len(df_pais[
                (df_pais['Año'] >= decada) & 
                (df_pais['Año'] < decada + 10)
            ]['Año'].unique())
            
            firma[f'registros_por_año_{decada}s'] = (
                registros_decada / años_en_decada if años_en_decada > 0 else 0
            )
        
        # 6. Variabilidad temporal: ¿datos consistentes o esporádicos?
        registros_por_año = df_pais.groupby('Año').size()
        firma['std_registros_por_año'] = registros_por_año.std()
        firma['cv_registros_por_año'] = (
            registros_por_año.std() / registros_por_año.mean() 
            if registros_por_año.mean() > 0 else 0
        )
        
        # 7. Completitud por tipo de elemento (agregado)
        for tipo_elemento in ['CO2', 'CH4', 'N2O', 'F-gases', 'CO2eq']:
            elementos_tipo = df_pais[
                df_pais['Elemento'].str.contains(tipo_elemento, case=False, na=False)
            ]
            firma[f'tiene_elementos_{tipo_elemento}'] = 1 if len(elementos_tipo) > 0 else 0
            firma[f'registros_{tipo_elemento}'] = len(elementos_tipo)
        
        firmas.append(firma)
    
    df_firmas = pd.DataFrame(firmas)
    paises = df_firmas['País']
    X = df_firmas.drop('País', axis=1)
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=paises
    )
    
    return X_scaled, X, scaler

escribir_informe("[4/7] Creando firmas de completitud...")
X_scaled, X_original, scaler = crear_firma_completitud(df, df_completitud)
escribir_informe(f"   ✓ Matriz de features: {X_scaled.shape}")
escribir_informe("")

# =====================================================================
# PASO 5: CLUSTERING
# =====================================================================

def encontrar_k_optimo(X, k_max=15):
    """Encuentra k óptimo usando elbow + silhouette"""
    k_range = range(2, min(k_max, len(X) // 2))
    
    if len(k_range) == 0:
        return 2, []
    
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        
        if k < len(X):
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(0)
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inercia', fontsize=12)
    axes[0].set_title('Método del Codo', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Análisis de Silhouette', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_completitud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if silhouettes:
        k_optimo = k_range[np.argmax(silhouettes)]
        k_optimo = max(k_optimo, 3)
    else:
        k_optimo = 3
    
    return k_optimo, list(zip(k_range, inertias, silhouettes))

escribir_informe("[5/7] Buscando k óptimo y aplicando clustering...")
k_optimo, metricas = encontrar_k_optimo(X_scaled)

escribir_informe(f"   ✓ K óptimo: {k_optimo}")
escribir_informe(f"   ✓ Silhouette score: {metricas[k_optimo-2][2]:.3f}")
escribir_informe("")

kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_completitud['Cluster'] = clusters

# =====================================================================
# PASO 6: CARACTERIZACIÓN DE CLUSTERS
# =====================================================================

escribir_informe("[6/7] Caracterizando clusters...")
escribir_informe("")
escribir_informe("## 🔍 CARACTERIZACIÓN DE CLUSTERS")
escribir_informe("")

for cluster_id in sorted(df_completitud['Cluster'].unique()):
    paises_cluster = df_completitud[df_completitud['Cluster'] == cluster_id]['País'].tolist()
    cluster_data = df_completitud[df_completitud['Cluster'] == cluster_id]
    
    escribir_informe(f"### Cluster {cluster_id} - {len(paises_cluster)} países")
    escribir_informe("")
    
    escribir_informe(f"**Países**: {', '.join(paises_cluster[:10])}")
    if len(paises_cluster) > 10:
        escribir_informe(f"... y {len(paises_cluster) - 10} más")
    escribir_informe("")
    
    escribir_informe("**Cobertura temporal:**")
    escribir_informe(f"- Año inicio promedio: {cluster_data['Año_min'].mean():.0f}")
    escribir_informe(f"- Rango de inicio: {cluster_data['Año_min'].min():.0f} - {cluster_data['Año_min'].max():.0f}")
    escribir_informe(f"- Continuidad promedio: {cluster_data['Continuidad_%'].mean():.1f}%")
    escribir_informe("")
    
    escribir_informe("**Cobertura de datos:**")
    escribir_informe(f"- Productos (promedio): {cluster_data['N_productos'].mean():.1f} / {df['Producto'].nunique()}")
    escribir_informe(f"- Elementos (promedio): {cluster_data['N_elementos'].mean():.1f} / {df['Elemento'].nunique()}")
    escribir_informe(f"- Densidad (promedio): {cluster_data['Densidad_datos_%'].mean():.1f}%")
    escribir_informe("")
    
    pct_pre1990 = (cluster_data['Tiene_pre1990'].sum() / len(cluster_data) * 100)
    escribir_informe(f"**Datos pre-1990:** {cluster_data['Tiene_pre1990'].sum()}/{len(cluster_data)} países ({pct_pre1990:.1f}%)")
    escribir_informe("")

# =====================================================================
# PASO 7: ANÁLISIS DE OVERLAP
# =====================================================================

def analizar_overlap_temporal(df, df_completitud):
    """Encuentra pares con overlap y calcula correlaciones"""
    resultados_overlap = []
    
    escribir_informe("[7/7] Analizando overlap y correlaciones...")
    escribir_informe("")
    
    for cluster_id in sorted(df_completitud['Cluster'].unique()):
        paises_cluster = df_completitud[df_completitud['Cluster'] == cluster_id]['País'].tolist()
        
        if len(paises_cluster) < 2:
            continue
        
        escribir_informe(f"### Overlap en Cluster {cluster_id}")
        escribir_informe("")
        
        for pais1, pais2 in combinations(paises_cluster, 2):
            df_p1 = df[df['Área'] == pais1].copy()
            df_p2 = df[df['Área'] == pais2].copy()
            
            años_p1 = set(df_p1['Año'].unique())
            años_p2 = set(df_p2['Año'].unique())
            años_comunes = años_p1 & años_p2
            
            if len(años_comunes) < 5:
                continue
            
            prods_p1 = set(df_p1['Producto'].unique())
            prods_p2 = set(df_p2['Producto'].unique())
            prods_comunes = prods_p1 & prods_p2
            
            if len(prods_comunes) == 0:
                continue
            
            try:
                df_p1_comun = df_p1[
                    (df_p1['Año'].isin(años_comunes)) &
                    (df_p1['Producto'].isin(prods_comunes))
                ]
                df_p2_comun = df_p2[
                    (df_p2['Año'].isin(años_comunes)) &
                    (df_p2['Producto'].isin(prods_comunes))
                ]
                
                pivot_p1 = df_p1_comun.pivot_table(
                    index='Año',
                    columns=['Producto', 'Elemento'],
                    values='Valor',
                    aggfunc='sum'
                )
                
                pivot_p2 = df_p2_comun.pivot_table(
                    index='Año',
                    columns=['Producto', 'Elemento'],
                    values='Valor',
                    aggfunc='sum'
                )
                
                cols_comunes = pivot_p1.columns.intersection(pivot_p2.columns)
                
                if len(cols_comunes) > 0:
                    corr = pivot_p1[cols_comunes].corrwith(pivot_p2[cols_comunes], axis=0).mean()
                    
                    años_solo_p1 = años_p1 - años_p2
                    años_solo_p2 = años_p2 - años_p1
                    
                    resultado = {
                        'Cluster': cluster_id,
                        'País_1': pais1,
                        'País_2': pais2,
                        'Años_overlap': len(años_comunes),
                        'Años_min_común': min(años_comunes),
                        'Años_max_común': max(años_comunes),
                        'Productos_comunes': len(prods_comunes),
                        'Correlación': corr,
                        'Años_solo_P1': len(años_solo_p1),
                        'Años_solo_P2': len(años_solo_p2),
                        'Rango_solo_P1': f"{min(años_solo_p1) if años_solo_p1 else 'N/A'}-{max(años_solo_p1) if años_solo_p1 else 'N/A'}",
                        'Rango_solo_P2': f"{min(años_solo_p2) if años_solo_p2 else 'N/A'}-{max(años_solo_p2) if años_solo_p2 else 'N/A'}",
                    }
                    
                    resultados_overlap.append(resultado)
            
            except Exception as e:
                continue
        
        # Mostrar top correlaciones
        cluster_overlaps = [r for r in resultados_overlap if r['Cluster'] == cluster_id]
        
        if cluster_overlaps:
            df_cluster_overlap = pd.DataFrame(cluster_overlaps).sort_values('Correlación', ascending=False)
            
            escribir_informe(f"**Top 5 pares con mayor correlación:**")
            escribir_informe("")
            
            for idx, row in df_cluster_overlap.head(5).iterrows():
                escribir_informe(f"- **{row['País_1']} ↔ {row['País_2']}**")
                escribir_informe(f"  - Correlación: {row['Correlación']:.3f}")
                escribir_informe(f"  - Overlap: {row['Años_overlap']} años ({row['Años_min_común']}-{row['Años_max_común']})")
                escribir_informe(f"  - Productos comunes: {row['Productos_comunes']}")
                
                if row['Años_solo_P1'] > 0:
                    escribir_informe(f"  - 🔍 {row['País_1']} tiene datos exclusivos: {row['Rango_solo_P1']}")
                if row['Años_solo_P2'] > 0:
                    escribir_informe(f"  - 🔍 {row['País_2']} tiene datos exclusivos: {row['Rango_solo_P2']}")
                escribir_informe("")
    
    if resultados_overlap:
        df_overlap = pd.DataFrame(resultados_overlap)
        CSV_PATH = os.path.join(RESULTS_DIR, 'overlap_correlaciones.csv')
        df_overlap.to_csv(CSV_PATH, index=False)
        escribir_informe(f"✓ Análisis de overlap guardado en 'overlap_correlaciones.csv'")
        escribir_informe("")
        return df_overlap
    else:
        escribir_informe("⚠️  No se encontraron overlaps significativos")
        escribir_informe("")
        return pd.DataFrame()

df_overlap = analizar_overlap_temporal(df, df_completitud)

# =====================================================================
# IDENTIFICACIÓN DE CANDIDATOS PARA IMPUTACIÓN
# =====================================================================

escribir_informe("## 🎯 CANDIDATOS PARA IMPUTACIÓN")
escribir_informe("")

if len(df_overlap) > 0:
    # Filtrar pares prometedores
    candidatos = df_overlap[
        (df_overlap['Correlación'] > 0.70) &
        (df_overlap['Años_overlap'] >= 10) &
        ((df_overlap['Años_solo_P1'] > 0) | (df_overlap['Años_solo_P2'] > 0))
    ].copy()
    
    if len(candidatos) > 0:
        escribir_informe(f"Se identificaron **{len(candidatos)} pares de países** candidatos para imputación:")
        escribir_informe("")
        escribir_informe("### Criterios de selección:")
        escribir_informe("- Correlación > 0.70")
        escribir_informe("- Overlap temporal ≥ 10 años")
        escribir_informe("- Al menos uno tiene datos exclusivos")
        escribir_informe("")
        
        # Ordenar por correlación y mostrar top 20
        candidatos_sorted = candidatos.sort_values('Correlación', ascending=False)
        
        escribir_informe("### Top candidatos:")
        escribir_informe("")
        escribir_informe("| País Donante | País Receptor | Correlación | Años Overlap | Período a Imputar |")
        escribir_informe("|--------------|---------------|-------------|--------------|-------------------|")
        
        for idx, row in candidatos_sorted.head(20).iterrows():
            # Determinar quién es donante y quién receptor
            if row['Años_solo_P1'] > row['Años_solo_P2']:
                donante = row['País_1']
                receptor = row['País_2']
                periodo = row['Rango_solo_P1']
            else:
                donante = row['País_2']
                receptor = row['País_1']
                periodo = row['Rango_solo_P2']
            
            escribir_informe(f"| {donante} | {receptor} | {row['Correlación']:.3f} | {row['Años_overlap']} | {periodo} |")
        
        escribir_informe("")
        
        # Guardar candidatos
        CSV_PATH = os.path.join(RESULTS_DIR, 'candidatos_imputacion.csv')
        candidatos_sorted.to_csv(CSV_PATH, index=False)
        escribir_informe("✓ Candidatos guardados en 'candidatos_imputacion.csv'")
        escribir_informe("")
    else:
        escribir_informe("⚠️  No se encontraron pares que cumplan los criterios de imputación")
        escribir_informe("")
else:
    escribir_informe("⚠️  No hay datos de overlap disponibles")
    escribir_informe("")

# =====================================================================
# ANÁLISIS DE PRODUCTOS Y ELEMENTOS FALTANTES
# =====================================================================

escribir_informe("## 🔍 ANÁLISIS DE DATOS FALTANTES POR PRODUCTO/ELEMENTO")
escribir_informe("")

# Encontrar productos/elementos con más gaps
productos_elemento = df.groupby(['Producto', 'Elemento']).size().reset_index(name='Registros')
productos_elemento_esperados = len(df['Área'].unique()) * (df['Año'].max() - df['Año'].min() + 1)

escribir_informe("### Productos/Elementos con más datos faltantes:")
escribir_informe("")

# Calcular completitud por producto-elemento
completitud_prod_elem = []
for _, row in productos_elemento.iterrows():
    pct_completitud = (row['Registros'] / productos_elemento_esperados) * 100
    completitud_prod_elem.append({
        'Producto': row['Producto'],
        'Elemento': row['Elemento'],
        'Registros': row['Registros'],
        'Completitud_%': pct_completitud
    })

df_completitud_prod = pd.DataFrame(completitud_prod_elem).sort_values('Completitud_%')

escribir_informe("| Producto | Elemento | Registros | Completitud |")
escribir_informe("|----------|----------|-----------|-------------|")
for idx, row in df_completitud_prod.head(15).iterrows():
    escribir_informe(f"| {row['Producto']} | {row['Elemento']} | {row['Registros']} | {row['Completitud_%']:.1f}% |")

escribir_informe("")
CSV_PATH = os.path.join(RESULTS_DIR, 'completitud_productos_elementos.csv')
df_completitud_prod.to_csv(CSV_PATH, index=False)
escribir_informe("✓ Guardado en 'completitud_productos_elementos.csv'")
escribir_informe("")

# =====================================================================
# GUARDAR RESULTADOS FINALES
# =====================================================================

escribir_informe("## 📁 ARCHIVOS GENERADOS")
escribir_informe("")

df_completitud_final = df_completitud.sort_values(['Cluster', 'Densidad_datos_%'], ascending=[True, False])
CSV_PATH = os.path.join(RESULTS_DIR, 'paises_agrupados_por_completitud.csv')
df_completitud_final.to_csv(CSV_PATH, index=False)

archivos_generados = [
    ('datos_normalizados.csv', 'Datos limpios y normalizados con información geográfica'),
    ('informacion_geografica_paises.csv', 'Tabla de referencia: nombres originales, normalizados y coordenadas'),
    ('completitud_por_pais.csv', 'Análisis detallado de cada país'),
    ('paises_agrupados_por_completitud.csv', f'Países agrupados en {k_optimo} clusters'),
    ('overlap_correlaciones.csv', 'Pares de países con overlap y correlaciones'),
    ('candidatos_imputacion.csv', 'Pares candidatos para imputación'),
    ('completitud_productos_elementos.csv', 'Completitud por producto/elemento'),
    ('clustering_completitud.png', 'Visualización del clustering'),
]

for archivo, descripcion in archivos_generados:
    escribir_informe(f"- **{archivo}**: {descripcion}")

escribir_informe("")

# =====================================================================
# RESUMEN EJECUTIVO
# =====================================================================

escribir_informe("## 📋 RESUMEN EJECUTIVO")
escribir_informe("")

escribir_informe(f"### Datos procesados:")
escribir_informe(f"- **{len(df):,}** registros de emisiones")
escribir_informe(f"- **{df['Área'].nunique()}** países")
escribir_informe(f"- **{df['Producto'].nunique()}** productos")
escribir_informe(f"- **{df['Elemento'].nunique()}** elementos de emisión")
escribir_informe(f"- Período: **{df['Año'].min()}-{df['Año'].max()}**")
escribir_informe("")

escribir_informe(f"### Clustering:")
escribir_informe(f"- Se identificaron **{k_optimo} grupos** de países con patrones similares de disponibilidad de datos")
escribir_informe(f"- Silhouette score: **{metricas[k_optimo-2][2]:.3f}**")
escribir_informe("")

if len(df_overlap) > 0:
    escribir_informe(f"### Oportunidades de imputación:")
    n_candidatos = len(candidatos) if len(candidatos) > 0 else 0
    escribir_informe(f"- **{n_candidatos}** pares de países candidatos para imputación")
    escribir_informe(f"- Correlación promedio: **{candidatos['Correlación'].mean():.3f}**" if n_candidatos > 0 else "- No aplica")
    escribir_informe("")

escribir_informe(f"### Calidad de datos:")
paises_completos = (df_completitud['Densidad_datos_%'] > 80).sum()
paises_incompletos = (df_completitud['Densidad_datos_%'] < 50).sum()

escribir_informe(f"- Países con alta completitud (>80%): **{paises_completos}** ({paises_completos/len(df_completitud)*100:.1f}%)")
escribir_informe(f"- Países con baja completitud (<50%): **{paises_incompletos}** ({paises_incompletos/len(df_completitud)*100:.1f}%)")
escribir_informe(f"- Densidad promedio de datos: **{df_completitud['Densidad_datos_%'].mean():.1f}%**")
escribir_informe("")

# =====================================================================
# RECOMENDACIONES
# =====================================================================

escribir_informe("## 💡 RECOMENDACIONES")
escribir_informe("")

escribir_informe("### Para imputación de datos:")
escribir_informe("")
escribir_informe("1. **Priorizar pares con correlación > 0.80** y overlap > 15 años")
escribir_informe("2. **Validar temporalmente**: Ocultar 10% de datos conocidos y verificar precisión de imputación")
escribir_informe("3. **Imputar solo productos que existían** en el período a imputar (verificar contra productos nuevos post-1990)")
escribir_informe("4. **Documentar**: Crear columna 'Imputado' = True para transparencia")
escribir_informe("5. **Análisis de sensibilidad**: Probar diferentes países donantes y comparar resultados")
escribir_informe("")

escribir_informe("### Para análisis futuros:")
escribir_informe("")
escribir_informe("1. **Análisis por región geográfica**: Los clusters actuales podrían refinarse con información geográfica")
escribir_informe("2. **Factores contextuales**: Incorporar variables económicas, políticas o climáticas")
escribir_informe("3. **Series temporales**: Aplicar modelos ARIMA o Prophet para productos con datos continuos")
escribir_informe("4. **Validación cruzada**: Realizar validación k-fold temporal antes de imputación final")
escribir_informe("")

# =====================================================================
# GUARDAR INFORME
# =====================================================================

escribir_informe("=" * 80)
escribir_informe(f"INFORME GENERADO EXITOSAMENTE")
escribir_informe("=" * 80)

# Guardar informe en markdown
with open(INFORME_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(informe))

print(f"\n✅ Informe completo guardado en '{INFORME_MD}'")
print(f"✅ Todos los archivos CSV generados exitosamente")
print("\n" + "=" * 80)