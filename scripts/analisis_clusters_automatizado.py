import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
import os
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

RUTA_DATOS = '../results/datos_normalizados.csv'
RUTA_CLUSTERS = '../results/paises_agrupados_por_completitud.csv'
RUTA_SALIDA = '../results/analisis_clusters/'
WINDOW_SIZE = 5  # Ventana temporal para secuencias

# Crear directorio de salida
Path(RUTA_SALIDA).mkdir(parents=True, exist_ok=True)

# Sistema de logging dual (consola + markdown)
log_lines = []
INFORME_MD = os.path.join(RUTA_SALIDA, f'informe_analisis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')

def log(texto, consola=True, markdown=True):
    """Escribe en consola y/o acumula en log markdown"""
    if consola:
        print(texto)
    if markdown:
        log_lines.append(texto)

def guardar_log():
    """Guarda el log acumulado en markdown"""
    with open(INFORME_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"\n📄 Informe markdown guardado en: {INFORME_MD}")

# =====================================================================
# MAPEO DE PRODUCTOS (unificación de categorías)
# =====================================================================

MAP_PRODUCTOS = {
    # Categorías iguales
    'Residuos agrícolas': 'Residuos agrícolas',
    'Cultivo del arroz': 'Cultivo del arroz',
    'Quemado de residuos agrícolas': 'Quemado de residuos agrícolas',
    'Fermentación entérica': 'Fermentación entérica',
    'Gestión del estiércol': 'Gestión del estiércol',
    'Estiércol depositado en las pasturas': 'Estiércol depositado en las pasturas',
    'Estiércol aplicado a los suelos': 'Estiércol aplicado a los suelos',
    'Fertilizantes sintéticos': 'Fertilizantes sintéticos',
    'Energía': 'Energía',
    'IPPU': 'IPPU',
    'Desechos': 'Desechos',
    'Otro': 'Otro',
    'Emisiones derivadas de los cultivos': 'Emisiones derivadas de los cultivos',
    'Emissiones derivadas del sector ganadero': 'Emissiones derivadas del sector ganadero',
    'IPCC Agricultura': 'IPCC Agricultura',
    'Suelos agrícolas': 'Suelos agrícolas',

    # Suelos y uso de la tierra → Suelos agrícolas
    'Suelos orgánicos drenados': 'Suelos agrícolas',
    'Suelos orgánicos drenados (CO2)': 'Suelos agrícolas',
    'Suelos orgánicos drenados (N2O)': 'Suelos agrícolas',
    'Tierras forestales': 'Suelos agrícolas',
    'Conversión neta de bosques': 'Suelos agrícolas',
    'Cambios de uso de la tierra': 'Suelos agrícolas',
    'Emisiones en tierras agrícolas': 'Suelos agrícolas',
    'LULUCF': 'Suelos agrícolas',
    'AFOLU': 'Suelos agrícolas',
    'Emisiones totales incluyendo LULUCF': 'Suelos agrícolas',
    'Emisiones totales excluyendo LULUCF': 'Suelos agrícolas',

    # Incendios → Residuos agrícolas
    'Incendios de sabana': 'Residuos agrícolas',
    'Incendios en suelos de turba': 'Residuos agrícolas',
    'Incendios forestales': 'Residuos agrícolas',
    'Incendios en los bosques tropicales húmedos': 'Residuos agrícolas',

    # Energía
    'On-farm energy use': 'Energía',
    'Tanques de combustible internacional': 'Energía',

    # IPPU (procesos industriales)
    'Fabricación de fertilizantes': 'IPPU',
    'Fabricación de pesticidas': 'IPPU',
    'Envasado alimentario': 'IPPU',
    'Transformación\xa0de alimentos': 'IPPU',

    # Otros sistemas
    'Sistemas agroalimentarios': 'Otro',
    'Farm gate': 'Otro',
    'Pre y\xa0post-producción': 'Otro',
    'Venta de alimentos': 'Otro',
    'Consumo\xa0de alimentos en los hogares': 'Otro',

    # Desechos
    'Eliminación de desechos de sistemas agroalimentarios': 'Desechos',
}

# =====================================================================
# FUNCIONES AUXILIARES BÁSICAS
# =====================================================================

def cargar_y_preparar_datos():
    """Carga datos y clusters, realiza merge"""
    log("=" * 80)
    log("# ANÁLISIS AUTOMATIZADO POR CLUSTER")
    log("=" * 80)
    log("")
    log(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    log("## [1/6] Cargando datos...")
    log("")
    
    df = pd.read_csv(RUTA_DATOS, sep=',')
    df_cluster = pd.read_csv(RUTA_CLUSTERS, sep=',')
    
    log(f"- ✓ Datos cargados: **{len(df):,}** registros")
    log(f"- ✓ Información de clusters: **{len(df_cluster)}** países")
    
    # Merge
    df = df.merge(
        df_cluster[['País', 'Cluster']], 
        left_on='Área', 
        right_on='País', 
        how='left'
    )
    df = df.drop(columns='País')
    
    # Aplicar mapeo de productos
    df['Producto'] = df['Producto'].replace(MAP_PRODUCTOS)
    
    # Seleccionar columnas relevantes
    df = df[['Área', 'Producto', 'Elemento', 'Año', 'Valor', 'Cluster']]
    
    # Eliminar registros sin año o valor
    df = df.dropna(subset=['Año', 'Valor'])
    
    log(f"- ✓ Datos preparados: **{len(df):,}** registros válidos")
    log(f"- ✓ Productos únicos después del mapeo: **{df['Producto'].nunique()}**")
    log("")
    
    return df

def analizar_completitud_cluster(df_cluster):
    """Analiza elementos presentes/faltantes por producto en un cluster"""
    
    elements_per_product = (
        df_cluster.groupby("Producto")["Elemento"]
        .unique()
        .apply(set)
        .to_dict()
    )
    
    all_elements = set.union(*elements_per_product.values())
    
    incomplete = {
        prod: all_elements - elems
        for prod, elems in elements_per_product.items()
        if elems != all_elements
    }
    
    # Resumen tabular
    summary = []
    for prod, elems in elements_per_product.items():
        summary.append({
            "Producto": prod,
            "Elementos_presentes": len(elems),
            "Elementos_esperados": len(all_elements),
            "Porcentaje_completo": round(100 * len(elems) / len(all_elements), 2)
        })
    df_summary = pd.DataFrame(summary).sort_values("Porcentaje_completo")
    
    return {
        'all_elements': all_elements,
        'incomplete': incomplete,
        'summary': df_summary
    }

def detectar_duplicados_y_limpiar(df_cluster):
    """Detecta y elimina duplicados promediando valores"""
    
    duplicates = df_cluster.duplicated(
        subset=["Área", "Producto", "Elemento", "Año"], 
        keep=False
    )
    
    if duplicates.any():
        log(f"   ⚠️  {duplicates.sum()} filas duplicadas detectadas. Promediando...")
        df_cluster = (
            df_cluster.groupby(["Área", "Producto", "Elemento", "Año"], as_index=False)
            .agg({"Valor": "mean", "Cluster": "first"})
        )
    
    return df_cluster

def crear_indice_completo(df_cluster):
    """Crea índice completo de todas las combinaciones posibles"""
    
    all_areas = df_cluster["Área"].unique()
    all_products = df_cluster["Producto"].unique()
    all_elements = df_cluster["Elemento"].unique()
    all_years = sorted(df_cluster["Año"].unique())
    
    full_index = pd.MultiIndex.from_product(
        [all_areas, all_products, all_elements, all_years],
        names=["Área", "Producto", "Elemento", "Año"]
    )
    
    df_full = (
        df_cluster.set_index(["Área", "Producto", "Elemento", "Año"])
        .reindex(full_index)
        .reset_index()
    )
    
    missing_mask = df_full["Valor"].isna()
    missing = df_full[missing_mask]
    
    stats = {
        'total_combinations': len(df_full),
        'missing_count': missing_mask.sum(),
        'complete_count': len(df_full) - missing_mask.sum(),
        'missing_by_country': missing.groupby("Área").size().to_dict()
    }
    
    return df_full, missing, stats

def crear_mapeos_categoricos(df_cluster):
    """Crea diccionarios de mapeo para convertir categorías a índices"""
    
    area_to_idx = {area: i for i, area in enumerate(df_cluster["Área"].unique())}
    prod_to_idx = {prod: i for i, prod in enumerate(df_cluster["Producto"].unique())}
    elem_to_idx = {elem: i for i, elem in enumerate(df_cluster["Elemento"].unique())}
    year_to_idx = {year: i for i, year in enumerate(sorted(df_cluster["Año"].unique()))}
    
    # Inversos
    idx_to_area = {v: k for k, v in area_to_idx.items()}
    idx_to_prod = {v: k for k, v in prod_to_idx.items()}
    idx_to_elem = {v: k for k, v in elem_to_idx.items()}
    idx_to_year = {v: k for k, v in year_to_idx.items()}
    
    return {
        'area_to_idx': area_to_idx,
        'prod_to_idx': prod_to_idx,
        'elem_to_idx': elem_to_idx,
        'year_to_idx': year_to_idx,
        'idx_to_area': idx_to_area,
        'idx_to_prod': idx_to_prod,
        'idx_to_elem': idx_to_elem,
        'idx_to_year': idx_to_year,
    }

def crear_tensor_4d(df_cluster, mapeos):
    """Crea tensor 4D: (áreas, productos, elementos, años)"""
    
    df_cluster = df_cluster.copy()
    df_cluster["area_idx"] = df_cluster["Área"].map(mapeos['area_to_idx'])
    df_cluster["prod_idx"] = df_cluster["Producto"].map(mapeos['prod_to_idx'])
    df_cluster["elem_idx"] = df_cluster["Elemento"].map(mapeos['elem_to_idx'])
    df_cluster["year_idx"] = df_cluster["Año"].map(mapeos['year_to_idx'])
    
    num_areas = len(mapeos['area_to_idx'])
    num_prods = len(mapeos['prod_to_idx'])
    num_elems = len(mapeos['elem_to_idx'])
    num_years = len(mapeos['year_to_idx'])
    
    tensor = torch.full(
        (num_areas, num_prods, num_elems, num_years),
        float('nan'),
        dtype=torch.float32
    )
    
    for _, row in df_cluster.iterrows():
        a = int(row["area_idx"])
        p = int(row["prod_idx"])
        e = int(row["elem_idx"])
        y = int(row["year_idx"])
        tensor[a, p, e, y] = row["Valor"]
    
    return tensor

def normalizar_tensor(tensor):
    """Normaliza tensor ignorando NaN"""
    
    mask = ~torch.isnan(tensor)
    mean_val = torch.sum(tensor[mask]) / mask.sum()
    std_val = torch.sqrt(torch.sum(((tensor[mask] - mean_val) ** 2)) / mask.sum())
    
    epsilon = 1e-8
    std_val = std_val + epsilon
    
    tensor_norm = (tensor - mean_val) / std_val
    
    stats = {
        'mean': mean_val.item(),
        'std': std_val.item(),
        'nan_count': torch.isnan(tensor_norm).sum().item(),
        'inf_count': torch.isinf(tensor_norm).sum().item()
    }
    
    return tensor_norm, stats

def generar_secuencias(tensor, window=5):
    """Crea secuencias para entrenamiento temporal"""
    
    X, y = [], []
    num_years = tensor.shape[-1]
    
    for t in range(num_years - window):
        X.append(tensor[..., t:t+window])
        y.append(tensor[..., t+window])
    
    return torch.stack(X), torch.stack(y)

# =====================================================================
# NUEVAS FUNCIONES: ANÁLISIS AVANZADOS
# =====================================================================

def analizar_correlacion_temporal(tensor, mapeos, cluster_id, output_path):
    """
    Analiza correlaciones temporales entre países, productos y elementos
    """
    log("   📊 Analizando correlaciones temporales...")
    
    # 1. Correlación entre países (promediando productos y elementos)
    tensor_np = tensor.numpy()
    mask = ~np.isnan(tensor_np)
    
    # Promediar sobre productos y elementos para cada país y año
    paises_series = []
    paises_nombres = []
    
    for i, pais in mapeos['idx_to_area'].items():
        serie_pais = np.nanmean(tensor_np[i, :, :, :], axis=(0, 1))
        if not np.all(np.isnan(serie_pais)):
            paises_series.append(serie_pais)
            paises_nombres.append(pais)
    
    # Calcular matriz de correlación entre países
    n_paises = len(paises_series)
    corr_matrix = np.zeros((n_paises, n_paises))
    
    for i in range(n_paises):
        for j in range(n_paises):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Correlación de Pearson ignorando NaN
                mask_ij = ~np.isnan(paises_series[i]) & ~np.isnan(paises_series[j])
                if mask_ij.sum() > 2:
                    corr, _ = stats.pearsonr(
                        paises_series[i][mask_ij],
                        paises_series[j][mask_ij]
                    )
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = np.nan
    
    # Guardar matriz de correlación
    df_corr = pd.DataFrame(
        corr_matrix,
        index=paises_nombres,
        columns=paises_nombres
    )
    df_corr.to_csv(output_path)
    
    # Crear heatmap
    fig = px.imshow(
        corr_matrix,
        x=paises_nombres,
        y=paises_nombres,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f'Cluster {cluster_id}: Correlación temporal entre países',
        labels={'color': 'Correlación'}
    )
    fig.update_xaxes(tickangle=45)
    fig.write_html(output_path.replace('.csv', '.html'))
    
    # Estadísticas
    corr_flat = corr_matrix[~np.isnan(corr_matrix)]
    corr_flat = corr_flat[corr_flat != 1.0]  # Excluir diagonal
    
    stats_corr = {
        'correlacion_media': np.mean(corr_flat),
        'correlacion_mediana': np.median(corr_flat),
        'correlacion_std': np.std(corr_flat),
        'correlacion_min': np.min(corr_flat),
        'correlacion_max': np.max(corr_flat),
        'n_pares_analizados': len(corr_flat)
    }
    
    log(f"      - Correlación media entre países: **{stats_corr['correlacion_media']:.3f}**")
    log(f"      - Rango: [{stats_corr['correlacion_min']:.3f}, {stats_corr['correlacion_max']:.3f}]")
    
    return stats_corr

def detectar_outliers(tensor, mapeos, cluster_id, output_path):
    """
    Detecta outliers usando Z-score y IQR
    """
    log("   🔍 Detectando outliers...")
    
    tensor_np = tensor.numpy()
    mask = ~np.isnan(tensor_np)
    valores_validos = tensor_np[mask]
    
    if len(valores_validos) == 0:
        log("      ⚠️  No hay valores válidos para detectar outliers")
        return {}
    
    # Método 1: Z-score (valores con |z| > 3)
    z_scores = np.abs(stats.zscore(valores_validos, nan_policy='omit'))
    outliers_zscore = valores_validos[z_scores > 3]
    
    # Método 2: IQR (Interquartile Range)
    Q1 = np.percentile(valores_validos, 25)
    Q3 = np.percentile(valores_validos, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = valores_validos[(valores_validos < lower_bound) | (valores_validos > upper_bound)]
    
    # Encontrar índices de outliers extremos (z > 4)
    indices_outliers = np.where(mask)
    z_scores_full = np.abs((tensor_np - np.nanmean(tensor_np)) / np.nanstd(tensor_np))
    outliers_extremos = z_scores_full > 4
    
    # Convertir índices a DataFrame
    outliers_data = []
    for a, p, e, y in zip(*np.where(outliers_extremos)):
        outliers_data.append({
            'País': mapeos['idx_to_area'][a],
            'Producto': mapeos['idx_to_prod'][p],
            'Elemento': mapeos['idx_to_elem'][e],
            'Año': mapeos['idx_to_year'][y],
            'Valor': tensor_np[a, p, e, y],
            'Z_score': z_scores_full[a, p, e, y]
        })
    
    df_outliers = pd.DataFrame(outliers_data)
    if len(df_outliers) > 0:
        df_outliers = df_outliers.sort_values('Z_score', ascending=False)
        df_outliers.to_csv(output_path, index=False)
    
    stats_outliers = {
        'n_outliers_zscore': len(outliers_zscore),
        'n_outliers_iqr': len(outliers_iqr),
        'n_outliers_extremos': len(outliers_data),
        'pct_outliers': 100 * len(outliers_zscore) / len(valores_validos),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    log(f"      - Outliers (Z-score > 3): **{stats_outliers['n_outliers_zscore']}** ({stats_outliers['pct_outliers']:.2f}%)")
    log(f"      - Outliers (IQR): **{stats_outliers['n_outliers_iqr']}**")
    log(f"      - Outliers extremos (Z > 4): **{stats_outliers['n_outliers_extremos']}**")
    
    return stats_outliers

def analizar_calidad_datos(tensor, df_cluster, mapeos, cluster_id, output_path):
    """
    Análisis exhaustivo de calidad de datos
    """
    log("   ✅ Analizando calidad de datos...")
    
    metricas = {}
    
    # 1. Completitud general
    total_celdas = tensor.numel()
    celdas_validas = (~torch.isnan(tensor)).sum().item()
    metricas['completitud_%'] = 100 * celdas_validas / total_celdas
    
    # 2. Completitud por dimensión
    metricas['completitud_por_pais'] = {}
    for i, pais in mapeos['idx_to_area'].items():
        tensor_pais = tensor[i, :, :, :]
        completitud = 100 * (~torch.isnan(tensor_pais)).sum().item() / tensor_pais.numel()
        metricas['completitud_por_pais'][pais] = completitud
    
    # 3. Consistencia temporal (gaps consecutivos)
    gaps_consecutivos = []
    for i in range(tensor.shape[0]):  # Por país
        for p in range(tensor.shape[1]):  # Por producto
            for e in range(tensor.shape[2]):  # Por elemento
                serie = tensor[i, p, e, :].numpy()
                # Detectar secuencias de NaN
                is_nan = np.isnan(serie)
                gaps = np.diff(np.concatenate(([False], is_nan, [False])).astype(int))
                gap_starts = np.where(gaps == 1)[0]
                gap_ends = np.where(gaps == -1)[0]
                gap_lengths = gap_ends - gap_starts
                if len(gap_lengths) > 0:
                    gaps_consecutivos.extend(gap_lengths)
    
    if gaps_consecutivos:
        metricas['gap_medio'] = np.mean(gaps_consecutivos)
        metricas['gap_maximo'] = np.max(gaps_consecutivos)
        metricas['n_gaps'] = len(gaps_consecutivos)
    else:
        metricas['gap_medio'] = 0
        metricas['gap_maximo'] = 0
        metricas['n_gaps'] = 0
    
    # 4. Variabilidad temporal (coeficiente de variación por serie)
    cvs = []
    for i in range(tensor.shape[0]):
        for p in range(tensor.shape[1]):
            for e in range(tensor.shape[2]):
                serie = tensor[i, p, e, :].numpy()
                serie_valida = serie[~np.isnan(serie)]
                if len(serie_valida) > 2:
                    cv = np.std(serie_valida) / np.mean(serie_valida) if np.mean(serie_valida) != 0 else 0
                    cvs.append(abs(cv))
    
    metricas['cv_medio'] = np.mean(cvs) if cvs else 0
    metricas['cv_std'] = np.std(cvs) if cvs else 0
    
    # 5. Densidad temporal (años con datos vs años totales)
    años_con_datos_por_pais = {}
    for i, pais in mapeos['idx_to_area'].items():
        tensor_pais = tensor[i, :, :, :]
        años_con_datos = (~torch.isnan(tensor_pais)).any(dim=(0, 1)).sum().item()
        años_con_datos_por_pais[pais] = años_con_datos
    
    metricas['densidad_temporal_media'] = np.mean(list(años_con_datos_por_pais.values()))
    
    # Guardar métricas
    df_metricas = pd.DataFrame([metricas])
    df_metricas.to_csv(output_path, index=False)
    
    log(f"      - Completitud general: **{metricas['completitud_%']:.2f}%**")
    log(f"      - Gap promedio: **{metricas['gap_medio']:.1f}** años")
    log(f"      - Coeficiente de variación medio: **{metricas['cv_medio']:.3f}**")
    
    return metricas

def analizar_pca_temporal(tensor, mapeos, cluster_id, output_path):
    """
    PCA sobre series temporales para encontrar patrones principales
    """
    log("   🎯 Realizando análisis PCA...")
    
    # Preparar datos: cada fila es un país, columnas son (producto, elemento, año)
    tensor_np = tensor.numpy()
    n_areas = tensor.shape[0]
    
    # Aplanar a 2D: (países, features_temporales)
    X = tensor_np.reshape(n_areas, -1)
    
    # Eliminar columnas con todos NaN
    mask_columnas = ~np.all(np.isnan(X), axis=0)
    X_clean = X[:, mask_columnas]
    
    # Imputar NaN con media (necesario para PCA)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_clean)
    
    # PCA
    n_components = min(5, X_imputed.shape[0], X_imputed.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_imputed)
    
    # Guardar resultados
    df_pca = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=[mapeos['idx_to_area'][i] for i in range(n_areas)]
    )
    df_pca['País'] = df_pca.index
    df_pca.to_csv(output_path, index=False)
    
    # Visualización
    fig = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        text='País',
        title=f'Cluster {cluster_id}: PCA de patrones temporales',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'}
    )
    fig.update_traces(textposition='top center')
    fig.write_html(output_path.replace('.csv', '.html'))
    
    log(f"      - Varianza explicada (PC1): **{pca.explained_variance_ratio_[0]*100:.1f}%**")
    log(f"      - Varianza explicada acumulada (3 PCs): **{pca.explained_variance_ratio_[:3].sum()*100:.1f}%**")
    
    return {
        'varianza_explicada': pca.explained_variance_ratio_.tolist(),
        'n_components': n_components
    }

# =====================================================================
# FUNCIONES DE VISUALIZACIÓN
# =====================================================================

def generar_heatmap_nan(tensor, mapeos, cluster_id, output_path):
    """Genera heatmap interactivo de valores faltantes por año"""
    
    nan_per_year = torch.isnan(tensor).sum(dim=2)
    num_areas, num_prods, num_years = nan_per_year.shape
    
    data = []
    for a in range(num_areas):
        for p in range(num_prods):
            for y in range(num_years):
                data.append({
                    "Área": mapeos['idx_to_area'][a],
                    "Producto": mapeos['idx_to_prod'][p],
                    "Año": mapeos['idx_to_year'][y],
                    "NaN_count": int(nan_per_year[a, p, y])
                })
    
    df_plot = pd.DataFrame(data)
    
    fig = px.density_heatmap(
        df_plot,
        x="Producto",
        y="Área",
        z="NaN_count",
        animation_frame="Año",
        color_continuous_scale="Reds",
        labels={"NaN_count": "Valores faltantes"},
        title=f"Cluster {cluster_id}: Valores faltantes por año"
    )
    
    fig.update_layout(height=800, width=1200, xaxis_tickangle=-45)
    fig.write_html(output_path)

def generar_estadisticas_completitud(tensor, cluster_id, output_path):
    """Genera estadísticas de completitud por dimensión"""
    
    stats = {}
    
    # Por área
    nan_por_area = torch.isnan(tensor).sum(dim=(1,2,3))
    total_por_area = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
    completitud_area = 100 * (1 - nan_por_area / total_por_area)
    
    # Por producto
    nan_por_prod = torch.isnan(tensor).sum(dim=(0,2,3))
    total_por_prod = tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
    completitud_prod = 100 * (1 - nan_por_prod / total_por_prod)
    
    # Por elemento
    nan_por_elem = torch.isnan(tensor).sum(dim=(0,1,3))
    total_por_elem = tensor.shape[0] * tensor.shape[1] * tensor.shape[3]
    completitud_elem = 100 * (1 - nan_por_elem / total_por_elem)
    
    # Por año
    nan_por_year = torch.isnan(tensor).sum(dim=(0,1,2))
    total_por_year = tensor.shape[0] * tensor.shape[1] * tensor.shape[2]
    completitud_year = 100 * (1 - nan_por_year / total_por_year)
    
    stats = {
        'completitud_area': completitud_area.numpy(),
        'completitud_producto': completitud_prod.numpy(),
        'completitud_elemento': completitud_elem.numpy(),
        'completitud_año': completitud_year.numpy(),
    }
    
    pd.DataFrame({
        'Dimensión': ['Área', 'Producto', 'Elemento', 'Año'],
        'Completitud_promedio_%': [
            stats['completitud_area'].mean(),
            stats['completitud_producto'].mean(),
            stats['completitud_elemento'].mean(),
            stats['completitud_año'].mean()
        ]
    }).to_csv(output_path, index=False)
    
    return stats

# =====================================================================
# PROCESAMIENTO POR CLUSTER
# =====================================================================

def procesar_cluster(df, cluster_id):
    """Procesa un cluster completo con todos los análisis"""
    
    log("")
    log("=" * 80)
    log(f"## CLUSTER {cluster_id}")
    log("=" * 80)
    log("")
    
    # Filtrar datos del cluster
    df_cluster = df[df['Cluster'] == cluster_id].copy()
    
    if len(df_cluster) == 0:
        log(f"⚠️  Cluster {cluster_id} vacío, omitiendo...")
        return None
    
    log(f"### Información general")
    log("")
    log(f"- **Registros**: {len(df_cluster):,}")
    log(f"- **Países**: {df_cluster['Área'].nunique()}")
    log(f"- **Productos**: {df_cluster['Producto'].nunique()}")
    log(f"- **Elementos**: {df_cluster['Elemento'].nunique()}")
    log(f"- **Período**: {df_cluster['Año'].min():.0f} - {df_cluster['Año'].max():.0f}")
    log("")
    
    # Crear directorio
    cluster_dir = os.path.join(RUTA_SALIDA, f'cluster_{cluster_id}')
    Path(cluster_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Análisis de completitud
    log("### [1/10] Análisis de completitud por producto")
    log("")
    completitud = analizar_completitud_cluster(df_cluster)
    completitud['summary'].to_csv(
        os.path.join(cluster_dir, 'completitud_productos.csv'),
        index=False
    )
    
    # Mostrar top 5 productos menos completos
    top_incompletos = completitud['summary'].head(5)
    log("**Productos con menor completitud:**")
    log("")
    for _, row in top_incompletos.iterrows():
        log(f"- {row['Producto']}: {row['Porcentaje_completo']:.1f}%")
    log("")
    
    # 2. Limpiar duplicados
    log("### [2/10] Limpieza de duplicados")
    log("")
    df_cluster = detectar_duplicados_y_limpiar(df_cluster)
    log("")
    
    # 3. Índice completo
    log("### [3/10] Creación de índice completo")
    log("")
    df_full, missing, stats_missing = crear_indice_completo(df_cluster)
    
    log(f"- **Combinaciones totales**: {stats_missing['total_combinations']:,}")
    log(f"- **Valores faltantes**: {stats_missing['missing_count']:,}")
    log(f"- **Completitud**: {100 * stats_missing['complete_count'] / stats_missing['total_combinations']:.2f}%")
    log("")
    
    missing.to_csv(
        os.path.join(cluster_dir, 'valores_faltantes.csv'),
        index=False
    )
    
    # 4. Crear mapeos
    log("### [4/10] Creación de mapeos categóricos")
    log("")
    mapeos = crear_mapeos_categoricos(df_cluster)
    log(f"- Mapeos creados para {len(mapeos['area_to_idx'])} áreas")
    log("")
    
    # 5. Crear tensor 4D
    log("### [5/10] Creación de tensor 4D")
    log("")
    tensor = crear_tensor_4d(df_cluster, mapeos)
    log(f"- **Shape**: {tensor.shape}")
    log(f"- **Memoria**: {tensor.element_size() * tensor.nelement() / 1024 / 1024:.2f} MB")
    log("")
    
    # 6. Normalizar
    log("### [6/10] Normalización del tensor")
    log("")
    tensor_norm, norm_stats = normalizar_tensor(tensor)
    log(f"- **Media**: {norm_stats['mean']:.4f}")
    log(f"- **Desviación estándar**: {norm_stats['std']:.4f}")
    log(f"- **NaN tras normalización**: {norm_stats['nan_count']}")
    log("")
    
    # 7. Generar secuencias
    log("### [7/10] Generación de secuencias temporales")
    log("")
    X, y = generar_secuencias(tensor_norm, window=WINDOW_SIZE)
    log(f"- **Shape X**: {X.shape}")
    log(f"- **Shape y**: {y.shape}")
    log(f"- **Secuencias generadas**: {len(X)}")
    log("")
    
    # 8. NUEVO: Análisis de correlación temporal
    log("### [8/10] Análisis de correlación temporal")
    log("")
    stats_corr = analizar_correlacion_temporal(
        tensor,
        mapeos,
        cluster_id,
        os.path.join(cluster_dir, 'correlacion_temporal.csv')
    )
    log("")
    
    # 9. NUEVO: Detección de outliers
    log("### [9/10] Detección de outliers")
    log("")
    stats_outliers = detectar_outliers(
        tensor,
        mapeos,
        cluster_id,
        os.path.join(cluster_dir, 'outliers.csv')
    )
    log("")
    
    # 10. NUEVO: Análisis de calidad de datos
    log("### [10/10] Análisis de calidad de datos")
    log("")
    metricas_calidad = analizar_calidad_datos(
        tensor,
        df_cluster,
        mapeos,
        cluster_id,
        os.path.join(cluster_dir, 'metricas_calidad.csv')
    )
    log("")
    
    # BONUS: PCA
    log("### Análisis PCA (Bonus)")
    log("")
    stats_pca = analizar_pca_temporal(
        tensor,
        mapeos,
        cluster_id,
        os.path.join(cluster_dir, 'pca_temporal.csv')
    )
    log("")
    
    # Guardar tensores
    torch.save({
        'tensor_original': tensor,
        'tensor_normalizado': tensor_norm,
        'X': X,
        'y': y,
        'mapeos': mapeos,
        'norm_stats': norm_stats
    }, os.path.join(cluster_dir, 'tensors.pt'))
    
    # Generar visualizaciones
    log("### Generación de visualizaciones")
    log("")
    log("   📊 Generando gráficos...")
    
    generar_heatmap_nan(
        tensor, 
        mapeos, 
        cluster_id,
        os.path.join(cluster_dir, 'heatmap_nan.html')
    )
    
    generar_estadisticas_completitud(
        tensor,
        cluster_id,
        os.path.join(cluster_dir, 'estadisticas_completitud.csv')
    )
    
    log(f"   ✓ Heatmap guardado: `heatmap_nan.html`")
    log(f"   ✓ Estadísticas guardadas: `estadisticas_completitud.csv`")
    log("")
    
    # Guardar metadatos
    metadata = {
        'cluster_id': cluster_id,
        'n_registros': len(df_cluster),
        'n_paises': df_cluster['Área'].nunique(),
        'n_productos': df_cluster['Producto'].nunique(),
        'n_elementos': df_cluster['Elemento'].nunique(),
        'año_min': int(df_cluster['Año'].min()),
        'año_max': int(df_cluster['Año'].max()),
        'tensor_shape': list(tensor.shape),
        'completitud_%': 100 * stats_missing['complete_count'] / stats_missing['total_combinations'],
        'mean_normalized': norm_stats['mean'],
        'std_normalized': norm_stats['std'],
        'window_size': WINDOW_SIZE,
        'n_sequences': len(X),
        # Nuevas métricas
        'correlacion_media_paises': stats_corr.get('correlacion_media', None),
        'n_outliers': stats_outliers.get('n_outliers_zscore', None),
        'cv_medio': metricas_calidad.get('cv_medio', None),
        'gap_medio': metricas_calidad.get('gap_medio', None),
        'varianza_pc1': stats_pca['varianza_explicada'][0] if stats_pca['varianza_explicada'] else None,
    }
    
    pd.DataFrame([metadata]).to_csv(
        os.path.join(cluster_dir, 'metadata.csv'),
        index=False
    )
    
    log(f"✅ **Cluster {cluster_id} procesado completamente**")
    log("")
    
    return metadata

# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================

def main():
    inicio = datetime.now()
    
    # Cargar datos
    df = cargar_y_preparar_datos()
    
    # Obtener clusters únicos
    clusters = sorted(df['Cluster'].dropna().unique())
    log(f"## [2/6] Clusters detectados")
    log("")
    log(f"**Total de clusters**: {len(clusters)}")
    log(f"**IDs**: {clusters}")
    log("")
    
    # Procesar cada cluster
    log(f"## [3/6] Procesamiento de clusters")
    log("")
    resultados = []
    
    for cluster_id in clusters:
        metadata = procesar_cluster(df, int(cluster_id))
        if metadata:
            resultados.append(metadata)
    
    # Resumen consolidado
    log("")
    log("=" * 80)
    log("## [4/6] Resumen consolidado")
    log("=" * 80)
    log("")
    
    df_resumen = pd.DataFrame(resultados)
    df_resumen.to_csv(
        os.path.join(RUTA_SALIDA, 'resumen_todos_clusters.csv'),
        index=False
    )
    
    log("### Estadísticas por cluster")
    log("")
    log("| Cluster | Países | Productos | Completitud (%) | Corr. Media | Outliers |")
    log("|---------|--------|-----------|-----------------|-------------|----------|")
    for _, row in df_resumen.iterrows():
        log(f"| {int(row['cluster_id'])} | {row['n_paises']} | {row['n_productos']} | "
            f"{row['completitud_%']:.2f} | {row['correlacion_media_paises']:.3f} | {row['n_outliers']} |")
    log("")
    
    # Visualización comparativa
    log("## [5/6] Visualizaciones comparativas")
    log("")
    
    # Gráfico de barras
    fig1 = px.bar(
        df_resumen,
        x='cluster_id',
        y=['n_paises', 'n_productos', 'n_elementos'],
        barmode='group',
        title='Comparación de dimensiones por cluster',
        labels={'value': 'Cantidad', 'variable': 'Dimensión', 'cluster_id': 'Cluster'}
    )
    fig1.write_html(os.path.join(RUTA_SALIDA, 'comparacion_dimensiones.html'))
    
    # Gráfico de completitud
    fig2 = px.bar(
        df_resumen,
        x='cluster_id',
        y='completitud_%',
        title='Completitud de datos por cluster',
        labels={'completitud_%': 'Completitud (%)', 'cluster_id': 'Cluster'},
        color='completitud_%',
        color_continuous_scale='RdYlGn'
    )
    fig2.write_html(os.path.join(RUTA_SALIDA, 'comparacion_completitud.html'))
    
    # Gráfico de calidad (correlación vs completitud)
    fig3 = px.scatter(
        df_resumen,
        x='completitud_%',
        y='correlacion_media_paises',
        size='n_paises',
        text='cluster_id',
        title='Calidad de datos: Completitud vs Correlación',
        labels={
            'completitud_%': 'Completitud (%)',
            'correlacion_media_paises': 'Correlación promedio entre países',
            'n_paises': 'Número de países'
        }
    )
    fig3.update_traces(textposition='top center')
    fig3.write_html(os.path.join(RUTA_SALIDA, 'calidad_datos.html'))
    
    log("✓ Visualizaciones guardadas:")
    log("  - `comparacion_dimensiones.html`")
    log("  - `comparacion_completitud.html`")
    log("  - `calidad_datos.html`")
    log("")
    
    # Resumen final
    fin = datetime.now()
    duracion = (fin - inicio).total_seconds()
    
    log("=" * 80)
    log("## [6/6] Resumen de ejecución")
    log("=" * 80)
    log("")
    log(f"- **Inicio**: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"- **Fin**: {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"- **Duración**: {duracion:.2f} segundos ({duracion/60:.2f} minutos)")
    log(f"- **Clusters procesados**: {len(resultados)}/{len(clusters)}")
    log(f"- **Total de países analizados**: {df['Área'].nunique()}")
    log(f"- **Total de registros procesados**: {len(df):,}")
    log("")
    
    log("### Archivos generados")
    log("")
    log("**Por cada cluster:**")
    log("- `tensors.pt` - Tensores PyTorch")
    log("- `completitud_productos.csv` - Completitud por producto")
    log("- `valores_faltantes.csv` - Combinaciones faltantes")
    log("- `correlacion_temporal.csv` / `.html` - Matriz de correlación entre países")
    log("- `outliers.csv` - Valores atípicos detectados")
    log("- `metricas_calidad.csv` - Métricas de calidad de datos")
    log("- `pca_temporal.csv` / `.html` - Análisis de componentes principales")
    log("- `heatmap_nan.html` - Heatmap animado de valores faltantes")
    log("- `estadisticas_completitud.csv` - Completitud por dimensión")
    log("- `metadata.csv` - Metadatos del cluster")
    log("")
    log("**Archivos globales:**")
    log("- `resumen_todos_clusters.csv` - Comparación de todos los clusters")
    log("- `comparacion_dimensiones.html` - Gráfico comparativo de dimensiones")
    log("- `comparacion_completitud.html` - Gráfico de completitud")
    log("- `calidad_datos.html` - Scatter plot de calidad")
    log(f"- `{os.path.basename(INFORME_MD)}` - Este informe")
    log("")
    
    log("=" * 80)
    log("✅ **ANÁLISIS COMPLETADO EXITOSAMENTE**")
    log("=" * 80)
    
    # Guardar log
    guardar_log()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("")
        log("=" * 80)
        log("❌ **ERROR EN LA EJECUCIÓN**")
        log("=" * 80)
        log("")
        log(f"```")
        log(f"{type(e).__name__}: {str(e)}")
        log(f"```")
        log("")
        guardar_log()
        raise