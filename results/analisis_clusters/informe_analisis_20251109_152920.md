================================================================================
# ANÁLISIS AUTOMATIZADO POR CLUSTER
================================================================================

**Fecha**: 2025-11-09 15:29:20

## [1/6] Cargando datos...

- ✓ Datos cargados: **2,375,400** registros
- ✓ Información de clusters: **280** países
- ✓ Datos preparados: **2,375,400** registros válidos
- ✓ Productos únicos después del mapeo: **16**

## [2/6] Clusters detectados

**Total de clusters**: 5
**IDs**: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]

## [3/6] Procesamiento de clusters


================================================================================
## CLUSTER 0
================================================================================

### Información general

- **Registros**: 220,841
- **Países**: 30
- **Productos**: 16
- **Elementos**: 9
- **Período**: 1990 - 2022

### [1/10] Análisis de completitud por producto

**Productos con menor completitud:**

- Cultivo del arroz: 33.3%
- Fermentación entérica: 33.3%
- Fertilizantes sintéticos: 44.4%
- Emissiones derivadas del sector ganadero: 55.6%
- Estiércol depositado en las pasturas: 55.6%

### [2/10] Limpieza de duplicados

   ⚠️  186183 filas duplicadas detectadas. Promediando...

### [3/10] Creación de índice completo

- **Combinaciones totales**: 142,560
- **Valores faltantes**: 61,363
- **Completitud**: 56.96%

### [4/10] Creación de mapeos categóricos

- Mapeos creados para 30 áreas

### [5/10] Creación de tensor 4D

- **Shape**: torch.Size([30, 16, 9, 33])
- **Memoria**: 0.54 MB

### [6/10] Normalización del tensor

- **Media**: 3520.3823
- **Desviación estándar**: 29429.8945
- **NaN tras normalización**: 61363

### [7/10] Generación de secuencias temporales

- **Shape X**: torch.Size([28, 30, 16, 9, 5])
- **Shape y**: torch.Size([28, 30, 16, 9])
- **Secuencias generadas**: 28

### [8/10] Análisis de correlación temporal

   📊 Analizando correlaciones temporales...
      - Correlación media entre países: **0.134**
      - Rango: [-0.962, 0.977]

### [9/10] Detección de outliers

   🔍 Detectando outliers...
      - Outliers (Z-score > 3): **478** (0.59%)
      - Outliers (IQR): **13335**
      - Outliers extremos (Z > 4): **325**

### [10/10] Análisis de calidad de datos

   ✅ Analizando calidad de datos...
      - Completitud general: **56.96%**
      - Gap promedio: **14.3** años
      - Coeficiente de variación medio: **0.300**

### Análisis PCA (Bonus)

   🎯 Realizando análisis PCA...
      - Varianza explicada (PC1): **96.2%**
      - Varianza explicada acumulada (3 PCs): **99.3%**

### Generación de visualizaciones

   📊 Generando gráficos...
   ✓ Heatmap guardado: `heatmap_nan.html`
   ✓ Estadísticas guardadas: `estadisticas_completitud.csv`

✅ **Cluster 0 procesado completamente**


================================================================================
## CLUSTER 1
================================================================================

### Información general

- **Registros**: 1,870,737
- **Países**: 193
- **Productos**: 16
- **Elementos**: 9
- **Período**: 1961 - 2022

### [1/10] Análisis de completitud por producto

**Productos con menor completitud:**

- Cultivo del arroz: 33.3%
- Fermentación entérica: 33.3%
- Fertilizantes sintéticos: 44.4%
- Emissiones derivadas del sector ganadero: 55.6%
- Estiércol depositado en las pasturas: 55.6%

### [2/10] Limpieza de duplicados

   ⚠️  1194944 filas duplicadas detectadas. Promediando...

### [3/10] Creación de índice completo

- **Combinaciones totales**: 1,723,104
- **Valores faltantes**: 751,862
- **Completitud**: 56.37%

### [4/10] Creación de mapeos categóricos

- Mapeos creados para 193 áreas

### [5/10] Creación de tensor 4D

- **Shape**: torch.Size([193, 16, 9, 62])
- **Memoria**: 6.57 MB

### [6/10] Normalización del tensor

- **Media**: 26757.5664
- **Desviación estándar**: 267295.7812
- **NaN tras normalización**: 751862

### [7/10] Generación de secuencias temporales

- **Shape X**: torch.Size([57, 193, 16, 9, 5])
- **Shape y**: torch.Size([57, 193, 16, 9])
- **Secuencias generadas**: 57

### [8/10] Análisis de correlación temporal

   📊 Analizando correlaciones temporales...
      - Correlación media entre países: **0.560**
      - Rango: [-0.908, 1.000]

### [9/10] Detección de outliers

   🔍 Detectando outliers...
      - Outliers (Z-score > 3): **5672** (0.58%)
      - Outliers (IQR): **185318**
      - Outliers extremos (Z > 4): **4103**

### [10/10] Análisis de calidad de datos

   ✅ Analizando calidad de datos...
      - Completitud general: **56.37%**
      - Gap promedio: **50.3** años
      - Coeficiente de variación medio: **0.464**

### Análisis PCA (Bonus)

   🎯 Realizando análisis PCA...
      - Varianza explicada (PC1): **89.3%**
      - Varianza explicada acumulada (3 PCs): **99.1%**

### Generación de visualizaciones

   📊 Generando gráficos...
   ✓ Heatmap guardado: `heatmap_nan.html`
   ✓ Estadísticas guardadas: `estadisticas_completitud.csv`

✅ **Cluster 1 procesado completamente**


================================================================================
## CLUSTER 2
================================================================================

### Información general

- **Registros**: 179,279
- **Países**: 29
- **Productos**: 16
- **Elementos**: 9
- **Período**: 1961 - 2022

### [1/10] Análisis de completitud por producto

**Productos con menor completitud:**

- Cultivo del arroz: 33.3%
- Fermentación entérica: 33.3%
- Fertilizantes sintéticos: 44.4%
- Emissiones derivadas del sector ganadero: 55.6%
- Estiércol depositado en las pasturas: 55.6%

### [2/10] Limpieza de duplicados

   ⚠️  119662 filas duplicadas detectadas. Promediando...

### [3/10] Creación de índice completo

- **Combinaciones totales**: 258,912
- **Valores faltantes**: 170,223
- **Completitud**: 34.25%

### [4/10] Creación de mapeos categóricos

- Mapeos creados para 29 áreas

### [5/10] Creación de tensor 4D

- **Shape**: torch.Size([29, 16, 9, 62])
- **Memoria**: 0.99 MB

### [6/10] Normalización del tensor

- **Media**: 23.3240
- **Desviación estándar**: 244.6646
- **NaN tras normalización**: 170223

### [7/10] Generación de secuencias temporales

- **Shape X**: torch.Size([57, 29, 16, 9, 5])
- **Shape y**: torch.Size([57, 29, 16, 9])
- **Secuencias generadas**: 57

### [8/10] Análisis de correlación temporal

   📊 Analizando correlaciones temporales...
      - Correlación media entre países: **0.091**
      - Rango: [-0.958, 0.986]

### [9/10] Detección de outliers

   🔍 Detectando outliers...
      - Outliers (Z-score > 3): **677** (0.76%)
      - Outliers (IQR): **16961**
      - Outliers extremos (Z > 4): **402**

### [10/10] Análisis de calidad de datos

   ✅ Analizando calidad de datos...
      - Completitud general: **34.25%**
      - Gap promedio: **47.6** años
      - Coeficiente de variación medio: **0.529**

### Análisis PCA (Bonus)

   🎯 Realizando análisis PCA...
      - Varianza explicada (PC1): **78.9%**
      - Varianza explicada acumulada (3 PCs): **97.8%**

### Generación de visualizaciones

   📊 Generando gráficos...
   ✓ Heatmap guardado: `heatmap_nan.html`
   ✓ Estadísticas guardadas: `estadisticas_completitud.csv`

✅ **Cluster 2 procesado completamente**


================================================================================
## CLUSTER 3
================================================================================

### Información general

- **Registros**: 89,963
- **Países**: 22
- **Productos**: 16
- **Elementos**: 9
- **Período**: 1990 - 2022

### [1/10] Análisis de completitud por producto

**Productos con menor completitud:**

- Cultivo del arroz: 33.3%
- Fermentación entérica: 33.3%
- Fertilizantes sintéticos: 44.4%
- Emissiones derivadas del sector ganadero: 55.6%
- Estiércol depositado en las pasturas: 55.6%

### [2/10] Limpieza de duplicados

   ⚠️  69574 filas duplicadas detectadas. Promediando...

### [3/10] Creación de índice completo

- **Combinaciones totales**: 104,544
- **Valores faltantes**: 68,421
- **Completitud**: 34.55%

### [4/10] Creación de mapeos categóricos

- Mapeos creados para 22 áreas

### [5/10] Creación de tensor 4D

- **Shape**: torch.Size([22, 16, 9, 33])
- **Memoria**: 0.40 MB

### [6/10] Normalización del tensor

- **Media**: 406.8466
- **Desviación estándar**: 3634.0312
- **NaN tras normalización**: 68421

### [7/10] Generación de secuencias temporales

- **Shape X**: torch.Size([28, 22, 16, 9, 5])
- **Shape y**: torch.Size([28, 22, 16, 9])
- **Secuencias generadas**: 28

### [8/10] Análisis de correlación temporal

   📊 Analizando correlaciones temporales...
      - Correlación media entre países: **-0.003**
      - Rango: [-0.995, 0.982]

### [9/10] Detección de outliers

   🔍 Detectando outliers...
      - Outliers (Z-score > 3): **339** (0.94%)
      - Outliers (IQR): **6640**
      - Outliers extremos (Z > 4): **315**

### [10/10] Análisis de calidad de datos

   ✅ Analizando calidad de datos...
      - Completitud general: **34.55%**
      - Gap promedio: **24.5** años
      - Coeficiente de variación medio: **0.345**

### Análisis PCA (Bonus)

   🎯 Realizando análisis PCA...
      - Varianza explicada (PC1): **80.5%**
      - Varianza explicada acumulada (3 PCs): **99.4%**

### Generación de visualizaciones

   📊 Generando gráficos...
   ✓ Heatmap guardado: `heatmap_nan.html`
   ✓ Estadísticas guardadas: `estadisticas_completitud.csv`

✅ **Cluster 3 procesado completamente**


================================================================================
## CLUSTER 4
================================================================================

### Información general

- **Registros**: 14,580
- **Países**: 6
- **Productos**: 16
- **Elementos**: 9
- **Período**: 1961 - 1999

### [1/10] Análisis de completitud por producto

**Productos con menor completitud:**

- Cultivo del arroz: 33.3%
- Fermentación entérica: 33.3%
- Fertilizantes sintéticos: 44.4%
- Emissiones derivadas del sector ganadero: 55.6%
- Estiércol depositado en las pasturas: 55.6%

### [2/10] Limpieza de duplicados

   ⚠️  3078 filas duplicadas detectadas. Promediando...

### [3/10] Creación de índice completo

- **Combinaciones totales**: 33,696
- **Valores faltantes**: 21,438
- **Completitud**: 36.38%

### [4/10] Creación de mapeos categóricos

- Mapeos creados para 6 áreas

### [5/10] Creación de tensor 4D

- **Shape**: torch.Size([6, 16, 9, 39])
- **Memoria**: 0.13 MB

### [6/10] Normalización del tensor

- **Media**: 19138.9043
- **Desviación estándar**: 157996.3438
- **NaN tras normalización**: 21438

### [7/10] Generación de secuencias temporales

- **Shape X**: torch.Size([34, 6, 16, 9, 5])
- **Shape y**: torch.Size([34, 6, 16, 9])
- **Secuencias generadas**: 34

### [8/10] Análisis de correlación temporal

   📊 Analizando correlaciones temporales...
      - Correlación media entre países: **0.704**
      - Rango: [0.295, 0.947]

### [9/10] Detección de outliers

   🔍 Detectando outliers...
      - Outliers (Z-score > 3): **39** (0.32%)
      - Outliers (IQR): **2327**
      - Outliers extremos (Z > 4): **37**

### [10/10] Análisis de calidad de datos

   ✅ Analizando calidad de datos...
      - Completitud general: **36.38%**
      - Gap promedio: **23.6** años
      - Coeficiente de variación medio: **0.263**

### Análisis PCA (Bonus)

   🎯 Realizando análisis PCA...
      - Varianza explicada (PC1): **99.4%**
      - Varianza explicada acumulada (3 PCs): **100.0%**

### Generación de visualizaciones

   📊 Generando gráficos...
   ✓ Heatmap guardado: `heatmap_nan.html`
   ✓ Estadísticas guardadas: `estadisticas_completitud.csv`

✅ **Cluster 4 procesado completamente**


================================================================================
## [4/6] Resumen consolidado
================================================================================

### Estadísticas por cluster

| Cluster | Países | Productos | Completitud (%) | Corr. Media | Outliers |
|---------|--------|-----------|-----------------|-------------|----------|
| 0 | 30 | 16 | 56.96 | 0.134 | 478 |
| 1 | 193 | 16 | 56.37 | 0.560 | 5672 |
| 2 | 29 | 16 | 34.25 | 0.091 | 677 |
| 3 | 22 | 16 | 34.55 | -0.003 | 339 |
| 4 | 6 | 16 | 36.38 | 0.704 | 39 |

## [5/6] Visualizaciones comparativas

✓ Visualizaciones guardadas:
  - `comparacion_dimensiones.html`
  - `comparacion_completitud.html`
  - `calidad_datos.html`

================================================================================
## [6/6] Resumen de ejecución
================================================================================

- **Inicio**: 2025-11-09 15:29:20
- **Fin**: 2025-11-09 15:31:51
- **Duración**: 151.18 segundos (2.52 minutos)
- **Clusters procesados**: 5/5
- **Total de países analizados**: 280
- **Total de registros procesados**: 2,375,400

### Archivos generados

**Por cada cluster:**
- `tensors.pt` - Tensores PyTorch
- `completitud_productos.csv` - Completitud por producto
- `valores_faltantes.csv` - Combinaciones faltantes
- `correlacion_temporal.csv` / `.html` - Matriz de correlación entre países
- `outliers.csv` - Valores atípicos detectados
- `metricas_calidad.csv` - Métricas de calidad de datos
- `pca_temporal.csv` / `.html` - Análisis de componentes principales
- `heatmap_nan.html` - Heatmap animado de valores faltantes
- `estadisticas_completitud.csv` - Completitud por dimensión
- `metadata.csv` - Metadatos del cluster

**Archivos globales:**
- `resumen_todos_clusters.csv` - Comparación de todos los clusters
- `comparacion_dimensiones.html` - Gráfico comparativo de dimensiones
- `comparacion_completitud.html` - Gráfico de completitud
- `calidad_datos.html` - Scatter plot de calidad
- `informe_analisis_20251109_152920.md` - Este informe

================================================================================
✅ **ANÁLISIS COMPLETADO EXITOSAMENTE**
================================================================================