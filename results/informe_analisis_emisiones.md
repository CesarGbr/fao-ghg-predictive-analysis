================================================================================
ANÁLISIS Y NORMALIZACIÓN DE DATOS DE EMISIONES
================================================================================

[0/7] Cargando datos desde 'Emisiones_Totales_S_Todos_los_Datos_(Normalizado).csv'...
   ✓ Datos cargados: 2,397,366 registros
   ✓ Columnas: Código del área, Código del área (M49), Área, Código del producto, Producto, Código del elemento, Elemento, Código del año, Año, Código fuente, Fuente, Unidad, Valor, Símbolo, Nota

[1/7] Normalizando y limpiando datos...

   ⚠️  Eliminando 21966 registros con años > 2025
       Años encontrados: [np.int64(2030), np.int64(2050)]
   ✓ Años válidos: 1961 - 2022

   📦 Normalizando productos...
   ✓ No se detectaron duplicados en productos
   ✓ Productos únicos: 43 → 43

   🧪 Normalizando elementos de emisión...
   ✓ No se detectaron duplicados en elementos
   ✓ Elementos únicos: 9 → 9

   🌍 Normalizando nombres de países...
   🔄 Geocodificando 280 países con Nominatim...
      (Esto puede tomar varios minutos...)
       Procesados 10/280...
       Procesados 20/280...
       Procesados 30/280...
       Procesados 40/280...
       Procesados 50/280...
       Procesados 60/280...
       Procesados 70/280...
       Procesados 80/280...
       Procesados 90/280...
       Procesados 100/280...
       Procesados 110/280...
       Procesados 120/280...
       Procesados 130/280...
       Procesados 140/280...
       Procesados 150/280...
       Procesados 160/280...
       Procesados 170/280...
       Procesados 180/280...
       Procesados 190/280...
       Procesados 200/280...
       Procesados 210/280...
       Procesados 220/280...
       Procesados 230/280...
       Procesados 240/280...
       Procesados 250/280...
       Procesados 260/280...
       Procesados 270/280...
       Procesados 280/280...
   ✓ Nombres normalizados: 185
   ✓ Países geocodificados: 260/280
   ✓ Información geográfica guardada en 'informacion_geografica_paises.csv'

   📍 Ejemplos de normalización:
      'Afganistán' → 'افغانستان' (33.77, 66.24) [N/A]
      'Albania' → 'Shqipëria' (41.00, 20.00) [Europa]
      'Alemania' → 'Deutschland' (51.16, 10.45) [N/A]
      'Andorra' → 'Andorra' (42.54, 1.57) [Europa]
      'Angola' → 'Angola' (-11.88, 17.57) [África]
   ✓ Países únicos: 280

   💾 Datos normalizados guardados en 'datos_normalizados.csv'

[2/7] Generando estadísticas descriptivas...

## 📊 ESTADÍSTICAS GENERALES

- **Total de registros**: 2,375,400
- **Países únicos**: 280
- **Productos únicos**: 43
- **Elementos únicos**: 9
- **Rango temporal**: 1961 - 2022 (62 años)
- **Valores nulos**: 0 (0.00%)

### 📈 Distribución de valores de emisiones

- **Media**: 36,010.98
- **Mediana**: 15.11
- **Desviación estándar**: 534,596.32
- **Mínimo**: -12,498,697.59
- **Máximo**: 53,510,727.43

### 🌍 Top 10 países por emisiones totales

1. **Mundo**: 14,562,971,593.23
2. **Países No-Anexo I**: 9,515,805,695.08
3. **Asia**: 6,108,548,228.43
4. **Países Anexo I**: 4,980,392,514.92
5. **OECD**: 4,519,128,951.70
6. **Américas**: 3,932,214,138.62
7. **Asia oriental**: 2,869,480,847.10
8. **Europa**: 2,394,593,102.58
9. **China**: 2,370,473,183.43
10. **China, Continental**: 2,282,867,064.27

### 📦 Top 10 productos por emisiones totales

1. **Emisiones totales incluyendo LULUCF**: 16,929,965,848.12
2. **Emisiones totales excluyendo LULUCF**: 16,206,212,973.28
3. **Energía**: 14,720,188,273.56
4. **Sistemas agroalimentarios**: 6,080,145,908.61
5. **Emisiones en tierras agrícolas**: 4,471,070,196.56
6. **IPCC Agricultura**: 4,307,943,843.40
7. **AFOLU**: 2,948,808,831.66
8. **Farm gate**: 2,867,886,675.32
9. **Emissiones derivadas del sector ganadero**: 2,629,294,116.16
10. **Fermentación entérica**: 2,039,043,538.39

### 🧪 Elementos de emisión por volumen total

1. **Emisiones (CO2eq) (AR5)**: 41,164,594,197.12
2. **Emisiones (CO2)**: 25,864,645,677.75
3. **Emisiones (CO2eq) proveniente de CH4 (AR5)**: 12,219,466,250.22
4. **Emisiones (CO2eq) proveniente de N2O (AR5)**: 5,208,748,759.17
5. **Emisiones (CO2eq) proveniente de F-gases (AR5)**: 592,860,662.61
6. **Emisiones (CH4)**: 457,137,175.74
7. **Emisiones (N2O)**: 20,065,862.92
8. **Emisiones directas (N2O)**: 10,142,857.20
9. **Emisiones indirectas (N2O)**: 2,809,727.35

### 📅 Evolución temporal de emisiones

| Década | Emisiones Totales |
|--------|-------------------|
| 1960s | 1,852,537,313.86 |
| 1970s | 2,646,982,655.74 |
| 1980s | 3,005,762,101.84 |
| 1990s | 20,466,485,974.38 |
| 2000s | 23,192,765,779.97 |
| 2010s | 26,185,953,870.46 |
| 2020s | 8,189,983,473.85 |

### 🗺️ Cobertura de datos por región

- **Promedio de registros por país**: 8484
- **Mediana**: 9193
- **País con más registros**: Mundo (12,268)
- **País con menos registros**: Territorio de las Islas del Pacífico (474)

[3/7] Analizando completitud por país...

   ✓ Análisis de 280 países completado

### 📊 Estadísticas de completitud

- **Países con datos pre-1990**: 228 (81.4%)
- **Países con datos post-1990**: 280 (100.0%)
- **Países con ambos períodos**: 228
- **Cobertura promedio de productos**: 90.4%
- **Densidad promedio de datos**: 75.2%

[4/7] Creando firmas de completitud...
   ✓ Matriz de features: (280, 33)

[5/7] Buscando k óptimo y aplicando clustering...
   ✓ K óptimo: 5
   ✓ Silhouette score: 0.633

[6/7] Caracterizando clusters...

## 🔍 CARACTERIZACIÓN DE CLUSTERS

### Cluster 0 - 30 países

**Países**: Armenia, Azerbaiyán, Belarús, Bélgica, Bosnia y Herzegovina, Chequia, Croacia, Eritrea, Eslovaquia, Eslovenia
... y 20 más

**Cobertura temporal:**
- Año inicio promedio: 1994
- Rango de inicio: 1990 - 2006
- Continuidad promedio: 100.0%

**Cobertura de datos:**
- Productos (promedio): 40.6 / 43
- Elementos (promedio): 9.0 / 9
- Densidad (promedio): 111.3%

**Datos pre-1990:** 0/30 países (0.0%)

### Cluster 1 - 193 países

**Países**: Afganistán, Albania, Alemania, Angola, Antigua y Barbuda, Arabia Saudita, Argelia, Argentina, Australia, Austria
... y 183 más

**Cobertura temporal:**
- Año inicio promedio: 1961
- Rango de inicio: 1961 - 1961
- Continuidad promedio: 100.0%

**Cobertura de datos:**
- Productos (promedio): 40.8 / 43
- Elementos (promedio): 9.0 / 9
- Densidad (promedio): 70.6%

**Datos pre-1990:** 193/193 países (100.0%)

### Cluster 2 - 29 países

**Países**: Andorra, Anguila, Antillas Neerlandesas (ex), Aruba, Ascensión, Santa Elena y Tristán de Acuña, Bermudas, China, RAE de Macao, Groenlandia, Guadalupe, Guayana Francesa
... y 19 más

**Cobertura temporal:**
- Año inicio promedio: 1961
- Rango de inicio: 1961 - 1961
- Continuidad promedio: 100.0%

**Cobertura de datos:**
- Productos (promedio): 32.3 / 43
- Elementos (promedio): 8.0 / 9
- Densidad (promedio): 60.1%

**Datos pre-1990:** 29/29 países (100.0%)

### Cluster 3 - 22 países

**Países**: Gibraltar, Guam, Isla de Man, Isla Norfolk, Islas Anglonormandas, Islas Caimán, Islas Malvinas (Falkland Islands), Islas Marianas del Norte, Islas Marshall, Islas Svalbard y Jan Mayen
... y 12 más

**Cobertura temporal:**
- Año inicio promedio: 1992
- Rango de inicio: 1990 - 2012
- Continuidad promedio: 100.0%

**Cobertura de datos:**
- Productos (promedio): 28.7 / 43
- Elementos (promedio): 7.6 / 9
- Densidad (promedio): 97.3%

**Datos pre-1990:** 0/22 países (0.0%)

### Cluster 4 - 6 países

**Países**: Bélgica-Luxemburgo, Checoslovaq, Etiopía RDP, Territorio de las Islas del Pacífico, URSS, Yugoslav RFS

**Cobertura temporal:**
- Año inicio promedio: 1961
- Rango de inicio: 1961 - 1961
- Continuidad promedio: 100.0%

**Cobertura de datos:**
- Productos (promedio): 36.7 / 43
- Elementos (promedio): 8.3 / 9
- Densidad (promedio): 36.0%

**Datos pre-1990:** 6/6 países (100.0%)

[7/7] Analizando overlap y correlaciones...

### Overlap en Cluster 0

**Top 5 pares con mayor correlación:**

- **Etiopía ↔ Turkmenistán**
  - Correlación: 0.690
  - Overlap: 30 años (1993-2022)
  - Productos comunes: 40
  - 🔍 Turkmenistán tiene datos exclusivos: 1992-1992

- **Kazajstán ↔ Asia central**
  - Correlación: 0.688
  - Overlap: 31 años (1992-2022)
  - Productos comunes: 42

- **Uzbekistán ↔ Asia central**
  - Correlación: 0.659
  - Overlap: 31 años (1992-2022)
  - Productos comunes: 41

- **Letonia ↔ Lituania**
  - Correlación: 0.641
  - Overlap: 31 años (1992-2022)
  - Productos comunes: 40

- **Chequia ↔ Eslovaquia**
  - Correlación: 0.630
  - Overlap: 30 años (1993-2022)
  - Productos comunes: 41

### Overlap en Cluster 1

**Top 5 pares con mayor correlación:**

- **China ↔ China, Continental**
  - Correlación: 0.980
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 42

- **Los países menos desarrollados ↔ Import netos alim en Des**
  - Correlación: 0.958
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 42

- **África ↔ Import netos alim en Des**
  - Correlación: 0.958
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 42

- **Oceanía ↔ Australia y Nueva Zelandia**
  - Correlación: 0.949
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 42

- **India ↔ Asia meridional**
  - Correlación: 0.936
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 42

### Overlap en Cluster 2

**Top 5 pares con mayor correlación:**

- **Anguila ↔ Islas Turcas y Caicos**
  - Correlación: 0.868
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 27

- **China, RAE de Macao ↔ Kiribati**
  - Correlación: 0.760
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 32

- **Kiribati ↔ Maldivas**
  - Correlación: 0.748
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 30

- **Anguila ↔ Kiribati**
  - Correlación: 0.741
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 27

- **Anguila ↔ Islas Vírgenes Británicas**
  - Correlación: 0.728
  - Overlap: 62 años (1961-2022)
  - Productos comunes: 26

### Overlap en Cluster 3

**Top 5 pares con mayor correlación:**

- **Gibraltar ↔ Islas Anglonormandas**
  - Correlación: 0.913
  - Overlap: 33 años (1990-2022)
  - Productos comunes: 16

- **Isla de Man ↔ Islas Anglonormandas**
  - Correlación: 0.896
  - Overlap: 33 años (1990-2022)
  - Productos comunes: 16

- **Islas Anglonormandas ↔ Sudán**
  - Correlación: 0.896
  - Overlap: 11 años (2012-2022)
  - Productos comunes: 16
  - 🔍 Islas Anglonormandas tiene datos exclusivos: 1990-2011

- **Islas Anglonormandas ↔ Islas Marshall**
  - Correlación: 0.893
  - Overlap: 32 años (1991-2022)
  - Productos comunes: 16
  - 🔍 Islas Anglonormandas tiene datos exclusivos: 1990-1990

- **Islas Anglonormandas ↔ Islas Vírgenes de los Estados Unidos**
  - Correlación: 0.878
  - Overlap: 33 años (1990-2022)
  - Productos comunes: 16

### Overlap en Cluster 4

**Top 5 pares con mayor correlación:**

- **Etiopía RDP ↔ Territorio de las Islas del Pacífico**
  - Correlación: 0.833
  - Overlap: 30 años (1961-1990)
  - Productos comunes: 26
  - 🔍 Etiopía RDP tiene datos exclusivos: 1991-1992

- **Territorio de las Islas del Pacífico ↔ URSS**
  - Correlación: 0.760
  - Overlap: 30 años (1961-1990)
  - Productos comunes: 26
  - 🔍 URSS tiene datos exclusivos: 1991-1991

- **Territorio de las Islas del Pacífico ↔ Yugoslav RFS**
  - Correlación: 0.728
  - Overlap: 30 años (1961-1990)
  - Productos comunes: 26
  - 🔍 Yugoslav RFS tiene datos exclusivos: 1991-1991

- **Checoslovaq ↔ URSS**
  - Correlación: 0.694
  - Overlap: 31 años (1961-1991)
  - Productos comunes: 40
  - 🔍 Checoslovaq tiene datos exclusivos: 1992-1992

- **Checoslovaq ↔ Yugoslav RFS**
  - Correlación: 0.554
  - Overlap: 31 años (1961-1991)
  - Productos comunes: 39
  - 🔍 Checoslovaq tiene datos exclusivos: 1992-1992

✓ Análisis de overlap guardado en 'overlap_correlaciones.csv'

## 🎯 CANDIDATOS PARA IMPUTACIÓN

Se identificaron **20 pares de países** candidatos para imputación:

### Criterios de selección:
- Correlación > 0.70
- Overlap temporal ≥ 10 años
- Al menos uno tiene datos exclusivos

### Top candidatos:

| País Donante | País Receptor | Correlación | Años Overlap | Período a Imputar |
|--------------|---------------|-------------|--------------|-------------------|
| Islas Anglonormandas | Sudán | 0.896 | 11 | 1990-2011 |
| Islas Anglonormandas | Islas Marshall | 0.893 | 32 | 1990-1990 |
| África septentrional | Sudán (ex) | 0.849 | 51 | 2012-2022 |
| Etiopía RDP | Territorio de las Islas del Pacífico | 0.833 | 30 | 1991-1992 |
| África occidental | Sudán (ex) | 0.801 | 51 | 2012-2022 |
| Burkina Faso | Sudán (ex) | 0.782 | 51 | 2012-2022 |
| Omán | Sudán (ex) | 0.782 | 51 | 2012-2022 |
| Islas Anglonormandas | Islas Marianas del Norte | 0.768 | 32 | 1990-1990 |
| URSS | Territorio de las Islas del Pacífico | 0.760 | 30 | 1991-1991 |
| Gibraltar | Sudán | 0.741 | 11 | 1990-2011 |
| Egipto | Sudán (ex) | 0.731 | 51 | 2012-2022 |
| Yugoslav RFS | Territorio de las Islas del Pacífico | 0.728 | 30 | 1991-1991 |
| Senegal | Sudán (ex) | 0.721 | 51 | 2012-2022 |
| África | Sudán (ex) | 0.720 | 51 | 2012-2022 |
| Ghana | Sudán (ex) | 0.716 | 51 | 2012-2022 |
| Islas Caimán | Sudán | 0.713 | 11 | 1990-2011 |
| Nigeria | Sudán (ex) | 0.707 | 51 | 2012-2022 |
| Polinesia | Sudán (ex) | 0.704 | 51 | 2012-2022 |
| Los países menos desarrollados | Sudán (ex) | 0.702 | 51 | 2012-2022 |
| Uganda | Sudán (ex) | 0.700 | 51 | 2012-2022 |

✓ Candidatos guardados en 'candidatos_imputacion.csv'

## 🔍 ANÁLISIS DE DATOS FALTANTES POR PRODUCTO/ELEMENTO

### Productos/Elementos con más datos faltantes:

| Producto | Elemento | Registros | Completitud |
|----------|----------|-----------|-------------|
| Suelos agrícolas | Emisiones (CO2eq) proveniente de CH4 (AR5) | 9 | 0.1% |
| Suelos agrícolas | Emisiones (CH4) | 9 | 0.1% |
| Tanques de combustible internacional | Emisiones (CH4) | 33 | 0.2% |
| Tanques de combustible internacional | Emisiones (N2O) | 33 | 0.2% |
| Tanques de combustible internacional | Emisiones (CO2eq) proveniente de N2O (AR5) | 33 | 0.2% |
| Tanques de combustible internacional | Emisiones (CO2eq) proveniente de CH4 (AR5) | 33 | 0.2% |
| Tanques de combustible internacional | Emisiones (CO2) | 33 | 0.2% |
| Tanques de combustible internacional | Emisiones (CO2eq) (AR5) | 33 | 0.2% |
| Incendios forestales | Emisiones (CO2) | 93 | 0.5% |
| Incendios de sabana | Emisiones (CO2) | 278 | 1.6% |
| IPCC Agricultura | Emisiones (CO2) | 1349 | 7.8% |
| Fabricación de fertilizantes | Emisiones (N2O) | 2601 | 15.0% |
| Fabricación de fertilizantes | Emisiones (CO2eq) proveniente de N2O (AR5) | 2601 | 15.0% |
| Otro | Emisiones (CO2eq) proveniente de CH4 (AR5) | 2733 | 15.7% |
| Fabricación de fertilizantes | Emisiones (CO2) | 3671 | 21.1% |

✓ Guardado en 'completitud_productos_elementos.csv'

## 📁 ARCHIVOS GENERADOS

- **datos_normalizados.csv**: Datos limpios y normalizados con información geográfica
- **informacion_geografica_paises.csv**: Tabla de referencia: nombres originales, normalizados y coordenadas
- **completitud_por_pais.csv**: Análisis detallado de cada país
- **paises_agrupados_por_completitud.csv**: Países agrupados en 5 clusters
- **overlap_correlaciones.csv**: Pares de países con overlap y correlaciones
- **candidatos_imputacion.csv**: Pares candidatos para imputación
- **completitud_productos_elementos.csv**: Completitud por producto/elemento
- **clustering_completitud.png**: Visualización del clustering

## 📋 RESUMEN EJECUTIVO

### Datos procesados:
- **2,375,400** registros de emisiones
- **280** países
- **43** productos
- **9** elementos de emisión
- Período: **1961-2022**

### Clustering:
- Se identificaron **5 grupos** de países con patrones similares de disponibilidad de datos
- Silhouette score: **0.633**

### Oportunidades de imputación:
- **20** pares de países candidatos para imputación
- Correlación promedio: **0.762**

### Calidad de datos:
- Países con alta completitud (>80%): **64** (22.9%)
- Países con baja completitud (<50%): **7** (2.5%)
- Densidad promedio de datos: **75.2%**

## 💡 RECOMENDACIONES

### Para imputación de datos:

1. **Priorizar pares con correlación > 0.80** y overlap > 15 años
2. **Validar temporalmente**: Ocultar 10% de datos conocidos y verificar precisión de imputación
3. **Imputar solo productos que existían** en el período a imputar (verificar contra productos nuevos post-1990)
4. **Documentar**: Crear columna 'Imputado' = True para transparencia
5. **Análisis de sensibilidad**: Probar diferentes países donantes y comparar resultados

### Para análisis futuros:

1. **Análisis por región geográfica**: Los clusters actuales podrían refinarse con información geográfica
2. **Factores contextuales**: Incorporar variables económicas, políticas o climáticas
3. **Series temporales**: Aplicar modelos ARIMA o Prophet para productos con datos continuos
4. **Validación cruzada**: Realizar validación k-fold temporal antes de imputación final

================================================================================
INFORME GENERADO EXITOSAMENTE
================================================================================