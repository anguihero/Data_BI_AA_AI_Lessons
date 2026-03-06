---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 07: Aprendizaje No Supervisado

## 🎯 Objetivo de la Sesión
Dominar técnicas de clustering, específicamente K-Means, para descubrir patrones ocultos en datos sin etiquetas. Aprenderás a segmentar clientes en grupos naturales sin conocimiento previo de las categorías.

## 📚 Conceptos Teóricos

### ¿Qué es Aprendizaje No Supervisado?
A diferencia del aprendizaje supervisado, en el no supervisado trabajamos con datos **sin etiquetas**. El objetivo es descubrir estructura, patrones o agrupaciones naturales en los datos.

**Diferencias clave**:
| Aspecto | Supervisado | No Supervisado |
|---------|-------------|----------------|
| Datos | Etiquetados (X, y) | No etiquetados (solo X) |
| Objetivo | Predecir etiqueta | Encontrar estructura |
| Ejemplos | Clasificación, Regresión | Clustering, Reducción dimensionalidad |
| Evaluación | Métricas directas (accuracy, R²) | Métricas indirectas (silhouette, inertia) |

### Tipos de Aprendizaje No Supervisado

#### 1. Clustering (Agrupamiento)
Agrupar datos similares en clusters.
- **K-Means**: Particional, esférico, requiere K predefinido
- **Hierarchical Clustering**: Jerárquico, no requiere K
- **DBSCAN**: Basado en densidad, bueno para formas arbitrarias

#### 2. Reducción de Dimensionalidad
Reducir número de features manteniendo información.
- **PCA (Principal Component Analysis)**: Componentes principales
- **t-SNE**: Visualización de datos de alta dimensión
- **UMAP**: Alternativa moderna a t-SNE

#### 3. Detección de Anomalías
Identificar observaciones atípicas.
- **Isolation Forest**
- **One-Class SVM**
- **Autoencoders**

### K-Means: El Rey del Clustering

#### ¿Cómo Funciona K-Means?

**Algoritmo**:
1. **Inicialización**: Selecciona K centroides aleatorios
2. **Asignación**: Asigna cada punto al centroide más cercano
3. **Actualización**: Recalcula centroides como la media de puntos asignados
4. **Repetir**: Pasos 2-3 hasta convergencia (centroides no cambian)

```
Iteración 1:     Iteración 2:     Iteración N:
  •  •           •  •             •  •
 • X •          • X •            • X •
• • • •        • • • •          • • • •
  • X          • • X •          • • X •
              (convergencia)
```

#### Parámetros Clave

**1. n_clusters (K)**
- El número de clusters a formar
- Debe especificarse antes del entrenamiento
- Usar Elbow Method o Silhouette Score para encontrar K óptimo

**2. init**
- `'k-means++'` (default): Inicialización inteligente, converge más rápido
- `'random'`: Centroides aleatorios

**3. max_iter**
- Número máximo de iteraciones
- Default: 300

**4. n_init**
- Número de veces que ejecutar K-Means con diferentes inicializaciones
- Default: 10, selecciona la mejor

#### Ventajas y Desventajas

**✅ Ventajas**:
- Simple y fácil de implementar
- Rápido y escalable (O(n))
- Funciona bien en datasets grandes
- Interpretable

**❌ Desventajas**:
- Requiere especificar K previamente
- Sensible a outliers
- Asume clusters esféricos de tamaño similar
- Sensible a inicialización (mitigado con k-means++)
- Solo funciona con features numéricas

### Determinando el Número Óptimo de Clusters

#### 1. Elbow Method (Método del Codo)

Grafica la **inertia** (suma de distancias al cuadrado dentro de clusters) vs K.

```python
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias)
# Busca el "codo" donde la reducción de inertia disminuye
```

**Interpretación**:
- Inertia siempre disminuye con más K
- El "codo" es donde agregar más clusters aporta poco
- No siempre hay un codo claro

#### 2. Silhouette Score

Mide qué tan similar es un objeto a su propio cluster vs otros clusters.

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Seleccionar K con mayor silhouette score
```

**Rango de Silhouette Score**:
- **1**: Clusters perfectamente densos y separados
- **0**: Clusters se sobreponen
- **-1**: Puntos asignados al cluster incorrecto

**Interpretación**:
- 0.71-1.0: Estructura fuerte
- 0.51-0.70: Estructura razonable
- 0.26-0.50: Estructura débil
- < 0.25: Sin estructura significativa

#### 3. Domain Knowledge (Conocimiento del Negocio)

A veces el número de clusters lo dicta el negocio:
- Segmentar clientes en 3 categorías: premium, medio, básico
- Dividir productos en categorías predefinidas

### Preprocesamiento para K-Means

#### Feature Scaling es CRÍTICO
K-Means usa distancia euclidiana, por lo que features con rangos grandes dominan.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Ejemplo sin scaling**:
- Feature 1: Ingresos en millones [0-10]
- Feature 2: Edad en años [20-80]
- Ingresos dominará el clustering (mayor varianza)

**Después de scaling**:
- Ambas features tienen media=0, std=1
- Contribución equitativa al clustering

### Interpretación y Aplicación de Clusters

Una vez obtenidos los clusters, el trabajo real es darles significado.

#### Análisis de Perfiles de Cluster

```python
# Agregar cluster labels al dataset original
df['cluster'] = kmeans.labels_

# Analizar características promedio por cluster
cluster_profiles = df.groupby('cluster').mean()
```

#### Nombrar Clusters

Basado en las características, asigna nombres descriptivos:
- **Cluster 0**: "Clientes de Alto Valor" (altos ingresos, alta frecuencia)
- **Cluster 1**: "Clientes Ocasionales" (bajos ingresos, baja frecuencia)
- **Cluster 2**: "Clientes en Crecimiento" (ingresos medios, alta frecuencia)

### Casos de Uso de K-Means

#### 1. Segmentación de Clientes (RFM Analysis)
- **R**ecency: Qué tan reciente fue la última compra
- **F**requency: Qué tan seguido compra
- **M**onetary: Cuánto gasta

Clusters: VIP, Leales, En Riesgo, Perdidos

#### 2. Compresión de Imágenes
Reducir paleta de colores agrupando píxeles similares

#### 3. Detección de Anomalías
Puntos lejanos de todos los centroides son outliers

#### 4. Sistemas de Recomendación
Agrupar usuarios con comportamientos similares

#### 5. Organización de Documentos
Agrupar textos por tópicos similares

### Limitaciones y Alternativas

**Cuándo NO usar K-Means**:
- Clusters de formas no esféricas → Usar DBSCAN
- Clusters de tamaños muy diferentes → Usar Gaussian Mixture Models
- Features categóricas → Usar K-Modes
- Clusters jerárquicos → Usar Hierarchical Clustering

## 🏆 Reto: Segmentación de Clientes con K-Means

### Escenario
Eres Data Scientist en una empresa de e-commerce. El equipo de marketing quiere segmentar a los clientes para personalizar campañas. Tienes datos de:
- Ingresos anuales
- Gasto anual en la plataforma
- Frecuencia de compra (visitas/mes)
- Antigüedad como cliente (meses)
- Edad

### Objetivos del Reto

#### 1. Preparación de Datos
- Cargar/crear dataset de clientes
- EDA: distribuciones, correlaciones
- Feature scaling (StandardScaler)

#### 2. Determinar K Óptimo
- Implementar Elbow Method
- Calcular Silhouette Scores para K = 2 a 10
- Seleccionar mejor K

#### 3. Entrenamiento de K-Means
- Entrenar modelo con K óptimo
- Obtener labels y centroides

#### 4. Análisis de Clusters
- Perfilar cada cluster (características promedio)
- Visualizar clusters (PCA para 2D)
- Nombrar clusters según perfiles

#### 5. Insights de Negocio
- ¿Qué estrategia de marketing usarías para cada segmento?
- ¿Qué segmento tiene mayor potencial de crecimiento?
- ¿Algún cluster en riesgo de churn?

## 💡 Tips para el Éxito
1. **Siempre escala antes de K-Means**: StandardScaler es tu amigo
2. **El "codo" no siempre es obvio**: Combina Elbow + Silhouette + Business sense
3. **Interpreta los clusters**: Números sin significado no ayudan al negocio
4. **Visualiza en 2D**: Usa PCA para reducir a 2 dimensiones y visualizar
5. **Valida con conocimiento del dominio**: ¿Los clusters tienen sentido de negocio?

## 📊 Criterios de Evaluación
- ✅ Preprocesamiento adecuado (scaling)
- ✅ Método sistemático para elegir K
- ✅ Interpretación clara de clusters
- ✅ Visualizaciones informativas
- ✅ Insights accionables de negocio

## 📖 Recursos Complementarios
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [K-Means Visualization](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
- [Customer Segmentation Guide](https://towardsdatascience.com/customer-segmentation-using-k-means-clustering-d33964f238c3)
- [Choosing K in K-Means](https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/)

## 🔍 Conceptos que Dominarás
- Diferencia entre aprendizaje supervisado y no supervisado
- Algoritmo K-Means y su funcionamiento interno
- Elbow Method y Silhouette Score
- Feature scaling para clustering
- Interpretación y perfilado de clusters
- Aplicación de clustering a problemas de negocio

## 🚀 Siguiente Paso
En la Sesión 08, darás un salto cuántico hacia el futuro: La Era de los Transformers, donde usarás modelos de lenguaje pre-entrenados de HuggingFace para análisis de sentimientos y generación de texto.

---
**¡Descubre patrones ocultos en tus datos! 🔍📊**
