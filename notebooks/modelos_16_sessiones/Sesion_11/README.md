---
Autor: anmmunozsa@outlook.es
---

# Sesión 11: Clustering y Segmentación

## 🎯 Objetivo de la Sesión

Agrupar datos sin etiquetas usando algoritmos de clustering, y evaluar objetivamente la calidad de los grupos obtenidos.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido y contexto: diferencia clave con todo lo visto hasta ahora (no hay variable objetivo) |
| 2 | 25 min | K-Means: teoría, inicialización (`k-means++`), iteración hasta convergencia |
| 3 | 20 min | Método del Codo (Elbow Method) para elegir K |
| 4 | 20 min | Clustering Jerárquico: dendrograma, tipos de `linkage` |
| 5 | 20 min | DBSCAN: clusters de forma arbitraria y detección de ruido/outliers |
| 6 | 15 min | Métricas internas: Silhouette Score, Davies-Bouldin, Calinski-Harabasz |
| 7 | 10 min | Iteración manual de K-Means, segmentación y perfilado |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16). Referencia teórica: [fundamentos de aprendizaje automatico.md § 2.1 y 2.4](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md).

### Recorrido real del notebook

- K-Means con `make_blobs`, centroides, inercia y Método del Codo.
- Segmentación de `Mall_Customers.csv` con escalado.
- Clustering jerárquico, dendrograma, DBSCAN y detección de ruido.
- Silhouette, Davies-Bouldin y perfilado descriptivo de clusters.
- Laboratorio de una iteración manual de K-Means: distancias, asignación, actualización de centroides e inercia.
- Revisión de `n_clusters`, `init`, `n_init`, `eps`, `min_samples`, `linkage` y atributos aprendidos.

**Por cada método (K-Means, Jerárquico, DBSCAN), cubrir:**
- Teoría técnica: hiperparámetros clave (`n_clusters`, `linkage`, `eps`/`min_samples`).
- Fortalezas y debilidades (K-Means requiere definir K a priori y asume clusters esféricos; Jerárquico es costoso en datasets grandes pero no requiere K de antemano; DBSCAN detecta forma arbitraria y ruido pero es sensible a `eps`).
- Código ya escrito ejecutando cada algoritmo y graficando los clusters resultantes (2D con Matplotlib/Seaborn).
- Resumen para dummies.

**Importante — recordar el escalado:** K-Means y DBSCAN son sensibles a la escala; recordar aplicar `StandardScaler` (visto en Sesión 07) antes de clusterizar.

**Aplicaciones del mundo real a mencionar:** segmentación de clientes para marketing, agrupación de documentos, detección de anomalías (puntos sin cluster claro en DBSCAN).

## 📦 Dataset(s)

- **`sklearn.datasets.make_blobs()`** (sintético) para introducir K-Means: al conocer de antemano los centros y la separación de los grupos, es la forma más clara de ver "en vivo" cómo itera el algoritmo y por qué funciona.
- **"Mall Customer Segmentation Data"** (real) para la parte central de la sesión y el reto: [vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) en Kaggle (descargado con la API de la Sesión 04), con **fuente remota alternativa** (CSV crudo, sin necesidad de Kaggle) en el [repositorio de SteffiPeTaffy](https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv) leído directamente con `pd.read_csv(url)`.

## 🏆 Retos de Práctica

- **Básico:** aplicar K-Means con K=3 sobre datos generados con `make_blobs` y graficar los clusters resultantes junto a los centroides.
- **Medio:** sobre `Mall_Customers.csv`, usar el Método del Codo (con `Annual Income` y `Spending Score`) para elegir el K óptimo, y calcular el Silhouette Score para justificar la elección.
- **Avanzado:** comparar K-Means, Clustering Jerárquico y DBSCAN sobre `Mall_Customers.csv`, generando una tabla con sus métricas internas (Silhouette, Davies-Bouldin) y un perfil descriptivo de cada cluster (¿quiénes son los clientes de "alto ingreso, bajo gasto" vs. "bajo ingreso, alto gasto"?).

## ✅ Criterios de Evaluación

- Escalado correcto de las variables antes de clusterizar.
- Uso del Método del Codo o Silhouette Score para justificar el número de clusters.
- Interpretación de negocio de al menos 2 clusters resultantes ("¿quiénes son los clientes del cluster 1?").
- Explicación de cómo el escalado y los hiperparámetros modifican la segmentación.

## 🔗 Prerrequisitos

Sesión 07 (escalado de variables) y Sesión 06 (visualización de resultados).

## 🚀 Siguiente Paso

En la Sesión 12 seguimos en el mundo no supervisado: Reducción de Dimensionalidad y Selección de Variables.
