---
Autor: anmmunozsa@outlook.es
Fecha de actualización: 2026-07-01
---

# Ruta de Aprendizaje — Modelos de Analítica

## 1. Propósito

Esta ruta desarrolla competencias prácticas de análisis de datos y modelado en Python. Comienza con ciencia de datos, Colab, Markdown y programación; continúa con Pandas, visualización, estadística y aprendizaje automático; integra pipelines y optimización; y cierra con redes neuronales, CNN y procesamiento de lenguaje natural.

- **Duración:** 16 sesiones de 2 horas, 32 horas en total.
- **Entorno:** Google Colab.
- **Metodología:** explicación conceptual, fundamento técnico, código ejecutable paso a paso, interpretación y retos.
- **Material:** 16 notebooks, uno por sesión.

## 2. Progresión

```text
Fundamentos
S01 Ciencia de datos + Colab + Markdown
  ↓
Programación y datos
S02 Python I → S03 Python II → S04 Pandas I + Kaggle → S05 Wrangling → S06 Visualización
  ↓
Preparación y modelos
S07 Estadística + Preprocesamiento → S08 Regresión → S09 Clasificación → S10 Ensambles
  ↓
Aprendizaje no supervisado
S11 Clustering → S12 Reducción de dimensionalidad y selección
  ↓
Integración
S13 Pipelines → S14 Optimización de hiperparámetros
  ↓
Deep Learning y NLP
S15 Redes neuronales + CNN → S16 LLM y NLP aplicado
```

La secuencia introduce primero el lenguaje y las estructuras de datos. Después desarrolla preparación, evaluación y modelos clásicos. Los pipelines y la optimización aparecen cuando el estudiante ya conoce preprocesamiento y varios estimadores. Redes neuronales y NLP cierran la ruta como extensiones hacia datos no tabulares.

## 3. Detalle de las sesiones

### Sesión 01 — Ciencia de Datos, Python, Colab y Markdown

**Objetivo:** comprender el ciclo de un proyecto de datos y comenzar a trabajar en un notebook reproducible.

Se estudian la relación entre negocio, estadística, programación y comunicación; la formulación de preguntas analíticas; Big Data y sus 7 V; CRISP-DM; y las diferencias entre aprendizaje supervisado, no supervisado y por refuerzo. Se revisan Python y Colab y se realiza un microanálisis separando pregunta, datos, cálculo e interpretación. El notebook cierra con un taller guiado de Markdown que incluye encabezados, listas, tablas, enlaces, imágenes, fórmulas, bloques de código, plantilla de análisis y lista de verificación.

**Material:** [README](Sesion_01/README.md) · [Notebook](Sesion_01/Sesion_01_Ciencia_de_datos.ipynb) · [Bienvenida y acuerdos](Sesion_01/bienvenida_saludo_acuerdos.md) · [Imagen de bienvenida](Sesion_01/Saludo_Bienvenida.png)

### Sesión 02 — Bases de Python I

**Objetivo:** escribir programas básicos con tipos, operadores, decisiones, iteraciones y funciones.

Se trabajan `str`, `int`, `float` y `bool`; conversiones; operadores aritméticos, relacionales y lógicos; `if`/`elif`/`else`; ciclos `for` y `while`; y funciones con parámetros y retorno. El laboratorio profundiza en métodos de texto, operaciones numéricas, reglas booleanas, parámetros predeterminados, anotaciones de tipo y validación con `ValueError`.

**Material:** [README](Sesion_02/README.md) · [Notebook](Sesion_02/Sesion_02_Python_I.ipynb)

### Sesión 03 — Bases de Python II

**Objetivo:** seleccionar y manipular estructuras de datos apropiadas para análisis.

Se comparan listas, tuplas, diccionarios y conjuntos; se practican comprehensions, `*args`, `**kwargs`, `lambda`, excepciones y archivos. El laboratorio distingue mutación de copia, emplea métodos de diccionarios y sets, construye una función configurable y explica el costo práctico de buscar elementos en distintas colecciones.

**Material:** [README](Sesion_03/README.md) · [Notebook](Sesion_03/Sesion_03_Python_II.ipynb)

### Sesión 04 — Pandas I y obtención de datos

**Objetivo:** cargar, inspeccionar, tipar, seleccionar y filtrar datos tabulares.

Se crean `Series` y `DataFrame`; se usan `head`, `shape`, `info`, `describe` y `dtypes`; se leen archivos CSV; y se descarga `world_country.csv` con `kagglehub`. Se practican `loc`, `iloc` y filtros booleanos. El laboratorio aplica accesores `.str`, `.dt` y `.cat`, conversión explícita de tipos, `query`, `assign` y parámetros de `read_csv` útiles en Colab.

**Material:** [README](Sesion_04/README.md) · [Notebook](Sesion_04/Sesion_04_Pandas_I_Kaggle.ipynb)

### Sesión 05 — Pandas II: Data Wrangling

**Objetivo:** transformar, combinar y resumir tablas de manera controlada.

Se desarrollan `groupby`, `agg`, `merge`, `concat`, `pivot_table`, valores faltantes, operaciones vectorizadas, `map`, `apply` y fechas. El laboratorio compara `agg` con `transform`, calcula diferencias respecto a la media de un grupo y audita uniones con `validate` e `indicator`.

**Material:** [README](Sesion_05/README.md) · [Notebook](Sesion_05/Sesion_05_Pandas_II_Wrangling.ipynb)

### Sesión 06 — Visualización con Matplotlib, Seaborn, Plotly y Folium

**Objetivo:** construir visualizaciones estáticas, interactivas y geoespaciales.

Se elaboran histogramas, boxplots, dispersión, barras, mapas de correlación, gráficos interactivos y mapas. Se comparan Matplotlib, Seaborn, Plotly Express y Folium. El laboratorio separa preparación de datos, geometría y presentación; revisa `Figure`/`Axes`, `hue`, transparencia, hover, escalas, títulos y unidades.

**Material:** [README](Sesion_06/README.md) · [Notebook](Sesion_06/Sesion_06_Visualizacion.ipynb)

### Sesión 07 — Estadística aplicada y preprocesamiento

**Objetivo:** describir los datos y preparar variables sin introducir fuga de información.

Se estudian medidas de centro y dispersión, distribuciones, Pearson y Spearman, outliers mediante IQR, imputación, codificación y escalado. El laboratorio diferencia el tratamiento de variables numéricas, categóricas, ordinales y temporales; muestra las fórmulas de estandarización e IQR; y separa `fit` de `transform` usando únicamente entrenamiento para aprender parámetros.

**Material:** [README](Sesion_07/README.md) · [Notebook](Sesion_07/Sesion_07_Estadistica_Preprocesamiento.ipynb)

### Sesión 08 — Métodos de regresión

**Objetivo:** predecir valores continuos y evaluar el error de los modelos.

Con `load_diabetes` se aplican train/test split, regresión lineal, expansión polinómica, Ridge y Lasso. Se comparan MAE, MSE, RMSE y R². El laboratorio deriva la intuición del MSE, calcula gradientes de pendiente e intercepto y ejecuta descenso por gradiente, relacionando tasa de aprendizaje, pérdida, coeficientes y regularización.

**Material:** [README](Sesion_08/README.md) · [Notebook](Sesion_08/Sesion_08_Regresion.ipynb)

### Sesión 09 — Métodos de clasificación

**Objetivo:** predecir categorías y seleccionar métricas y umbrales según el problema.

Se entrenan Regresión Logística, KNN, SVM y Árbol de Decisión sobre `load_breast_cancer`; se estudian matriz de confusión, accuracy, precision, recall y F1; y se presenta clasificación multiclase con `load_wine`. El laboratorio explica sigmoide y log-loss, inspecciona probabilidades y compara varios umbrales de decisión.

**Material:** [README](Sesion_09/README.md) · [Notebook](Sesion_09/Sesion_09_Clasificacion.ipynb)

### Sesión 10 — Ensambles y benchmarking

**Objetivo:** comparar modelos bajo un protocolo común y elegir candidatos estables.

Se estudian Random Forest, Gradient Boosting y XGBoost, junto con validación cruzada y benchmarking. La tabla comparativa considera promedio, desviación y tiempo. El laboratorio conecta residuos, pérdida cuadrática y boosting con descenso por gradiente en el espacio de funciones, y usa folds estratificados comunes para una comparación justa.

**Material:** [README](Sesion_10/README.md) · [Notebook](Sesion_10/Sesion_10_Ensambles_Benchmarking.ipynb)

### Sesión 11 — Clustering y segmentación

**Objetivo:** descubrir grupos sin etiquetas y evaluar su cohesión y separación.

Se aplican K-Means, Método del Codo, clustering jerárquico y DBSCAN sobre datos sintéticos y `Mall_Customers.csv`. Se calculan Silhouette y Davies-Bouldin y se perfilan los grupos. El laboratorio reproduce una iteración de K-Means mediante distancias, asignaciones, centroides e inercia, y revisa los hiperparámetros de cada algoritmo.

**Material:** [README](Sesion_11/README.md) · [Notebook](Sesion_11/Sesion_11_Clustering.ipynb)

### Sesión 12 — Reducción de dimensionalidad y selección de variables

**Objetivo:** reducir complejidad y seleccionar información relevante.

Se desarrollan PCA, t-SNE, métodos de filtro, RFE y Lasso sobre `load_diabetes`. El laboratorio centra los datos, construye una matriz de covarianza y obtiene eigenvectores; luego inspecciona componentes y varianza explicada. Se diferencia entre transformar variables mediante componentes y seleccionar variables originales.

**Material:** [README](Sesion_12/README.md) · [Notebook](Sesion_12/Sesion_12_Reduccion_Dimensionalidad.ipynb)

### Sesión 13 — Pipeline completo de Machine Learning

**Objetivo:** encapsular preprocesamiento y modelo en un flujo reproducible.

Se explica data leakage; se construyen `Pipeline` y `ColumnTransformer`; se valida el flujo completo con `cross_val_score`; y se persiste con `joblib`. El laboratorio profundiza en el contrato `fit`/`transform`/`predict`, en `get_params`/`set_params`, en la sintaxis `paso__parametro` y en los atributos aprendidos.

**Material:** [README](Sesion_13/README.md) · [Notebook](Sesion_13/Sesion_13_Pipeline_End_to_End.ipynb)

### Sesión 14 — Optimización de hiperparámetros

**Objetivo:** explorar configuraciones de un pipeline y comparar costo con desempeño.

Se aplican `GridSearchCV`, `RandomizedSearchCV` y Optuna a un `RandomForestRegressor`. Se comparan R² y tiempo. El laboratorio formula el objetivo de validación cruzada, calcula el número de ajustes requerido por una grilla e inspecciona `cv_results_`, incluyendo ranking, media, desviación, tiempo y parámetros.

**Material:** [README](Sesion_14/README.md) · [Notebook](Sesion_14/Sesion_14_Tuning_Hiperparametros.ipynb)

### Sesión 15 — Redes neuronales y CNN

**Objetivo:** comprender el entrenamiento de redes y clasificar imágenes MNIST.

Se presentan perceptrón, MLP, activaciones, MNIST, capas convolucionales, pooling, Dropout, entrenamiento y matriz de confusión. El laboratorio relaciona logits, softmax y entropía cruzada; calcula gradientes con `GradientTape`; aplica una actualización SGD; y revisa Adam, tasa de aprendizaje, batch, épocas y parámetros entrenables.

**Material:** [README](Sesion_15/README.md) · [Notebook](Sesion_15/Sesion_15_Redes_Neuronales_CNN_MNIST.ipynb)

### Sesión 16 — LLM y NLP aplicado

**Objetivo:** usar modelos preentrenados para clasificación, extracción de entidades y generación controlada.

Se introducen tokenización, embeddings y atención; se configura Gemini con Secrets de Colab; se practican prompts zero-shot y few-shot; y se utilizan pipelines de Hugging Face para sentimiento y NER. El laboratorio inspecciona tokens, IDs, padding, truncamiento y `attention_mask`, y explica parámetros de inferencia como longitud, temperatura, `top_p`, dispositivo y tamaño de lote.

**Material:** [README](Sesion_16/README.md) · [Notebook](Sesion_16/Sesion_16_LLM_NLP.ipynb)

## 4. Formato estándar de notebook (Sesiones 1–16)

Cada notebook organiza el aprendizaje con la siguiente estructura:

1. Título, objetivo y tabla de contenido.
2. Introducción conceptual en lenguaje sencillo.
3. Teoría técnica e intuición matemática.
4. Fortalezas, debilidades y criterios de uso.
5. Código distribuido en celdas con una responsabilidad clara.
6. Métodos y propiedades relacionados con el tipo de dato u objeto.
7. Parámetros, hiperparámetros y atributos aprendidos.
8. Función objetivo o pérdida cuando corresponde.
9. Interpretación del resultado y limitaciones.
10. Retos básico, medio y avanzado.

El descenso por gradiente se desarrolla en regresión, regresión logística, boosting y redes neuronales. K-Means y PCA presentan sus propias funciones objetivo y mecanismos de optimización, sin atribuirles un procedimiento que no utilizan en las implementaciones estudiadas.

## 5. Datos y dependencias

| Sesiones | Datos principales | Dependencias adicionales |
|---|---|---|
| 1–3 | Datos creados en el notebook | Python estándar |
| 4 y 6 | World Coordinates | `kagglehub`, Plotly, Folium |
| 5 | Tablas sintéticas de ventas | Pandas |
| 7 | Credit EDA Case Study | `kagglehub`, scikit-learn |
| 8, 12–14 | `load_diabetes` | scikit-learn, Optuna en S14 |
| 9–10 | `load_breast_cancer`, `load_wine` | XGBoost en S10 |
| 11 | `make_blobs`, Mall Customers | SciPy, `kagglehub` |
| 15 | MNIST | TensorFlow/Keras, GPU opcional |
| 16 | Reseñas y noticia definidas en el notebook | Gemini, Transformers, modelo descargado |

Las credenciales se almacenan mediante Secrets de Colab. Los tokens y archivos de autenticación no deben incluirse en el repositorio ni escribirse directamente en las celdas.

## 6. Evaluación práctica

Cada sesión finaliza con retos en tres niveles:

- **Básico:** reproduce la técnica central.
- **Medio:** compara alternativas o combina varios pasos.
- **Avanzado:** adapta el flujo, justifica decisiones y comunica resultados.

La evaluación considera ejecución correcta, interpretación, elección de métricas, prevención de fuga de información, uso justificado de hiperparámetros y claridad de la documentación.

## 7. Verificación antes de impartir

Los notebooks tienen estructura JSON válida y celdas con sintaxis Python válida. Antes de cada edición del curso se debe ejecutar `Runtime > Run all` en un entorno limpio de Google Colab para comprobar:

- instalación y compatibilidad de librerías;
- acceso a Kaggle y disponibilidad de archivos;
- tiempo de ejecución y memoria;
- disponibilidad de GPU para MNIST;
- vigencia del SDK y del modelo configurado para Gemini;
- descarga y carga de modelos de Hugging Face.
