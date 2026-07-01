---
Autor: anmmunozsa@outlook.es
Fecha: 2026-07-01
Estado: Paso 1 (ruta) y Paso 2 (detalle por sesión) completos — Paso 3 (notebooks) pendiente
---

# Ruta de Aprendizaje — Modelos de Analítica (16 Sesiones x 2 horas)

## 0. Objetivo General

Formar a los participantes en un recorrido **práctico y escrito en Python** que va desde los fundamentos de ciencia de datos hasta la construcción de un pipeline completo de Machine Learning con selección y ajuste de modelos, pasando por redes neuronales/CNN, y cerrando con una introducción aplicada a LLMs/NLP. Cada sesión combina teoría + práctica guiada + retos, con un **formato de notebook estandarizado** (ver sección 4).

**Duración total:** 16 sesiones × 2 horas = 32 horas.
**Modalidad de entrega:** 100% Google Colab. **Ninguna herramienta requiere costo ni tarjeta de crédito** — requisito explícito confirmado en esta revisión.
**Documento de referencia teórica base:** [background/esp/fundamentos de aprendizaje automatico.md](../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md) y [background/esp/fundamentos de llm.md](../../background/esp/fundamentos%20de%20llm.md).

### Cambios incorporados en esta revisión (v2)

1. **Nueva sesión de Redes Neuronales y CNN** (Sesión 15): arquitecturas de red desde el perceptrón hasta una red neuronal convolucional entrenada para clasificar dígitos MNIST.
2. **APIs 100% gratuitas**: para la Sesión 16 se descarta cualquier API de pago. Se usará la **API gratuita de Google Gemini** (a través de Google AI Studio — se obtiene una API key sin tarjeta de crédito, y encaja naturalmente con Google Colab) como opción principal, y **Hugging Face `transformers`** (modelo descargado y ejecutado localmente en el notebook, sin API ni costo) para clasificación de texto y NER.
3. **Kaggle API**: se agrega un instructivo de registro en Kaggle y uso de su API en Python (`kagglehub` / `kaggle.json`) para descargar datasets. Se integra en la **Sesión 4** (donde se enseña la carga de datos), de modo que desde ahí en adelante cada sesión de modelado puede usar datasets de Kaggle que tú selecciones.
4. **Folium se fusiona en la Sesión 6**: la sesión de visualización ahora cubre Matplotlib/Seaborn, **Plotly** (gráficos interactivos) **y** Folium (mapas con marcadores y mapas de calor) en un solo bloque dedicado exclusivamente a "generar visualizaciones" (sin profundizar en interpretación estadística, que queda en la Sesión 7). Esto libera el cupo que ahora ocupa la nueva sesión de redes neuronales, manteniendo el total en 16 sesiones.
5. **Sesión 1 ampliada**: ahora incluye explícitamente "¿Qué es Python?", instalación/uso vía Google Colab (sin instalación local requerida) y bases de Markdown — sigue siendo la única sesión **sin notebook** (es introductoria/conceptual).
6. **Formato estándar de notebook** definido para las Sesiones 2–16 (ver sección 4): storytelling en Markdown con tabla de contenido, explicación teórica "para dummies" + técnica detallada por método (fortalezas/debilidades), código ya escrito ejecutándose mientras se explica, resumen aplicado final, y retos de práctica en 3 niveles (básico/medio/avanzado).

---

## 1. Lógica de la Progresión

```text
[Fundamentos]              [Herramientas]                    [Prep. + Supervisado]                              [No Supervisado]        [Integración]                    [Deep Learning + Cierre]
S01 DS+Python+Markdown → S02-03 Python → S04 Pandas+Kaggle API → S05 Pandas Wrangling → S06 Viz (Matplotlib+Folium)
                                                                                              ↓
                                                        S07 Estadística+Preprocesamiento → S08 Regresión → S09 Clasificación → S10 Ensambles+Benchmarking
                                                                                              ↓
                                                        S11 Clustering → S12 Reducción de Dimensionalidad
                                                                                              ↓
                                                        S13 Pipeline End-to-End → S14 Tuning (Grid/Random/Bayesiano)
                                                                                              ↓
                                                        S15 Redes Neuronales y CNN (MNIST) → S16 LLM/NLP aplicado (APIs gratuitas)
```

**Por qué este orden:**

- Python y Pandas van primero porque son prerrequisito de todo lo demás. La Sesión 4 (Pandas I) es también el mejor momento para enseñar la API de Kaggle, porque "cargar datos" es exactamente el tema de esa sesión, y de ahí en adelante cada sesión de modelado puede alimentarse de datasets reales que tú elijas en Kaggle.
- La Sesión 6 concentra **toda** la generación de gráficos (estadísticos con Matplotlib/Seaborn y geoespaciales con Folium) como una sola habilidad de "visualización", separada de la interpretación estadística profunda (correlación, distribuciones, outliers), que se enseña en la Sesión 7 justo antes de empezar a modelar.
- Regresión → Clasificación → Ensambles/Benchmarking sigue la progresión de complejidad de modelos supervisados, cerrando con benchmarking (comparación objetiva de modelos), que es justo lo pedido como "ajuste de varios modelos y selección".
- Clustering y reducción de dimensionalidad (no supervisado) se dejan después de dominar el supervisado, para contrastar "con etiqueta" vs "sin etiqueta".
- Pipeline completo y tuning de hiperparámetros integran todo el bloque de ML clásico (requieren ya conocer regresión, clasificación y preprocesamiento).
- **Redes Neuronales/CNN se ubica justo antes de LLM** porque son la puerta de entrada natural al Deep Learning: primero se entienden arquitecturas de red "clásicas" (perceptrón, red densa, CNN) entrenando un modelo propio (MNIST), y luego se da el salto conceptual a los Transformers/LLMs en la Sesión 16, que son también redes neuronales pero ya pre-entrenadas y consumidas vía API o modelo descargado.

---

## 2. Tabla Resumen de las 16 Sesiones

| # | Título de Sesión | Bloque temático | Objetivo de aprendizaje | Temas clave | Material base a reutilizar |
| --- | --- | --- | --- | --- | --- |
| 1 | Bienvenida: Ciencia de Datos, Python y Markdown en Colab | Fundamentos *(sin notebook)* | Entender qué es la ciencia de datos, qué es Python y cómo documentar análisis en Markdown, todo dentro de Google Colab | ¿Qué es la ciencia de datos?, tipos de aprendizaje (sup/no sup/refuerzo), CRISP-DM, ¿qué es Python?, uso de Google Colab (sin instalación local), sintaxis Markdown | `img/crispdm_process.png`, `exercises/000_notebook_markdown.ipynb`, `exercises/Markdown/taller_de_markdown.ipynb`, `Retos_AA_8/Sesion_01` |
| 2 | Bases de Python I | Herramientas | Escribir programas Python básicos: variables, tipos, operadores, control de flujo | Variables, tipos de dato, condicionales, bucles, funciones simples | `notebooks/DS_route/001_python_basic_shortstory.ipynb`, `Retos_AA_8/Sesion_02` |
| 3 | Bases de Python II | Herramientas | Usar estructuras de datos y funciones avanzadas para preparar el terreno a Pandas | Listas, tuplas, diccionarios, comprehensions, funciones, manejo de errores, lectura de archivos | `notebooks/DS_route/001_python_basic_longstory.ipynb`, `Retos_AA_8/Sesion_03` |
| 4 | Pandas I — Series, DataFrames y Obtención de Datos | Herramientas | Cargar, explorar y filtrar datos tabulares, incluyendo datasets descargados de Kaggle | Series, DataFrame, lectura CSV/Excel, indexing, filtrado, tipos de variable, **registro en Kaggle y uso de su API (`kagglehub`/`kaggle.json`) para descargar datasets** | `notebooks/DS_route/002_pandas_shortstory.ipynb`, `Retos_AA_8/Sesion_04`, `Retos_AA_8/Sesion_05` |
| 5 | Pandas II — Data Wrangling | Herramientas | Transformar y combinar datasets reales | GroupBy, merge/join, pivot, valores faltantes, creación de columnas (feature engineering básico) | `exercises/data_wrangling/datawrangling.ipynb` |
| 6 | Visualización de Datos: Matplotlib, Seaborn, Plotly y Folium | Herramientas | Generar visualizaciones estadísticas (estáticas e interactivas) y geoespaciales (solo generación de gráficos, sin profundizar en interpretación estadística) | Histogramas, boxplots, scatter, gráficos de barras, heatmaps de correlación (Matplotlib/Seaborn); **gráficos interactivos con Plotly**; **mapas con Folium: marcadores y mapas de calor** | `notebooks/DS_route/003_EDA_Exploracion.ipynb` (como referencia de gráficos), *(nuevo)* dataset con coordenadas para Folium/Plotly |
| 7 | Estadística Aplicada y Preprocesamiento de Datos | Preparación | Interpretar estadísticamente los datos y prepararlos para modelar | Estadística descriptiva, distribuciones, correlación, outliers (IQR), imputación de faltantes, encoding categórico, escalado | `notebooks/backup/001_distribuciones_probabilidad.ipynb`, `notebooks/backup/002_analisis_correlacion_anova.ipynb`, doc base secciones 0 y 3.7 |
| 8 | Métodos de Regresión | Modelos supervisados | Predecir valores continuos y evaluar con métricas de regresión | Regresión lineal simple/múltiple, polinómica, Ridge/Lasso, MAE/MSE/RMSE/R² | `notebooks/DS_route/004_Analisis_de_regresion.ipynb`, `notebooks/backup/003_regresion_lineal_multiple.ipynb`, `notebooks/backup/regularizacion_modelos_lineales.ipynb` |
| 9 | Métodos de Clasificación | Modelos supervisados | Predecir categorías y evaluar con matriz de confusión y métricas | Regresión logística, KNN, SVM, árboles de decisión, accuracy/precision/recall/F1/AUC-ROC | `notebooks/DS_route/005_Analisis_de_clasificacion.ipynb`, `exercises/classification/Taller_lvl_Rookie_Titanic.ipynb`, `exercises/classification/Taller_lvl_Rookie_Bank_Loan.ipynb` |
| 10 | Ensambles y Benchmarking de Modelos | Modelos supervisados | Comparar múltiples algoritmos de forma objetiva para seleccionar el mejor | Random Forest, Boosting, XGBoost, validación cruzada k-fold, tabla comparativa de modelos (benchmarking) | `notebooks/backup/modelos_arboles_machine_learning.ipynb`, `notebooks/DS_route/007_Modelos_Analiticos.ipynb` |
| 11 | Clustering y Segmentación | No supervisado | Agrupar datos sin etiquetas y evaluar la calidad de los clusters | K-Means, jerárquico, DBSCAN, método del codo, Silhouette Score | `notebooks/DS_route/006_Analisis_de_Segmentacion.ipynb`, `notebooks/backup/clustering_tecnicas_niveles.ipynb`, `Retos_AA_8/Sesion_07` |
| 12 | Reducción de Dimensionalidad y Selección de Variables | No supervisado | Simplificar datasets de alta dimensionalidad preservando información relevante | PCA, varianza explicada, t-SNE (visual), métodos de selección de variables (filtro/wrapper/embedded) | `notebooks/backup/reduccion_dimensionalidad.ipynb`, `notebooks/backup/seleccion_variables_niveles.ipynb` |
| 13 | Pipeline Completo de ML (End-to-End) | Integración | Construir un pipeline reproducible que encapsule preprocesamiento + modelo | `Pipeline`, `ColumnTransformer`, `cross_val_score`, prevención de data leakage | `notebooks/backup/pipeline_columntransformer_preprocesamiento.ipynb` |
| 14 | Optimización de Hiperparámetros | Integración | Afinar el mejor modelo del benchmarking con distintas estrategias de búsqueda | Grid Search, Random Search, Optimización Bayesiana (Optuna), comparación de eficiencia | `notebooks/backup/validacion_y_busqueda_hiperparametros.ipynb` |
| 15 | Arquitecturas de Redes Neuronales y CNN (MNIST) | Deep Learning | Entender arquitecturas de red neuronal y entrenar una CNN para clasificar dígitos escritos a mano | Perceptrón, redes densas (MLP), funciones de activación, backpropagation (intuición), capas convolucionales y de pooling, entrenamiento de una CNN con el dataset MNIST | `notebooks/backup/fundamentos_redes_neuronales.ipynb`, `notebooks/backup/cnn_redes_convolucionales.ipynb` |
| 16 | LLMs y NLP Aplicado en Python (100% gratuito) | Cierre / Proyecto integrador | Usar un LLM vía API gratuita y un modelo Hugging Face para tareas de NLP reales | Prompting básico con **API gratuita de Google Gemini** (Google AI Studio, sin tarjeta de crédito), pipeline de Hugging Face `transformers` (modelo local) para clasificación de texto y NER | `background/esp/fundamentos de llm.md`, `exercises/nlp/` (a construir), `resources/Sanchez - Clasificacion de textos... BERT.pdf` |

---

## 3. Detalle por Bloque Temático

### Bloque A — Fundamentos y Herramientas (Sesiones 1–6)

Ciencia de datos, Python, Markdown, Pandas (incluyendo obtención de datos vía Kaggle API) y toda la generación de visualizaciones (estadísticas + geoespaciales con Folium) en un solo bloque de herramientas.

### Bloque B — Preparación y Modelos Supervisados (Sesiones 7–10)

Estadística aplicada + preprocesamiento, seguido de regresión → clasificación → ensambles/benchmarking. El benchmarking (S10) responde directamente al requerimiento de "ajuste de varios modelos y benchmarking para seleccionar modelos".

### Bloque C — Modelos No Supervisados (Sesiones 11–12)

Clustering y reducción de dimensionalidad.

### Bloque D — Integración y Optimización (Sesiones 13–14)

Pipeline completo (`Pipeline`/`ColumnTransformer`) y las estrategias de tuning pedidas: grilla (Grid Search), aleatoria (Random Search) y bayesiana (Optuna). Nota: "estocástica" se cubre conceptualmente dentro de Random Search (muestreo estocástico del espacio de hiperparámetros), evitando duplicar contenido.

### Bloque E — Deep Learning y Cierre (Sesiones 15–16)

Redes neuronales y CNN (MNIST) como introducción a Deep Learning, seguido del cierre con LLMs/NLP usando exclusivamente herramientas gratuitas.

---

## 4. Formato Estándar de Notebook (Sesiones 2–16)

Para que cada notebook sea "para dummies pero aplicable" y mantenga una narrativa consistente, **todos los notebooks de las Sesiones 2 a 16** seguirán esta misma estructura (la Sesión 1 no tiene notebook, es introductoria). El nivel de detalle técnico de cada método (fórmulas, hiperparámetros, fortalezas/debilidades) sí debe ser riguroso; lo que se simplifica es el *empaquetado y la narrativa*, no el contenido técnico. La profundización adicional queda como iniciativa propia de quien estudia, no como exigencia del curso.

1. **Portada + Tabla de Contenido**: título de la sesión, objetivo de aprendizaje, y un índice en Markdown con enlaces (anchors) a cada sección del notebook.
2. **Introducción general (para dummies)**: qué es el tema de la sesión explicado en términos simples, por qué importa, y 2-3 ejemplos de aplicación en el mundo real.
3. **Por cada método/técnica de la sesión** (ej. en la sesión de clasificación: Logística, KNN, SVM, Árbol de Decisión):
   - Explicación teórica y técnica detallada: cómo funciona, fórmula/intuición matemática, hiperparámetros clave.
   - Fortalezas y debilidades del método (tabla comparativa cuando aplique).
   - Celdas de código **ya escritas** que se ejecutan mientras se explica cada parámetro/paso (no se deja código para el final; se intercala explicación + ejecución).
   - Un **resumen "para dummies"** al cierre del método: lo mínimo indispensable para poder aplicarlo correctamente sin dominar toda la teoría.
4. **Ejemplos sencillos de aplicación práctica**: casos reales cortos que muestran cómo se usaría el método fuera del aula.
5. **Sección de práctica (retos)**: ejercicios en 3 niveles — **básico**, **medio** y **avanzado** — para que cada estudiante practique según su ritmo.

Este formato se aplicará de manera consistente en el Paso 3 (construcción de notebooks), y se detallará sesión por sesión en el Paso 2.

---

## 5. Riesgos / Puntos a Decidir Contigo

1. **Densidad de contenido:** Sesiones como la 10 (Ensambles+Benchmarking) o la 15 (Redes Neuronales+CNN) son ambiciosas para 2 horas. En el Paso 2 se acotará el alcance práctico de cada una (ej. en S15 se usará una CNN pequeña y pocas épocas de entrenamiento para que corra en tiempo de clase).
2. **Dataset para Folium (Sesión 6):** no hay aún un dataset geoespacial en `data/`; se debe elegir o generar uno (ej. ciudades con ventas, sismos, población) — puede salir de Kaggle una vez cubierta la Sesión 4.
3. **Datasets de Kaggle para Sesiones 7-14:** mencionas que tú los buscarás; en el Paso 2 dejaré marcado en cada sesión qué características debería tener el dataset ideal (tamaño, tipo de variables, si es para regresión/clasificación/clustering) para que la búsqueda en Kaggle sea dirigida.
4. **Proveedor de LLM (Sesión 16):** se asume **Google Gemini API (free tier)** como principal por ser gratuita, sin tarjeta de crédito y coherente con el entorno de Google Colab. Si prefieres otra opción 100% gratuita (ej. Groq con modelos open-source), lo cambiamos.
5. **Nivel de partida real de los estudiantes:** si ya tienen experiencia en Python/Excel, las Sesiones 2–3 podrían comprimirse y liberar tiempo para el Bloque D o E.

---

## 6. Detalle por Sesión (Paso 2 — completo)

Cada sesión tiene ahora su propia carpeta con un `README.md` que documenta: objetivo específico, agenda minuto a minuto, contenido técnico a cubrir, dataset(s) sugeridos, retos de práctica en 3 niveles (básico/medio/avanzado) y criterios de evaluación.

| # | Sesión | Detalle |
| --- | --- | --- |
| 1 | Bienvenida: DS, Python y Markdown *(sin notebook)* | [Sesion_01/README.md](Sesion_01/README.md) |
| 2 | Bases de Python I | [Sesion_02/README.md](Sesion_02/README.md) |
| 3 | Bases de Python II | [Sesion_03/README.md](Sesion_03/README.md) |
| 4 | Pandas I + Kaggle API | [Sesion_04/README.md](Sesion_04/README.md) |
| 5 | Pandas II — Data Wrangling | [Sesion_05/README.md](Sesion_05/README.md) |
| 6 | Visualización: Matplotlib, Seaborn, Plotly y Folium | [Sesion_06/README.md](Sesion_06/README.md) |
| 7 | Estadística Aplicada y Preprocesamiento | [Sesion_07/README.md](Sesion_07/README.md) |
| 8 | Métodos de Regresión | [Sesion_08/README.md](Sesion_08/README.md) |
| 9 | Métodos de Clasificación | [Sesion_09/README.md](Sesion_09/README.md) |
| 10 | Ensambles y Benchmarking de Modelos | [Sesion_10/README.md](Sesion_10/README.md) |
| 11 | Clustering y Segmentación | [Sesion_11/README.md](Sesion_11/README.md) |
| 12 | Reducción de Dimensionalidad y Selección de Variables | [Sesion_12/README.md](Sesion_12/README.md) |
| 13 | Pipeline Completo de ML (End-to-End) | [Sesion_13/README.md](Sesion_13/README.md) |
| 14 | Optimización de Hiperparámetros | [Sesion_14/README.md](Sesion_14/README.md) |
| 15 | Arquitecturas de Redes Neuronales y CNN (MNIST) | [Sesion_15/README.md](Sesion_15/README.md) |
| 16 | LLMs y NLP Aplicado (100% gratuito) | [Sesion_16/README.md](Sesion_16/README.md) |

## 7. Próximos Pasos

- **Paso 3 (pendiente):** construir los 15 notebooks (Sesiones 2–16) de estudio, práctica y taller en formato Jupyter/Colab, uno dentro de cada carpeta `Sesion_XX/`, siguiendo al pie de la letra el formato estándar de la sección 4 y el detalle documentado en cada README.
