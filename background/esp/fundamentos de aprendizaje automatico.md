# Fundamentos de Aprendizaje Automático

**Autor**: AMMS

**Repositorio**: [Repositorio de clases de BI, AA y AI](https://github.com/anguihero/Data_BI_AA_AI_Lessons)

**Fecha Actualización**: 2026/06/12

Este documento presenta una guía sobre los conceptos fundamentales del aprendizaje automático, incluyendo métodos supervisados y no supervisados, algoritmos principales, métricas de evaluación y desafíos comunes. El contenido está diseñado para servir como referencia para estudiantes y profesionales interesados en comprender los principios del machine learning, con énfasis en la intuición práctica y los criterios de decisión reales.

---

# Introducción a Métodos de Aprendizaje Automático

El Aprendizaje Automático (Machine Learning) es un campo de la inteligencia artificial que se centra en el desarrollo de programas informáticos que pueden acceder a datos y utilizarlos para aprender por sí mismos. Se divide fundamentalmente en dos grandes categorías basadas en el tipo de datos de entrenamiento: Supervisado y No Supervisado.

El uso de métodos de aprendizaje automático es necesario cuando los problemas son demasiado complejos para ser resueltos mediante reglas explícitas o cuando los datos contienen patrones que no son evidentes a simple vista. Estos métodos permiten automatizar tareas, mejorar la toma de decisiones y descubrir información valiosa a partir de grandes volúmenes de datos.

![alt text](https://scikit-learn.org/1.3/_static/ml_map.png)

## Tipos de Aprendizaje Automático

El campo del Machine Learning se organiza en tres grandes paradigmas según la naturaleza de los datos y el objetivo del aprendizaje:

| Paradigma | Datos de Entrada | Objetivo | Ejemplo |
|---|---|---|---|
| **Supervisado** | Datos etiquetados (entrada + salida conocida) | Aprender la función que mapea entradas a salidas | Predecir si un email es spam |
| **No Supervisado** | Datos sin etiquetas | Descubrir patrones o estructuras ocultas | Segmentar clientes por comportamiento |
| **Por Refuerzo** | Retroalimentación de acciones (recompensa/penalización) | Maximizar una recompensa acumulada | Entrenar un agente para jugar ajedrez |

**Ejemplos de aplicación de Aprendizaje Supervisado:**
1. **Diagnóstico médico:** Clasificar imágenes médicas para detectar enfermedades como cáncer o neumonía.
2. **Detección de fraude:** Identificar transacciones bancarias sospechosas en tiempo real.
3. **Reconocimiento de voz:** Convertir audio en texto en asistentes virtuales como Siri o Alexa.

**Ejemplos de aplicación de Aprendizaje No Supervisado:**
1. **Segmentación de clientes:** Agrupar usuarios según sus hábitos de compra para campañas de marketing personalizadas.
2. **Análisis de temas en textos:** Descubrir automáticamente temas recurrentes en grandes colecciones de documentos.
3. **Detección de anomalías:** Identificar patrones inusuales en sensores industriales para mantenimiento predictivo.

Estos ejemplos muestran cómo el aprendizaje automático puede aportar valor en distintos sectores, ayudando a resolver problemas complejos y optimizar procesos.

---

## 0. Conceptos básicos previos al aprendizaje automático

Antes de construir un modelo de aprendizaje automático, es fundamental tener dominio sobre ciertos conceptos estadísticos y matemáticos que forman la base de la mayoría de los algoritmos. Esta sección cubre los fundamentos esenciales que todo profesional de datos debe conocer.

### 0.1 Estadística Descriptiva y Probabilidad

La estadística descriptiva es el conjunto de técnicas que nos permite **resumir, organizar y analizar datos** para extraer información relevante antes de aplicar modelos de aprendizaje automático. Es el primer paso para comprender la naturaleza de los datos y detectar posibles problemas como valores atípicos, sesgos o distribuciones inusuales.

#### Medidas de Tendencia Central

Las medidas de tendencia central indican el punto alrededor del cual se agrupan los datos:

- **Media (Promedio):** Es la suma de todos los valores dividida por el número de observaciones. Es sensible a valores extremos (outliers).
  
  $$ \text{Media} = \frac{1}{n} \sum_{i=1}^{n} x_i $$

- **Mediana:** Es el valor central cuando los datos están ordenados. Es robusta frente a valores atípicos. Si el número de observaciones es par, se promedia los dos valores centrales.

- **Moda:** Es el valor que aparece con mayor frecuencia en el conjunto de datos. Un conjunto puede tener múltiples modas (distribución bimodal o multimodal).

> 💡 **¿Cuándo usar cada una?** La media es ideal cuando los datos son simétricos y sin outliers. La mediana es preferible con datos asimétricos (como salarios o precios inmobiliarios). La moda es útil para variables categóricas o discretas.

#### Medidas de Dispersión

Las medidas de dispersión muestran qué tan dispersos o concentrados están los datos respecto a la tendencia central:

- **Varianza:** Mide el promedio de las diferencias al cuadrado respecto a la media.

  $$ \text{Varianza} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$

- **Desviación estándar:** Es la raíz cuadrada de la varianza, lo que la hace más interpretable porque está en las mismas unidades que los datos.

  $$ \text{Desviación estándar} = \sqrt{\text{Varianza}} $$

- **Rango:** Diferencia entre el valor máximo y mínimo.

- **Cuartiles e IQR (Rango Intercuartílico):** Los cuartiles dividen los datos ordenados en cuatro partes iguales (Q1 = 25%, Q2 = mediana = 50%, Q3 = 75%). El IQR mide la dispersión del 50% central de los datos:

  $$ \text{IQR} = Q3 - Q1 $$

  El IQR es especialmente útil para detectar **outliers**: valores menores a $Q1 - 1.5 \cdot \text{IQR}$ o mayores a $Q3 + 1.5 \cdot \text{IQR}$ se consideran atípicos. Este criterio es la base del diagrama de caja (boxplot).

- **Asimetría (Skewness):** Indica si la distribución está "sesgada" hacia uno de los lados. Una distribución con sesgo positivo tiene una cola larga a la derecha (muchos valores bajos y pocos muy altos, como salarios). Una distribución con sesgo negativo tiene la cola larga a la izquierda.

#### Distribuciones de Probabilidad

Las distribuciones de probabilidad describen cómo se distribuyen los valores de una variable aleatoria. Son fundamentales para modelar incertidumbre y realizar inferencias estadísticas.

- **Distribución Normal (Gaussiana):** Es la más común en estadística y machine learning. Tiene forma de campana y se caracteriza por su media y desviación estándar.

  $$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$

  Una propiedad fundamental es la **regla empírica (68-95-99.7)**: el 68% de los datos cae dentro de ±1σ de la media, el 95% dentro de ±2σ y el 99.7% dentro de ±3σ.

  **Ejemplos de uso:**
  1. Modelar errores de medición en sensores.
  2. Evaluar el rendimiento académico de estudiantes.
  3. Analizar fluctuaciones de precios en mercados financieros.

- **Distribución Binomial:** Modela el número de éxitos en una secuencia de ensayos independientes con dos posibles resultados (éxito/fracaso).

  $$ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} $$

  **Ejemplos de uso:**
  1. Predecir el número de clientes que compran un producto en una campaña.
  2. Analizar resultados de pruebas médicas (positivo/negativo).
  3. Estimar la probabilidad de fallos en componentes electrónicos.

- **Distribución Poisson:** Modela el número de eventos que ocurren en un intervalo de tiempo o espacio fijo, cuando los eventos son raros y ocurren de forma independiente.

  $$ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

  **Ejemplos de uso:**
  1. Contar llamadas recibidas en un call center por hora.
  2. Analizar la llegada de clientes a una tienda.
  3. Modelar la aparición de errores en sistemas informáticos.

Estas herramientas permiten entender la estructura y el comportamiento de los datos, facilitando la selección de modelos y técnicas adecuadas para el análisis posterior.

---

### 0.2 Correlación

La **correlación** es una medida estadística que indica la fuerza y dirección de la relación entre dos variables. Es clave para identificar patrones y dependencias antes de construir modelos predictivos, pero es importante recordar que **correlación no implica causalidad**.

> 📌 **Ejemplo clásico:** Las ciudades con más heladerías tienen más ahogados. Esto no significa que las heladerías causen ahogamientos; ambas variables están correlacionadas porque ambas aumentan en verano. La variable confusora es la temperatura.

#### Interpretación de la Correlación

El coeficiente de correlación varía entre -1 y +1:

- **+1:** Correlación positiva perfecta (ambas variables aumentan juntas).
- **0:** Sin correlación (no existe relación aparente).
- **-1:** Correlación negativa perfecta (una variable aumenta mientras la otra disminuye).

Valores cercanos a ±1 indican relaciones fuertes; valores cercanos a 0 indican relaciones débiles.

| Rango absoluto | Interpretación |
|---|---|
| 0.9 – 1.0 | Correlación muy fuerte |
| 0.7 – 0.9 | Correlación fuerte |
| 0.5 – 0.7 | Correlación moderada |
| 0.3 – 0.5 | Correlación débil |
| 0.0 – 0.3 | Correlación muy débil o nula |

#### Cálculo de la Correlación según el tipo de variable

**1. Ambas Variables Numéricas**

- **Coeficiente de Correlación de Pearson (r):** Mide la relación lineal entre dos variables continuas.

  $$
  r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
  $$

- **Coeficiente de Correlación de Spearman (ρ):** Mide la relación entre los rangos de dos variables, útil cuando la relación no es lineal o los datos no son normales.

  $$
  \rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
  $$

  Donde $d_i$ es la diferencia entre los rangos de cada par de observaciones.

**2. Ambas Variables Categóricas**

- **Prueba Chi-Cuadrado ($\chi^2$):** Evalúa si existe una asociación significativa entre dos variables categóricas usando tablas de contingencia.

- **V de Cramer:** Mide la fuerza de la asociación entre variables categóricas.

  $$
  V = \sqrt{ \frac{\chi^2}{n \cdot \min(k-1, r-1)} }
  $$

  Donde $\chi^2$ es el estadístico Chi-Cuadrado, $n$ es el número total de observaciones, $k$ es el número de columnas y $r$ el número de filas.

**3. Una Variable Numérica y una Categórica**

- **Eta Cuadrado ($\eta^2$):** Mide la proporción de la varianza en la variable numérica explicada por la variable categórica.

#### Uso de la Correlación en Machine Learning

- **Selección de Variables (Feature Selection):** Se calcula la correlación entre la variable dependiente y las independientes para identificar las más relevantes y eliminar variables poco informativas.
- **Detección de Multicolinealidad:** Se analiza la matriz de correlación entre variables independientes (también conocida como **heatmap de correlación**). Correlaciones altas entre predictores (ej. > 0.8) pueden causar inestabilidad en los coeficientes del modelo. Se recomienda eliminar o combinar variables altamente correlacionadas para mejorar la robustez del modelo.
- **Análisis Exploratorio (EDA):** Una matriz de correlación visualizada como mapa de calor es una herramienta estándar al inicio de cualquier proyecto de ML para identificar relaciones relevantes rápidamente.

La correlación es una herramienta esencial para el análisis exploratorio y la construcción de modelos predictivos sólidos en machine learning.

---

### 0.3 Tipos de Variables y Datos

Entender el tipo de variable que se está analizando es fundamental, ya que determina qué técnicas estadísticas, gráficos y transformaciones son aplicables. En Machine Learning, esta clasificación influye directamente en la elección de algoritmos y en cómo se preparan los datos.

#### Variables Numéricas

Las variables numéricas representan cantidades medibles:

- **Continuas:** Pueden tomar cualquier valor dentro de un rango. Por ejemplo: temperatura, peso, precio, ingresos. Su distribución puede aproximarse con una curva continua.
- **Discretas:** Solo toman valores enteros o un conjunto contable de valores. Por ejemplo: número de hijos, cantidad de visitas a un sitio web, número de productos vendidos.

#### Variables Categóricas

Las variables categóricas representan grupos o etiquetas:

- **Nominales:** Las categorías no tienen un orden intrínseco. Por ejemplo: género, país, tipo de producto, color. No tiene sentido decir que una categoría es "mayor" que otra.
- **Ordinales:** Las categorías sí tienen un orden natural, aunque los intervalos entre ellas no sean necesariamente iguales. Por ejemplo: nivel educativo (primaria < secundaria < universidad), satisfacción del cliente (1=Malo, 2=Regular, 3=Bueno, 4=Excelente).

#### Variables Temporales

Las variables de tiempo (fechas, timestamps) tienen características especiales como estacionalidad, tendencias y componentes cíclicos. En ML se suelen descomponer en características derivadas como mes, día de la semana, hora, días desde un evento, etc.

#### Relevancia para Machine Learning

| Tipo de Variable | Técnicas de Visualización | Técnicas de Preprocesamiento |
|---|---|---|
| Numérica continua | Histograma, Boxplot, Scatter | Normalización, Estandarización |
| Numérica discreta | Histograma, Gráfico de barras | Binning, uso directo |
| Categórica nominal | Gráfico de barras, Pie chart | One-Hot Encoding |
| Categórica ordinal | Gráfico de barras | Ordinal / Label Encoding |
| Temporal | Línea de tiempo, Heatmap calendario | Extracción de características temporales |

---

## 1. Aprendizaje Supervisado (Supervised Learning)

El aprendizaje supervisado utiliza un conjunto de datos donde las entradas están emparejadas con sus salidas deseadas (o **"etiquetas"**). Es como enseñarle a un niño mostrándole un objeto (la entrada) y diciéndole qué es (la salida o etiqueta). El objetivo del modelo es aprender la función que mapea la entrada a la salida.

En términos matemáticos, se busca aprender una función $f$ tal que:

$$\hat{y} = f(X)$$

Donde $X$ es la matriz de características (features), $\hat{y}$ es la predicción y $y$ es el valor real. El modelo aprende ajustando sus parámetros para minimizar la diferencia entre $\hat{y}$ e $y$ en los datos de entrenamiento.

El proceso general del aprendizaje supervisado es:

1. Recopilar datos históricos con entradas y salidas conocidas.
2. Dividir los datos en conjunto de **entrenamiento** y **prueba** (generalmente 70-80% / 20-30%).
3. Entrenar el modelo con los datos de entrenamiento.
4. Evaluar el modelo con los datos de prueba (datos que no vio durante el entrenamiento).
5. Ajustar el modelo y repetir hasta obtener el desempeño deseado.

### 1.1 Clasificación

Los modelos de clasificación son fundamentales en el aprendizaje automático porque permiten asignar elementos a categorías específicas basándose en sus características. Estos modelos analizan datos de entrada y, mediante algoritmos como árboles de decisión, regresión logística o redes neuronales, determinan a qué clase pertenece cada ejemplo. Por ejemplo, en un sistema de detección de spam, el modelo clasifica los correos electrónicos como "spam" o "no spam" según su contenido y otros atributos.

La característica principal de los modelos de clasificación es que la salida es una **categoría discreta**, es decir, el resultado pertenece a un conjunto limitado de clases predefinidas. Esto los diferencia de los modelos de regresión, donde la salida es un valor continuo. Los modelos de clasificación se utilizan en aplicaciones como reconocimiento de imágenes, diagnóstico médico y análisis de sentimientos, donde es esencial identificar a qué grupo pertenece cada dato analizado.

Se distinguen dos grandes subtipos:
- **Clasificación Binaria:** Solo existen dos clases posibles (spam/no spam, fraude/no fraude, enfermo/sano).
- **Clasificación Multiclase:** Existen tres o más clases posibles (clasificar dígitos del 0 al 9, identificar especie de planta, detectar tipo de objeto en imagen).

![alt text](https://www.themachinelearners.com/wp-content/uploads/2021/01/1_aE8XLyApqvaQA9B7MWjjlA.png)

1. **Detección de fraude en transacciones:** Clasificar transacciones financieras como fraudulentas o legítimas basándose en patrones de gasto y comportamiento del usuario.

2. **Diagnóstico médico:** Clasificar imágenes radiológicas para determinar si hay presencia de enfermedades como neumonía, tumores o fracturas.

3. **Análisis de sentimientos:** Clasificar reseñas de productos o comentarios en redes sociales como positivos, negativos o neutrales.

4. **Reconocimiento facial:** Identificar y clasificar personas en imágenes para sistemas de seguridad o verificación de identidad.

5. **Filtrado de contenido:** Clasificar contenido web o publicaciones en redes sociales como apropiado o inapropiado para distintas audiencias.


### 1.2 Regresión

Los modelos de regresión son fundamentales en el análisis predictivo porque permiten estimar relaciones cuantitativas entre variables. A diferencia de los modelos de clasificación, que asignan categorías discretas, los modelos de regresión generan una salida **continua**, lo que los hace ideales para tareas como predicción de precios, estimación de demanda, o análisis de tendencias. Estos modelos pueden ser simples, como la regresión lineal, donde se asume una relación lineal entre variables, o más complejos, como la regresión polinómica, Ridge, Lasso o incluso regresión basada en redes neuronales, que capturan relaciones no lineales y múltiples interacciones.

La salida **continua** de un modelo de regresión significa que el valor predicho puede tomar cualquier número dentro de un rango, lo que permite una mayor precisión en contextos donde los resultados no se limitan a clases discretas. Por ejemplo, al predecir la temperatura, el ingreso mensual o el tiempo de espera, se requiere un modelo que pueda ajustarse a variaciones sutiles. Además, los modelos de regresión permiten evaluar el impacto de cada variable independiente sobre la variable dependiente, lo que los convierte en herramientas poderosas para la interpretación y la toma de decisiones informadas en entornos científicos, económicos y operativos.


![alt text](https://pub.mdpi-res.com/ijerph/ijerph-15-02907/article_deploy/html/images/ijerph-15-02907-g001.png?1570846772)

Los modelos de regresión son fundamentales en muchas aplicaciones prácticas donde necesitamos predecir valores numéricos continuos.

Algunos ejemplos destacados incluyen:

1. **Predicción de precios inmobiliarios:** Estimar el valor de una vivienda basándose en características como ubicación, tamaño, número de habitaciones, antigüedad y amenidades cercanas.

![alt text](https://drek4537l1klr.cloudfront.net/serrano/v-4/Figures/image028.png)

2. **Pronóstico de ventas:** Predecir los ingresos futuros de una empresa considerando datos históricos, estacionalidad, tendencias del mercado y variables económicas.

3. **Estimación de consumo energético:** Calcular la demanda energética de edificios basándose en factores como clima, aislamiento, ocupación y sistemas instalados.

4. **Predicción de rendimiento de cultivos:** Estimar la producción agrícola según variables como precipitaciones, temperatura, tipo de suelo y fertilizantes utilizados.

5. **Valoración del impacto ambiental:** Modelar la relación entre emisiones contaminantes y factores como actividad industrial, población y medidas de control para predecir niveles futuros de contaminación.

Estos ejemplos demuestran cómo la regresión permite tomar decisiones informadas en áreas tan diversas como finanzas, planificación urbana, sostenibilidad y agricultura, proporcionando estimaciones numéricas precisas basadas en datos.

### 1.3 Algoritmos

A continuación se presenta un resumen de los algoritmos más utilizados en aprendizaje supervisado. Cada uno tiene fortalezas particulares; la elección del algoritmo adecuado depende del tipo de problema, la cantidad de datos disponible y el nivel de interpretabilidad requerido.

|Algoritmo|Tipo de Tarea Principal|Descripción Breve|Hiperparámetros Clave|
|-----------------:|----------------:|-----------------:|---------------|
|Regresión Lineal| Regresión|Modela la relación entre una variable dependiente (salida) y una o más variables independientes (entradas) ajustando la mejor línea recta a los datos.|No tiene hiperparámetros de aprendizaje. Los parámetros se calculan directamente (mínimos cuadrados). A menudo, solo se considera la tasa de aprendizaje si se usa Descenso de Gradiente.|
|Regresión Logística|Clasificación|Utiliza la función logística para estimar la probabilidad de que una instancia pertenezca a una clase. A pesar de su nombre, es un modelo de clasificación binaria (dos clases).|1. C (o λ): Inverso de la fuerza de regularización. Valores más pequeños especifican una regularización más fuerte. 2. Penalty (Penalización): Tipo de regularización aplicada (L1 o L2). 3. Solver: Algoritmo a utilizar en la optimización (ej. liblinear, saga, lbfgs).|
|Árboles de Decisión|Clasificación / Regresión|Crean un modelo que predice el valor de una variable objetivo (salida) dividiendo el conjunto de datos de entrenamiento en subconjuntos basados en los valores de las características (entradas), formando una estructura similar a un árbol.|1. max_depth: Profundidad máxima del árbol. 2. min_samples_split: Número mínimo de muestras requeridas para dividir un nodo interno. 3. criterion: Función para medir la calidad de una división (ej. gini o entropía para clasificación).|
|Random Forest|Clasificación / Regresión|Es un método de ensamble que construye múltiples árboles de decisión y combina sus predicciones para mejorar la precisión y evitar el sobreajuste.|1. n_estimators: Número de árboles en el bosque. 2. max_features: Número de características a considerar para la mejor división en cada nodo. 3. max_depth: Profundidad máxima de cada árbol individual.|
|Support Vector Machine (SVM)|Clasificación / Regresión|Encuentra un hiperplano óptimo que separa las clases en un espacio de alta dimensión, maximizando el margen entre el hiperplano y los puntos de datos más cercanos (vectores de soporte).|1. C: Parámetro de regularización. Penaliza los errores de clasificación. 2. kernel: Tipo de función de núcleo (linear, poly, rbf (Radial Basis Function), sigmoid). 3. gamma (γ): Parámetro del kernel rbf (define cuánta influencia tiene un solo ejemplo de entrenamiento).|
|K-Nearest Neighbors (KNN)|Clasificación / Regresión|Es un algoritmo no paramétrico que clasifica un nuevo punto basándose en la mayoría de las clases de sus K vecinos más cercanos en el espacio de características.| 1. n_neighbors (K): El número de vecinos a considerar. 2. weights: Función de ponderación utilizada en la predicción (uniform o distance). 3. metric: Métrica de distancia a utilizar (euclidean, manhattan, minkowski).|
|Boosting|Clasificación / Regresión|Crea un modelo fuerte a partir de una secuencia de modelos débiles, corrigiendo los errores del modelo anterior en cada iteración.|1. n_estimators (Número de modelos débiles). 2. learning_rate (Peso de cada modelo). 3. max_depth (Si usa árboles débiles).|
|XGBoost (eXtreme Gradient Boosting)|Clasificación / Regresión|Un algoritmo de boosting muy eficiente y popular que mejora iterativamente un conjunto de modelos débiles (típicamente árboles) para formar un modelo predictivo fuerte.|1. n_estimators: Número de rondas de aumento (número de árboles). 2. learning_rate (η): Tasa de aprendizaje o tamaño del paso en cada iteración. 3. max_depth: Profundidad máxima de cada árbol. 4. gamma (γ): Mínima pérdida de reducción necesaria para hacer una división adicional. 5. subsample: Fracción de muestras aleatorias a utilizar para entrenar cada árbol.|

#### Guía de Selección de Algoritmo

Más allá de los parámetros técnicos, elegir el algoritmo correcto requiere entender sus fortalezas y limitaciones en contextos reales:

| Algoritmo | Fortalezas | Debilidades | Mejor Caso de Uso |
|---|---|---|---|
| Regresión Lineal | Muy interpretable, rápido, funciona bien con relaciones lineales | No captura relaciones no lineales | Predecir ventas o precios con pocas variables numéricas |
| Regresión Logística | Probabilidades calibradas, interpretable, rápido | Solo captura fronteras de decisión lineales | Scoring crediticio, detección de fraude simple |
| Árbol de Decisión | Muy interpretable, maneja datos mixtos, no requiere escalado | Propenso a sobreajuste, inestable ante pequeños cambios en datos | Cuando la interpretabilidad es prioritaria para el negocio |
| Random Forest | Robusto, maneja bien datos ruidosos y variables irrelevantes | Menos interpretable, lento con muchos árboles | Clasificación general con datos de tamaño mediano |
| SVM | Efectivo en alta dimensionalidad, robusto a outliers | Lento con grandes datasets, difícil de interpretar | Clasificación de texto, bioinformática, datasets pequeños |
| KNN | Simple, no asume distribución, fácil de entender | Lento en predicción, sensible a la escala y a outliers | Sistemas de recomendación, datasets pequeños |
| XGBoost | Muy preciso, maneja valores faltantes, regularización integrada | Requiere tuning cuidadoso, menos interpretable | Competencias de ML, datasets tabulares complejos |


### 1.4 Métricas de Validación

Las métricas de validación son herramientas fundamentales para **evaluar el desempeño de los modelos** de aprendizaje automático. Permiten cuantificar qué tan bien un modelo realiza tareas de clasificación o regresión, ayudando a **comparar diferentes algoritmos** y **ajustar sus parámetros** para obtener mejores resultados.

![alt text](https://db0dce98.rocketcdn.me/es/files/2024/08/Schema-model_evaluation-42-42.png)

En clasificación, las métricas como la exactitud, precisión, recall y F1-score ayudan a entender si el modelo identifica correctamente las clases, lo cual es crucial en aplicaciones como la detección de fraude (donde es importante minimizar los falsos positivos) o el diagnóstico médico (donde es vital reducir los falsos negativos).

#### Matriz de Confusión

La **Matriz de Confusión** es la base de casi todas las métricas de clasificación. Resume en una tabla los cuatro posibles resultados de una predicción binaria:

|  | Predicho: Positivo | Predicho: Negativo |
|---|---|---|
| **Real: Positivo** | **TP** (Verdadero Positivo) | **FN** (Falso Negativo) |
| **Real: Negativo** | **FP** (Falso Positivo) | **TN** (Verdadero Negativo) |

- **TP (True Positive):** El modelo predijo positivo y era positivo. ✅
- **TN (True Negative):** El modelo predijo negativo y era negativo. ✅
- **FP (False Positive) — Error Tipo I:** El modelo predijo positivo pero era negativo. ❌ (falsa alarma)
- **FN (False Negative) — Error Tipo II:** El modelo predijo negativo pero era positivo. ❌ (caso perdido)

> 💡 **Ejemplo práctico:** En un test de detección de una enfermedad grave, un **Falso Negativo** (decirle a alguien enfermo que está sano) puede ser mortal. Por eso en medicina se prioriza minimizar los FN, lo que equivale a maximizar el **Recall**. En cambio, en un filtro de spam, un **Falso Positivo** (clasificar un email legítimo como spam) es el error más costoso, por lo que se prioriza la **Precisión**.

![alt text](https://almablog-media.s3.ap-south-1.amazonaws.com/image_14_4f4fc2cf7d.png)

En regresión, métricas como el error cuadrático medio (MSE), el error absoluto medio (MAE) y el coeficiente de determinación (R²) permiten medir la diferencia entre los valores predichos y los reales. Por ejemplo, al predecir el precio de una vivienda, un bajo MSE indica que el modelo realiza estimaciones cercanas a los valores reales.

![alt text](https://miro.medium.com/1*5fnmYVHLTC8mGxybHm4XkA.png)

Seleccionar la métrica adecuada depende del problema y del impacto de los errores. Por eso, entender y aplicar correctamente estas métricas es esencial para desarrollar modelos robustos y útiles en la práctica.


|Métrica|Tipo de Uso|Cómo se Calcula|Fortalezas y Debilidades|
|-------|-----------|---------------|------------------------|
|Exactitud (Accuracy)|Clasificación|$\frac{TP + TN}{TP + TN + FP + FN}$ — Proporción de predicciones correctas sobre el total.|Fácil de interpretar, pero puede ser engañosa en conjuntos de datos desbalanceados.|
|Precisión (Precision)|Clasificación|$\frac{TP}{TP + FP}$ — De todo lo que predije como positivo, ¿cuánto era realmente positivo?|Útil cuando el costo de los falsos positivos es alto; puede ignorar falsos negativos.|
|Recall (Sensibilidad)|Clasificación|$\frac{TP}{TP + FN}$ — De todos los positivos reales, ¿cuántos identifiqué correctamente?|Importante cuando el costo de los falsos negativos es alto; puede ignorar falsos positivos.|
|F1-Score|Clasificación|Media armónica entre precisión y recall: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$|Equilibra precisión y recall; útil en datos desbalanceados.|
|AUC-ROC|Clasificación|Área bajo la curva ROC, que compara la tasa de verdaderos positivos vs. falsos positivos a distintos umbrales de clasificación.|Evalúa el rendimiento en todos los umbrales; robusta en datos desbalanceados.|
|MSE (Error Cuadrático Medio)|Regresión|$\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$|Penaliza fuertemente los errores grandes; sensible a valores atípicos.|
|RMSE (Raíz del MSE)|Regresión|$\sqrt{MSE}$|Misma unidad que la variable objetivo; más interpretable que el MSE puro.|
|MAE (Error Absoluto Medio)|Regresión|$\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$|Menos sensible a valores atípicos que MSE; fácil de interpretar.|
|R² (Coeficiente de Determinación)|Regresión|Proporción de la varianza de $y$ explicada por el modelo: $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$|Indica el ajuste global del modelo; puede ser negativo si el modelo es peor que predecir siempre la media.|

#### ¿Cuándo usar cada métrica de clasificación?

El contexto del negocio determina cuál error es más costoso, y por tanto qué métrica priorizar:

| Contexto | Métrica Prioritaria | Razón |
|---|---|---|
| Detección de enfermedades graves | **Recall** | Minimizar falsos negativos (no perder un caso real) |
| Filtro de spam | **Precision** | Minimizar falsos positivos (no descartar emails legítimos) |
| Detección de fraude bancario | **F1-Score / AUC-ROC** | Balance entre no bloquear clientes y atrapar fraudes |
| Dataset balanceado, clasificación general | **Accuracy** | Métrica simple cuando las clases están bien balanceadas |
| Comparar modelos con distintos umbrales | **AUC-ROC** | Evalúa el desempeño global independientemente del umbral elegido |

---

## 2. Aprendizaje No Supervisado (Unsupervised Learning)

El aprendizaje no supervisado trabaja con datos **sin etiquetas**. El modelo debe encontrar patrones, estructuras o relaciones ocultas dentro de los datos por sí mismo. No existe una "respuesta correcta" que guíe el aprendizaje; en cambio, el algoritmo busca organizar los datos según su propia estructura interna.

El aprendizaje no supervisado es especialmente valioso cuando:
- No se tienen etiquetas disponibles (caso muy común en la práctica real).
- Se quiere explorar y entender la estructura de los datos antes de definir el problema supervisado.
- Se busca comprimir o simplificar datos de alta dimensionalidad.
- Se quiere descubrir segmentos naturales dentro de los datos sin imponer categorías a priori.

El uso principal es:

![alt text](https://scikit-learn.org/0.18/_images/sphx_glr_plot_cluster_comparison_001.png)

* **Clustering (Agrupamiento)**: Agrupar datos similares entre sí.
* **Reducción de Dimensionalidad**: Simplificar los datos preservando la mayor información posible.
* **Detección de Anomalías**: Identificar observaciones que difieren significativamente del patrón general.
* **Reglas de Asociación**: Descubrir relaciones frecuentes entre items (ejemplo: análisis de cesta de mercado).

### 2.1 Clustering

El **clustering** o agrupamiento es la tarea de dividir un conjunto de datos en grupos (clusters) donde los datos dentro de un mismo grupo son más similares entre sí que con los datos de otros grupos. Es una de las técnicas más utilizadas para segmentación, exploración de datos y detección de patrones.

**Ejemplos de aplicación:**
1. **Segmentación de clientes:** Identificar grupos de usuarios con comportamientos de compra similares para personalizar estrategias de marketing.
2. **Agrupación de documentos:** Organizar automáticamente colecciones de noticias o artículos por temática sin etiquetas previas.
3. **Análisis genómico:** Agrupar genes con patrones de expresión similares para descubrir funciones biológicas relacionadas.
4. **Detección de anomalías:** Puntos que no pertenecen a ningún cluster claro pueden ser anomalías o fraudes.
5. **Compresión de imágenes:** Reducir la paleta de colores de una imagen agrupando colores similares con K-Means.

|Algoritmo|Tipo de Tarea Principal|Descripción Breve|Hiperparámetros Clave|
|-----------------:|----------------:|-----------------:|---------------|
|K-Means|Clustering|Divide los datos en K grupos, donde K es un número predefinido. Asigna cada punto al centroide (punto central) más cercano e itera hasta convergencia.|1. n_clusters (K): El número de clusters que se deben formar. 2. init: Método de inicialización de los centroides (k-means++ o random). 3. max_iter: Número máximo de iteraciones para el algoritmo.|
|Clustering Jerárquico|Clustering|Construye una jerarquía de clusters visualizable en un dendrograma. Puede ser aglomerativo (comenzar con puntos individuales y unirlos) o divisivo (comenzar con un cluster grande y dividirlo).|1. n_clusters: El número de clusters al que detener el proceso. 2. linkage: Criterio de conexión entre conjuntos de observaciones (ward, average, complete). 3. affinity (o metric): Métrica de distancia utilizada (euclidean, manhattan).|
|Clustering Basado en Densidad (DBSCAN)|Clustering|Identifica clusters basándose en la densidad de los puntos de datos. Puede encontrar clusters de formas arbitrarias y distingue automáticamente el ruido (outliers), que se marcan como puntos sin cluster asignado.|1. eps (ϵ): La distancia máxima entre dos muestras para que se consideren vecinas. 2. min_samples: El número mínimo de puntos para que una región se considere densa. 3. metric: Métrica de distancia a utilizar.|

#### El Método del Codo (Elbow Method)

Un desafío clave en K-Means es determinar el valor óptimo de K. El **método del codo** consiste en calcular la **inercia** (suma de distancias al cuadrado entre cada punto y su centroide) para distintos valores de K y graficarlos. El punto donde la curva empieza a disminuir más lentamente — formando un "codo" — indica el K óptimo. Complementarlo con el **Silhouette Score** da una visión más completa de la calidad del agrupamiento.

### 2.2 Reducción de Dimensionalidad

La **reducción de dimensionalidad** es el proceso de reducir el número de variables (características) de un conjunto de datos, preservando la mayor cantidad posible de información relevante. Es crucial cuando se trabaja con datos de alta dimensionalidad (muchas columnas), ya que:

- Reduce el tiempo de entrenamiento de los modelos.
- Combate la **"maldición de la dimensionalidad"** (*curse of dimensionality*): en espacios de alta dimensión, los datos se vuelven muy dispersos y las métricas de distancia pierden significado.
- Facilita la visualización de los datos (reducir a 2D o 3D para graficar).
- Puede eliminar ruido y variables redundantes, mejorando la precisión del modelo.

#### Análisis de Componentes Principales (PCA)

El **PCA (Principal Component Analysis)** es el algoritmo de reducción de dimensionalidad más popular. Su objetivo es encontrar las **componentes principales**, que son nuevas variables (combinaciones lineales de las originales) que capturan la máxima varianza de los datos, organizadas de mayor a menor importancia.

**Funcionamiento paso a paso:**
1. Estandarizar los datos (media 0, varianza 1) — obligatorio para que PCA no esté sesgado por la escala.
2. Calcular la matriz de covarianza.
3. Obtener los vectores propios (eigenvectors) y valores propios (eigenvalues).
4. Seleccionar los k vectores propios con los valores propios más grandes — estas son las k **componentes principales**.
5. Proyectar los datos en el nuevo espacio de k dimensiones.

La **varianza explicada acumulada** por las componentes indica qué porcentaje de la información original se conserva. Se recomienda seleccionar el número de componentes que acumulen al menos el **95% de la varianza total**.

**Aplicaciones:**
1. **Visualización:** Reducir datasets con cientos de variables a 2D para explorar grupos visualmente antes de clustering.
2. **Eliminación de ruido:** Las últimas componentes (menor varianza) suelen capturar ruido, no señal útil.
3. **Preprocesamiento:** Reducir la dimensionalidad antes de aplicar algoritmos sensibles como KNN o SVM.
4. **Reconocimiento facial:** El método "Eigenfaces" usa PCA para representar rostros en un espacio reducido.

#### t-SNE

El **t-SNE** (t-Distributed Stochastic Neighbor Embedding) es una técnica de reducción de dimensionalidad **no lineal** especialmente útil para visualización en 2D o 3D. A diferencia de PCA (que es lineal), t-SNE preserva las estructuras locales de los datos, revelando clusters que PCA no siempre muestra. Su limitación principal es que es computacionalmente costoso y los ejes del resultado no tienen interpretación directa — se usa solo para exploración visual.

### 2.3 Reglas de Asociación

Las **reglas de asociación** buscan patrones frecuentes de co-ocurrencia en grandes conjuntos de datos transaccionales. Son la base del análisis de **cesta de mercado** (*market basket analysis*).

**Concepto clave:** "Los clientes que compran pañales también tienden a comprar cerveza los viernes."

Las reglas tienen la forma: **Si {A} → entonces {B}**, y se evalúan con tres métricas principales:

- **Soporte (Support):** Frecuencia con la que aparece el conjunto de items en el total de transacciones.

  $$ \text{Soporte}(A \rightarrow B) = \frac{\text{transacciones con A y B}}{\text{total de transacciones}} $$

- **Confianza (Confidence):** De las veces que aparece A, ¿con qué frecuencia también aparece B?

  $$ \text{Confianza}(A \rightarrow B) = \frac{\text{transacciones con A y B}}{\text{transacciones con A}} $$

- **Lift:** Mide cuánto más probable es B dado A, comparado con la probabilidad base de B. Un lift > 1 indica una asociación positiva real (no aleatoria).

  $$ \text{Lift}(A \rightarrow B) = \frac{\text{Confianza}(A \rightarrow B)}{\text{Soporte}(B)} $$

**Algoritmos principales:** Apriori, FP-Growth.

**Aplicaciones:**
1. **Retail y e-commerce:** Recomendaciones del tipo "Los clientes que compraron X también compraron Y".
2. **Colocación de productos:** Decidir qué productos ubicar juntos en la tienda física.
3. **Salud:** Descubrir combinaciones de síntomas o medicamentos que co-ocurren frecuentemente en registros clínicos.
4. **Marketing:** Identificar combinaciones de servicios que los clientes suelen contratar en conjunto para diseñar paquetes.

### 2.4 Métricas de Validación

Evaluar la calidad del aprendizaje no supervisado es más complejo que en el supervisado, ya que no existe una "respuesta correcta" con la cual comparar. Las métricas **internas** evalúan la calidad usando solo los datos; las **externas** comparan con etiquetas conocidas (cuando existen).

|Métrica|Tipo de Uso|Cómo se Calcula|Fortalezas y Debilidades|
|-------|-----------|---------------|------------------------|
|Silhouette Score|Clustering|Promedio de la diferencia entre la distancia intra-cluster y la distancia al cluster más cercano. Rango: [-1, 1]. Valores cercanos a 1 son ideales.|Evalúa separación y cohesión sin necesidad de etiquetas verdaderas; intuitivo.|
|Davies-Bouldin Index|Clustering|Promedio de la relación entre la dispersión intra-cluster y la distancia inter-cluster.|Menor valor indica mejor clustering; sensible a clusters de diferente tamaño.|
|Calinski-Harabasz Index|Clustering|Relación entre la dispersión entre clusters y la dispersión dentro de los clusters.|Mayor valor indica mejor clustering; favorece clusters compactos y bien separados.|
|Homogeneidad|Clustering con etiquetas|Mide si cada cluster contiene solo miembros de una sola clase real.|Útil si se conocen las etiquetas verdaderas; no aplicable en clustering puro sin etiquetas.|
|Completeness|Clustering con etiquetas|Mide si todos los miembros de una clase están en el mismo cluster.|Complementa la homogeneidad; útil con etiquetas verdaderas.|
|Varianza Explicada|Reducción de Dimensionalidad|Proporción de la varianza total capturada por los componentes seleccionados.|Indica cuánta información se conserva; no mide interpretabilidad de las componentes.|



---

## 3. Desafios del aprendizaje automatico

La aplicación práctica del machine learning enfrenta numerosos desafíos que van más allá de simplemente seleccionar un algoritmo. Estos obstáculos son los que diferencian un proyecto de ML exitoso de uno que falla en producción. Comprender estos desafíos y sus soluciones es fundamental para cualquier profesional de datos.

### 3.1 Subajuste y Sobreajuste

![alt text](https://media.licdn.com/dms/image/v2/D4E22AQFLsYgMYO-H7Q/feedshare-shrink_800/B4EZkk_dW5KYAg-/0/1757262241175?e=2147483647&v=beta&t=FSDUbc3ZebNr4pO1yqRu6awd1VHCny45aOzxyFwMcoY)

El **subajuste (underfitting)** ocurre cuando un modelo es demasiado simple para capturar los patrones relevantes de los datos, lo que resulta en un bajo desempeño tanto en el conjunto de entrenamiento como en el de prueba. El **sobreajuste (overfitting)** sucede cuando el modelo es demasiado complejo y aprende detalles o ruido específico del conjunto de entrenamiento, perdiendo capacidad de generalización y mostrando alto desempeño en entrenamiento pero bajo en prueba. Ninguno de estos escenarios es deseado, ya que impiden que el modelo sea útil en datos nuevos.

| Situación | Error en Entrenamiento | Error en Prueba | Solución Típica |
|---|---|---|---|
| **Underfitting** | Alto | Alto | Modelo más complejo, más features, menos regularización |
| **Ajuste óptimo** | Bajo-moderado | Bajo-moderado | ✅ Modelo listo para producción |
| **Overfitting** | Muy bajo | Alto | Más datos, regularización L1/L2, reducción de complejidad |

### 3.2 Desbalanceo de Clases

![alt text](https://cdn.sanity.io/images/31qskqlc/production/a680f0dd5ab72cd0dfb06effd8cdbfa0858ac6a8-850x647.webp?fit=max&auto=format)

El **desbalanceo de clase** aparece cuando algunas clases están representadas por muchas más muestras que otras. Esto puede llevar a que el modelo ignore las clases minoritarias, afectando la precisión y utilidad en aplicaciones críticas (por ejemplo, detección de fraude o enfermedades raras).

> 📌 **Ejemplo:** En detección de fraude, el 99% de transacciones son legítimas y el 1% son fraudes. Un modelo que prediga siempre "no fraude" tendría 99% de Accuracy pero no detectaría ningún caso real de fraude.

**Estrategias para manejar el desbalanceo:**
1. **Resampling:** Aplicar oversampling (duplicar muestras minoritarias o generar sintéticas con SMOTE) o undersampling (reducir muestras de la clase mayoritaria).
2. **Pesos de clase:** La mayoría de algoritmos en scikit-learn permiten asignar mayor penalización a errores en la clase minoritaria con `class_weight='balanced'`.
3. **Métricas adecuadas:** Usar F1-Score, AUC-ROC o Recall en lugar de Accuracy cuando hay desbalance.
4. **Algoritmos robustos:** XGBoost incluye el parámetro `scale_pos_weight` para manejar el desbalance directamente durante el entrenamiento.

### 3.3 Calidad y Cantidad de Datos

La **calidad** de los datos (presencia de errores, valores faltantes, ruido) y la **cantidad** insuficiente de datos pueden limitar la capacidad del modelo para aprender patrones útiles. Datos pobres o escasos suelen llevar a modelos poco confiables y resultados engañosos.

**Problemas de calidad más comunes:**

| Problema | Descripción | Impacto en el Modelo |
|---|---|---|
| **Valores faltantes** | Observaciones con campos vacíos o NA | Errores en entrenamiento; sesgo si los faltantes no son aleatorios |
| **Outliers** | Valores extremos alejados del resto | Distorsionan la media y afectan algoritmos sensibles a escala |
| **Duplicados** | Registros repetidos en el dataset | Sobreajuste, evaluación artificialmente inflada |
| **Errores tipográficos** | Inconsistencias en categorías ("Colombia", "colombia", "COLOMBIA") | Fragmentación artificial de categorías |
| **Sesgo de muestreo** | Los datos no representan bien la población objetivo | El modelo aprende patrones sesgados que no generalizan |
| **Data leakage** | Filtración de información del futuro al entrenamiento | Métricas artificialmente altas que no se replican en producción |

**Sobre la cantidad de datos:** Como regla general, se necesita al menos 10 veces más muestras que features para modelos simples. Algoritmos complejos como redes neuronales profundas requieren millones de muestras. Cuando los datos son escasos:
- Usar algoritmos más simples (regresión lineal, árboles poco profundos).
- Aplicar técnicas de aumento de datos (data augmentation) cuando aplica.
- Considerar transferencia de aprendizaje (transfer learning) si hay modelos pre-entrenados disponibles.

### 3.4 Tuneo de Hiperparámetros

El **ajuste de hiperparámetros** implica explorar diferentes configuraciones para encontrar la combinación que optimiza el desempeño del modelo. Es una forma de explorar soluciones y evitar tanto el subajuste como el sobreajuste.

La distinción clave es:
- **Parámetros:** Se aprenden automáticamente durante el entrenamiento (ej. los coeficientes de una regresión, los pesos de una red neuronal).
- **Hiperparámetros:** Se definen **antes** del entrenamiento y controlan el proceso de aprendizaje (ej. número de árboles en Random Forest, tasa de aprendizaje, profundidad máxima).

#### 3.4.1 algoritmos de tuneo de Hiperparámetros

| Método de Tuneo de Hiperparámetros | Cómo Funciona | Fortalezas | Debilidades |
|------------------------------------|---------------|------------|-------------|
| Grid Search                       | Explora todas las combinaciones posibles de hiperparámetros en una cuadrícula definida. | Exhaustivo; garantiza encontrar la mejor combinación en el espacio definido. | Costoso computacionalmente; no escala bien con muchos hiperparámetros. |
| Random Search                     | Selecciona combinaciones aleatorias de hiperparámetros dentro los rangos definidos. | Más eficiente que grid search; puede encontrar buenas combinaciones rápidamente. | Puede pasar por alto la mejor combinación; resultados dependen del azar. |
| Bayesian Optimization             | Modela la función objetivo y selecciona los hiperparámetros basándose en resultados previos para maximizar el desempeño. | Eficiente; requiere menos evaluaciones; aprende de iteraciones anteriores. | Más complejo de implementar; depende de la calidad del modelo probabilístico. |
| Hyperband                        | Utiliza técnicas de muestreo y parada temprana para asignar recursos de manera eficiente entre configuraciones. | Rápido; ahorra recursos; bueno para grandes espacios de búsqueda. | Puede descartar buenas configuraciones prematuramente; requiere ajuste de parámetros propios. |
| Optuna                            | Algoritmo de optimización automática que ajusta el espacio de búsqueda dinámicamente y utiliza técnicas avanzadas como pruning. | Flexible; eficiente; fácil de integrar con frameworks modernos. | Puede requerir configuración avanzada; resultados dependen de la definición del espacio de búsqueda. |

### 3.5 Estrategias de Muestreo y Validación

![alt text](https://towardsdatascience.com/wp-content/uploads/2021/02/1l3NEnB5bThd0uqxVe5Mbqg.jpeg)

Para abordar estos desafíos, se emplean varias estrategias:

| Método| Descripción| Cuándo Usar| Fortalezas| Debilidades|
|----------|----------|----------|----------|----------|
| Partición de datos| Separar el conjunto en datos de entrenamiento y prueba.| Siempre, para evaluar la capacidad de generalización del modelo.| Fácil de implementar; rápida evaluación.| Puede depender de la partición elegida; no aprovecha todos los datos para entrenamiento.|
| Validación cruzada| Divide los datos en varias particiones y alterna entrenamiento/validación en cada una.| Cuando se necesita una evaluación robusta y se dispone de pocos datos.| Reduce la varianza de la evaluación; usa todos los datos para entrenamiento y validación.| Más costosa computacionalmente; puede ser lenta con grandes conjuntos de datos.|
| Subsampling y Oversampling| Disminuir o aumentar muestras de clases para equilibrar el conjunto.| Cuando hay desbalance de clases en el conjunto de datos.| Mejora el balance de clases; fácil de aplicar.| Puede eliminar información útil (subsampling) o causar sobreajuste (oversampling).|
| SMOTE (Synthetic Minority Over-sampling)  | Genera ejemplos sintéticos de la clase minoritaria interpolando entre muestras existentes.| Cuando la clase minoritaria es muy pequeña y el oversampling tradicional no es suficiente.   | Mejora el balance sin duplicar datos; reduce el sobreajuste.| Puede generar ejemplos poco realistas; requiere cuidado en la aplicación.|

#### Validación Cruzada k-Fold en detalle

La **validación cruzada k-fold** es la técnica estándar para evaluar modelos de manera robusta. El proceso es:

1. Dividir el conjunto de datos en k particiones (folds) iguales.
2. Usar k-1 folds para entrenamiento y 1 fold para validación.
3. Repetir k veces, usando cada fold como validación exactamente una vez.
4. Reportar el **promedio** y la **desviación estándar** del desempeño en los k folds.

Valores comunes de k son **5** y **10**. Un caso especial es el **Leave-One-Out Cross-Validation (LOOCV)**, donde k es igual al número de muestras — muy preciso pero computacionalmente muy costoso para datasets grandes.

> ⚠️ Siempre aplicar las transformaciones de preprocesamiento (escalado, imputación) **dentro** de cada fold y no antes de la división. Aplicarlas antes causa *data leakage* y genera evaluaciones artificialmente optimistas.

Estas técnicas ayudan a construir modelos más robustos y confiables, mitigando los problemas comunes en aprendizaje automático.


### 3.6 El Trade-off de Sesgo vs. Varianza (Bias-Variance Tradeoff)

Este es quizás el concepto más importante en machine learning para diagnosticar problemas en un modelo. El error total de un modelo se puede descomponer matemáticamente en tres componentes:

$$\text{Error Total} = \text{Sesgo}^2 + \text{Varianza} + \text{Error Irreducible}$$

- **Sesgo (Bias):** Es el error por suposiciones demasiado simplistas en el modelo. Un **alto sesgo** lleva al **underfitting**: el modelo no es suficientemente flexible para capturar la relación real entre las variables. *Ejemplo:* usar una regresión lineal para datos con una relación claramente cuadrática.

- **Varianza (Variance):** Es la sensibilidad del modelo a pequeñas fluctuaciones en los datos de entrenamiento. Una **alta varianza** lleva al **overfitting**: el modelo "memoriza" los datos de entrenamiento (incluido el ruido) y no generaliza bien a datos nuevos. *Ejemplo:* un árbol de decisión sin límite de profundidad que aprende cada excepción del training set.

- **Error Irreducible:** Es el ruido inherente al problema que ningún modelo puede eliminar, independientemente de su sofisticación.

| Característica | Alto Sesgo (Underfitting) | Alta Varianza (Overfitting) |
|---|---|---|
| Error en entrenamiento | Alto | Muy bajo |
| Error en prueba | Alto | Alto |
| Modelo típico | Demasiado simple (lineal con datos no lineales) | Demasiado complejo (árbol sin poda) |
| Síntoma clave | Error similar en train y test, ambos altos | Brecha grande entre error en train (bajo) y test (alto) |
| Solución | Aumentar complejidad, más features | Regularización, más datos, reducir complejidad |

El objetivo es encontrar el **punto de equilibrio** donde el error de validación es mínimo — la complejidad óptima del modelo.

**Estrategias para reducir el Sesgo:**
- Usar un modelo más complejo (mayor profundidad, más capas, más estimadores).
- Crear nuevas características relevantes (feature engineering).
- Reducir la regularización.

**Estrategias para reducir la Varianza:**
- Obtener más datos de entrenamiento.
- Aplicar regularización: L1 (Lasso), L2 (Ridge), Elastic Net, Dropout en redes neuronales.
- Usar técnicas de ensamble basadas en Bagging (como Random Forest), que promedian múltiples modelos de alta varianza para reducirla.
- Reducir la complejidad del modelo (limitar la profundidad de los árboles, menos neuronas).

### 3.7  Preprocesamiento de Datos (Data Preprocessing)

Los datos del mundo real casi nunca vienen "limpios" y listos para usar. El preprocesamiento es el paso de transformación y preparación que ocurre antes del entrenamiento. Es uno de los pasos más importantes y que más tiempo consume en un proyecto de ML (típicamente el **60-80% del tiempo total** del proyecto).

#### 3.7.1 Manejo de Valores Faltantes

Cuando un dataset tiene valores ausentes (NaN, None, vacíos), existen varias estrategias:

| Estrategia | Descripción | Cuándo Usar |
|---|---|---|
| **Eliminación de filas** | Descartar observaciones con valores faltantes | Cuando hay pocos faltantes (< 5%) y las observaciones son suficientes |
| **Eliminación de columnas** | Descartar variables con demasiados faltantes | Cuando la variable tiene > 50% de valores faltantes |
| **Imputación por media/mediana** | Reemplazar con estadístico central de la columna | Variables numéricas (media para distribuciones simétricas, mediana para asimétricas) |
| **Imputación por moda** | Reemplazar con el valor más frecuente | Variables categóricas |
| **Imputación con valor constante** | Reemplazar con un valor especial (ej. -1, "Desconocido") | Cuando el hecho de que falte tiene significado propio |
| **Imputación predictiva (KNN, Regresión)** | Predecir el valor faltante usando otras variables | Cuando los faltantes tienen patrones y hay suficientes datos |

> ⚠️ Siempre aplicar la imputación **después** de dividir en entrenamiento/prueba, para no filtrar información del conjunto de prueba al entrenamiento (*data leakage*).

#### 3.7.2 Normalización y Estandarización

Muchos algoritmos de ML son sensibles a la escala de las variables. Si una variable va de 0 a 1 y otra de 0 a 1.000.000, la segunda dominará el modelo injustamente.

- **Normalización (Min-Max Scaling):** Transforma los valores al rango [0, 1].

  $$ x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} $$

  Útil cuando se sabe que los datos tienen límites fijos y no hay outliers extremos.

- **Estandarización (Z-Score Standardization):** Transforma los datos para que tengan media 0 y desviación estándar 1.

  $$ x_{std} = \frac{x - \mu}{\sigma} $$

  Más robusta a outliers que la normalización. Es la transformación recomendada para la mayoría de algoritmos.

- **Escalado Robusto (Robust Scaling):** Usa la mediana y el IQR en lugar de media y desviación estándar. Muy robusto frente a outliers.

  $$ x_{rob} = \frac{x - \text{mediana}}{\text{IQR}} $$

| Algoritmo | ¿Necesita escalado? |
|---|---|
| Regresión Lineal / Logística (con regularización) | Sí |
| SVM, KNN, Redes Neuronales, PCA | Sí (crítico) |
| Árboles de Decisión, Random Forest, XGBoost | No (basados en particiones, no en distancias) |

#### 3.7.3 Codificación de Variables Categóricas

Los modelos matemáticos operan sobre números. Las variables categóricas deben transformarse:

- **Label Encoding:** Asigna un número entero a cada categoría. Simple pero introduce un orden artificial que puede confundir al modelo. Apropiado solo para variables ordinales o árboles de decisión.

  ```
  ['rojo', 'verde', 'azul'] → [0, 1, 2]
  ```

- **One-Hot Encoding (OHE):** Crea una columna binaria (0 o 1) para cada categoría. Elimina el orden artificial. Para evitar la trampa de la multicolinealidad, se elimina una columna (n-1 columnas para n categorías).

  ```
  color: ['rojo', 'verde', 'azul']
  → color_rojo: [1,0,0], color_verde: [0,1,0], color_azul: [0,0,1]
  ```

- **Ordinal Encoding:** Asigna números respetando el orden natural de la variable.

  ```
  ['Malo', 'Regular', 'Bueno', 'Excelente'] → [0, 1, 2, 3]
  ```

- **Target Encoding / Mean Encoding:** Reemplaza cada categoría con el promedio de la variable objetivo para esa categoría. Útil con alta cardinalidad, pero requiere cuidado para evitar data leakage (se debe calcular solo sobre los datos de entrenamiento).

#### 3.7.4 Ingeniería de Características (Feature Engineering)

La **ingeniería de características** es el proceso de usar el conocimiento del dominio para crear, transformar o seleccionar las variables que se utilizarán en el modelo. Es frecuentemente el factor con **mayor impacto** en el desempeño final, más que la elección del algoritmo.

**Técnicas comunes:**

| Técnica | Descripción | Ejemplo |
|---|---|---|
| **Creación de features** | Combinar variables existentes para crear nuevas con mayor poder predictivo | `precio_por_m2 = precio / superficie` |
| **Transformaciones matemáticas** | Aplicar log, raíz cuadrada, potencias para cambiar la distribución | `log(ingreso)` para reducir asimetría en ingresos |
| **Extracción temporal** | Descomponer fechas en componentes útiles | `mes`, `dia_semana`, `hora`, `es_fin_de_semana` desde un timestamp |
| **Interacciones** | Multiplicar o combinar variables para capturar efectos conjuntos | `edad * nivel_educativo` |
| **Binning** | Convertir variables continuas en categorías discretas | `edad → ['18-25', '26-35', '36-50', '50+']` |
| **Agregaciones** | Calcular estadísticos de grupos para enriquecer el dataset | `promedio_compras_por_cliente_en_los_ultimos_30_dias` |

#### 3.7.5 Selección de Características (Feature Selection)

Incluir demasiadas variables puede causar sobreajuste y ralentizar el entrenamiento. La **selección de características** busca el subconjunto óptimo:

- **Métodos de Filtro (Filter Methods):** Seleccionar variables basándose en correlación con la variable objetivo, varianza, o pruebas estadísticas (Chi-Cuadrado, ANOVA). Son independientes del modelo y muy rápidos.

- **Métodos Envolventes (Wrapper Methods):** Evaluar subconjuntos de variables entrenando el modelo real. Por ejemplo, **Recursive Feature Elimination (RFE)** entrena el modelo repetidamente eliminando las variables menos importantes. Son más precisos pero computacionalmente costosos.

- **Métodos Embebidos (Embedded Methods):** La selección ocurre durante el entrenamiento. Por ejemplo, la regularización **L1 (Lasso)** penaliza los coeficientes de variables poco relevantes llevándolos exactamente a cero, eliminándolas del modelo de forma automática.

---

## 4. Flujo de Trabajo Típico en Machine Learning

Un proyecto de Machine Learning exitoso sigue un proceso **iterativo y estructurado**. Conocer este flujo ayuda a organizar el trabajo, detectar problemas a tiempo y asegurar resultados reproducibles. En la práctica, casi siempre se vuelve a pasos anteriores al obtener nuevos insights.

### 4.1 Definición del Problema

Antes de tocar los datos, es fundamental responder:

- ¿Cuál es el **objetivo de negocio**? (reducir churn, detectar fraude, optimizar precios, segmentar clientes)
- ¿Es un problema de **clasificación, regresión, clustering** u otro?
- ¿Qué **métrica de éxito** se usará? ¿Cómo sabremos que el modelo es suficientemente bueno?
- ¿Cuáles son las **restricciones**? (tiempo de respuesta real, interpretabilidad requerida, privacidad de datos, costo computacional)
- ¿Qué **datos están disponibles** y cuáles necesitamos obtener?

Una buena definición del problema evita construir modelos técnicamente correctos que no responden la pregunta de negocio real.

### 4.2 Recopilación y Comprensión de Datos (EDA)

El **Análisis Exploratorio de Datos (EDA)** es el proceso de inspeccionar, visualizar y resumir los datos para:

- Entender su estructura (dimensiones, tipos de variables, valores faltantes).
- Identificar distribuciones, asimetrías y outliers.
- Descubrir relaciones entre variables (correlaciones, heatmaps).
- Formular hipótesis sobre qué variables podrían ser relevantes.

**Herramientas típicas del EDA:**
- Estadísticas descriptivas (`.describe()` en pandas).
- Histogramas y boxplots para ver distribuciones y detectar outliers.
- Matrices de correlación y heatmaps para relaciones entre variables.
- Scatter plots para relaciones bivariadas.
- Gráficos de barras para frecuencias de variables categóricas.
- Análisis de valores faltantes por variable.

### 4.3 Preprocesamiento y Preparación de Datos

Aplicar las técnicas descritas en la sección 3.7 en el siguiente orden recomendado:

1. Tratar valores faltantes (imputación o eliminación).
2. Detectar y manejar outliers (eliminar, capear o transformar).
3. Codificar variables categóricas (OHE, Label Encoding, etc.).
4. Crear nuevas características (feature engineering).
5. Escalar variables numéricas (después de la creación de features).
6. Seleccionar las características más relevantes (feature selection).
7. Dividir en conjunto de entrenamiento y prueba **antes** de aplicar cualquier transformación que use estadísticos del dataset completo.

### 4.4 Selección, Entrenamiento y Comparación de Modelos

1. **Crear una línea base (*baseline*):** Antes de modelos complejos, establecer una predicción de referencia simple (ej. predecir siempre la clase mayoritaria, o la media de la variable objetivo). El modelo real debe superar esta baseline para justificar su complejidad.
2. **Seleccionar candidatos:** Elegir 2-3 algoritmos apropiados para el tipo de problema usando la guía de la sección 1.3.
3. **Entrenar y comparar con validación cruzada:** Evaluar cada algoritmo con validación cruzada k-fold para obtener estimaciones robustas.
4. **Seleccionar el mejor modelo** según la métrica de éxito definida en la etapa de definición del problema.

### 4.5 Optimización y Evaluación Final

1. **Tuneo de hiperparámetros** del modelo seleccionado (Grid Search, Random Search u Optuna).
2. Evaluar el modelo optimizado en el conjunto de **prueba** (datos no vistos durante todo el proceso).
3. Analizar los errores: ¿dónde falla el modelo? ¿Hay patrones en los errores?
4. Revisar si hay signos de overfitting o underfitting (comparar métricas de entrenamiento vs. prueba).
5. Iterar si el desempeño no es satisfactorio — puede ser necesario volver a pasos anteriores (más datos, nuevas features, otro algoritmo).

### 4.6 Interpretación y Comunicación de Resultados

Un modelo técnicamente excelente que nadie puede entender tiene poco valor práctico. La interpretación incluye:

- Identificar y comunicar qué variables son las más importantes (**feature importance**).
- Mostrar ejemplos concretos de predicciones correctas e incorrectas.
- Traducir el desempeño a términos de negocio (ej. "el modelo detecta el 85% de los fraudes antes de que ocurran, reduciendo las pérdidas en un 40%").
- Identificar claramente los sesgos y limitaciones conocidas del modelo.

### 4.7 Despliegue y Monitoreo

- **Despliegue (Deployment):** Integrar el modelo en el sistema productivo (API REST, pipeline de datos batch, aplicación embebida).
- **Monitoreo (Monitoring):** Los datos del mundo real cambian con el tiempo. Este fenómeno se llama ***data drift*** (cambia la distribución de los inputs) o ***concept drift*** (cambia la relación entre inputs y outputs). Es crucial monitorear el desempeño del modelo en producción y re-entrenarlo periódicamente.

**El ciclo de vida de un modelo de ML nunca termina verdaderamente** — siempre hay nuevos datos, cambios en el contexto y oportunidades de mejora continua.
