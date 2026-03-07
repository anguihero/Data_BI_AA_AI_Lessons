# Fundamentos de Aprendizaje Automático

**Autor**: AMMS

**Repositorio**: [Repositorio de clases de BI, AA y AI](https://github.com/anguihero/Data_BI_AA_AI_Lessons)

**Fecha Actualización**: 2025/10/14

Este documento presenta una guía concisa sobre los conceptos fundamentales del aprendizaje automático, incluyendo métodos supervisados y no supervisados, algoritmos principales, métricas de evaluación y desafíos comunes. El contenido está diseñado para servir como referencia rápida para estudiantes y profesionales interesados en comprender los principios básicos del machine learning.

# Introducción a Métodos de Aprendizaje Automático

El Aprendizaje Automático (Machine Learning) es un campo de la inteligencia artificial que se centra en el desarrollo de programas informáticos que pueden acceder a datos y utilizarlos para aprender por sí mismos. Se divide fundamentalmente en dos grandes categorías basadas en el tipo de datos de entrenamiento: Supervisado y No Supervisado.

El uso de métodos de aprendizaje automático es necesario cuando los problemas son demasiado complejos para ser resueltos mediante reglas explícitas o cuando los datos contienen patrones que no son evidentes a simple vista. Estos métodos permiten automatizar tareas, mejorar la toma de decisiones y descubrir información valiosa a partir de grandes volúmenes de datos.

![alt text](https://scikit-learn.org/1.3/_static/ml_map.png)

**Ejemplos de aplicación de Aprendizaje Supervisado:**
1. **Diagnóstico médico:** Clasificar imágenes médicas para detectar enfermedades como cáncer o neumonía.
2. **Detección de fraude:** Identificar transacciones bancarias sospechosas en tiempo real.
3. **Reconocimiento de voz:** Convertir audio en texto en asistentes virtuales como Siri o Alexa.

**Ejemplos de aplicación de Aprendizaje No Supervisado:**
1. **Segmentación de clientes:** Agrupar usuarios según sus hábitos de compra para campañas de marketing personalizadas.
2. **Análisis de temas en textos:** Descubrir automáticamente temas recurrentes en grandes colecciones de documentos.
3. **Detección de anomalías:** Identificar patrones inusuales en sensores industriales para mantenimiento predictivo.

Estos ejemplos muestran cómo el aprendizaje automático puede aportar valor en distintos sectores, ayudando a resolver problemas complejos y optimizar procesos.

## 0. Conceptos basicos previos al aprendizaje automatico

### 0.1 Estadística Descriptiva y Probabilidad

La estadística descriptiva es el conjunto de técnicas que nos permite **resumir, organizar y analizar datos** para extraer información relevante antes de aplicar modelos de aprendizaje automático. Es el primer paso para comprender la naturaleza de los datos y detectar posibles problemas como valores atípicos, sesgos o distribuciones inusuales.

#### Medidas de Tendencia Central

Las medidas de tendencia central indican el punto alrededor del cual se agrupan los datos:

- **Media (Promedio):** Es la suma de todos los valores dividida por el número de observaciones. Es sensible a valores extremos (outliers).
  
  $$ \text{Media} = \frac{1}{n} \sum_{i=1}^{n} x_i $$

- **Mediana:** Es el valor central cuando los datos están ordenados. Es robusta frente a valores atípicos.

- **Moda:** Es el valor que aparece con mayor frecuencia en el conjunto de datos.

#### Medidas de Dispersión

Las medidas de dispersión muestran qué tan dispersos o concentrados están los datos respecto a la tendencia central:

- **Varianza:** Mide el promedio de las diferencias al cuadrado respecto a la media.

  $$ \text{Varianza} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$

- **Desviación estándar:** Es la raíz cuadrada de la varianza, lo que la hace más interpretable porque está en las mismas unidades que los datos.

  $$ \text{Desviación estándar} = \sqrt{\text{Varianza}} $$

- **Rango:** Diferencia entre el valor máximo y mínimo.

#### Distribuciones de Probabilidad

Las distribuciones de probabilidad describen cómo se distribuyen los valores de una variable aleatoria. Son fundamentales para modelar incertidumbre y realizar inferencias estadísticas.

- **Distribución Normal (Gaussiana):** Es la más común en estadística y machine learning. Tiene forma de campana y se caracteriza por su media y desviación estándar.

  $$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$

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

La **correlación** es una medida estadística que indica la fuerza y dirección de la relación entre dos variables. Es clave para identificar patrones y dependencias antes de construir modelos predictivos, pero es importante recordar que correlación no implica causalidad.

#### Interpretación de la Correlación

El coeficiente de correlación varía entre -1 y +1:

- **+1:** Correlación positiva perfecta (ambas variables aumentan juntas).
- **0:** Sin correlación (no existe relación aparente).
- **-1:** Correlación negativa perfecta (una variable aumenta mientras la otra disminuye).

Valores cercanos a ±1 indican relaciones fuertes; valores cercanos a 0 indican relaciones débiles.

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

#### Uso de la Correlación en Regresión

- **Selección de Variables:** Se calcula la correlación entre la variable dependiente y las independientes para identificar las más relevantes.
- **Detección de Multicolinealidad:** Se analiza la matriz de correlación entre variables independientes; correlaciones altas (ej. > 0.8) pueden causar inestabilidad en los coeficientes del modelo. Se recomienda eliminar o combinar variables altamente correlacionadas para mejorar la robustez del modelo.

La correlación es una herramienta esencial para el análisis exploratorio y la construcción de modelos predictivos sólidos en machine learning.

---

## 1. Aprendizaje Supervisado (Supervised Learning)

El aprendizaje supervisado utiliza un conjunto de datos donde las entradas están emparejadas con sus salidas deseadas (o **"etiquetas"**). Es como enseñarle a un niño mostrándole un objeto (la entrada) y diciéndole qué es (la salida o etiqueta). El objetivo del modelo es aprender la función que mapea la entrada a la salida. 

### 1.2 Clasificación

Los modelos de clasificación son fundamentales en el aprendizaje automático porque permiten asignar elementos a categorías específicas basándose en sus características. Estos modelos analizan datos de entrada y, mediante algoritmos como árboles de decisión, regresión logística o redes neuronales, determinan a qué clase pertenece cada ejemplo. Por ejemplo, en un sistema de detección de spam, el modelo clasifica los correos electrónicos como “spam” o “no spam” según su contenido y otros atributos.

La característica principal de los modelos de clasificación es que la salida es una **categoría discreta**, es decir, el resultado pertenece a un conjunto limitado de clases predefinidas. Esto los diferencia de los modelos de regresión, donde la salida es un valor continuo. Los modelos de clasificación se utilizan en aplicaciones como reconocimiento de imágenes, diagnóstico médico y análisis de sentimientos, donde es esencial identificar a qué grupo pertenece cada dato analizado

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

|Algoritmo|Tipo de Tarea Principal|Descripción Breve|Hiperparámetros Clave|
|-----------------:|----------------:|-----------------:|---------------|
|Regresión Lineal| Regresión|Modela la relación entre una variable dependiente (salida) y una o más variables independientes (entradas) ajustando la mejor línea recta a los datos.|	No tiene hiperparámetros de aprendizaje. Los parámetros se calculan directamente (mínimos cuadrados). A menudo, solo se considera la tasa de aprendizaje si se usa Descenso de Gradiente.|
|Regresión Logística|Clasificación|	Utiliza la función logística para estimar la probabilidad de que una instancia pertenezca a una clase. A pesar de su nombre, es un modelo de clasificación binaria (dos clases).|1. C (o λ): Inverso de la fuerza de regularización. Valores más pequeños especifican una regularización más fuerte. 2. Penalty (Penalización): Tipo de regularización aplicada (L1 o L2). 3. Solver: Algoritmo a utilizar en la optimización (ej. liblinear, saga, lbfgs).|
|Árboles de Decisión|Clasificación / Regresión|Crean un modelo que predice el valor de una variable objetivo (salida) dividiendo el conjunto de datos de entrenamiento en subconjuntos basados en los valores de las características (entradas), formando una estructura similar a un árbol.|1. max_depth: Profundidad máxima del árbol. 2. min_samples_split: Número mínimo de muestras requeridas para dividir un nodo interno. 3. criterion: Función para medir la calidad de una división (ej. gini o entropía para clasificación).|
|Random Forest|Clasificación / Regresión|Es un método de ensamble que construye múltiples árboles de decisión y combina sus predicciones para mejorar la precisión y evitar el sobreajuste.|1. n_estimators: Número de árboles en el bosque. 2. max_features: Número de características a considerar para la mejor división en cada nodo. 3. max_depth: Profundidad máxima de cada árbol individual.|
|Support Vector Machine (SVM)|Clasificación / Regresión|Encuentra un hiperplano óptimo que separa las clases en un espacio de alta dimensión, maximizando el margen entre el hiperplano y los puntos de datos más cercanos (vectores de soporte).|1. C: Parámetro de regularización. Penaliza los errores de clasificación. 2. kernel: Tipo de función de núcleo (linear, poly, rbf (Radial Basis Function), sigmoid). 3. gamma (γ): Parámetro del kernel rbf (define cuánta influencia tiene un solo ejemplo de entrenamiento).|
|K-Nearest Neighbors (KNN)|Clasificación / Regresión|Es un algoritmo no paramétrico que clasifica un nuevo punto basándose en la mayoría de las clases de sus K vecinos más cercanos en el espacio de características.| 1. n_neighbors (K): El número de vecinos a considerar. 2. weights: Función de ponderación utilizada en la predicción (uniform o distance). 3. metric: Métrica de distancia a utilizar (euclidean, manhattan, minkowski).|
|Boosting|Clasificación / Regresión|Crea un modelo fuerte a partir de una secuencia de modelos débiles, corrigiendo los errores del modelo anterior.|1. n_estimators (Número de modelos débiles). 2. learning_rate (Peso de cada modelo). 3. max_depth (Si usa árboles débiles).|
|XGBoost (eXtreme Gradient Boosting)|Clasificación / Regresión|Un algoritmo de boosting muy eficiente y popular que mejora iterativamente un conjunto de modelos débiles (típicamente árboles) para formar un modelo predictivo fuerte.|1. n_estimators: Número de rondas de aumento (número de árboles). 2. learning_rate (η): Tasa de aprendizaje o tamaño del paso en cada iteración. 3. max_depth: Profundidad máxima de cada árbol. 4. gamma (γ): Mínima pérdida de reducción necesaria para hacer una división adicional. 5. subsample: Fracción de muestras aleatorias a utilizar para entrenar cada árbol.|


### 1.4 Métricas de Validación

Las métricas de validación son herramientas fundamentales para **evaluar el desempeño de los modelos** de aprendizaje automático. Permiten cuantificar qué tan bien un modelo realiza tareas de clasificación o regresión, ayudando a **comparar diferentes algoritmos** y **ajustar sus parámetros** para obtener mejores resultados.

![alt text](https://db0dce98.rocketcdn.me/es/files/2024/08/Schema-model_evaluation-42-42.png)

En clasificación, las métricas como la exactitud, precisión, recall y F1-score ayudan a entender si el modelo identifica correctamente las clases, lo cual es crucial en aplicaciones como la detección de fraude (donde es importante minimizar los falsos positivos) o el diagnóstico médico (donde es vital reducir los falsos negativos).

![alt text](https://almablog-media.s3.ap-south-1.amazonaws.com/image_14_4f4fc2cf7d.png)

En regresión, métricas como el error cuadrático medio (MSE), el error absoluto medio (MAE) y el coeficiente de determinación (R²) permiten medir la diferencia entre los valores predichos y los reales. Por ejemplo, al predecir el precio de una vivienda, un bajo MSE indica que el modelo realiza estimaciones cercanas a los valores reales.

![alt text](https://miro.medium.com/1*5fnmYVHLTC8mGxybHm4XkA.png)

Seleccionar la métrica adecuada depende del problema y del impacto de los errores. Por eso, entender y aplicar correctamente estas métricas es esencial para desarrollar modelos robustos y útiles en la práctica.


|Métrica|Tipo de Uso|Cómo se Calcula|Fortalezas y Debilidades|
|-------|-----------|---------------|------------------------|
|Exactitud (Accuracy)|Clasificación|Proporción de predicciones correctas sobre el total de muestras.|Fácil de interpretar, pero puede ser engañosa en conjuntos de datos desbalanceados.|
|Precisión (Precision)|Clasificación|TP / (TP + FP), donde TP = verdaderos positivos, FP = falsos positivos.|Útil cuando el costo de los falsos positivos es alto; puede ignorar falsos negativos.|
|Recall (Sensibilidad)|Clasificación|TP / (TP + FN), donde FN = falsos negativos.|Importante cuando el costo de los falsos negativos es alto; puede ignorar falsos positivos.|
|F1-Score|Clasificación|Media armónica entre precisión y recall: 2 * (Precision * Recall) / (Precision + Recall).|Equilibra precisión y recall; útil en datos desbalanceados.|
|AUC-ROC|Clasificación|Área bajo la curva ROC, que compara la tasa de verdaderos positivos vs. falsos positivos.|Evalúa el rendimiento en todos los umbrales; robusta en datos desbalanceados.|
|MSE (Error Cuadrático Medio)|Regresión|Promedio de los cuadrados de las diferencias entre valores reales y predichos.|Penaliza fuertemente los errores grandes; sensible a valores atípicos.|
|MAE (Error Absoluto Medio)|Regresión|Promedio de los valores absolutos de las diferencias entre valores reales y predichos.|Menos sensible a valores atípicos que MSE; fácil de interpretar.|
|R² (Coeficiente de Determinación)|Regresión|Proporción de la varianza explicada por el modelo respecto a la varianza total.|Indica el ajuste global; puede ser negativo si el modelo es peor que la media.|



---


## 2. Aprendizaje No Supervisado (Unsupervised Learning)

El aprendizaje no supervisado trabaja con datos **sin etiquetas**. El modelo debe encontrar patrones, estructuras o relaciones ocultas dentro de los datos por sí mismo. 

El uso principal es:

![alt text](https://scikit-learn.org/0.18/_images/sphx_glr_plot_cluster_comparison_001.png)

* **Clustering (Agrupamiento)**: Agrupar datos similares.
* **Asociación**: Descubrir reglas que describen relaciones (ejemplo: análisis de cesta de mercado).
* **Reducción de Dimensionalidad**: Simplificar los datos.

### 2.1 Algoritmos

|Algoritmo|Tipo de Tarea Principal|Descripción Breve|Hiperparámetros Clave|
|-----------------:|----------------:|-----------------:|---------------|
|K-Means|Clustering|	Divide los datos en K grupos o clusters, donde K es un número predefinido. Asigna cada punto al centroide (punto central) más cercano.|1. n_clusters (K): El número de clusters que se deben formar. 2. init: Método de inicialización de los centroides (k-means++ o random). 3. max_iter: Número máximo de iteraciones para el algoritmo.|
|Clustering Jerárquico|Clustering|Construye una jerarquía de clusters. Puede ser aglomerativo (comenzar con puntos individuales y agruparlos) o divisivo (comenzar con un cluster grande y dividirlo).|1. n_clusters: El número de clusters a detener el proceso. 2. linkage: Criterio de conexión entre conjuntos de observaciones (ward, average, complete). 3. affinity (o metric): Métrica de distancia utilizada para calcular las distancias (euclidean, manhattan).|
|Clustering Basado en Densidad|Clustering|Identifica clusters basándose en la densidad de los puntos de datos. Es bueno para encontrar clusters de formas arbitrarias y es robusto al ruido (puntos atípicos).|1. eps (ϵ): La distancia máxima entre dos muestras para que se consideren vecinas. 2. min_samples: El número de muestras (o puntos) en un vecindario para que un punto se considere un punto central. 3. metric: Métrica de distancia a utilizar (euclidean, manhattan).|


### 2.2 Métricas de Validación

|Métrica|Tipo de Uso|Cómo se Calcula|Fortalezas y Debilidades|
|-------|-----------|---------------|------------------------|
|Silhouette Score|Clustering|Promedio de la diferencia entre la distancia intra-cluster y la distancia al cluster más cercano.|Evalúa la separación y cohesión de los clusters; requiere conocer los clusters.|
|Davies-Bouldin Index|Clustering|Promedio de la relación entre la dispersión intra-cluster y la distancia inter-cluster.|Menor valor indica mejor clustering; sensible a clusters de diferente tamaño.|
|Calinski-Harabasz Index|Clustering|Relación entre la dispersión entre clusters y la dispersión dentro de los clusters.|Mayor valor indica mejor clustering; favorece clusters compactos y bien separados.|
|Homogeneidad|Clustering|Mide si cada cluster contiene solo miembros de una sola clase.|Útil si se conocen las etiquetas verdaderas; no aplicable en clustering puro.|
|Completeness|Clustering|Mide si todos los miembros de una clase están en el mismo cluster.|Complementa la homogeneidad; útil con etiquetas verdaderas.|
|Varianza Explicada|Reducción de Dimensionalidad|Proporción de la varianza total capturada por los componentes seleccionados.|Indica cuánta información se conserva; no mide interpretabilidad.|




---

## 3. Desafios del aprendizaje automatico

### 3.1 Subajuste y Sobreajuste

![alt text](https://media.licdn.com/dms/image/v2/D4E22AQFLsYgMYO-H7Q/feedshare-shrink_800/B4EZkk_dW5KYAg-/0/1757262241175?e=2147483647&v=beta&t=FSDUbc3ZebNr4pO1yqRu6awd1VHCny45aOzxyFwMcoY)

El **subajuste (underfitting)** ocurre cuando un modelo es demasiado simple para capturar los patrones relevantes de los datos, lo que resulta en un bajo desempeño tanto en el conjunto de entrenamiento como en el de prueba. El **sobreajuste (overfitting)** sucede cuando el modelo es demasiado complejo y aprende detalles o ruido específico del conjunto de entrenamiento, perdiendo capacidad de generalización y mostrando alto desempeño en entrenamiento pero bajo en prueba. Ninguno de estos escenarios es deseado, ya que impiden que el modelo sea útil en datos nuevos.

### 3.2 Desbalanceo de Clases

![alt text](https://cdn.sanity.io/images/31qskqlc/production/a680f0dd5ab72cd0dfb06effd8cdbfa0858ac6a8-850x647.webp?fit=max&auto=format)

El **desbalanceo de clase** aparece cuando algunas clases están representadas por muchas más muestras que otras. Esto puede llevar a que el modelo ignore las clases minoritarias, afectando la precisión y utilidad en aplicaciones críticas (por ejemplo, detección de fraude o enfermedades raras).

### 3.3 Calidad y Cantidad de Datos

La **calidad** de los datos (presencia de errores, valores faltantes, ruido) y la **cantidad** insuficiente de datos pueden limitar la capacidad del modelo para aprender patrones útiles. Datos pobres o escasos suelen llevar a modelos poco confiables y resultados engañosos.

### 3.4 Tuneo de Hiperparámetros

El **ajuste de hiperparámetros** implica explorar diferentes configuraciones para encontrar la combinación que optimiza el desempeño del modelo. Es una forma de explorar soluciones y evitar tanto el subajuste como el sobreajuste.

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
| SMOTE (Synthetic Minority Over-sampling)  | Genera ejemplos sintéticos de la clase minoritaria.| Cuando la clase minoritaria es muy pequeña y el oversampling tradicional no es suficiente.   | Mejora el balance sin duplicar datos; reduce el sobreajuste.| Puede generar ejemplos poco realistas; requiere cuidado en la aplicación.|

Estas técnicas ayudan a construir modelos más robustos y confiables, mitigando los problemas comunes en aprendizaje automático.


### 3.6 El Trade-off de Sesgo vs. Varianza (Bias-Variance Tradeoff): 


Este es quizás el concepto más importante en machine learning para diagnosticar problemas en un modelo.

Sesgo (Bias): Es el error por suposiciones demasiado simples. Un alto sesgo lleva al underfitting (subajuste), donde el modelo es tan simple que no captura la relación real en los datos.

Varianza (Variance): Es el error por ser demasiado sensible a los datos de entrenamiento. Una alta varianza lleva al overfitting (sobreajuste), donde el modelo "memoriza" los datos de entrenamiento (incluido el ruido) y no generaliza bien a datos nuevos.

El objetivo es encontrar un balance entre ambos.

### 3.7  Preprocesamiento de Datos (Data Preprocessing)

Los datos del mundo real casi nunca vienen "limpios" y listos para usar. El preprocesamiento es el paso de limpieza y preparación.

Manejo de Valores Faltantes: Decidir qué hacer cuando faltan datos en tu dataset (eliminarlos, reemplazarlos con la media o mediana, etc.).

Normalización y Estandarización: Se usan para escalar las variables numéricas a un rango común. Esto es crucial para algoritmos que son sensibles a las diferentes escalas de los datos (como las redes neuronales o SVM).

Codificación de Variables Categóricas: Los modelos matemáticos necesitan números, no texto. Técnicas como One-Hot Encoding (crear columnas binarias para cada categoría) o Label Encoding (asignar un número a cada categoría) son esenciales.