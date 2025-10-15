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

---

## 1. Aprendizaje Supervisado (Supervised Learning)

El aprendizaje supervisado utiliza un conjunto de datos donde las entradas están emparejadas con sus salidas deseadas (o "etiquetas"). Es como enseñarle a un niño mostrándole un objeto (la entrada) y diciéndole qué es (la salida o etiqueta). El objetivo del modelo es aprender la función que mapea la entrada a la salida. 

Se utiliza para:

* Clasificación:  La salida es una categoría discreta (ejemplo: clasificar un correo como spam o no spam).

![alt text](https://www.themachinelearners.com/wp-content/uploads/2021/01/1_aE8XLyApqvaQA9B7MWjjlA.png)

* Regresión: La salida es un valor continuo (ejemplo: predecir el precio de una casa).
![alt text](https://pub.mdpi-res.com/ijerph/ijerph-15-02907/article_deploy/html/images/ijerph-15-02907-g001.png?1570846772)


### 1.1 Algoritmos 

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


### 1.2 Métricas de Validación

Las métricas de validación son herramientas fundamentales para evaluar el desempeño de los modelos de aprendizaje automático. Permiten cuantificar qué tan bien un modelo realiza tareas de clasificación o regresión, ayudando a comparar diferentes algoritmos y ajustar sus parámetros para obtener mejores resultados.

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

El aprendizaje no supervisado trabaja con datos sin etiquetas. El modelo debe encontrar patrones, estructuras o relaciones ocultas dentro de los datos por sí mismo. 

El uso principal es:

![alt text](https://www.researchgate.net/publication/371115662/figure/fig1/AS:11431281171335502@1688119614453/Simple-Illustration-of-Main-Types-of-Clustering-Models-Note-A-Center-based.png)

* Clustering (Agrupamiento): Agrupar datos similares.
* Asociación: Descubrir reglas que describen relaciones (ejemplo: análisis de cesta de mercado).
* Reducción de Dimensionalidad: Simplificar los datos.

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

![alt text](https://www.researchgate.net/publication/372486542/figure/fig1/AS:11431281200476307@1697939957710/Visual-representation-of-imbalanced-and-balanced-class-distributions-for-binary-and.png)

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

| Método                                   | Descripción                                                                                  | Cuándo Usar                                                                                  | Fortalezas                                                                                   | Debilidades                                                                                  |
|-------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Partición de datos                        | Separar el conjunto en datos de entrenamiento y prueba.                                       | Siempre, para evaluar la capacidad de generalización del modelo.                             | Fácil de implementar; rápida evaluación.                                                     | Puede depender de la partición elegida; no aprovecha todos los datos para entrenamiento.      |
| Validación cruzada                        | Divide los datos en varias particiones y alterna entrenamiento/validación en cada una.        | Cuando se necesita una evaluación robusta y se dispone de pocos datos.                       | Reduce la varianza de la evaluación; usa todos los datos para entrenamiento y validación.     | Más costosa computacionalmente; puede ser lenta con grandes conjuntos de datos.               |
| Subsampling y Oversampling                | Disminuir o aumentar muestras de clases para equilibrar el conjunto.                          | Cuando hay desbalance de clases en el conjunto de datos.                                     | Mejora el balance de clases; fácil de aplicar.                                               | Puede eliminar información útil (subsampling) o causar sobreajuste (oversampling).            |
| SMOTE (Synthetic Minority Over-sampling)  | Genera ejemplos sintéticos de la clase minoritaria.                                           | Cuando la clase minoritaria es muy pequeña y el oversampling tradicional no es suficiente.   | Mejora el balance sin duplicar datos; reduce el sobreajuste.                                 | Puede generar ejemplos poco realistas; requiere cuidado en la aplicación.                     |

Estas técnicas ayudan a construir modelos más robustos y confiables, mitigando los problemas comunes en aprendizaje automático.
