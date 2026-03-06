---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 06: Aprendizaje Supervisado

## 🎯 Objetivo de la Sesión
Implementar tus primeros modelos de Machine Learning supervisado: clasificación y regresión. Aprenderás a entrenar, evaluar y optimizar modelos usando Scikit-learn con datasets clásicos.

## 📚 Conceptos Teóricos

### ¿Qué es Aprendizaje Supervisado?
El aprendizaje supervisado es un tipo de Machine Learning donde el modelo aprende de datos etiquetados (features + labels) para hacer predicciones sobre datos nuevos.

**Tipos principales**:
- **Clasificación**: Predecir categorías discretas (¿sobrevivirá?, ¿es spam?, ¿tipo de flor?)
- **Regresión**: Predecir valores continuos (precio de casa, temperatura, ventas)

### Anatomía de un Problema de ML Supervisado

```
Datos de Entrada (X)          →    Modelo ML    →    Predicción (y)
[edad, ingresos, educación]   →    Algoritmo    →    [¿aprobará préstamo?]
```

**Componentes**:
1. **Features (X)**: Variables independientes, predictores, atributos
2. **Target (y)**: Variable dependiente, etiqueta, objetivo
3. **Modelo**: Algoritmo que aprende la relación X → y
4. **Métricas**: Forma de evaluar qué tan bien predice el modelo

### El Pipeline de Machine Learning

```
1. Obtener Datos
   ↓
2. Análisis Exploratorio (EDA)
   ↓
3. Preparación de Datos
   - Limpieza
   - Encoding de categóricas
   - Escaling de numéricas
   - Train/Test Split
   ↓
4. Entrenamiento del Modelo
   ↓
5. Evaluación
   ↓
6. Optimización (Hyperparameter Tuning)
   ↓
7. Predicción en Producción
```

### Clasificación: Predicción de Categorías

#### Algoritmos Comunes de Clasificación

**1. Logistic Regression**
- Simple y rápido
- Bueno para problemas linealmente separables
- Proporciona probabilidades

**2. Decision Trees**
- Fácil de interpretar
- No requiere escaling
- Propenso a overfitting

**3. Random Forest**
- Ensemble de árboles de decisión
- Robusto, maneja no-linealidad
- Reduce overfitting vs Decision Trees

**4. Support Vector Machines (SVM)**
- Bueno para espacios de alta dimensión
- Efectivo cuando # features > # muestras

**5. K-Nearest Neighbors (KNN)**
- Simple, basado en similitud
- Sensible a escala de features
- Lento en predicción para datasets grandes

#### Métricas de Clasificación

**Confusion Matrix**
```
                Predicho
                No   Sí
Actual  No     TN   FP
        Sí     FN   TP
```

- **TP (True Positive)**: Predijo Sí, era Sí ✓
- **TN (True Negative)**: Predijo No, era No ✓
- **FP (False Positive)**: Predijo Sí, era No ✗ (Error Tipo I)
- **FN (False Negative)**: Predijo No, era Sí ✗ (Error Tipo II)

**Métricas Derivadas**:
```python
Accuracy = (TP + TN) / Total            # % correctamente clasificados
Precision = TP / (TP + FP)              # De los predichos como positivos, % correctos
Recall = TP / (TP + FN)                 # De los realmente positivos, % detectados
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

**Cuándo usar qué métrica**:
- **Accuracy**: Clases balanceadas, todos los errores igual de importantes
- **Precision**: Cuando FP es costoso (ej: spam detection, no quieres marcar emails importantes como spam)
- **Recall**: Cuando FN es costoso (ej: detección de fraude, no quieres perder fraudes reales)
- **F1-Score**: Balance entre Precision y Recall

**ROC-AUC**:
- Área bajo la curva ROC (Receiver Operating Characteristic)
- Mide capacidad del modelo para discriminar entre clases
- Valores: 0.5 (aleatorio) a 1.0 (perfecto)

### Regresión: Predicción de Valores Continuos

#### Algoritmos Comunes de Regresión

**1. Linear Regression**
- Asume relación lineal entre X e y
- Simple, interpretable
- Fórmula: y = β₀ + β₁x₁ + β₂x₂ + ... + ε

**2. Ridge Regression (L2 Regularization)**
- Penaliza coeficientes grandes
- Reduce overfitting
- Bueno cuando hay multicolinealidad

**3. Lasso Regression (L1 Regularization)**
- Penaliza coeficientes grandes
- Puede llevar coeficientes a cero (feature selection)

**4. Polynomial Regression**
- Captura relaciones no-lineales
- Crea features polinómicos (x², x³, etc.)

**5. Random Forest Regressor**
- Ensemble de árboles
- Captura no-linealidad sin transformaciones

#### Métricas de Regresión

```python
# MAE (Mean Absolute Error) - Promedio de errores absolutos
MAE = mean(|y_real - y_pred|)

# MSE (Mean Squared Error) - Penaliza errores grandes
MSE = mean((y_real - y_pred)²)

# RMSE (Root Mean Squared Error) - MSE en unidades originales
RMSE = sqrt(MSE)

# R² (Coefficient of Determination) - % de varianza explicada
R² = 1 - (SS_res / SS_tot)  # 0 (malo) a 1 (perfecto)
```

**R² Interpretación**:
- R² = 0.85 → El modelo explica el 85% de la variabilidad de los datos
- R² < 0 → El modelo es peor que simplemente predecir la media

### Preparación de Datos

#### Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para test
    random_state=42,    # Reproducibilidad
    stratify=y          # Mantiene proporción de clases
)
```

#### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NUNCA fit en test!
```

**Cuándo escalar**:
- ✅ Necesario: KNN, SVM, Logistic Regression, Neural Networks
- ❌ No necesario: Tree-based algorithms (Random Forest, Decision Trees)

#### Encoding de Variables Categóricas
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# LabelEncoder: Ordinal (Bajo=0, Medio=1, Alto=2)
le = LabelEncoder()
y = le.fit_transform(y)

# OneHotEncoder: Nominal (Crea columnas binarias)
# ['Rojo', 'Verde', 'Azul'] → [1,0,0], [0,1,0], [0,0,1]
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X_categorical)

# Pandas: get_dummies (más simple)
X_encoded = pd.get_dummies(X, columns=['color'], drop_first=True)
```

### Overfitting vs Underfitting

**Underfitting** (High Bias):
- Modelo muy simple
- Bajo rendimiento en train Y test
- Solución: Modelo más complejo, más features

**Overfitting** (High Variance):
- Modelo muy complejo
- Alto rendimiento en train, bajo en test
- Solución: Regularización, más datos, menos features

**Just Right**:
- Balance entre bias y variance
- Buen rendimiento en train Y test
- Generaliza bien a datos nuevos

### Cross-Validation

En lugar de un solo train/test split, usa múltiples folds:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## 🏆 Reto de la Sesión

### Parte 1: Clasificación - Titanic Survival
Predice quién sobrevivió en el Titanic usando características como edad, clase, género.

**Features**:
- Pclass (clase de ticket)
- Sex (género)
- Age (edad)
- SibSp (# hermanos/cónyuge a bordo)
- Parch (# padres/hijos a bordo)
- Fare (tarifa pagada)

**Target**: Survived (1 = sobrevivió, 0 = falleció)

**Tareas**:
1. Carga y explora el dataset
2. Preprocesamiento: manejo de nulos, encoding
3. Entrena un modelo Random Forest
4. Evalúa con accuracy, precision, recall, F1
5. Matriz de confusión

### Parte 2: Regresión - Predicción de Precios de Casas
Predice el precio de casas usando características físicas.

**Features**:
- Área en pies cuadrados
- Número de habitaciones
- Número de baños
- Año de construcción
- Ubicación

**Target**: Precio de venta

**Tareas**:
1. Genera/carga dataset de casas
2. EDA: correlaciones, distribuciones
3. Entrena un modelo de Linear Regression
4. Evalúa con MAE, MSE, RMSE, R²
5. Visualiza: valores reales vs predichos

## 💡 Tips para el Éxito
1. **NUNCA hagas fit() en el test set**: Causaría data leakage
2. **Valida la distribución**: Usa stratify en clasificación con clases desbalanceadas
3. **Features engineering**: A veces crear nuevas features mejora más que cambiar el algoritmo
4. **Baseline model**: Siempre compara contra un modelo simple primero
5. **Interpreta resultados**: Un modelo perfecto (99.9% accuracy) puede indicar data leakage

## 📊 Criterios de Evaluación
- ✅ Preprocesamiento correcto de datos
- ✅ Train/test split apropiado
- ✅ Modelos entrenados sin errores
- ✅ Métricas calculadas e interpretadas correctamente
- ✅ Comprensión de trade-offs entre modelos

## 📖 Recursos Complementarios
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Supervised Learning Algorithms](https://scikit-learn.org/stable/supervised_learning.html)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [StatQuest ML Videos](https://www.youtube.com/c/joshstarmer)

## 🔍 Conceptos que Dominarás
- Clasificación vs Regresión
- Random Forest y Linear Regression
- Métricas de evaluación
- Train/test split
- Feature scaling y encoding
- Interpretación de resultados

## 🚀 Siguiente Paso
En la Sesión 07 explorarás Aprendizaje No Supervisado: clustering con K-Means para segmentar clientes sin etiquetas previas.

---
**¡Bienvenido al mundo del Machine Learning! 🤖📊**
