
# 📊 Guía Integral de Análisis y Pronóstico de Series de Tiempo

## 1️⃣ La Intuición Detrás de los Datos

### ¿Qué hace especial a una Serie de Tiempo?

Una serie de tiempo es una secuencia de observaciones donde **el orden importa**. A diferencia de otros datos, aquí el ayer afecta al hoy. Imagina un río: no es lo mismo saber la temperatura del agua en diferentes puntos que conocer cómo cambia minuto a minuto.

### 🔄 Componentes Principales

| Componente | Analogía | Ejemplo |
|---|---|---|
| **Tendencia** | La dirección del viento | Ventas crecientes año tras año |
| **Estacionalidad** | Las estaciones del año | Más helado en verano, más chocolate en invierno |
| **Ciclo** | Cambios climáticos a largo plazo | Booms y crisis económicas (sin periodo fijo) |
| **Residuo** | Ruido impredecible | Lluvia inesperada |

### 📈 Modelos: Aditivo vs. Multiplicativo

**Aditivo:** $Y_t = T_t + S_t + C_t + R_t$
- La estacionalidad es constante (±100 unidades siempre)
- Crecimiento lento y lineal

**Multiplicativo:** $Y_t = T_t \times S_t \times C_t \times R_t$
- La estacionalidad crece con la tendencia (5% si la base crece)
- Crecimiento acelerado o exponencial

---

## 2️⃣ Conceptos Fundamentales de Estabilidad

### 🎯 Estacionariedad: Las Reglas del Juego Fijas

Una serie es **estacionaria** cuando:
- Su media es constante (no sube ni baja)
- Su variabilidad no cambia con el tiempo
- No hay tendencia obvia

**¿Por qué importa?** La mayoría de modelos clásicos exigen que los datos sean estacionarios. Si los datos "juegan" distinto en 2023 que en 2024, el modelo se confunde.

### 🔧 Técnicas de Estabilización

**Diferenciación:** Restar hoy menos ayer
```
Diferencia_t = Y_t - Y_(t-1)
```
Convierte una tendencia creciente en variaciones estables.

**Transformaciones:** Aplicar logaritmo para suavizar variabilidad exponencial
```
Y_t_transformada = log(Y_t)
```

---

## 3️⃣ El Arsenal Tecnológico

### 📚 Librerías Esenciales

| Librería | Fortaleza | Caso de Uso |
|---|---|---|
| **Pandas** | Manipulación temporal y resample | Cambiar frecuencia (diaria → mensual) |
| **Statsmodels** | Tests (ADF) y ARIMA/SARIMA | Análisis estadístico riguroso |
| **Scikit-learn** | ML supervisado | Ventanas deslizantes, regresión |
| **Prophet** | Automatizado, festivos, cambios abruptos | Pronósticos rápidos en producción |
| **NeuralProphet** | Redes neuronales + componentes clásicas | Datos con patrones complejos |
| **Darts/Kats** | Interfaz unificada de modelos | Comparar múltiples enfoques fácilmente |

---

## 4️⃣ El Árbol Genealógico de Modelos

### 🧬 Modelos Clásicos

**AR (Auto-regressivo):**
```
Y_t = φ₁·Y_(t-1) + φ₂·Y_(t-2) + ... + ε_t
```
El hoy depende del ayer. Útil para datos con memoria corta.

**MA (Media Móvil):**
```
Y_t = μ + ε_t + θ₁·ε_(t-1) + θ₂·ε_(t-2) + ...
```
Corrige basándose en errores pasados.

**ARIMA/SARIMA:**
```
ARIMA(p, d, q): AR(p) + Diferenciación(d) + MA(q)
SARIMA(p,d,q)(P,D,Q,s): Añade estacionalidad
```
La navaja suiza: combina memoria, estabilidad y estacionalidad.

**Suavizado Exponencial (ETS):**
Da más peso a datos recientes. Perfecto para cambios graduales.

---


## 5️⃣ Otros Métodos y Modelos

### 🧬 Modelos Nueva Generación

#### 1️⃣ **LSTM (Long Short-Term Memory)**

**Concepto:** Red neuronal recurrente que "recuerda" patrones a largo plazo mediante compuertas de memoria.

**Bondades:**
- Captura dependencias temporales complejas y no lineales
- Maneja secuencias largas sin perder información del pasado
- Excelente con múltiples variables correlacionadas

**Debilidades:**
- Requiere grandes volúmenes de datos (mínimo 500+ observaciones)
- Computacionalmente costoso (GPUs recomendadas)
- Resultados menos interpretables ("caja negra")
- Sobreajuste frecuente sin regularización adecuada

**Metodología:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

modelo = Sequential([
    LSTM(50, activation='relu', input_shape=(lookback, n_features)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(1)
])
modelo.compile(optimizer='adam', loss='mse')
```

**Aporte:** Ideal para datos de alta frecuencia (minutos/horas) con patrones no lineales.

---

#### 2️⃣ **Transformer + Attention Mechanism**

**Concepto:** Arquitectura basada en mecanismo de atención que pondera qué momentos pasados son más relevantes para predecir el futuro.

**Bondades:**
- Paralelizable (entrenamientos muy rápidos)
- Identifica automáticamente períodos relevantes
- Superior a LSTM en secuencias muy largas (1000+ pasos)
- Excelente para series multivariantes

**Debilidades:**
- Mayor complejidad computacional durante predicción
- Requiere aún más datos que LSTM
- Implementación compleja (librerías especializadas)

**Metodología:**
```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

# Bloque Transformer simplificado
attention = MultiHeadAttention(num_heads=4, key_dim=64)
```

**Aporte:** Gold standard para datos financieros/meteorológicos de alta frecuencia.

---

#### 3️⃣ **GAM (Generalized Additive Models)**

**Concepto:** Modelo aditivo que aprende funciones suaves (splines) para cada variable independientemente.

**Bondades:**
- **Interpretable:** visualizas cómo cada variable afecta la predicción
- Sin necesidad de estacionariedad (como ARIMA)
- Flexible: captura relaciones no lineales pero sin overfitting
- Rápido entrenamiento

**Debilidades:**
- Asume componentes aditivas (menos flexible que redes neuronales)
- Menos preciso con interacciones complejas entre variables
- Requiere sintonización manual de suavidad

**Metodología:**
```python
import pymc as pm

with pm.Model():
    # s(x) = función suave vía GP (Gaussian Process)
    x = pm.data.Data('x', datos)
    trend = pm.gp.Marginal(cov_func=pm.gp.cov.ExpQuad(1, ls=1))
    f = trend.prior('f', X=x[:, None])
    y_obs = pm.Normal('y_obs', mu=f, sigma=sigma, observed=y)
```

**Aporte:** Mejor balance entre precisión e interpretabilidad. Ideal para reportes ejecutivos.

---

#### 4️⃣ **Random Forest / Gradient Boosting (XGBoost, LightGBM)**

**Concepto:** Métodos de ensemble que crean "ventanas deslizantes" con lags (Y_{t-1}, Y_{t-2}...) como características.

**Bondades:**
- Captura interacciones automáticamente
- Robusto a outliers y datos faltantes
- Muy rápido (especialmente LightGBM)
- Feature importance nativo

**Debilidades:**
- **Requiere ingeniería de features:** crear lags, medias móviles, etc. manualmente
- Extrapola mal (predice cercano a rango de entrenamiento)
- Ignora la naturaleza secuencial (trata como regresión clásica)
- Sesgado a períodos recientes (sin peso histórico)

**Metodología:**
```python
import pandas as pd
from xgboost import XGBRegressor

# Crear lags
df['lag1'] = df['valor'].shift(1)
df['lag7'] = df['valor'].shift(7)
df['ma7'] = df['valor'].rolling(7).mean()

X = df[['lag1', 'lag7', 'ma7']].dropna()
y = df['valor'].iloc[len(df)-len(X):]

modelo = XGBRegressor(n_estimators=100, learning_rate=0.05)
modelo.fit(X, y)
```

**Aporte:** Útil como benchmark rápido o cuando hay datos irregulares/multivariantes sin temporalidad crítica.

---

#### 5️⃣ **Hybridization: ARIMA + Machine Learning**

**Concepto:** Combina componente clásica determinista (ARIMA) con residuos aprendidos por ML.

**Metodología:**
```
1. Ajustar ARIMA(p,d,q) → obtener pronóstico F_arima y residuos R
2. Entrenar ML (LSTM/XGBoost) sobre residuos R
3. Predicción final = F_arima + F_residuos_ML
```

**Bondades:**
- Hereda interpretabilidad de ARIMA + flexibilidad de ML
- Mejor desempeño que modelos puros en datos mixtos
- Combina estabilidad estadística con aprendizaje

**Debilidades:**
- Mayor complejidad operacional
- Riesgo de acumular errores de ambos modelos

**Aporte:** Mejor en producción; balance entre robustez y precisión.

---

### 📊 Matriz Comparativa: Todos los Métodos

| Método | Datos Requeridos | Interpretabilidad | Velocidad | Complejidad | Mejor Para |
|---|---|---|---|---|---|
| **ARIMA/SARIMA** | Pequeño (100+) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Baja | Series limpias, univariantes |
| **LSTM** | Grande (500+) | ⭐ | ⭐⭐ | Muy Alta | Patrones complejos no lineales |
| **Transformer** | Muy Grande (1000+) | ⭐⭐ | ⭐⭐⭐ | Muy Alta | Alta frecuencia (fintech) |
| **GAM** | Medio (200+) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Media | Balance precisión/interpretabilidad |
| **XGBoost/LightGBM** | Medio (300+) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Media | Múltiples features, benchmark |
| **Hybrid** | Medio (300+) | ⭐⭐⭐ | ⭐⭐⭐ | Alta | Producción robusta |


---

## 🔄 Guía Práctica de Aplicación

### 🔄 Flujo de Trabajo Típico

```
1. CARGA → 2. VISUALIZACIÓN → 3. ESTACIONARIEDAD → 4. AJUSTE → 5. VALIDACIÓN
```

### 💻 Código: Detección de Estacionariedad

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Cargar datos
df = pd.read_csv('datos.csv', index_col='fecha', parse_dates=True)

# Test Dickey-Fuller
resultado = adfuller(df['valor'])
print(f"p-value: {resultado[1]}")
# Si p-value < 0.05: Data es estacionaria ✓
```

### 💻 Código: Ajuste de SARIMA

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Definir modelo SARIMA(1,1,1)(1,1,1,12) para datos mensuales
modelo = SARIMAX(
    df['valor'],
    order=(1, 1, 1),           # (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # (P, D, Q, s)
)

# Entrenar
resultado = modelo.fit(disp=False)

# Pronosticar próximos 12 meses
pronóstico = resultado.get_forecast(steps=12)
df_predicción = pronóstico.conf_int()
```

### 💻 Código: Prophet (Automatizado)

```python
from prophet import Prophet

# Preparar datos (columnas: 'ds' y 'y')
df_prophet = df.reset_index().rename(
    columns={'fecha': 'ds', 'valor': 'y'}
)

# Crear y entrenar
modelo = Prophet(yearly_seasonality=True, monthly_seasonality=True)
modelo.fit(df_prophet)

# Generar fechas futuras
futuro = modelo.make_future_dataframe(periods=12, freq='M')
pronóstico = modelo.predict(futuro)

# Visualizar
figura = modelo.plot(pronóstico)
```

---

## 📋 Tabla Comparativa: ¿Qué Modelo Elegir?

| Características | ARIMA/SARIMA | Exponential Smoothing | Prophet | Redes Neuronales |
|---|---|---|---|---|
| **Datos estacionarios necesarios** | Sí | No | No | No |
| **Maneja festivos/eventos** | No | No | **Sí** | Requiere preparación |
| **Interpretabilidad** | **Alta** | **Alta** | Media | Baja |
| **Velocidad de entrenamiento** | **Rápida** | **Rápida** | Media | Lenta |
| **Escalabilidad multivariante** | Limitada | Limitada | Media | **Excelente** |
| **Curva de aprendizaje** | Media | Baja | **Muy baja** | Alta |

---

## 🎓 Resumen

El análisis de series de tiempo combina **estadística clásica** (ARIMA) con **automatización moderna** (Prophet) y **flexibilidad de ML** (redes neuronales). Elige según tu contexto: datos limpios y estables → ARIMA; datos ruidosos con eventos → Prophet; patrones complejos multivariantes → redes neuronales.
