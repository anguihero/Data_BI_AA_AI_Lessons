---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 05: Domando Pandas

## 🎯 Objetivo de la Sesión
Dominar las operaciones avanzadas de Pandas que te convertirán en un maestro de la manipulación de datos: groupby, pivot tables, filtros complejos y manejo de valores nulos.

## 📚 Conceptos Teóricos

### ¿Por Qué Pandas es Fundamental?
Pandas es la librería más importante para Data Science en Python. Es la herramienta que usarás diariamente para:
- Limpiar y transformar datos
- Realizar análisis exploratorios
- Preparar datos para modelos de ML
- Generar reportes y agregaciones

> **"Si SQL es el lenguaje de las bases de datos, Pandas es el lenguaje de los DataFrames."**

### Anatomía de un DataFrame

```python
import pandas as pd

# Un DataFrame es una tabla con filas y columnas
df = pd.DataFrame({
    'nombre': ['Ana', 'Carlos', 'María'],
    'edad': [25, 30, 28],
    'ciudad': ['Bogotá', 'Medellín', 'Cali']
})
```

**Componentes clave**:
- **Index**: Etiquetas de fila (por defecto 0, 1, 2...)
- **Columns**: Nombres de columnas
- **Values**: Los datos en sí
- **dtypes**: Tipo de dato de cada columna

### GroupBy: El Poder de la Agregación

GroupBy permite dividir datos en grupos y aplicar funciones a cada grupo.

#### Sintaxis Básica
```python
# Agrupar por una columna
df.groupby('categoria').mean()

# Agrupar por múltiples columnas
df.groupby(['categoria', 'region']).sum()

# Agregar con múltiples funciones
df.groupby('categoria')['ventas'].agg(['sum', 'mean', 'count'])
```

#### Funciones de Agregación Comunes
- `sum()`: Suma
- `mean()`: Promedio
- `median()`: Mediana
- `count()`: Conteo
- `min()`, `max()`: Mínimo y máximo
- `std()`: Desviación estándar
- `var()`: Varianza
- `first()`, `last()`: Primer y último valor

#### Agregaciones Personalizadas
```python
# Función personalizada
def rango(x):
    return x.max() - x.min()

df.groupby('categoria')['precio'].agg([rango, 'mean'])

# Lambda
df.groupby('categoria')['precio'].agg(lambda x: x.quantile(0.75))
```

#### Transform vs Apply
```python
# Transform: Retorna un objeto del mismo tamaño que el grupo
df['precio_normalizado'] = df.groupby('categoria')['precio'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Apply: Puede retornar cualquier estructura
df.groupby('categoria').apply(lambda x: x.head(3))
```

### Pivot Tables: Agregación Multidimensional

Las tablas dinámicas son perfectas para resumir datos en formato de matriz.

```python
# Pivot table básica
pd.pivot_table(
    df,
    values='ventas',      # Valores a agregar
    index='categoria',    # Filas
    columns='mes',        # Columnas
    aggfunc='sum',        # Función de agregación
    fill_value=0          # Llenar nulos con 0
)
```

#### Pivot vs Pivot Table
- **pivot()**: Reorganiza datos sin agregación (requiere índices únicos)
- **pivot_table()**: Reorganiza Y agrega datos

```python
# pivot() - Datos ya agregados
df.pivot(index='fecha', columns='producto', values='ventas')

# pivot_table() - Agrega automáticamente
df.pivot_table(index='fecha', columns='producto', values='ventas', aggfunc='sum')
```

### Filtros Complejos

#### Filtros Básicos
```python
# Condición simple
df[df['edad'] > 25]

# Múltiples condiciones con &, |
df[(df['edad'] > 25) & (df['ciudad'] == 'Bogotá')]

# isin() para múltiples valores
df[df['ciudad'].isin(['Bogotá', 'Medellín'])]

# contains() para strings
df[df['nombre'].str.contains('Ana', case=False)]
```

#### Query: SQL-like Filtering
```python
# Sintaxis alternativa más legible
df.query('edad > 25 and ciudad == "Bogotá"')

# Con variables
min_edad = 25
df.query('edad > @min_edad')
```

#### Boolean Indexing Avanzado
```python
# Filtrado con múltiples condiciones
condicion1 = df['edad'] > 25
condicion2 = df['salario'] > 50000
condicion3 = df['ciudad'].isin(['Bogotá', 'Medellín'])

df[condicion1 & (condicion2 | condicion3)]
```

### Manejo de Valores Nulos

Los valores nulos (NaN, None, NULL) son inevitables en datos reales.

#### Detección de Nulos
```python
# Identificar nulos
df.isnull()        # Retorna DataFrame booleano
df.isna()          # Alias de isnull()
df.notnull()       # Inverso

# Conteo de nulos por columna
df.isnull().sum()

# Porcentaje de nulos
(df.isnull().sum() / len(df)) * 100
```

#### Eliminación de Nulos
```python
# Eliminar filas con cualquier nulo
df.dropna()

# Eliminar solo si TODAS las columnas son nulas
df.dropna(how='all')

# Eliminar columnas con nulos
df.dropna(axis=1)

# Eliminar si >50% son nulos
df.dropna(thresh=len(df)*0.5, axis=1)

# Solo en columnas específicas
df.dropna(subset=['columna1', 'columna2'])
```

#### Imputación de Nulos
```python
# Llenar con valor constante
df.fillna(0)

# Llenar con estadística
df['columna'].fillna(df['columna'].mean())

# Forward fill (propagar último valor válido)
df.fillna(method='ffill')

# Backward fill
df.fillna(method='bfill')

# Interpolar (útil para series temporales)
df['columna'].interpolate()

# Llenar por grupo
df['salario'].fillna(df.groupby('departamento')['salario'].transform('mean'))
```

### Operaciones Encadenadas (Method Chaining)

Pandas permite encadenar operaciones para código más limpio.

```python
# Sin encadenamiento (menos legible)
df_temp = df[df['edad'] > 25]
df_temp = df_temp.groupby('ciudad')['salario'].mean()
df_temp = df_temp.sort_values(ascending=False)

# Con encadenamiento (más legible)
resultado = (df
    .query('edad > 25')
    .groupby('ciudad')['salario']
    .mean()
    .sort_values(ascending=False)
)
```

### Window Functions

Cálculos sobre "ventanas" de datos, útiles para análisis de series temporales.

```python
# Rolling mean (media móvil)
df['media_movil'] = df['ventas'].rolling(window=7).mean()

# Cumulative sum (suma acumulativa)
df['ventas_acumuladas'] = df['ventas'].cumsum()

# Shift (desplazar valores)
df['ventas_ayer'] = df['ventas'].shift(1)
df['ventas_manana'] = df['ventas'].shift(-1)

# Cambio porcentual
df['cambio_pct'] = df['precio'].pct_change()
```

## 🏆 Reto: Análisis de Churn de Clientes

Tienes un dataset de churn de clientes de una empresa de telecomunicaciones. Debes realizar análisis de manipulación intensiva usando Pandas.

### Dataset
Columnas del dataset de churn:
- `customer_id`: ID único del cliente
- `tenure`: Meses como cliente
- `monthly_charges`: Cargo mensual
- `total_charges`: Cargo total
- `contract_type`: Tipo de contrato (Month-to-month, One year, Two year)
- `payment_method`: Método de pago
- `internet_service`: Tipo de servicio de internet
- `churn`: Si abandonó el servicio (Yes/No)

### Tareas del Reto

#### 1. Groupby: Análisis por Segmentos
- Calcular la tasa de churn por tipo de contrato
- Promedio de ingresos mensuales por método de pago
- Cliente con mayor tenure por tipo de internet

#### 2. Pivot Tables: Matrices de Agregación
- Crear tabla dinámica: churn (filas) vs contract_type (columnas) con conteo
- Crear tabla: internet_service (filas) vs payment_method (columnas) con promedio de monthly_charges

#### 3. Filtros Complejos
- Clientes con tenure > 12 meses Y monthly_charges > 70
- Clientes con contrato "Month-to-month" O que hayan hecho churn
- Clientes con internet service DSL con charges mayores al promedio

#### 4. Limpieza de Nulos
- Identificar columnas con valores nulos
- Imputar total_charges nulos con la mediana por tipo de contrato
- Eliminar filas donde customer_id sea nulo

## 💡 Tips para el Éxito
1. **Usa .head() constantemente**: Valida cada paso con una muestra pequeña
2. **Aprende los warnings**: Pandas te avisa cuando haces cosas ineficientes
3. **SettingWithCopyWarning**: Usa `.copy()` cuando trabajes con subsets
4. **Categoricals**: Convierte columnas repetitivas a tipo category para ahorrar memoria
5. **Docstrings de funciones**: Usa `df.groupby?` para ver documentación rápida

## 📊 Criterios de Evaluación
- ✅ Uso correcto de groupby con múltiples agregaciones
- ✅ Pivot tables bien estructuradas con índices y columnas apropiadas
- ✅ Filtros complejos con sintaxis limpia
- ✅ Manejo inteligente de valores nulos
- ✅ Código encadenado y legible

## 📖 Recursos Complementarios
- [Pandas GroupBy Documentation](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [Pandas Pivot Table Guide](https://pandas.pydata.org/docs/user_guide/reshaping.html)
- [Missing Data Handling](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Pandas Optimization Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)

## 🔍 Conceptos que Dominarás
- Split-Apply-Combine con groupby
- Agregaciones múltiples y personalizadas
- Pivot y pivot_table para análisis multidimensional
- Filtrado complejo con condiciones múltiples
- Estrategias de imputación de valores nulos
- Method chaining para código elegante

## 🚀 Siguiente Paso
En la Sesión 06 pasarás de manipular datos a construir modelos predictivos con Aprendizaje Supervisado: clasificación y regresión.

---
**¡Pandas es tu superpoder en Data Science! 🐼💪**
