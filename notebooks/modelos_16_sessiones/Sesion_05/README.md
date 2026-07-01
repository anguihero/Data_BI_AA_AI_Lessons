---
Autor: anmmunozsa@outlook.es
---

# Sesión 05: Pandas II — Data Wrangling

## 🎯 Objetivo de la Sesión

Transformar, combinar y limpiar datasets reales usando las operaciones de Pandas más usadas en un flujo de trabajo profesional.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 04 |
| 2 | 25 min | `groupby` + funciones de agregación (`sum`, `mean`, `count`, `agg` con múltiples funciones) |
| 3 | 20 min | Combinar datasets: `merge` (inner/left/right/outer), `concat` |
| 4 | 20 min | Tablas dinámicas: `pivot_table` |
| 5 | 20 min | Valores faltantes: `isna()`, `dropna()`, `fillna()` (introducción; el detalle de estrategias de imputación se profundiza en la Sesión 07) |
| 6 | 15 min | Creación de columnas derivadas: `apply`, `map`, operaciones vectorizadas, extracción de componentes de fecha (`dt.month`, `dt.dayofweek`) |
| 7 | 10 min | Laboratorio `agg`/`transform`, validación de uniones y reto |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16).

### Recorrido real del notebook

- `groupby` con agregaciones simples y múltiples.
- Combinación de tablas con `merge` y `concat`; reorganización con `pivot_table`.
- Diagnóstico y tratamiento inicial de valores faltantes.
- Creación de variables con operaciones vectorizadas, `map`, `apply` y accesores de fecha.
- Laboratorio de `agg` frente a `transform`, diferencias respecto a la media del grupo y uniones auditables con `validate` e `indicator`.

**Temas técnicos a cubrir con detalle:**
- `groupby` como el equivalente de "resumir por categoría" — comparar contra la alternativa manual con diccionarios vista en la Sesión 03.
- Tipos de `merge` (inner/left/right/outer) con diagramas tipo Venn para visualizar la diferencia.
- `pivot_table` como forma de "reformatear" datos para reportes (equivalente a tablas dinámicas de Excel, dato que conecta con la experiencia previa de muchos estudiantes).
- Diferencia entre `apply` (fila/columna completa) y `map` (elemento a elemento en una Serie).
- Por qué operar sobre fechas requiere convertir primero con `pd.to_datetime`.

**Aplicaciones del mundo real a mencionar:** consolidar reportes de ventas por región y mes, cruzar una tabla de clientes con una de transacciones, preparar variables de fecha para un modelo de series de tiempo.

## 📦 Dataset(s)

- Tablas sintéticas de ventas, vendedores, ventas mensuales y regiones construidas en el notebook. Esto permite concentrarse en las operaciones y observar con claridad el efecto de cada transformación.

## 🏆 Retos de Práctica

- **Básico:** calcular el salario promedio agrupado por una columna categórica (ej. país o rol) usando `groupby`.
- **Medio:** crear una tabla dinámica (`pivot_table`) que cruce dos variables categóricas mostrando el promedio de una variable numérica, y combinarla con un `merge` de una tabla auxiliar.
- **Avanzado:** construir un pipeline de wrangling completo: cargar dos datasets, unirlos con `merge`, crear al menos 2 columnas derivadas (una con `apply` y otra extrayendo componentes de fecha), y producir un resumen agregado final con `.agg()` usando múltiples funciones por columna.

## ✅ Criterios de Evaluación

- Uso correcto de `groupby` con al menos una función de agregación.
- Al menos un `merge` o `concat` ejecutado correctamente.
- Al menos una columna nueva creada con `apply`/`map`.
- Uso de `validate` o `indicator` para comprobar una unión.

## 🔗 Prerrequisitos

Sesión 04 (Series, DataFrame, carga de datos).

## 🚀 Siguiente Paso

En la Sesión 06 aprendemos a **visualizar** todo lo que ya sabemos transformar: Matplotlib, Seaborn, Plotly y mapas con Folium.
