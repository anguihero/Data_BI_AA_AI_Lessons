---
Autor: anmmunozsa@outlook.es
---

# Sesión 07: Estadística Aplicada y Preprocesamiento de Datos

## 🎯 Objetivo de la Sesión

Interpretar estadísticamente un dataset (tendencia central, dispersión, distribución, correlación) y dejarlo listo para modelar mediante las técnicas de preprocesamiento estándar.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 06 |
| 2 | 20 min | Estadística descriptiva: media, mediana, moda, varianza, desviación estándar, cuartiles/IQR |
| 3 | 15 min | Distribuciones de probabilidad (Normal, Binomial, Poisson) — nivel conceptual + regla empírica 68-95-99.7 |
| 4 | 20 min | Correlación: Pearson vs. Spearman, matriz de correlación/heatmap, correlación no implica causalidad |
| 5 | 15 min | Detección de outliers con IQR |
| 6 | 20 min | Manejo de valores faltantes: estrategias de imputación (media/mediana/moda, constante, predictiva) |
| 7 | 20 min | Codificación, escalado y laboratorio `fit`/`transform` por tipo de variable |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16). Referencia teórica directa: [fundamentos de aprendizaje automatico.md § 0](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md) y § 3.7.

### Recorrido real del notebook

- Carga selectiva y muestreo de `application_data.csv`.
- Estadística descriptiva, distribuciones, Pearson/Spearman y outliers con IQR.
- Imputación, One-Hot/Ordinal Encoding y escalado.
- Laboratorio por tipo de variable: resúmenes robustos para numéricas, frecuencias para categóricas y separación explícita entre `fit` y `transform`.
- Fórmulas de estandarización e IQR y revisión de los parámetros aprendidos por imputador y escalador.

**Temas técnicos a cubrir con detalle (cada uno con fortalezas/debilidades):**
- Media vs. mediana vs. moda: cuándo cada una es engañosa (ej. media con outliers).
- IQR como criterio objetivo para outliers (`Q1 - 1.5·IQR`, `Q3 + 1.5·IQR`).
- Pearson (relación lineal) vs. Spearman (relación monótona/no lineal, basada en rangos).
- Estrategias de imputación y **cuándo aplicarlas después del train/test split** (advertencia de data leakage, doc base § 3.7.1).
- Label Encoding vs. One-Hot Encoding vs. Ordinal Encoding — el error común de aplicar Label Encoding a variables nominales.
- Qué algoritmos requieren escalado y cuáles no (tabla del doc base § 3.7.2), como anticipo a las Sesiones 8-11.

**Aplicaciones del mundo real a mencionar:** detectar transacciones atípicas con IQR, decidir si usar media o mediana en un análisis de salarios, preparar variables categóricas de encuestas para un modelo.

## 📦 Dataset(s)

- **Principal — "Credit EDA Case Study" (Kaggle)**: [venkatasubramanian/credit-eda-case-study](https://www.kaggle.com/datasets/venkatasubramanian/credit-eda-case-study?select=application_data.csv) (`application_data.csv`), descargado con la API de Kaggle. Es rico en variables numéricas y categóricas y tiene valores faltantes reales — ideal para practicar imputación, outliers y encoding en un caso de negocio realista (riesgo crediticio).
- **Alternativa simplificada (si el dataset de crédito resulta muy pesado/complejo para 2 horas):** `sklearn.datasets.load_iris()` — dataset pequeño, sin faltantes ni categóricas, útil como ejemplo "de bolsillo" para introducir cada técnica antes de aplicarla sobre el dataset de crédito.

## 🏆 Retos de Práctica

- **Básico:** sobre `application_data.csv`, calcular media, mediana, desviación estándar e IQR de 2 variables numéricas (ej. ingresos, monto del crédito), e identificar si tienen outliers.
- **Medio:** generar una matriz de correlación (Pearson) de las variables numéricas del dataset y aplicar imputación de valores faltantes (media/mediana para una variable numérica, moda para una categórica).
- **Avanzado:** construir un flujo completo de preprocesamiento manual sobre `application_data.csv`: imputar faltantes, codificar al menos 2 variables categóricas (una nominal con One-Hot, una ordinal con Ordinal Encoding) y escalar las variables numéricas con `StandardScaler`, dejando un DataFrame final listo para modelar (se reutilizará en la Sesión 09 de clasificación de riesgo crediticio si el instructor lo decide).

## ✅ Criterios de Evaluación

- Cálculo correcto de medidas de tendencia central y dispersión.
- Identificación correcta de outliers con IQR.
- Aplicación correcta de al menos un método de encoding y uno de escalado.
- Preprocesamiento ajustado únicamente con datos de entrenamiento.

## 🔗 Prerrequisitos

Sesiones 04-06 (carga, wrangling y visualización de datos).

## 🚀 Siguiente Paso

En la Sesión 08 usamos estos datos ya preparados para construir nuestros primeros modelos: Regresión.
