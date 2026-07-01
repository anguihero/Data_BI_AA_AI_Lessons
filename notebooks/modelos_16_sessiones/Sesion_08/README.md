---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 08: Métodos de Regresión

## 🎯 Objetivo de la Sesión

Construir y evaluar modelos de regresión para predecir valores continuos, entendiendo cuándo usar cada variante y cómo interpretar sus métricas.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 07 y contexto: ¿qué es un problema de regresión? |
| 2 | 15 min | `train_test_split`: por qué y cómo dividir los datos antes de modelar |
| 3 | 25 min | Regresión Lineal simple y múltiple: teoría, coeficientes, supuestos básicos |
| 4 | 15 min | Regresión Polinómica: cuándo una relación no es lineal |
| 5 | 20 min | Regularización: Ridge (L2) y Lasso (L1) — qué problema resuelven (multicolinealidad, overfitting) |
| 6 | 20 min | Métricas de regresión: MAE, MSE, RMSE, R² — cómo interpretarlas y cuál priorizar |
| 7 | 15 min | Reto guiado: comparar Lineal vs. Ridge vs. Lasso sobre el mismo dataset |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-2–16). Referencia teórica: [fundamentos de aprendizaje automatico.md § 1.2 y 1.3](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md).

**Por cada método (Lineal, Polinómica, Ridge, Lasso), cubrir:**
- Teoría técnica: fórmula, supuestos, hiperparámetros clave (`alpha` en Ridge/Lasso, `degree` en polinómica).
- Fortalezas y debilidades (interpretabilidad vs. capacidad de capturar no linealidad; Ridge reduce coeficientes vs. Lasso los lleva a cero — selección de variables implícita).
- Código ya escrito entrenando el modelo con `scikit-learn`, mientras se explica cada parámetro.
- Resumen para dummies: "úsalo cuando..." en una frase.

**Aplicaciones del mundo real a mencionar:** predicción de precios de vivienda, pronóstico de ventas, estimación de consumo energético (ver doc base § 1.2).

## 📦 Dataset(s)

- **`sklearn.datasets.load_diabetes()`** — dataset clásico de scikit-learn (avance de la enfermedad de diabetes), 10 variables numéricas ya estandarizadas y una variable objetivo continua. Se elige por ser pequeño, sin necesidad de preprocesamiento adicional (ya visto en Sesión 07), y permitir enfocar la sesión en el algoritmo y sus métricas en vez de en la limpieza de datos.

## 🏆 Retos de Práctica

- **Básico:** entrenar una Regresión Lineal simple (1 variable predictora, ej. `bmi`) sobre `load_diabetes` para predecir el avance de la enfermedad y calcular MAE/RMSE/R².
- **Medio:** entrenar una Regresión Lineal múltiple con las 10 variables de `load_diabetes`, comparar sus métricas contra el modelo simple del reto básico.
- **Avanzado:** comparar en una tabla Regresión Lineal, Ridge y Lasso sobre `load_diabetes`, probando al menos 3 valores de `alpha`, e identificar qué variables "elimina" Lasso (coeficientes en 0).

## ✅ Criterios de Evaluación

- `train_test_split` aplicado correctamente (sin fuga de datos).
- Al menos 2 modelos de regresión entrenados y comparados con las 4 métricas (MAE/MSE/RMSE/R²).
- Interpretación escrita de qué métrica es más relevante para el caso de uso elegido.

## 🔗 Prerrequisitos

Sesión 07 (preprocesamiento: encoding, escalado, manejo de faltantes).

## 🚀 Siguiente Paso

En la Sesión 09 pasamos de predecir números a predecir categorías: Métodos de Clasificación.
