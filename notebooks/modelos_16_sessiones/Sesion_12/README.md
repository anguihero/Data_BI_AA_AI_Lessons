---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 12: Reducción de Dimensionalidad y Selección de Variables

## 🎯 Objetivo de la Sesión

Simplificar datasets de alta dimensionalidad preservando la mayor información posible, y aplicar criterios objetivos para seleccionar las variables más relevantes.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido y contexto: la "maldición de la dimensionalidad" |
| 2 | 30 min | PCA (Análisis de Componentes Principales): teoría paso a paso, varianza explicada acumulada |
| 3 | 20 min | t-SNE: reducción no lineal para visualización 2D/3D, diferencias con PCA |
| 4 | 20 min | Métodos de Filtro: correlación con la variable objetivo, varianza, pruebas estadísticas (Chi-cuadrado, ANOVA) |
| 5 | 20 min | Métodos Envolventes y Embebidos: Recursive Feature Elimination (RFE), Lasso como selector implícito |
| 6 | 20 min | Reto guiado: reducir un dataset de muchas variables a sus componentes principales y visualizarlo en 2D |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-2–16). Referencia teórica: [fundamentos de aprendizaje automatico.md § 2.2 y 3.7.5](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md).

**Por cada técnica (PCA, t-SNE, Filtro, Wrapper/RFE, Embebido/Lasso), cubrir:**
- Teoría técnica: pasos del algoritmo (para PCA: estandarizar → matriz de covarianza → eigenvectores/eigenvalues → proyectar), hiperparámetros clave (`n_components`, `perplexity` en t-SNE).
- Fortalezas y debilidades (PCA es lineal e interpretable en varianza pero no en significado de componentes; t-SNE es solo para visualización, computacionalmente costoso, ejes sin interpretación directa).
- Código ya escrito ejecutando cada técnica.
- Resumen para dummies: "PCA para reducir y modelar, t-SNE solo para visualizar".

**Regla práctica a destacar:** seleccionar el número de componentes de PCA que acumulen al menos 95% de la varianza explicada (graficar la curva de varianza acumulada).

**Aplicaciones del mundo real a mencionar:** visualización de datasets con cientos de variables, eliminación de ruido antes de KNN/SVM, reconocimiento facial con Eigenfaces.

## 📦 Dataset(s)

- **`sklearn.datasets.load_diabetes()`** — se reutiliza el mismo dataset de la Sesión 08 para que la reducción de dimensionalidad y la selección de variables se sientan como una continuación natural de la regresión ya construida, en vez de partir de un dataset desconocido.

## 🏆 Retos de Práctica

- **Básico:** aplicar PCA a las 10 variables de `load_diabetes` y graficar la varianza explicada acumulada por componente.
- **Medio:** reducir `load_diabetes` a 2 componentes principales y visualizar los puntos coloreados según si el avance de la enfermedad está por encima o debajo de la mediana, comparando el resultado contra una proyección con t-SNE.
- **Avanzado:** aplicar un método de selección de variables (RFE o Lasso) sobre `load_diabetes`, comparar el subconjunto de variables elegido contra un ranking simple de correlación con la variable objetivo (método de filtro), y entrenar la Regresión Lineal de la Sesión 08 con las variables seleccionadas vs. con todas, comparando MAE/R².

## ✅ Criterios de Evaluación

- Aplicación correcta de PCA con estandarización previa.
- Interpretación correcta de la varianza explicada acumulada.
- Al menos un método de selección de variables aplicado y justificado.

## 🔗 Prerrequisitos

Sesión 07 (escalado) y Sesión 11 (clustering, para conectar con visualización 2D de grupos).

## 🚀 Siguiente Paso

En la Sesión 13 integramos todo lo aprendido en un Pipeline completo de Machine Learning.
