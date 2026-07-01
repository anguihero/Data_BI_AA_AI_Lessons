---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 09: Métodos de Clasificación

## 🎯 Objetivo de la Sesión

Construir y evaluar modelos de clasificación para predecir categorías, entendiendo la matriz de confusión y cómo elegir la métrica correcta según el contexto de negocio.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 08 y contexto: clasificación binaria vs. multiclase |
| 2 | 20 min | Regresión Logística: teoría, función sigmoide, probabilidades |
| 3 | 15 min | K-Nearest Neighbors (KNN): teoría, importancia del escalado |
| 4 | 15 min | Support Vector Machine (SVM): teoría, kernels (lineal vs. RBF) a nivel intuitivo |
| 5 | 15 min | Árbol de Decisión: teoría, interpretabilidad, riesgo de overfitting |
| 6 | 25 min | Matriz de confusión y métricas: accuracy, precision, recall, F1-score, AUC-ROC — cuándo priorizar cada una |
| 7 | 20 min | Reto guiado: entrenar y comparar 2 clasificadores sobre el mismo dataset |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-2–16). Referencia teórica: [fundamentos de aprendizaje automatico.md § 1.1, 1.3 y 1.4](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md).

**Por cada método (Logística, KNN, SVM, Árbol), cubrir:**
- Teoría técnica: intuición del algoritmo, hiperparámetros clave (`C`/`penalty` en Logística, `n_neighbors` en KNN, `kernel`/`C`/`gamma` en SVM, `max_depth`/`criterion` en Árbol).
- Fortalezas y debilidades (tabla comparativa del doc base § 1.3, "Guía de Selección de Algoritmo").
- Código ya escrito entrenando cada modelo con `scikit-learn`.
- Resumen para dummies: "úsalo cuando..." en una frase.

**Métricas:** construir la matriz de confusión manualmente sobre un ejemplo pequeño antes de usar `confusion_matrix` de sklearn, para que el concepto de TP/TN/FP/FN quede claro. Usar el ejemplo del doc base (diagnóstico médico → priorizar Recall; filtro de spam → priorizar Precision).

**Aplicaciones del mundo real a mencionar:** detección de fraude, diagnóstico médico, filtrado de spam, scoring crediticio.

## 📦 Dataset(s)

- **Binaria — `sklearn.datasets.load_breast_cancer()`**: dataset de diagnóstico de cáncer de mama (maligno/benigno), 30 variables numéricas, ideal para clasificación binaria sin necesidad de preprocesamiento adicional.
- **Multiclase — `sklearn.datasets.load_wine()`**: 3 clases de vino según su composición química, usado para mostrar que los mismos algoritmos (con ajustes menores) también resuelven problemas de más de 2 categorías.

## 🏆 Retos de Práctica

- **Básico:** entrenar una Regresión Logística sobre `load_breast_cancer` y calcular accuracy, precision, recall y F1.
- **Medio:** entrenar KNN y un Árbol de Decisión sobre `load_breast_cancer`, comparar sus matrices de confusión y decidir cuál es mejor según el contexto (¿importa más el recall o la precisión en un diagnóstico médico?).
- **Avanzado:** entrenar los 4 algoritmos (Logística, KNN, SVM, Árbol) sobre `load_breast_cancer`, graficar sus curvas ROC en un mismo gráfico y comparar su AUC-ROC; adicionalmente, repetir el ejercicio con Regresión Logística multinomial sobre `load_wine` para practicar clasificación multiclase.

## ✅ Criterios de Evaluación

- Al menos 2 modelos de clasificación entrenados y evaluados con matriz de confusión.
- Justificación escrita de qué métrica se prioriza y por qué, según el caso de uso.
- Identificación correcta de si el dataset está balanceado o no.

## 🔗 Prerrequisitos

Sesión 07 (preprocesamiento) y Sesión 08 (flujo general de entrenar/evaluar con `train_test_split`).

## 🚀 Siguiente Paso

En la Sesión 10 subimos de nivel con ensambles (Random Forest, Boosting, XGBoost) y aprendemos a hacer benchmarking entre múltiples modelos.
