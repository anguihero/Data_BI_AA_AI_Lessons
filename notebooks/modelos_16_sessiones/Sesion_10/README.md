---
Autor: anmmunozsa@outlook.es
---

# Sesión 10: Ensambles y Benchmarking de Modelos

## 🎯 Objetivo de la Sesión

Entender los métodos de ensamble (Random Forest, Boosting, XGBoost) y aprender a comparar objetivamente varios modelos con validación cruzada para elegir el mejor candidato.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 09 y motivación: ¿por qué combinar modelos? |
| 2 | 20 min | Random Forest: Bagging, cómo reduce la varianza de un árbol individual |
| 3 | 20 min | Boosting (conceptual) y Gradient Boosting: cómo corrige errores de forma secuencial |
| 4 | 20 min | XGBoost: por qué es tan usado, hiperparámetros clave (`n_estimators`, `learning_rate`, `max_depth`, `subsample`) |
| 5 | 20 min | Validación cruzada k-fold: `cross_val_score`, por qué es más robusta que un solo `train_test_split` |
| 6 | 20 min | **Benchmarking**: construir una tabla comparativa de 4-5 modelos (de las Sesiones 08-10) con su media y desviación estándar de la métrica elegida |
| 7 | 10 min | Laboratorio de residuos/boosting y selección justificada del modelo |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16). Referencia teórica: [fundamentos de aprendizaje automatico.md § 1.3 y 3.5](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md).

### Recorrido real del notebook

- Random Forest, Gradient Boosting y XGBoost sobre `load_breast_cancer`.
- Validación cruzada y función reutilizable para comparar modelos.
- Benchmarking de modelos individuales y ensambles mediante media y desviación.
- Laboratorio sobre residuos, pérdida cuadrática y la interpretación de boosting como descenso por gradiente en el espacio de funciones.
- Validación estratificada común y comparación de accuracy, F1, ROC-AUC y tiempo.

**Por cada método (Random Forest, Boosting, XGBoost), cubrir:**
- Teoría técnica: Bagging vs. Boosting (paralelo vs. secuencial), hiperparámetros clave y su efecto.
- Fortalezas y debilidades (robustez y manejo de ruido vs. menor interpretabilidad y necesidad de tuning cuidadoso).
- Código ya escrito entrenando cada modelo.
- Resumen para dummies.

**Benchmarking (parte central de la sesión):** construir una función/tabla que entrene N modelos con `cross_val_score` (k=5 o 10) y reporte media ± desviación estándar de la métrica relevante (accuracy/F1 para clasificación o RMSE/R² para regresión, según el dataset usado). Enfatizar que la **desviación estándar** importa tanto como la media: un modelo con mejor promedio pero mucha varianza puede ser menos confiable.

**Aplicaciones del mundo real a mencionar:** por qué XGBoost domina competencias de Kaggle en datos tabulares, cómo un equipo de datos decide en la práctica qué modelo llevar a producción.

## 📦 Dataset(s)

- **`sklearn.datasets.load_breast_cancer()`** — se mantiene el mismo dataset de la Sesión 09 a propósito, para poder comparar directamente los nuevos ensambles contra Logística/KNN/SVM/Árbol ya entrenados, y así dedicar el 100% del tiempo al benchmarking en vez de a entender un dataset nuevo.

## 🏆 Retos de Práctica

- **Básico:** entrenar un Random Forest sobre `load_breast_cancer` y comparar su accuracy/F1 contra el mejor modelo individual de la Sesión 09.
- **Medio:** entrenar Random Forest, Gradient Boosting y XGBoost sobre `load_breast_cancer`, y construir una tabla comparativa con `cross_val_score` (5-fold).
- **Avanzado:** construir una función reutilizable que reciba una lista de modelos (incluyendo los de la Sesión 09: Logística, KNN, SVM, Árbol) y el dataset `load_breast_cancer`, y devuelva un DataFrame ordenado por desempeño promedio con su desviación estándar — el "benchmarking automático".

## ✅ Criterios de Evaluación

- Al menos 2 modelos de ensamble entrenados correctamente.
- Uso correcto de `cross_val_score` (no solo un único split).
- Tabla de benchmarking con al menos 4 modelos comparados y una conclusión justificada sobre cuál elegir.
- Uso de los mismos folds y métricas para todos los candidatos.

## 🔗 Prerrequisitos

Sesión 09 (clasificación, matriz de confusión, métricas).

## 🚀 Siguiente Paso

En la Sesión 11 cambiamos de paradigma: Clustering, aprendizaje no supervisado sin etiquetas.
