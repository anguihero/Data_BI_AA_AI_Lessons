---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 13: Pipeline Completo de ML (End-to-End)

## 🎯 Objetivo de la Sesión

Construir un pipeline reproducible que encapsule todo el preprocesamiento y el modelo en un solo objeto de scikit-learn, evitando errores comunes como el data leakage.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido: todo lo hecho "a mano" en las Sesiones 07-12 y por qué conviene encapsularlo |
| 2 | 25 min | `Pipeline` de scikit-learn: encadenar pasos (`steps=[...]`), `fit`/`transform`/`predict` |
| 3 | 30 min | `ColumnTransformer`: aplicar transformaciones distintas a columnas numéricas y categóricas dentro del mismo pipeline |
| 4 | 20 min | Integrar el pipeline con `cross_val_score` — por qué esto previene el data leakage (el escalado/imputación se recalculan en cada fold) |
| 5 | 20 min | Recrear con `Pipeline` uno de los flujos manuales de sesiones anteriores (ej. regresión o clasificación) y comparar resultados |
| 6 | 15 min | Reto guiado: pipeline completo sobre un dataset nuevo |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-2–16). Referencia teórica: [fundamentos de aprendizaje automatico.md § 3.5 y 4.3](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md).

**Contenido técnico a cubrir con detalle:**
- Por qué hacer `fit_transform` en todo el dataset antes del split es un error de data leakage (recordar la advertencia del doc base § 3.5), y cómo `Pipeline` + `cross_val_score` lo resuelve automáticamente.
- Anatomía de un `ColumnTransformer`: `("num", StandardScaler(), columnas_numericas)`, `("cat", OneHotEncoder(), columnas_categoricas)`.
- Cómo anidar: `Pipeline([("preprocessor", column_transformer), ("model", RandomForestClassifier())])`.
- Ventaja práctica: un pipeline se puede guardar (`joblib`) y reutilizar en producción con los mismos pasos exactos de entrenamiento.
- No es un "método nuevo" sino una forma de **organizar** todo lo ya aprendido — el resumen para dummies aquí es "una sola pieza que hace todo el proceso, para no repetir código ni cometer errores".

**Aplicaciones del mundo real a mencionar:** por qué en la industria casi nunca se preprocesa "a mano" fuera de un pipeline, y cómo esto facilita el despliegue (doc base § 4.7).

## 📦 Dataset(s)

- **`sklearn.datasets.load_diabetes()`** — se continúa con el mismo dataset de las Sesiones 08 y 12. Como `load_diabetes` es solo numérico, el `ColumnTransformer` se explica igual (rama numérica) y se muestra cómo se **extendería** con una rama categórica usando `application_data.csv` (Sesión 07) como ejemplo de referencia, sin necesidad de cargarlo de nuevo en esta sesión.

## 🏆 Retos de Práctica

- **Básico:** construir un `Pipeline` simple de 2 pasos (`StandardScaler` + Regresión Lineal) sobre `load_diabetes`.
- **Medio:** construir un `ColumnTransformer` con la rama numérica de `load_diabetes`, integrado en un `Pipeline` con Ridge o Random Forest Regressor, validado con `cross_val_score`.
- **Avanzado:** reconstruir como un único `Pipeline` el "modelo campeón" de regresión (Sesión 08) sobre `load_diabetes`, comparando sus métricas de `cross_val_score` contra el flujo manual original; documentar en Markdown cómo se extendería el `ColumnTransformer` si el dataset tuviera variables categóricas (usando `application_data.csv` como ejemplo conceptual).

## ✅ Criterios de Evaluación

- `Pipeline` funcional que incluye al menos preprocesamiento + modelo.
- Uso de `ColumnTransformer` para tratar numéricas y categóricas por separado.
- Validación con `cross_val_score` sobre el pipeline completo (no sobre datos ya preprocesados manualmente).

## 🔗 Prerrequisitos

Sesiones 07-10 (preprocesamiento, regresión, clasificación, benchmarking).

## 🚀 Siguiente Paso

En la Sesión 14 optimizamos el pipeline con búsqueda de hiperparámetros: Grid Search, Random Search y Optimización Bayesiana.
