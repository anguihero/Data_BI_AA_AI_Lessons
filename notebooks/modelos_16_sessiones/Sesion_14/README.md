---
Autor: anmmunozsa@outlook.es
---

# Sesión 14: Optimización de Hiperparámetros

## 🎯 Objetivo de la Sesión

Afinar el mejor modelo/pipeline obtenido hasta ahora usando distintas estrategias de búsqueda de hiperparámetros, comparando su eficiencia y resultado.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido y contexto: diferencia entre parámetros (se aprenden) e hiperparámetros (se definen antes) |
| 2 | 25 min | Grid Search (`GridSearchCV`): búsqueda exhaustiva en una grilla, costo computacional |
| 3 | 25 min | Random Search (`RandomizedSearchCV`): muestreo aleatorio/estocástico del espacio de búsqueda, por qué suele ser más eficiente que Grid Search |
| 4 | 30 min | Optimización Bayesiana con Optuna: cómo aprende de iteraciones previas para explorar mejor el espacio, `study.optimize()` |
| 5 | 20 min | Comparar los 3 métodos: tiempo de ejecución vs. mejora en la métrica, sobre el pipeline de la Sesión 13 |
| 6 | 10 min | Laboratorio de presupuesto/`cv_results_` y cierre del bloque |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16). Referencia teórica: [fundamentos de aprendizaje automatico.md § 3.4](../../../background/esp/fundamentos%20de%20aprendizaje%20automatico.md).

### Recorrido real del notebook

- Pipeline base con `RandomForestRegressor` y `load_diabetes`.
- Búsqueda con `GridSearchCV`, `RandomizedSearchCV` y Optuna.
- Comparación de mejor R² y tiempo de ejecución.
- Laboratorio sobre función objetivo de validación cruzada, tamaño de la grilla y presupuesto de ajustes.
- Exploración de `cv_results_`: ranking, media, desviación, tiempo y parámetros.
- Revisión de `scoring`, `cv`, `refit`, `n_jobs`, `n_iter`, `random_state` y `n_trials`.

**Por cada estrategia (Grid Search, Random Search, Optuna/Bayesiana), cubrir:**
- Teoría técnica: cómo funciona, parámetros de configuración (`param_grid`, `n_iter`, `n_trials`).
- Fortalezas y debilidades (tabla del doc base § 3.4.1: Grid es exhaustivo pero costoso; Random es más eficiente pero puede pasar por alto el óptimo; Bayesiana aprende de resultados previos pero es más compleja de configurar).
- Código ya escrito ejecutando cada búsqueda sobre el mismo pipeline/modelo, para comparar resultados en igualdad de condiciones.
- Resumen para dummies: "Grid si tienes pocas combinaciones y tiempo; Random si tienes muchas combinaciones; Bayesiana si quieres lo mejor de ambos mundos y tienes Optuna instalado".

Random Search representa la búsqueda estocástica mediante muestreo aleatorio del espacio de hiperparámetros.

**Aplicaciones del mundo real a mencionar:** cómo equipos de ML deciden cuánto tiempo de cómputo invertir en tuning antes de llevar un modelo a producción.

## 📦 Dataset(s)

- **`sklearn.datasets.load_diabetes()`** — se optimiza el mismo `Pipeline` construido en la Sesión 13, para que el tuning se aplique sobre algo ya conocido y se puedan comparar métricas antes/después del ajuste.

## 🏆 Retos de Práctica

- **Básico:** usar `GridSearchCV` para afinar 1-2 hiperparámetros de un modelo simple (ej. `n_neighbors` en KNN o `max_depth` en un Árbol).
- **Medio:** usar `RandomizedSearchCV` sobre un espacio de búsqueda más amplio (ej. Random Forest con `n_estimators`, `max_depth`, `max_features`), y comparar el tiempo de ejecución y el resultado contra Grid Search.
- **Avanzado:** usar Optuna para optimizar el pipeline completo de la Sesión 13 (incluyendo el modelo campeón del benchmarking de la Sesión 10), reportando en una tabla final el mejor score y tiempo de cada una de las 3 estrategias.

## ✅ Criterios de Evaluación

- Implementación correcta de al menos 2 de las 3 estrategias de tuning.
- Comparación explícita de tiempo de ejecución y mejora en la métrica.
- Selección justificada de la configuración final de hiperparámetros.
- Evaluación final separada del proceso de búsqueda para evitar sobreajuste al conjunto de prueba.

## 🔗 Prerrequisitos

Sesión 13 (Pipeline completo) y Sesión 10 (benchmarking, para saber qué modelo optimizar).

## 🚀 Siguiente Paso

En la Sesión 15 damos el salto a Deep Learning: arquitecturas de redes neuronales y una CNN para clasificar dígitos MNIST.
