---
Autor: anmmunozsa@outlook.es
---

# Sesión 03: Bases de Python II

## 🎯 Objetivo de la Sesión

Dominar las estructuras de datos y funciones avanzadas de Python necesarias para trabajar con Pandas en las siguientes sesiones.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 02 |
| 2 | 25 min | Listas y tuplas: creación, indexado, slicing, métodos comunes (`append`, `sort`, etc.) |
| 3 | 20 min | Diccionarios y sets: pares clave-valor, acceso, iteración |
| 4 | 20 min | Comprehensions (listas y diccionarios) |
| 5 | 20 min | Funciones avanzadas: `*args`/`**kwargs`, funciones `lambda` |
| 6 | 15 min | Manejo de excepciones: `try`/`except`/`finally` |
| 7 | 10 min | Archivos y laboratorio de elección, mutación y búsqueda en colecciones |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16).

### Recorrido real del notebook

- Listas, tuplas, diccionarios y conjuntos; mutabilidad y acceso.
- Comprehensions, `*args`, `**kwargs`, `lambda` y manejo de excepciones.
- Lectura y escritura de un archivo de texto con `with open(...)`.
- Laboratorio de elección de colecciones, copia frente a mutación, métodos `update`/`get`, pertenencia y costo práctico de búsqueda.

**Temas técnicos a cubrir con detalle:**
- Diferencia lista (mutable) vs. tupla (inmutable) y cuándo usar cada una.
- Diccionarios como la estructura más cercana a un "registro" de datos (puente conceptual hacia una fila de DataFrame).
- Comprehensions como forma "pythónica" de transformar colecciones (equivalente conceptual a lo que luego se hará con `.apply()` en Pandas).
- Manejo de errores como buena práctica antes de trabajar con datos reales (archivos que no existen, tipos inesperados).

**Aplicaciones del mundo real a mencionar:** procesar una lista de ventas, contar frecuencia de palabras (diccionario), leer un archivo de configuración, manejar errores al leer un CSV corrupto.

## 📦 Dataset(s)

No aplica — sigue siendo fundamentos de lenguaje, pero se introduce la lectura de un archivo `.csv` o `.txt` simple como preparación a Pandas (puede ser un archivo de ejemplo generado en el propio notebook).

## 🏆 Retos de Práctica

- **Básico:** dado un diccionario de estudiante→calificaciones, calcular el promedio de cada uno.
- **Medio:** usar comprehensions para filtrar y transformar una lista de transacciones (ej. quedarse solo con las mayores a cierto monto y convertirlas a otra moneda).
- **Avanzado:** construir una función que lea un archivo `.csv` línea por línea (sin Pandas), maneje errores de formato con `try/except`, y devuelva un diccionario resumen (conteos, totales).

## ✅ Criterios de Evaluación

- Uso correcto de al menos 2 estructuras de datos (lista/diccionario/set).
- Al menos una comprehension funcional.
- Manejo de al menos un caso de excepción con `try/except`.
- Justificación de la estructura elegida para representar o buscar datos.

## 🔗 Prerrequisitos

Sesión 02 (variables, tipos, control de flujo, funciones).

## 🚀 Siguiente Paso

En la Sesión 04 damos el salto a Pandas: Series, DataFrames, y cómo obtener datasets reales (incluyendo la API de Kaggle).
