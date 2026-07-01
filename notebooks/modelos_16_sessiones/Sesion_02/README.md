---
Autor: anmmunozsa@outlook.es
---

# Sesión 02: Bases de Python I

## 🎯 Objetivo de la Sesión

Escribir programas Python básicos usando variables, tipos de datos, operadores y estructuras de control de flujo, sentando las bases para todo el curso.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de Colab/Markdown y contexto de la sesión |
| 2 | 25 min | Variables y tipos de datos: `int`, `float`, `str`, `bool`; conversión de tipos (`type()`, casting) |
| 3 | 20 min | Operadores aritméticos, de comparación y lógicos |
| 4 | 25 min | Estructuras condicionales: `if` / `elif` / `else` |
| 5 | 25 min | Bucles: `for` y `while`, `break`/`continue` |
| 6 | 15 min | Funciones, parámetros y laboratorio de métodos según el tipo de dato |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16): portada + TOC, introducción para dummies, teoría, código intercalado y resúmenes aplicados.

### Recorrido real del notebook

- Variables y tipos `str`, `int`, `float` y `bool`; `type()` y conversión.
- Operadores aritméticos, comparaciones y lógica booleana.
- Condicionales, ciclos `for`/`while` y funciones con parámetros y `return`.
- Laboratorio de métodos de texto (`strip`, `lower`, `replace`), operaciones numéricas, booleanos y validación de parámetros con `ValueError`.

**Temas técnicos a cubrir con detalle:**
- Tipos de datos primitivos y mutabilidad básica.
- Precedencia de operadores.
- Diferencia entre `if/elif/else` anidado vs. encadenado.
- Cuándo usar `for` (iteración conocida) vs. `while` (condición de parada).
- Anatomía de una función: parámetros posicionales, valor de retorno, alcance de variables (scope) a nivel introductorio.

**Aplicaciones del mundo real a mencionar:** validación de formularios, cálculo de descuentos/impuestos, contadores de eventos, automatización de tareas repetitivas.

## 📦 Dataset(s)

No aplica — sesión de fundamentos de lenguaje, sin datos externos.

## 🏆 Retos de Práctica

- **Básico:** calculadora de propinas (variables + operadores + `input`).
- **Medio:** clasificador de números (par/impar, positivo/negativo/cero) usando condicionales anidados, y una función que determine si un número es primo.
- **Avanzado:** simulador simple de cajero automático (bucle `while` con menú de opciones, validación de saldo, límite de intentos).

## ✅ Criterios de Evaluación

- Uso correcto de al menos 3 tipos de datos distintos.
- Al menos una función propia con parámetros y `return`.
- Código que corre sin errores en Colab.
- Uso apropiado de métodos o funciones según el tipo de dato.

## 🔗 Prerrequisitos

Sesión 01 (Google Colab en funcionamiento).

## 🚀 Siguiente Paso

En la Sesión 03 avanzamos a estructuras de datos (listas, diccionarios) y manejo de errores — la base directa para trabajar con Pandas.
