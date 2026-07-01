---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 04: Pandas I — Series, DataFrames y Obtención de Datos (Kaggle API)

## 🎯 Objetivo de la Sesión

Cargar, explorar y filtrar datos tabulares con Pandas, y aprender a obtener datasets reales tanto de archivos locales/Drive como de Kaggle mediante su API en Python.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 03 y motivación: ¿por qué Pandas? |
| 2 | 20 min | Series y DataFrame: creación, `.head()`, `.info()`, `.describe()`, `.shape`, `.dtypes` |
| 3 | 15 min | Lectura de datos: CSV, Excel (`read_csv`, `read_excel`) desde archivo local subido a Colab o desde Google Drive |
| 4 | 25 min | **Registro en Kaggle + API**: crear cuenta, generar token (`kaggle.json`), configurar credenciales en Colab, descargar un dataset con `kagglehub` u `opendatasets` |
| 5 | 20 min | Indexing y filtrado: `.loc`, `.iloc`, filtrado booleano, selección de columnas |
| 6 | 15 min | Tipos de variable en la práctica: numéricas (continua/discreta), categóricas (nominal/ordinal), temporales — cómo identificarlas con `.dtypes` |
| 7 | 15 min | Reto guiado: cargar un dataset descargado de Kaggle y hacer una primera exploración |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-2–16).

**Temas técnicos a cubrir con detalle:**
- Series vs. DataFrame: relación con listas/diccionarios ya vistos en Python.
- `.loc` (por etiqueta) vs. `.iloc` (por posición) — fortalezas/debilidades y errores comunes.
- Filtrado booleano encadenado (`&`, `|`) y por qué `and`/`or` de Python no funcionan igual en Pandas.
- **Instructivo Kaggle paso a paso**: (1) crear cuenta en kaggle.com, (2) ir a "Account" → "Create New Token" para descargar `kaggle.json`, (3) subir el token a Colab (`files.upload()` o Secrets de Colab), (4) instalar `kagglehub`/`kaggle`, (5) `kagglehub.dataset_download("usuario/nombre-dataset")` o `kaggle datasets download -d usuario/nombre-dataset`. Incluir nota de seguridad: nunca subir `kaggle.json` a un repositorio público.
- Tabla resumen de tipos de variable (doc base sección 0.3) aplicada con `.dtypes` y `.select_dtypes()`.

**Aplicaciones del mundo real a mencionar:** cómo cualquier analista de datos en la práctica descarga datasets públicos de Kaggle para prototipar modelos antes de tener acceso a datos productivos.

## 📦 Dataset(s)

- Dataset local ya incluido en el repo para la primera parte: `data/titanic/Titanic_Dataset.csv` o `data/loan/loan_data.csv`.
- Un dataset a elección del estudiante/instructor descargado en vivo desde Kaggle (criterio sugerido: tabular, < 50MB, con mezcla de variables numéricas y categóricas) para practicar el flujo completo de la API.

## 🏆 Retos de Práctica

- **Básico:** cargar `Titanic_Dataset.csv` y responder con código: ¿cuántas filas/columnas tiene?, ¿qué columnas tienen valores nulos?, ¿cuáles son categóricas y cuáles numéricas?
- **Medio:** descargar un dataset propio desde Kaggle usando la API, y reproducir el mismo análisis exploratorio básico (`.info()`, `.describe()`, `.dtypes`).
- **Avanzado:** con el dataset de Kaggle descargado, filtrar registros con al menos dos condiciones combinadas (`&`/`|`) y seleccionar solo un subconjunto de columnas relevantes, exportando el resultado a un nuevo CSV con `.to_csv()`.

## ✅ Criterios de Evaluación

- Descarga exitosa de al menos un dataset vía Kaggle API.
- Uso correcto de `.loc`/`.iloc` y filtrado booleano.
- Identificación correcta del tipo de al menos 3 variables del dataset.

## 🔗 Prerrequisitos

Sesión 03 (estructuras de datos, manejo de archivos y errores).

## 🚀 Siguiente Paso

En la Sesión 05 profundizamos en Data Wrangling: combinar, agrupar y transformar datasets con Pandas.
