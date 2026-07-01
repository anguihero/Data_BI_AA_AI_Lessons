---
Autor: anmmunozsa@outlook.es
---

# Sesión 06: Visualización de Datos — Matplotlib, Seaborn, Plotly y Folium

## 🎯 Objetivo de la Sesión

Generar visualizaciones estáticas, interactivas y geoespaciales a partir de un DataFrame. Esta sesión es **puramente de generación de gráficos**: la interpretación estadística profunda (qué significa una correlación, cómo detectar outliers, etc.) se deja para la Sesión 07.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido de la Sesión 05 y por qué visualizar es tan importante como transformar datos |
| 2 | 25 min | Matplotlib: `plot`, `bar`, `hist`, `boxplot`, `scatter`, subplots (`plt.subplots`), títulos/etiquetas/leyendas |
| 3 | 20 min | Seaborn: `heatmap` (matriz de correlación), `pairplot`, `countplot`, `boxplot` con estilo mejorado |
| 4 | 25 min | Plotly (Plotly Express): gráficos **interactivos** — `px.scatter`, `px.bar`, `px.line`, `px.histogram`, hover con tooltips |
| 5 | 25 min | Folium: mapa base, **marcadores** individuales, `folium.plugins.HeatMap` para mapas de calor |
| 6 | 15 min | Laboratorio Figure/Axes y reto con gráficos estáticos, interactivos y geográficos |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-1–16). Por tratarse de una sesión centrada en herramientas de graficación, cada bloque presenta cuándo usar una librería o tipo de gráfico, sus parámetros y un ejemplo ejecutable.

### Recorrido real del notebook

- Matplotlib: histogramas, boxplots, dispersión, barras y control `Figure`/`Axes`.
- Seaborn: histogramas, boxplots y mapa de correlación.
- Plotly Express: dispersión interactiva y `scatter_geo`.
- Folium: mapa base, marcadores y `HeatMap` con coordenadas de países.
- Tabla para elegir gráfico y laboratorio que separa datos, geometría y presentación; incluye `hue`, `marker`, `errorbar`, títulos, ejes y cuadrícula.

**Temas técnicos a cubrir con detalle:**
- Matplotlib como la base de todo el ecosistema de graficación en Python (Seaborn y Pandas `.plot()` lo usan por debajo).
- Seaborn como capa de alto nivel para gráficos estadísticos con mejor estética por defecto.
- Plotly Express para interactividad (zoom, hover, filtrar por leyenda) — clave quando se comparte el gráfico fuera del notebook (HTML embebible).
- Folium (envoltorio Python de Leaflet.js): estructura básica `folium.Map(location=[lat, lon], zoom_start=...)`, añadir `folium.Marker`, y usar `HeatMap` para densidad de puntos.
- Tabla comparativa: "¿qué gráfico uso según lo que quiero mostrar?" (comparación, composición, distribución, relación, evolución temporal, ubicación geográfica).

**Aplicaciones del mundo real a mencionar:** reportes ejecutivos con Matplotlib/Seaborn, dashboards interactivos internos con Plotly, mapas de calor de siniestralidad/ventas/entregas con Folium.

## 📦 Dataset(s)

- Un DataFrame sintético de salarios para los gráficos estadísticos y otro de ventas mensuales para el laboratorio paso a paso.
- **Geoespacial — "World Coordinates" (Kaggle)**: [qramkrishna/world-coordinates](https://www.kaggle.com/datasets/qramkrishna/world-coordinates?select=world_country.csv) (`world_country.csv`, columnas `latitude`/`longitude` por país), descargado con la API de Kaggle enseñada en la Sesión 04. **Fuente remota alternativa** (sin necesidad de Kaggle): CSV crudo de [q-viper/state-location-coordinates](https://github.com/q-viper/state-location-coordinates) leído directamente con `pd.read_csv(url)`.

## 🏆 Retos de Práctica

- **Básico:** con un dataset tabular, generar un histograma y un boxplot con Matplotlib, y un heatmap de correlación con Seaborn.
- **Medio:** recrear uno de los gráficos anteriores en versión interactiva con Plotly Express, agregando color por una variable categórica.
- **Avanzado:** con el dataset "World Coordinates", crear un mapa Folium con marcadores por país (popup con el nombre) y un segundo mapa con `HeatMap` mostrando la densidad de puntos; adicionalmente, recrear la misma vista con `px.scatter_geo` o `px.choropleth` de Plotly para comparar ambos enfoques.

## ✅ Criterios de Evaluación

- Al menos 2 gráficos estáticos correctamente etiquetados (título, ejes).
- Al menos 1 gráfico interactivo con Plotly.
- Un mapa Folium funcional con marcadores o mapa de calor.

## 🔗 Prerrequisitos

Sesión 05 (Data Wrangling — se necesita un DataFrame limpio para graficar).

## 🚀 Siguiente Paso

En la Sesión 07 interpretamos estadísticamente los datos (distribuciones, correlación, outliers) y los dejamos listos para modelar.
