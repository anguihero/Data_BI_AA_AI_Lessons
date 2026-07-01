---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 16: LLMs y NLP Aplicado en Python (100% gratuito)

## 🎯 Objetivo de la Sesión

Usar un modelo de lenguaje (LLM) vía API gratuita y un modelo descargado de Hugging Face para resolver tareas reales de NLP (clasificación de texto y reconocimiento de entidades nombradas — NER), cerrando el curso con un proyecto integrador corto.

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido: de CNN (Sesión 15) a Transformers — ambos son redes neuronales, pero los LLM ya vienen pre-entrenados |
| 2 | 15 min | Fundamentos rápidos de NLP: tokenización, embeddings (nivel intuitivo, sin álgebra) |
| 3 | 20 min | **API gratuita de Google Gemini**: crear API key en Google AI Studio (sin tarjeta de crédito), instalar `google-generativeai`, primer prompt |
| 4 | 20 min | Prompt Engineering básico: zero-shot, few-shot, Chain-of-Thought — aplicado a un caso real (ej. clasificar el sentimiento de reseñas) |
| 5 | 25 min | Hugging Face `transformers`: `pipeline("text-classification")` y `pipeline("ner")` con un modelo descargado localmente (sin API, sin costo) |
| 6 | 20 min | Comparar los dos enfoques (LLM vía API vs. modelo Hugging Face local) sobre el mismo texto: velocidad, calidad, control |
| 7 | 10 min | Cierre del curso: recapitulación de la ruta completa (Sesiones 1-16) y próximos pasos sugeridos |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-2–16). Referencia teórica: [fundamentos de llm.md](../../../background/esp/fundamentos%20de%20llm.md).

**Por cada enfoque (API Gemini, Hugging Face local), cubrir:**
- Teoría técnica: qué es un LLM decoder-only (Gemini/GPT) vs. un modelo encoder-only (BERT, usado típicamente en clasificación/NER de Hugging Face), diferencia entre "consumir" un modelo por API vs. "descargarlo y correrlo" localmente.
- Fortalezas y debilidades (API: no requiere GPU propia ni descarga, pero depende de conexión e impone límites de uso gratuito; Hugging Face local: control total y sin límites de llamadas, pero requiere descargar el modelo y más recursos de cómputo en Colab).
- Código ya escrito ejecutando cada enfoque paso a paso.
- Resumen para dummies: "si necesitas texto generado o razonamiento flexible, usa la API; si necesitas clasificar o extraer entidades de forma rápida y repetible, usa un modelo de Hugging Face".

**Instructivo Google Gemini (gratuito) paso a paso:** (1) entrar a [aistudio.google.com](https://aistudio.google.com) con una cuenta de Google, (2) "Get API key" → crear una key gratuita (sin tarjeta de crédito, con cuota gratuita diaria), (3) guardar la key en Colab usando **Secrets** (🔑 en la barra lateral) en vez de escribirla en texto plano, (4) `pip install google-generativeai`, (5) `genai.configure(api_key=...)` y `model.generate_content("...")`.

**Instructivo Hugging Face `transformers`:** `pip install transformers`, `from transformers import pipeline`, `clf = pipeline("text-classification", model="...")`, `ner = pipeline("ner", model="...", grouped_entities=True)` — todo corre localmente en el runtime de Colab, sin necesidad de token ni cuenta (para modelos públicos).

**Aplicaciones del mundo real a mencionar:** clasificación automática de tickets de soporte, extracción de nombres de personas/empresas/lugares de noticias o contratos, chatbots simples, resumen de documentos.

## 📦 Dataset(s)

- Textos cortos de ejemplo (reseñas de producto, tickets de soporte, noticias) generados dentro del propio notebook o tomados de `exercises/AI_Impact_Student_Life_2026 - Copy.csv` si contiene texto libre.
- Opcional: un dataset de texto descargado de Kaggle (usando lo aprendido en la Sesión 04) para el reto avanzado.

## 🏆 Retos de Práctica

- **Básico:** usar la API de Gemini para clasificar el sentimiento (positivo/negativo/neutral) de 5 reseñas de producto escritas a mano.
- **Medio:** usar un pipeline de Hugging Face (`text-classification`) para hacer la misma tarea sobre los mismos 5 textos, y comparar los resultados contra los de Gemini.
- **Avanzado:** aplicar `pipeline("ner")` sobre un texto más largo (ej. una noticia) para extraer personas/organizaciones/lugares, y luego pedirle a Gemini (con few-shot prompting) que genere un resumen de una línea del mismo texto — comparando ambas salidas en una tabla de "proyecto integrador" que cierre el curso.

## ✅ Criterios de Evaluación

- Uso exitoso y gratuito de la API de Gemini (key configurada de forma segura, no expuesta en texto plano).
- Uso exitoso de al menos un pipeline de Hugging Face (clasificación o NER).
- Comparación escrita entre ambos enfoques (API vs. modelo local).

## 🔗 Prerrequisitos

Sesión 15 (redes neuronales, como base conceptual de los Transformers) y Sesión 01 (Markdown, para documentar el proyecto integrador final).

## 🚀 Siguiente Paso

Fin de la ruta de 16 sesiones. Como proyecto de continuidad autodirigida, se sugiere explorar `exercises/nlp/`, `exercises/generative_ai/` y `exercises/computer_vision/` para profundizar según el interés de cada estudiante.
