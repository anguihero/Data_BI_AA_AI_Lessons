---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 15: Arquitecturas de Redes Neuronales y CNN (MNIST)

## 🎯 Objetivo de la Sesión

Entender la arquitectura de una red neuronal desde el perceptrón hasta una red convolucional (CNN), y entrenar un modelo propio capaz de clasificar dígitos escritos a mano (MNIST).

## 🗺️ Agenda (2 horas)

| Bloque | Duración | Contenido |
| --- | --- | --- |
| 1 | 10 min | Repaso rápido: de ML clásico a Deep Learning, ¿qué cambia? |
| 2 | 15 min | El Perceptrón: intuición, límite de separabilidad lineal |
| 3 | 20 min | Redes densas (MLP): capas, funciones de activación (ReLU, Sigmoide, Softmax), intuición de backpropagation (sin derivar matemáticamente) |
| 4 | 15 min | Cargar y explorar el dataset MNIST (imágenes 28x28, 10 clases) |
| 5 | 25 min | Capas convolucionales y de pooling: `Conv2D`, `MaxPooling2D` — qué "ve" un filtro convolucional |
| 6 | 25 min | Construir y entrenar una CNN pequeña con Keras/TensorFlow sobre MNIST (pocas épocas, con GPU de Colab) |
| 7 | 10 min | Evaluar el modelo: accuracy, matriz de confusión, visualizar predicciones correctas e incorrectas |

## 📚 Contenido y Estructura del Notebook

Sigue el [formato estándar](../ruta.md#4-formato-estándar-de-notebook-sesiones-2–16). Referencia teórica: `notebooks/backup/fundamentos_redes_neuronales.ipynb` y `notebooks/backup/cnn_redes_convolucionales.ipynb` como base a actualizar.

**Por cada bloque conceptual (Perceptrón, MLP, CNN), cubrir:**
- Teoría técnica: qué resuelve cada arquitectura y por qué la anterior no bastaba (perceptrón solo separa linealmente → MLP con capas ocultas captura no linealidad → CNN aprovecha la estructura espacial de una imagen en vez de "aplanarla" ingenuamente).
- Fortalezas y debilidades (MLP sobre imágenes ignora la posición relativa de los píxeles; CNN es más eficiente y precisa en imágenes pero requiere más cómputo).
- Código ya escrito: construcción capa por capa en Keras (`Sequential`), explicando cada línea (`Conv2D(filters, kernel_size, activation)`, `MaxPooling2D`, `Flatten`, `Dense`).
- Resumen para dummies: "una CNN aprende a reconocer bordes, luego formas, luego dígitos completos — capa por capa".

**Importante para que la sesión funcione en 2 horas:** usar una arquitectura pequeña (2-3 capas convolucionales) y pocas épocas (5-10), activar GPU en Colab (`Entorno de ejecución > Cambiar tipo de entorno > GPU`) para que el entrenamiento tome minutos, no horas.

**Aplicaciones del mundo real a mencionar:** reconocimiento óptico de caracteres (OCR), clasificación de imágenes médicas, visión por computador en general — conectando con `exercises/computer_vision/` como posible siguiente paso fuera del curso.

## 📦 Dataset(s)

- **MNIST**, disponible directamente en `tensorflow.keras.datasets.mnist` o `torchvision.datasets.MNIST` — no requiere descarga manual ni Kaggle.

## 🏆 Retos de Práctica

- **Básico:** entrenar un MLP simple (sin convoluciones, con `Flatten` + `Dense`) sobre MNIST y reportar su accuracy.
- **Medio:** construir una CNN con 2 capas convolucionales + pooling, entrenarla y comparar su accuracy contra el MLP del reto básico.
- **Avanzado:** experimentar variando la arquitectura (agregar una capa convolucional más, cambiar el número de filtros, agregar `Dropout`) y documentar en una tabla cómo cambia el accuracy y el tiempo de entrenamiento; visualizar al menos 5 ejemplos donde el modelo se equivoca y proponer una hipótesis de por qué.

## ✅ Criterios de Evaluación

- CNN entrenada exitosamente con accuracy razonable (> 95% es alcanzable fácilmente en MNIST).
- Evaluación con matriz de confusión y visualización de errores.
- Comprensión demostrada de la diferencia entre MLP y CNN (explicada en una celda de Markdown propia).

## 🔗 Prerrequisitos

Sesión 09 (clasificación y métricas) — el flujo de entrenar/evaluar es conceptualmente el mismo, cambia la arquitectura del modelo.

## 🚀 Siguiente Paso

En la Sesión 16, la última del curso, damos el salto de "entrenar nuestras propias redes" a "usar modelos pre-entrenados": LLMs y NLP aplicado.
