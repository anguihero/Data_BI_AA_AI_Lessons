---
Autor: anmmunozsa@outlook.es  
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 08: La Era de los Transformers

## Guía para Usar la API de HuggingFace en Google Colab

### 🎯 Objetivo de la Sesión

Introducirte al mundo de los **Transformers** y los **Large Language Models (LLMs)** usando la biblioteca de **HuggingFace**, una de las herramientas más poderosas y accesibles del ecosistema de Deep Learning moderno.

En esta sesión trabajarás con **modelos pre-entrenados de última generación** para realizar 3 tareas fundamentales:
1. **Análisis de Sentimiento** (Sentiment Analysis)
2. **Resumen de Texto** (Text Summarization)
3. **Generación de Texto** (Text Generation) con GPT-2 o BERT

Todo esto sin entrenar nada desde cero, solo usando APIs y modelos ya disponibles. ¡Bienvenido al paradigma del **Transfer Learning**!

---

## 📚 Conceptos Teóricos

### 1. ¿Qué Son los Transformers?

Los **Transformers** son una arquitectura de red neuronal introducida en 2017 por el paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) de Vaswani et al.

**Características clave**:
- Usan un mecanismo de **atención** para procesar secuencias de texto de forma eficiente
- No dependen de redes recurrentes (RNN) ni convolucionales (CNN)
- Permiten **paralelización** masiva en el entrenamiento
- Son la base de modelos como **BERT**, **GPT**, **T5**, **BART**, **RoBERTa**, y más

**Ventaja sobre RNNs**:
- Las RNNs procesan texto secuencialmente (palabra por palabra)
- Los Transformers procesan **todo el texto simultáneamente** usando atención

### 2. ¿Qué es el Mecanismo de Atención?

La **Atención** (Attention) permite al modelo enfocarse en las partes relevantes del texto para una tarea específica.

**Ejemplo**:
```
Texto: "El banco cerca del río cerró a las 5 PM"
Pregunta: "¿Qué cerró?"
```

Un modelo con atención puede identificar que "banco" (institución financiera) está relacionado con "cerró", no con "río" (orilla del río).

**Tipos de Atención en Transformers**:
1. **Self-Attention**: Cada palabra atiende a todas las demás palabras en la secuencia
2. **Multi-Head Attention**: Múltiples mecanismos de atención en paralelo para capturar diferentes relaciones

### 3. Principales Arquitecturas de Transformers

| Modelo | Arquitectura | Tarea Principal | Pre-entrenamiento |
|--------|--------------|-----------------|-------------------|
| **BERT** | Encoder | Comprensión de texto | Masked Language Model |
| **GPT-2/GPT-3** | Decoder | Generación de texto | Causal Language Model |
| **T5** | Encoder-Decoder | Text-to-Text | Span Corruption |
| **BART** | Encoder-Decoder | Resumen, traducción | Denoising Autoencoder |
| **RoBERTa** | Encoder | Comprensión (robusta) | BERT mejorado |

**BERT (Bidirectional Encoder Representations from Transformers)**:
- Lee el texto en **ambas direcciones** (izquierda-derecha y derecha-izquierda)
- Ideal para: clasificación, NER, question answering
- Pre-entrenamiento: predice palabras enmascaradas aleatoriamente

**GPT-2 (Generative Pre-trained Transformer 2)**:
- Lee el texto **unidireccionalmente** (izquierda a derecha)
- Ideal para: generación de texto, completación de frases
- Pre-entrenamiento: predice la siguiente palabra

**T5 (Text-to-Text Transfer Transformer)**:
- Convierte **todas las tareas** en formato texto → texto
- Ejemplos:
  - Traducción: "translate English to Spanish: Hello" → "Hola"
  - Resumen: "summarize: [long text]" → "[short summary]"

### 4. Transfer Learning y Fine-Tuning

**Transfer Learning** es el paradigma de:
1. Pre-entrenar un modelo en una tarea general (ejemplo: leer todo Internet)
2. **Ajustar** (fine-tune) el modelo para una tarea específica (ejemplo: sentimientos sobre productos de tu empresa)

**Ventajas**:
- No necesitas millones de datos etiquetados
- No necesitas GPUs masivas para entrenar desde cero
- Puedes lograr resultados state-of-the-art con pocos recursos

**Pipeline HuggingFace**:
```python
from transformers import pipeline

# Carga un modelo pre-entrenado
classifier = pipeline("sentiment-analysis")

# Úsalo directamente
resultado = classifier("Me encantó esta película")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 5. HuggingFace: El Hub de Transformers

[HuggingFace](https://huggingface.co/) es una plataforma que alberga:
- **+200,000 modelos pre-entrenados** (BERT, GPT-2, T5, LLaMA, etc.)
- **+30,000 datasets** (IMDb, SQuAD, GLUE, etc.)
- **Biblioteca Transformers**: API unificada para trabajar con cualquier modelo

**Ventajas**:
- Compatible con PyTorch, TensorFlow, JAX
- Modelos listos para usar con `pipeline()`
- Comunidad activa con modelos en 100+ idiomas

**Estructura de un modelo en HuggingFace**:
```
modelo
├── pytorch_model.bin (pesos pre-entrenados)
├── config.json (configuración del modelo)
├── tokenizer_config.json (configuración del tokenizer)
└── vocab.txt (vocabulario)
```

### 6. Tokenización en Transformers

Los Transformers no procesan texto directamente, primero lo **tokenizan** (convierten a números).

**Proceso**:
1. **Input**: "Hola, ¿cómo estás?"
2. **Tokens**: ["Hola", ",", "¿", "cómo", "estás", "?"]
3. **IDs**: [101, 5492, 1010, 1234, 5678, 999, 102]

**Tipos de tokenizers**:
- **Word-level**: Cada palabra = 1 token (vocabulario gigante)
- **Character-level**: Cada carácter = 1 token (secuencias largas)
- **Subword (BPE, WordPiece)**: Balance entre palabras y caracteres

Ejemplo de **WordPiece** (usado por BERT):
```
"unhappiness" → ["un", "##happiness"]
```

### 7. Tareas que Puedes Resolver con Transformers

| Tarea | Descripción | Modelo Recomendado |
|-------|-------------|--------------------|
| **Sentiment Analysis** | Determinar si el texto es positivo/negativo/neutro | BERT, RoBERTa, DistilBERT |
| **Named Entity Recognition (NER)** | Detectar nombres, lugares, organizaciones | BERT, RoBERTa |
| **Question Answering** | Responder preguntas dado un contexto | BERT, RoBERTa, ALBERT |
| **Text Summarization** | Generar resúmenes de textos largos | BART, T5, Pegasus |
| **Translation** | Traducir de un idioma a otro | T5, MarianMT, mBART |
| **Text Generation** | Generar texto continuando una frase | GPT-2, GPT-3, GPT-Neo |
| **Zero-Shot Classification** | Clasificar sin ejemplos de entrenamiento | BART, DeBERTa |

---

## 🛠️ Instalación y Configuración

### Requisitos
- **Python 3.7+**
- **Google Colab** (recomendado para este ejercicio) o entorno local con GPU

### Instalación en Google Colab

```python
# Instalar HuggingFace Transformers y dependencias
!pip install transformers torch sentencepiece

# Opcional: instalar datasets (para cargar datasets de HuggingFace)
!pip install datasets
```

**Nota**: Google Colab viene con PyTorch pre-instalado, pero asegúrate de tener la versión más reciente.

### Verificar Instalación

```python
import transformers
import torch

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📋 Estructura del Ejercicio

### Ejercicio 1: Análisis de Sentimiento

**Objetivo**: Clasificar textos en español/inglés como positivos, negativos o neutros.

**Modelo sugerido**: 
- `distilbert-base-uncased-finetuned-sst-2-english` (inglés)
- `nlptown/bert-base-multilingual-uncased-sentiment` (multilingüe, incluye español)

**Tarea**:
1. Cargar un pipeline de sentiment analysis
2. Analizar 5 frases de ejemplo
3. Interpretar el `score` (confianza del modelo)

### Ejercicio 2: Resumen de Texto

**Objetivo**: Generar un resumen automático de un texto largo.

**Modelo sugerido**:
- `facebook/bart-large-cnn` (inglés, entrenado en CNN/DailyMail)
- `t5-small` (inglés, generalista)

**Tarea**:
1. Cargar un pipeline de summarization
2. Proporcionar un artículo de 200-300 palabras
3. Generar un resumen de 50 palabras

### Ejercicio 3: Generación de Texto

**Objetivo**: Completar una frase inicial usando un modelo de lenguaje.

**Modelo sugerido**:
- `gpt2` (inglés)
- `DeepESP/gpt2-spanish` (español)

**Tarea**:
1. Cargar un pipeline de text generation
2. Proporcionar un prompt inicial
3. Generar 3 continuaciones diferentes
4. Ajustar parámetros: `max_length`, `temperature`, `top_k`, `top_p`

---

## 🎯 Retos de la Sesión

### Reto 1: Análisis de Sentimiento de Reseñas
Usa el pipeline de sentiment analysis para analizar reseñas de productos y determinar el **sentimiento promedio**.

### Reto 2: Resumen Automático de Noticias
Recopila 3 noticias de texto largo y genera resúmenes automáticos. Compara la calidad de diferentes modelos (BART vs T5).

### Reto 3: Generador de Ideas Creativas
Usa GPT-2 para completar frases creativas. Experimenta con diferentes valores de `temperature` (0.7, 1.0, 1.5) y observa cómo cambia la creatividad.

### Reto Bonus: Fine-Tuning Básico
Si tienes tiempo, intenta hacer fine-tuning de un modelo de sentiment analysis en un dataset pequeño en español (IMDb en español o similar).

---

## 🔑 Parámetros Importantes en Generación de Texto

### 1. `max_length`
Longitud máxima de la secuencia generada (en tokens).
```python
generator("Había una vez", max_length=50)
```

### 2. `temperature`
Controla la aleatoriedad:
- **Baja (0.5-0.7)**: Texto predecible y coherente
- **Media (0.8-1.0)**: Balance entre creatividad y coherencia
- **Alta (1.2-1.5)**: Texto muy creativo/aleatorio

```python
generator("Había una vez", temperature=1.2)
```

### 3. `top_k`
Considera solo las `k` palabras más probables en cada paso.
```python
generator("Había una vez", top_k=50)
```

### 4. `top_p` (Nucleus Sampling)
Considera palabras cuya probabilidad acumulada sea `p`.
```python
generator("Había una vez", top_p=0.9)
```

### 5. `num_return_sequences`
Número de secuencias diferentes a generar.
```python
generator("Había una vez", num_return_sequences=3)
```

---

## 📝 Checklist de Aprendizaje

Al finalizar esta sesión, deberías poder:

- [ ] Explicar qué son los Transformers y cómo funcionan (mecanismo de atención)
- [ ] Distinguir entre arquitecturas Encoder (BERT) y Decoder (GPT-2)
- [ ] Entender qué es Transfer Learning y por qué es revolucionario
- [ ] Usar la API `pipeline()` de HuggingFace para 3 tareas diferentes
- [ ] Interpretar el output de modelos (labels, scores)
- [ ] Ajustar hiperparámetros de generación (`temperature`, `top_k`, `top_p`)
- [ ] Identificar cuál modelo usar para cada tipo de tarea
- [ ] Buscar modelos en HuggingFace Hub

---

## 🚀 Recursos Adicionales

### Documentación Oficial
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Tutorials de HuggingFace](https://huggingface.co/course)

### Papers Fundamentales
- [Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

### Cursos Recomendados
- [HuggingFace Course](https://huggingface.co/course) (gratis)
- [Fast.ai NLP Course](https://www.fast.ai/)
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

### Comunidad
- [HuggingFace Forums](https://discuss.huggingface.co/)
- [Discord de HuggingFace](https://discord.gg/huggingface)

---

## 💡 Tips y Mejores Prácticas

### 1. Empieza con Modelos Pequeños
- Usa `distilbert` o `t5-small` antes de probar modelos gigantes
- Más rápido para iterar y aprender

### 2. Entiende el Idioma del Modelo
- Muchos modelos están entrenados solo en inglés
- Busca modelos multilingües: `bert-base-multilingual-cased`

### 3. Monitorea el Uso de Memoria
- Modelos grandes (GPT-3, T5-large) pueden exceder la RAM de Colab
- Usa `torch.cuda.empty_cache()` para liberar memoria

### 4. Guarda Resultados Localmente
```python
# Guardar outputs
import json
with open('resultados.json', 'w') as f:
    json.dump(resultados, f)
```

### 5. Experimenta con Diferentes Modelos
- No todos los modelos funcionan igual para todos los textos
- Compara 2-3 modelos diferentes y elige el mejor

---

## ⚠️ Consideraciones Éticas

### Sesgos en Modelos Pre-entrenados
Los modelos de lenguaje aprenden de texto de Internet, que contiene **sesgos humanos**:
- Estereotipos de género, raza, religión
- Lenguaje ofensivo o tóxico

**Ejemplo**: GPT-2 puede completar la frase "The doctor is a..." con "man" más frecuentemente que "woman".

**Responsabilidad**:
- Siempre revisa los outputs antes de usarlos en producción
- Implementa filtros de contenido inapropiado
- Documenta limitaciones conocidas

### Uso Responsable
- No uses generadores de texto para desinformación
- No hagas fine-tuning con datos sensibles sin consentimiento
- Respeta licencias de modelos (MIT, Apache, CC)

---

## 🎓 Preguntas de Reflexión

1. **¿Por qué los Transformers reemplazaron a las RNNs en NLP?**
2. **¿Cuál es la diferencia fundamental entre BERT y GPT-2?**
3. **¿Qué significa "pre-entrenado"? ¿Qué gano al usar un modelo pre-entrenado?**
4. **¿En qué contexto usarías sentiment analysis en una empresa real?**
5. **¿Qué riesgos existen al usar un modelo de generación de texto sin supervisión?**

---

## 🏆 Criterios de Éxito

Has completado exitosamente esta sesión si puedes:

1. ✅ Ejecutar los 3 pipelines (sentiment, summarization, generation) sin errores
2. ✅ Explicar cómo funciona el mecanismo de atención en una frase
3. ✅ Identificar cuándo usar BERT vs GPT-2
4. ✅ Encontrar un modelo específico en HuggingFace Hub para una tarea nueva
5. ✅ Generar texto creativo ajustando `temperature` y `top_k`

---

## 🔮 Próximos Pasos

Después de dominar esta sesión, considera explorar:

1. **Fine-Tuning**: Ajustar modelos para tus datos específicos
2. **RAG (Retrieval-Augmented Generation)**: Combinar búsqueda con generación
3. **LangChain**: Framework para construir aplicaciones con LLMs
4. **Prompt Engineering**: Optimizar las instrucciones para LLMs poderosos (GPT-3.5, GPT-4)
5. **Modelos Open-Source Grandes**: LLaMA, Falcon, Mistral

---

**¡Felicitaciones!** Has llegado al final del Curso de Analítica Avanzada. Ahora tienes las herramientas para trabajar con datos, entrenar modelos clásicos de Machine Learning, y usar los modelos más avanzados de IA del mundo. 

🚀 **El siguiente paso es tuyo: ¡construye algo increíble!** 🚀

---
