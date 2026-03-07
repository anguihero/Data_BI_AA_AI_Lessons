
# Mastering LLMs: De las Redes Neuronales a la Optimización de Modelos Fundacionales

## 🧠 1. Bases del Análisis de Texto y NLP


### Fundamentos Clásicos del Análisis de Texto

El análisis de texto tradicional se basaba en técnicas estadísticas simples pero efectivas. Los **n-gramas** —secuencias de n palabras consecutivas— permitían capturar patrones locales en el texto. Por ejemplo, los bigramas (2-gramas) como "machine learning" identificaban coocurrencias frecuentes, mientras que los trigramas revelaban estructuras sintácticas comunes. Estas técnicas formaban la base de modelos como Bag-of-Words (BoW), donde se contaba la frecuencia de palabras ignorando el orden. Con TF-IDF (Term Frequency-Inverse Document Frequency), se ponderaban las palabras según su relevancia en documentos específicos versus el corpus completo.


#### Ejemplos Prácticos de N-gramas y BoW

**Ejemplo de Unigramas (n=1)**:
```
Texto: "el gato come pescado el gato duerme"
Unigramas: ["el", "gato", "come", "pescado", "el", "gato", "duerme"]
Frecuencia: {"el": 2, "gato": 2, "come": 1, "pescado": 1, "duerme": 1}
```

**Ejemplo de Bigramas (n=2)**:
```
Texto: "el gato come pescado"
Bigramas: ["el gato", "gato come", "come pescado"]
Frecuencia: {"el gato": 1, "gato come": 1, "come pescado": 1}
```

**Ejemplo de Trigramas (n=3)**:
```
Texto: "el gato come pescado fresco"
Trigramas: ["el gato come", "gato come pescado", "come pescado fresco"]
```

**Bag-of-Words (BoW) - Ejemplo Simple**:
```python
# Documento 1: "el gato es blanco"
# Documento 2: "el perro es negro"

Vocabulario: ["el", "gato", "es", "blanco", "perro", "negro"]
Vector Doc1:  [1,   1,     1,  1,       0,      0]
Vector Doc2:  [1,   0,     1,  0,       1,      1]
# Cada posición = frecuencia de la palabra
```

**TF-IDF - Comparación de Relevancia**:
```
Corpus: 3 documentos sobre películas

Doc A: "película acción explosiones"  (muy común)
Doc B: "película drama emoción"       (menos común)
Doc C: "película acción coreografía"  (menos común)

Palabra "acción":
- Aparece en 2 documentos → menos relevante
- TF-IDF bajo (aparece frecuentemente en todo el corpus)

Palabra "coreografía":
- Aparece en 1 documento → más relevante
- TF-IDF alto (rara en el corpus, distingue Doc C)
```


**Técnicas descriptivas clásicas** como análisis de frecuencia, matriz de coocurrencia y análisis de similitud del coseno permitían tareas básicas pero importantes. El cálculo de similitud entre documentos mediante distancia euclidiana o coseno era fundamental para recuperación de información. Las técnicas de stemming y lematización reducían palabras a su raíz morfológica, mejorando la generalización. Estas aproximaciones eran computacionalmente eficientes y ampliamente utilizadas en buscadores web, detección de spam y clasificación de textos.


#### Ejemplos Prácticos de Técnicas Descriptivas Clásicas

**Análisis de Frecuencia - Ejemplo Simple**:
```
Texto: "el perro corre el gato salta el perro juega"

Frecuencia de palabras:
- "el": 3 veces
- "perro": 2 veces
- "corre": 1 vez
- "gato": 1 vez
- "salta": 1 vez
- "juega": 1 vez

Palabras más frecuentes = más relevantes para este documento
```

**Matriz de Coocurrencia - Palabras que Aparecen Juntas**:
```
Documentos:
Doc A: "gato blanco come pescado"
Doc B: "perro negro come carne"
Doc C: "gato negro juega solo"

Matriz de coocurrencia (¿qué palabras aparecen juntas?):
    gato  perro  negro  come
gato  [  0     0      1      1  ]
perro [  0     0      1      1  ]
negro [  1     1      0      1  ]
come  [  1     1      1      0  ]

Insight: "gato" y "negro" aparecen juntos → posible relación
```

**Similitud del Coseno - Comparar Documentos**:
```
Doc 1: "café caliente delicioso"
Doc 2: "café frío amargo"
Doc 3: "perro blanco grande"

Vector Doc 1: [café:1, caliente:1, delicioso:1, frío:0, amargo:0, perro:0, blanco:0, grande:0]
Vector Doc 2: [café:1, caliente:0, delicioso:0, frío:1, amargo:1, perro:0, blanco:0, grande:0]
Vector Doc 3: [café:0, caliente:0, delicioso:0, frío:0, amargo:0, perro:1, blanco:1, grande:1]

Similitud coseno Doc1-Doc2: 0.71 (muy similares - ambos sobre café)
Similitud coseno Doc1-Doc3: 0.00 (ningunos topic en común)
```

**Stemming y Lematización - Reducir Palabras a Raíz**:
```
Palabras originales: "corriendo", "corrió", "corre", "carrera"

Stemming (trunca agresivamente):
- "corriendo" → "corr"
- "corrió" → "corr"
- "corre" → "corr"
- "carrera" → "carr"

Lematización (reduce a forma canónica):
- "corriendo" → "correr"
- "corrió" → "correr"
- "corre" → "correr"
- "carrera" → "carrera" (palabra diferente)

Beneficio: variantes del mismo verbo se agrupan como equivalentes
```

**Distancia Euclidiana - Proximidad en Espacio Vectorial**:
```
Documentos como puntos en 2D (simplificado):

      Eje Y (tema: tecnología)
      |
    5 |     Doc A (tecnología: 5)
    4 |
    3 |               Doc B (tecnología: 3, deportes: 4)
    2 |
    1 |     Doc C (deportes: 5)
    0 |_____________________ Eje X (tema: deportes)
      0  1  2  3  4  5

Distancia Doc A a Doc B = √[(5-3)² + (0-4)²] = √20 ≈ 4.5 (medianamente similar)
Distancia Doc A a Doc C = √[(5-0)² + (0-5)²] = √50 ≈ 7.1 (muy diferente)

Documentos cercanos = similares en contenido
```


Los **casos de uso iniciales** incluían filtrado de correo no deseado, categorización automática de documentos, búsqueda de palabras clave y análisis de sentimientos primitivos basados en léxicos. Los motores de búsqueda tempranos dependían casi exclusivamente de estas técnicas. La clasificación de textos mediante Naive Bayes y Máquinas de Vectores de Soporte (SVM) demostraba que el procesamiento estadístico simple podía ser efectivo. Sin embargo, estas aproximaciones tenían limitaciones fundamentales: no capturaban significado semántico profundo, eran sensibles a variaciones morfológicas y requerían ingeniería manual intensiva de características.

**Limitaciones que impulsaron la innovación**: la maldición de la dimensionalidad hacía crecer exponencialmente el espacio de características; la falta de contexto llevaba a ambigüedad semántica irresoluble; y la incapacidad de modelar secuencias largas impedía capturar dependencias sintácticas complejas. Las técnicas clásicas no podían diferenciar entre "banco" (institución financiera) y "banco" (asiento) sin contexto suficiente, ni comprendían relaciones semánticas como sinonimia o analogía. Estas limitaciones quedaron evidentes con el crecimiento exponencial de datos textuales en internet.

---

### Evolución hacia Arquitecturas Inteligentes

La llegada de **redes neuronales recurrentes (RNN)** a principios de los 2000s marcó un cambio paradigmático. Estas arquitecturas procesaban sequences manteniendo un estado oculto que capturaba información contextual a través del tiempo. A diferencia de n-gramas que solo miraban ventanas locales fijas, las RNN podían aprender dependencias a largo plazo (teóricamente). El entrenamiento era costoso pero los resultados superiores justificaban el esfuerzo computacional.

**LSTMs y GRUs** (1997 y 2014) resolvieron el problema del desvanecimiento de gradientes que limitaba a las RNN simples. Estos mecanismos con compuertas (gates) permitían mantener información relevante a través de secuencias largas, mejorando dramáticamente el desempeño en traducción automática, generación de texto y análisis de sentimientos. Los avances en GPU computing y frameworks como TensorFlow hicieron viables estos entrenamientos a escala.

La **revolución de los Transformers en 2017** cambió completamente el paradigma. El mecanismo de Self-Attention permitía que el modelo ponderara automáticamente la importancia de cada palabra respecto a todas las demás, independientemente de su distancia. Esto eliminó la dependencia secuencial de RNN, permitiendo paralelización masiva. Modelos como BERT (2019) y GPT (2018 en adelante) demostraron que el pre-entrenamiento no supervisado en enormes corpus de texto transfería conocimiento lingüístico reutilizable.

**Aplicaciones inteligentes emergentes** incluyen sistemas de pregunta-respuesta que comprenden contexto complejo, traducción neural multilingüe prácticamente en tiempo real, resumen automático que captura ideas globales, y generación de texto coherente para asistentes virtuales. El reconocimiento de entidades nombradas pasó de heurísticas a aprendizaje profundo. Los modelos fundacionales actuales pueden realizar cientos de tareas sin reentrenamiento, mediante adaptación con pocas muestras o instrucciones simples.

El salto de complejidad fue exponencial: de contar palabras a entender matices semánticos, ironía, contexto cultural e incluso razonamiento lógico multi-paso. Los avances en infraestructura (TPUs, sistemas distribuidos), datos masivos (internet, repositorios públicos) y algoritmos de optimización (Adam, learning rate scheduling avanzado) convergieron para posibilitar modelos con miles de millones de parámetros capaces de tareas previamente consideradas exclusivamente humanas.



### 1.1 Flujo de Procesamiento de Texto

#### Tokenización


La tokenización convierte texto crudo en unidades procesables que el modelo puede procesar:

**Tokenización por palabras**: división simple separando por espacios
```
Texto: "Machine learning es fascinante"
Tokens: ["Machine", "learning", "es", "fascinante"]
```
Limitación: palabras raras o morfológicamente complejas se pierden.

**Subword tokenization**: divide palabras en subunidades para mejor generalización

- **BPE (Byte-Pair Encoding)**:
```
Texto: "unbelievable"
Pasos iterativos: "un" + "believ" + "able"
Tokens: ["un", "believ", "able"]
```

- **WordPiece** (usado en BERT):
```
Texto: "playing"
Tokens: ["play", "##ing"]
(## indica continuación)
```

- **SentencePiece** (usado en modelos multilingües):
```
Texto: "こんにちは" (japonés)
Tokens: ["▁こ", "ん", "に", "ち", "は"]
(▁ representa espacios)
```

**Impacto de tokens especiales**:
```
Texto original: "Hola mundo"

Con tokens especiales:
[CLS] Hola mundo [SEP]
[CLS]  = inicio de secuencia (clasificación)
[SEP]  = separador entre frases
[PAD]  = relleno para longitud uniforme
[UNK]  = palabra desconocida
[MASK] = token enmascarado (preentrenamiento)
```

Ejemplo completo en BERT:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.encode("Machine learning is great")
# Salida: [101, 2573, 4500, 2003, 1996, 1999, 102]
# 101=[CLS], 102=[SEP], resto son IDs de palabras/subpalabras
```


#### Embeddings y Espacios Vectoriales
Los embeddings transforman tokens en vectores densos de $d$ dimensiones:

$$\text{embedding}(token) \in \mathbb{R}^d$$

Propiedades: cercanía semántica, operaciones algebraicas, captura de relaciones.

### 1.2 Evolución Arquitectónica

| Arquitectura | Año | Características | Limitación |
|---|---|---|---|
| RNN | 2000s | Secuencia temporal | Vanishing gradient |
| LSTM | 1997 | Compuertas de control | Dependencia secuencial |
| GRU | 2014 | LSTM simplificado | Lento en secuencias largas |
| **Transformers** | **2017** | **Self-Attention paralizable** | **O(n²) en memoria** |

---

## ⚙️ 2. Anatomía de los Transformers

### 2.1 Mecanismo de Self-Attention

Dado un token query $Q$, key $K$ y value $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde $d_k$ es la dimensión de las claves.

### 2.2 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Con $h$ cabezas paralelas procesando diferentes espacios de representación.

### 2.3 Positional Encoding

Inyecta información posicional mediante:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### 2.4 Arquitecturas Principales

| Tipo | Ejemplo | Componentes | Caso de Uso |
|---|---|---|---|
| **Encoder-Only** | BERT | Masked LM | Clasificación, extracción |
| **Decoder-Only** | GPT-4 | Causal LM | Generación autoregresiva |
| **Encoder-Decoder** | T5, BART | Ambos | Traducción, resumen |

---

## 🏆 3. Familias de Modelos Fundacionales


### 3.1 Origen de los Modelos LLM: Del Perceptrón a los Transformers

#### Raíces Históricas: El Viaje de las Redes Neuronales

**1958: El Perceptrón - Primera Neural Red**
```
Entrada → [w₁, w₂] pesos → Suma → Activación → Salida (0 o 1)

Ejemplo dummy:
¿Es un email SPAM?
- Palabra "gratis" detectada: peso = +0.8 (indicador SPAM)
- Palabra "importante" detectada: peso = -0.6 (indicador legítimo)
- Suma: 0.8 - 0.6 = 0.2 → Predicción: SPAM

Limitación: solo problemas linealmente separables
```

**1980s: Redes Neuronales Profundas (Deep Learning)**
```
Entrada → [Capa 1] → [Capa 2] → [Capa 3] → Salida

Ejemplo dummy de capas:
Texto: "El gato duerme"

Capa 1 (características básicas):
- Detecta letras, sonidos
- Salida: patrones simples

Capa 2 (características intermedias):
- Detecta palabras, partes del habla
- Salida: "sustantivo", "verbo"

Capa 3 (características abstractas):
- Entiende semántica
- Salida: "animal pequeño realiza acción de reposo"
```

**2000s: RNNs - El Primer Paso hacia Secuencias**
```
Arquitectura con memoria (estado oculto h):

Posición 0: h₀ = 0 (sin información previa)
Posición 1: "El" → h₁ = f(w·"El" + h₀) = información sobre artículo
Posición 2: "gato" → h₂ = f(w·"gato" + h₁) = contexto: arte+sustantivo
Posición 3: "duerme" → h₃ = f(w·"duerme" + h₂) = contexto: animal realiza verbo

Ventaja: cada token ve todo lo anterior
Desventaja: gradientes desaparecen en secuencias largas
```

**1997: LSTMs - Memoria a Largo Plazo**
```
Mecanismo de compuertas (gates) que controlan el flujo:

┌─────────────────────────────────────────┐
│ Estado de Celda (Memoria a Largo Plazo) │
│              C = [1.5, 0.3, -0.8]       │
└─────────────────────────────────────────┘

Compuerta de Olvido: "¿Qué olvido?" → 0.1 (olvida 90% de info vieja)
Compuerta de Entrada: "¿Qué añado?" → 0.9 (añade 90% de info nueva)
Compuerta de Salida: "¿Qué muestro?" → 0.7 (muestra 70% de memoria)

Resultado: información importante persiste en secuencias de 100+ tokens

Ejemplo:
"Una empresa fue fundada en 1985... [50 tokens después] ... la empresa (LSTM recuerda año)"
```

**2017: Transformers - El Gran Salto (Sin RNN)**
```
Self-Attention: cada token atiende a TODOS los demás simultáneamente

Texto: "El banco es grande"

Query (Q): "¿Cuál es el contexto relevante?"
↓
Palabras clave (K): ["El", "banco", "es", "grande"]
↓
Valores (V): [definición artículo, institución financiera, verbo ser, adjetivo]

Self-Attention calcula pesos de importancia:
- "banco" mirando "El": peso = 0.1 (poco relevante)
- "banco" mirando "banco": peso = 0.9 (MUY relevante - es él mismo)
- "banco" mirando "es": peso = 0.3 (medianamente relevante)
- "banco" mirando "grande": peso = 0.2 (contexto descriptivo)

Resultado: "banco" = instancia de institución financiera (no mueble)

Ventaja crucial: Paralelización masiva (procesa todo simultáneamente)
```

---

#### Evolución de Capacidades: De Básico a Fondacional

**Generación 1 (2018): GPT Original - 117M parámetros**
```
Capacidades dummy:
- Entrada: "El cielo es"
- Salida: "azul" (completa la oración)
- Precisión: ~70% en tareas simples
- Casos de uso: autocompletado básico, escritura simple

Contexto limitado:
Solo ve 512 tokens anteriores = aproximadamente 350 palabras
```

**Generación 2 (2019): BERT - Bidireccional - 340M parámetros**
```
Cambio fundamental: Ve contexto ANTES y DESPUÉS

Caso dummy:
Frase: "El banco [MASK] grande"

BERT ve:
- Contexto izquierdo: "El banco"
- Contexto derecho: "grande"
- Puede inferir: [MASK] = "es" (verbos comunes con bancos y adjetivos)

Mejora: 85% en tareas de clasificación (vs 70% antes)
Caso de uso: análisis de sentimientos, extracción de información
```

**Generación 3 (2020): GPT-3 - 175B parámetros (1000x más grande)**
```
Salto exponencial en capacidades:

With 1.5B params (GPT-1):
- "Cuál es la capital de Francia?" → Sin respuesta clara

With 175B params (GPT-3):
- "Cuál es la capital de Francia?" → "París"
- "Escribe un poema sobre gatos" → Poema coherente de 10 versos
- "Traduce al francés: Hello world" → "Bonjour le monde"

Capacidad emergente: Few-shot learning
    Enseñas 2 ejemplos → entiende el patrón
```

**Generación 4 (2023-2024): GPT-4, Llama 3, Claude - Multi-modal**
```
Capacidades avanzadas:

Razonamiento multi-paso:
"Un tren sale a 100 km/h, otro a 80. ¿Se encuentran en 2 horas?"
→ Divide en pasos lógicos:
    1. Distancia cerrada por hora: 100-80 = 20 km
    2. En 2 horas: 20 × 2 = 40 km
    3. Respuesta: Sí, se encuentran si están separados ≥40 km

Comprensión de imágenes (GPT-4V):
- Recibe imagen de un gráfico
- Extrae datos numéricos
- Genera insights

Conocimiento factual:
Entrenado en datos hasta 2024 (vs 2021 en GPT-3)
```

---

### 3.2 Familias de Modelos Fundacionales: Genealogía Completa

#### Linaje GPT (Decoder-Only)

```
                                        2018: GPT-1 (117M)
                                                        ↓
                                        2019: GPT-2 (1.5B)
                                                        ↓
                                        2020: GPT-3 (175B)
                                            ↙          ↖
                        2021:                    2023:
                GPT-3.5 (175B)           GPT-4 (?)
                        ↓                       ↓
                ChatGPT                  ChatGPT+
                (Nov 2022)           (Mar 2023)


Familia abierta (similares a GPT):
├─ Llama 1 (Meta, 2023): 7B-65B
├─ Llama 2 (Meta, 2023): 7B-70B ← Instrucciones de Meta
├─ Llama 3 (Meta, 2024): 8B-70B ← MÁS ACURADO
├─ Mistral 7B (2023): 7B ← Eficiente para mobile
├─ Mixtral 8x7B (2023): 47B MoE ← Sparse (más barato)
├─ Falcon 180B (2023): 180B ← RLG de TII
└─ Qwen (Alibaba, 2024): 7B-72B ← Multilingüe

Capacidades por tamaño (dummy):
- 7B: Responde preguntas básicas, resume textos
- 13B: Entiende contexto complejo, escribe código
- 70B: Razonamiento multi-paso, comprende matices
- 175B+: Creatividad, adaptación a nuevas tareas
```

#### Linaje BERT (Encoder-Only)

```
                                        2019: BERT (340M)
                                                        ↓
                        ┌───────────────┼───────────────┐
                        ↓               ↓               ↓
                RoBERTa         DistilBERT       ALBERT
                (2019)          (2019)           (2019)
                Más fuerte      Más rápido       Parámetros compartidos

Familia BERT:
├─ BERT base: 110M (rápido, preciso)
├─ BERT large: 340M (más preciso, lento)
├─ XLM-RoBERTa: Multilingüe (111 idiomas)
├─ DeBERTa: Attention mejorado (1.3B)
└─ E5-large: Embeddings especializados

Casos de uso:
- BERT → Clasificación de texto
- Embedding BERT → Búsqueda semántica
- Multilingüe BERT → Soporte 100+ idiomas

Limitación: NO genera texto nuevo
(Solo entiende y analiza)
```

#### Linaje T5/BART (Encoder-Decoder)

```
                                2020: T5 (11B)
                                        Encoder → Decoder
                                                
                                        2019: BART (400M)
                                        Encoder → Decoder

Familias:
├─ T5 (Google): Traducción, resumen, QA
├─ BART (Facebook): Resumen, parafraseo
├─ mT5 (Google): Multilingüe (101 idiomas)
└─ mBART (Facebook): Multilingüe

Ventaja única:
Combina lo mejor de dos mundos:
- Encoder: comprende contexto profundo
- Decoder: genera texto coherente

Dummy case:
Entrada: "Explicar la relatividad en una oración"
↓ Encoder procesa: extrae concepto
↓ Decoder genera: "La relatividad muestra que espacio y tiempo son relativos"

Performance:
├─ Precisión (tipo BERT): 88%
├─ Generación (tipo GPT): 82%
└─ Traducción especializada: 95%
```

---

### 3.3 Comparativa de Familias

```python
# Dummy: Mismo problema, diferentes modelos

Pregunta: "¿Cuál es el sentimiento de: 'Ese producto es genial'?"

# BERT (Encoder-Only)
bert = load_model("bert-base")
result = bert.classify(text)
output: {"sentimiento": "POSITIVO", "confianza": 0.94}
→ Solo CLASIFICA, no EXPLICA

# GPT-4 (Decoder-Only)
gpt4 = load_model("gpt-4")
result = gpt4.generate("Analiza sentimiento de: 'Ese producto es genial'")
output: """
El sentimiento es POSITIVO. La palabra "genial" es un adjetivo
de fuerte valoración positiva. El emisor expresa satisfacción.
"""
→ CLASIFICA y EXPLICA con razonamiento

# T5 (Encoder-Decoder)
t5 = load_model("t5-base")
result = t5.generate("sentiment: Ese producto es genial")
output: "POSITIVO"
→ Híbrido, especializado en tareas específicas
```

---

### 3.4 Tendencias Emergentes (2024+)

```
┌─────────────────────────────────────┐
│ Modelos Especializados              │
├─────────────────────────────────────┤
│ CodeLlama: Solo código              │
│ LLaVA: Visión + Lenguaje            │
│ Whisper: Audio → Texto              │
│ Dall-E 3: Texto → Imagen            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Modelos Eficientes                  │
├─────────────────────────────────────┤
│ Phi-3 (3B): MÍNIMO viable          │
│ TinyLlama (1.1B): Mobile-friendly   │
│ Gemini Nano (4B): On-device         │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Arquitecturas Novedosas             │
├─────────────────────────────────────┤
│ MoE (Mixture of Experts): Selectivo │
│ Recurrent (RecycleToken): Memoria   │
│ KAN (Kolmogorov Networks): Sparse   │
└─────────────────────────────────────┘
```


| Modelo | Organización | Parámetros | Tipo | Acceso |
|---|---|---|---|---|
| GPT-4o | OpenAI | Unknown | Decoder | API |
| Llama 3 | Meta | 8B-70B | Decoder | Open source |
| Claude 3.5 | Anthropic | Private | Decoder | API |
| Mistral 7B | Mistral AI | 7B | Decoder | Open source |
| Mixtral 8x7B | Mistral AI | 47B (MoE) | Decoder | Open source |
| Falcon 180B | TII | 180B | Decoder | Open source |

### 3.1 Pre-training vs Fine-tuning

**Pre-training**: Entrenamiento masivo en corpus diverso (predecir tokens siguientes)
**Fine-tuning**: Adaptación a tareas específicas con datos curados

---

## 🔧 4. Técnicas de Optimización y Eficiencia

### 4.1 Cuantización

Reduce precisión para menor consumo de memoria:

| Técnica | Formato Original | Formato Cuantizado | Compresión |
|---|---|---|---|
| INT8 | FP32 (4 bytes) | INT8 (1 byte) | 4x |
| NF4 | FP32 | FP4 + escala | 8x |
| GGUF | FP32 | Mixto (4b/8b) | 4-8x |
| AWQ | FP32 | INT4 (esquema) | 8x |

```python
# Ejemplo: cargar modelo cuantizado
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B",
    quantization_config=bnb_config
)
```

### 4.2 Destilación de Modelos

Entrena un modelo pequeño (estudiante) para replicar comportamiento del maestro:

```python
from torch.nn import KLDivLoss
import torch.nn.functional as F

# Pérdida de destilación
kl_loss = KLDivLoss(reduction="batchmean")
logits_student = student_model(input_ids)
logits_teacher = teacher_model(input_ids).detach()

loss_distill = kl_loss(
    F.log_softmax(logits_student/T, dim=-1),
    F.softmax(logits_teacher/T, dim=-1)
)
```

### 4.3 LoRA y QLoRA

Adapta modelos sin actualizar todos los parámetros:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # rango bajo
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
# Solo ~1-2% parámetros entrenables
```

---

## 🚀 5. Estrategias de Implementación

### 5.1 RAG (Retrieval-Augmented Generation)

Combina recuperación de información con generación:

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(docs, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectorstore.as_retriever(k=3)
)
```

### 5.2 Prompt Engineering Avanzado

**Chain-of-Thought (CoT)**:
```
Pregunta: ¿Cuál es 15 + 2 × 3?
Pensemos paso a paso:
1. Primero multiplicamos: 2 × 3 = 6
2. Luego sumamos: 15 + 6 = 21
Respuesta: 21
```

**Few-shot Prompting**:
```
Ejemplo 1: Input → Output esperado
Ejemplo 2: Input → Output esperado
Nuevo Input: ?
```

---

## 📚 Referencias Técnicas

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2019)
- *Language Models are Few-Shot Learners* (Brown et al., 2020)
- Documentación oficial: HuggingFace Transformers, PEFT, bitsandbytes
