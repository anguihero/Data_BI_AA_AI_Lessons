---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 02: Python Foundations I (Junior)

## 🎯 Objetivo de la Sesión
Dominar los fundamentos esenciales de Python que todo Data Scientist debe conocer. Esta sesión establece las bases sólidas sobre las cuales construirás modelos complejos de Machine Learning.

## 📚 Conceptos Teóricos

### ¿Por Qué Python para Data Science?
Python se ha convertido en el lenguaje dominante en Data Science por varias razones:
- **Sintaxis clara y legible**: Código que se lee casi como inglés
- **Ecosistema robusto**: NumPy, Pandas, Scikit-learn, TensorFlow
- **Comunidad activa**: Millones de desarrolladores compartiendo soluciones
- **Versatilidad**: Desde análisis exploratorio hasta producción en la nube

### Estructuras de Datos Fundamentales

#### 1. Listas
Las listas son colecciones ordenadas y mutables que pueden contener elementos de diferentes tipos.

```python
numeros = [1, 2, 3, 4, 5]
mixta = [1, "texto", 3.14, True]
```

**Operaciones clave**:
- Indexación y slicing
- Append, extend, insert
- Remove, pop, clear
- Sort, reverse

#### 2. Diccionarios
Estructuras de datos tipo clave-valor, extremadamente útiles para representar datos estructurados.

```python
persona = {
    "nombre": "Ana",
    "edad": 28,
    "profesion": "Data Scientist"
}
```

**Operaciones clave**:
- Acceso por clave
- Keys(), values(), items()
- Get() con valor por defecto
- Update, pop, popitem

#### 3. Tuplas
Colecciones ordenadas e inmutables (no se pueden modificar después de creadas).

```python
coordenadas = (10.5, 20.3)
```

#### 4. Sets
Colecciones no ordenadas de elementos únicos.

```python
unicos = {1, 2, 3, 3, 4}  # Result: {1, 2, 3, 4}
```

### Control de Flujo

#### Condicionales (if, elif, else)
```python
if temperatura > 30:
    print("Hace calor")
elif temperatura > 20:
    print("Clima agradable")
else:
    print("Hace frío")
```

#### Bucles (for, while)
```python
# For loop
for i in range(5):
    print(i)

# While loop
contador = 0
while contador < 5:
    print(contador)
    contador += 1
```

### Funciones
Las funciones permiten encapsular lógica reutilizable.

```python
def calcular_promedio(numeros):
    """Calcula el promedio de una lista de números."""
    return sum(numeros) / len(numeros)
```

**Conceptos importantes**:
- Parámetros y argumentos
- Valores por defecto
- Return statements
- Docstrings
- Funciones lambda

### Manejo de Errores
El manejo de excepciones evita que tu programa falle abruptamente.

```python
try:
    resultado = 10 / 0
except ZeroDivisionError:
    print("No se puede dividir por cero")
finally:
    print("Este bloque siempre se ejecuta")
```

### Comprensiones de Listas
Una forma pythónica y eficiente de crear listas.

```python
# Forma tradicional
cuadrados = []
for i in range(10):
    cuadrados.append(i**2)

# Comprensión de lista
cuadrados = [i**2 for i in range(10)]

# Con condicional
pares = [i for i in range(10) if i % 2 == 0]
```

## 🏆 Reto: 6 Problemas Clásicos de Python

### Problema 1: Manipulación de Listas
**Objetivo**: Crear una función que reciba una lista de números y retorne una nueva lista con los números pares duplicados y los impares triplicados.

**Ejemplo**:
```
Input: [1, 2, 3, 4, 5]
Output: [3, 4, 9, 8, 15]
```

### Problema 2: Diccionarios Avanzados
**Objetivo**: Dada una lista de estudiantes con sus calificaciones (como diccionarios), calcular el promedio de cada estudiante y retornar un nuevo diccionario con el nombre del estudiante y su promedio.

**Ejemplo**:
```python
estudiantes = [
    {"nombre": "Ana", "calificaciones": [85, 90, 78]},
    {"nombre": "Carlos", "calificaciones": [92, 88, 95]}
]
# Resultado esperado: {"Ana": 84.33, "Carlos": 91.67}
```

### Problema 3: Control de Flujo - FizzBuzz Mejorado
**Objetivo**: Crea una función que imprima números del 1 al 100, pero:
- Si el número es divisible por 3, imprime "Fizz"
- Si es divisible por 5, imprime "Buzz"
- Si es divisible por ambos, imprime "FizzBuzz"
- Si es un número primo, imprime "Primo"
- En otro caso, imprime el número

### Problema 4: Funciones con Múltiples Parámetros
**Objetivo**: Crear una función que calcule estadísticas descriptivas básicas (media, mediana, desviación estándar) de una lista de números sin usar librerías externas.

### Problema 5: Manejo de Errores Robusto
**Objetivo**: Crear una función que solicite al usuario un número entero y maneje todos los posibles errores (ValueError, tipo incorrecto, etc.). La función debe seguir pidiendo hasta que reciba un valor válido.

### Problema 6: Comprensión de Listas Avanzada
**Objetivo**: Dada una lista de strings, crear una nueva lista que contenga solo las palabras que tengan más de 5 caracteres, convertidas a mayúsculas y ordenadas alfabéticamente.

**Ejemplo**:
```
Input: ["python", "es", "genial", "para", "datos"]
Output: ["GENIAL", "PYTHON"]
```

## 💡 Tips para Resolver los Problemas
1. **Lee el problema dos veces**: Asegúrate de entender EXACTAMENTE qué se pide
2. **Descompón el problema**: Divide problemas complejos en pasos pequeños
3. **Escribe pseudocódigo primero**: Planifica antes de codear
4. **Prueba con ejemplos simples**: Valida tu lógica con casos sencillos
5. **Refactoriza**: Una vez funcione, hazlo más elegante

## 📊 Criterios de Evaluación
- ✅ **Corrección**: El código resuelve el problema correctamente
- ✅ **Eficiencia**: Usa estructuras de datos apropiadas
- ✅ **Legibilidad**: Código claro con nombres descriptivos
- ✅ **Documentación**: Funciones documentadas con docstrings
- ✅ **Manejo de casos edge**: Considera inputs inusuales

## 📖 Recursos Complementarios
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python - Tutorials](https://realpython.com/)
- [Python Tutor - Visualiza tu código](https://pythontutor.com/)
- [LeetCode Python Problems](https://leetcode.com/)

## 🔍 Conceptos que Cubrirás
- Manipulación de listas y diccionarios
- Estructuras de control (if/else, loops)
- Definición y uso de funciones
- Try/except para manejo de errores
- List comprehensions
- Lógica algorítmica básica

## 🚀 Siguiente Paso
En la Sesión 03 profundizaremos en conceptos matemáticos y lógicos más avanzados de Python, incluyendo recursividad y algoritmos de ordenamiento.

---
**¡Estos 6 problemas son tu puerta de entrada al mundo Python! 🐍**
