---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 03: Python para Data Science - Lógica Matemática

## 🎯 Objetivo de la Sesión
Profundizar en conceptos avanzados de Python con enfoque matemático y algorítmico, esenciales para comprender el funcionamiento interno de librerías de Data Science y optimizar el rendimiento de tus análisis.

## 📚 Conceptos Teóricos

### ¿Por Qué Lógica Matemática en Data Science?
Los algoritmos de Machine Learning están fundamentados en conceptos matemáticos. Comprender cómo implementar lógica matemática en Python te permite:
- Entender mejor el funcionamiento interno de Scikit-learn, TensorFlow, etc.
- Crear soluciones personalizadas cuando las librerías estándar no son suficientes
- Optimizar código para grandes volúmenes de datos
- Comunicarte efectivamente con equipos de investigación

### Funciones Lambda

Las funciones lambda son funciones anónimas de una sola línea, perfectas para operaciones simples.

#### Sintaxis Básica
```python
# Función tradicional
def cuadrado(x):
    return x ** 2

# Función lambda equivalente
cuadrado = lambda x: x ** 2
```

#### Casos de Uso Comunes
```python
# Con map()
numeros = [1, 2, 3, 4, 5]
cuadrados = list(map(lambda x: x**2, numeros))

# Con filter()
pares = list(filter(lambda x: x % 2 == 0, numeros))

# Con sorted() - ordenamiento personalizado
palabras = ["python", "es", "genial"]
ordenadas = sorted(palabras, key=lambda x: len(x))

# En Pandas (muy común)
df['nueva_columna'] = df['columna'].apply(lambda x: x * 2 if x > 0 else 0)
```

#### Cuándo Usar Lambda vs Funciones Normales
- **Usa lambda**: Operaciones simples de una línea, callbacks, sorting keys
- **Usa funciones normales**: Lógica compleja, reutilización, legibilidad

### Recursividad

La recursividad es cuando una función se llama a sí misma. Es fundamental para ciertos algoritmos y estructuras de datos.

#### Anatomía de una Función Recursiva
```python
def factorial(n):
    # Caso base (condición de parada)
    if n == 0 or n == 1:
        return 1
    # Caso recursivo
    return n * factorial(n - 1)
```

Toda función recursiva necesita:
1. **Caso base**: Condición que detiene la recursión
2. **Caso recursivo**: La función llamándose a sí misma con un argumento modificado
3. **Progreso hacia el caso base**: Cada llamada debe acercarse a la condición de parada

#### Ejemplos Clásicos
```python
# Fibonacci
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Suma de lista recursiva
def suma_recursiva(lista):
    if len(lista) == 0:
        return 0
    return lista[0] + suma_recursiva(lista[1:])

# Búsqueda binaria recursiva
def busqueda_binaria(arr, objetivo, inicio, fin):
    if inicio > fin:
        return -1
    
    medio = (inicio + fin) // 2
    
    if arr[medio] == objetivo:
        return medio
    elif arr[medio] > objetivo:
        return busqueda_binaria(arr, objetivo, inicio, medio - 1)
    else:
        return busqueda_binaria(arr, objetivo, medio + 1, fin)
```

#### Recursividad vs Iteración
- **Recursividad**: Más elegante, natural para problemas divisibles
- **Iteración**: Más eficiente en memoria, evita stack overflow

### Tipos de Datos Complejos

#### Diccionarios Anidados
```python
empresa = {
    "departamentos": {
        "ventas": {
            "empleados": ["Ana", "Carlos"],
            "presupuesto": 100000
        },
        "IT": {
            "empleados": ["María", "Juan"],
            "presupuesto": 150000
        }
    }
}

# Acceso profundo
print(empresa["departamentos"]["ventas"]["empleados"][0])  # Ana
```

#### Listas de Tuplas
```python
# Coordenadas geográficas
ciudades = [
    ("Bogotá", 4.7110, -74.0721),
    ("Medellín", 6.2442, -75.5812),
    ("Cali", 3.4516, -76.5320)
]

# Ordenar por latitud
ciudades_ordenadas = sorted(ciudades, key=lambda x: x[1])
```

#### Comprensiones Complejas
```python
# Diccionario por comprensión
cuadrados_dict = {x: x**2 for x in range(10) if x % 2 == 0}

# Comprensión anidada (matriz)
matriz = [[i*j for j in range(5)] for i in range(5)]

# Set comprehension
unicos = {palabra.lower() for palabra in texto.split()}
```

### Algoritmos de Ordenamiento

Comprender algoritmos de ordenamiento te ayuda a:
- Entender complejidad temporal (Big O notation)
- Optimizar procesamiento de datos
- Responder preguntas técnicas en entrevistas

#### Bubble Sort (O(n²))
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

#### Selection Sort (O(n²))
```python
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

#### Quick Sort (O(n log n) promedio)
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    izquierda = [x for x in arr if x < pivot]
    medio = [x for x in arr if x == pivot]
    derecha = [x for x in arr if x > pivot]
    return quick_sort(izquierda) + medio + quick_sort(derecha)
```

### Complejidad Algorítmica (Big O)

| Notación | Nombre | Ejemplo |
|----------|--------|---------|
| O(1) | Constante | Acceso a elemento de lista por índice |
| O(log n) | Logarítmica | Búsqueda binaria |
| O(n) | Lineal | Recorrer una lista |
| O(n log n) | Log-lineal | Quick sort, Merge sort |
| O(n²) | Cuadrática | Bubble sort, nested loops |
| O(2ⁿ) | Exponencial | Fibonacci recursivo sin memoization |

## 🏆 Retos de la Sesión

### Reto 1: Funciones Lambda Avanzadas
Usa funciones lambda para:
1. Ordenar una lista de tuplas (nombre, edad, salario) por edad descendente
2. Filtrar palabras que empiecen con vocal
3. Transformar temperaturas de Celsius a Fahrenheit

### Reto 2: Recursividad Simple
Implementa recursivamente:
1. Cálculo de potencia (base^exponente)
2. Inversión de string ("hola" → "aloh")
3. Suma de dígitos de un número (123 → 6)

### Reto 3: Manejo de Tipos de Datos Complejos
Dado un diccionario anidado con datos de estudiantes:
1. Extraer todos los nombres en una sola lista
2. Calcular el promedio general de todas las calificaciones
3. Encontrar el estudiante con mejor promedio

### Reto 4: Algoritmos de Ordenamiento Básico
Implementa Bubble Sort desde cero y:
1. Ordena una lista de números
2. Cuenta cuántos intercambios realiza
3. Compara el tiempo con `sorted()` de Python para 1000 elementos

## 💡 Tips para el Éxito
1. **Dibuja la recursión**: Usa diagramas de árbol para visualizar llamadas recursivas
2. **Prueba con casos pequeños**: Valida tu lógica con inputs mínimos
3. **Mide el rendimiento**: Usa `time.time()` o `timeit` para comparar enfoques
4. **Piensa en complejidad**: Siempre pregúntate "¿Cuál es el Big O de esto?"

## 📊 Criterios de Evaluación
- ✅ Uso correcto de funciones lambda
- ✅ Implementación recursiva con caso base claro
- ✅ Manejo eficiente de estructuras anidadas
- ✅ Comprensión de trade-offs entre enfoques

## 📖 Recursos Complementarios
- [Python Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)
- [Visualizing Recursion](https://pythontutor.com/)
- [Big O Cheat Sheet](https://www.bigocheatsheet.com/)
- [Algorithm Visualizer](https://algorithm-visualizer.org/)

## 🚀 Siguiente Paso
En la Sesión 04 aplicarás todo este conocimiento para enfrentar uno de los mayores desafíos del Data Scientist: leer datos del mundo real (sucios, corruptos, anidados).

---
**¡Domina la lógica, domina el código!** 🧠💻
