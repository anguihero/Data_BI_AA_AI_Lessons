---
Autor: anmmunozsa@outlook.es
Nota: Este es material de código abierto diseñado para compartir y aprender colectivamente.
---

# Sesión 04: El Arte de Leer Datos

## 🎯 Objetivo de la Sesión
Dominar las técnicas para importar y limpiar datos del mundo real, que rara vez vienen en formatos perfectos. Esta es una de las habilidades más valoradas (y subestimadas) en Data Science.

## 📚 Conceptos Teóricos

### La Realidad de los Datos
En tutoriales académicos, los datasets son perfectos: CSV limpios, columnas bien nombradas, sin valores faltantes. En el mundo real, te encuentras con:
- **CSV corruptos**: Delimitadores inconsistentes, comillas mal cerradas, encoding extraño
- **JSON anidados**: Múltiples niveles de profundidad, listas dentro de diccionarios
- **Excel complejos**: Múltiples hojas, celdas combinadas, encabezados en filas no estándar
- **Datos semi-estructurados**: Logs, HTML, XML

> **"El 80% del tiempo de un Data Scientist se va en limpiar datos, no en modelar."**

### Formatos de Archivo Comunes

#### CSV (Comma-Separated Values)
El formato más común pero también el más problemático.

**Problemas típicos**:
```csv
# Delimitadores inconsistentes
"Nombre","Edad","Ciudad"
Juan,28,Bogotá
"María, Pérez",32,"Cali, Valle"  ← Coma dentro de datos

# Encoding incorrecto
Nombre,Ciudad
José,Bogotá  ← Puede verse como Jos� si el encoding es incorrecto

# Líneas corruptas
Nombre,Edad,Salario
Ana,28,50000
Carlos,ERROR,60000  ← Valor inválido
Laura,30  ← Falta columna
```

**Soluciones con Pandas**:
```python
# Handle encoding
df = pd.read_csv('datos.csv', encoding='latin1')  # o 'utf-8', 'iso-8859-1'

# Handle delimiters
df = pd.read_csv('datos.csv', sep=';')  # o '\t' para TSV

# Skip bad lines
df = pd.read_csv('datos.csv', on_bad_lines='skip')

# Specify data types
df = pd.read_csv('datos.csv', dtype={'columna': str})

# Handle missing values
df = pd.read_csv('datos.csv', na_values=['', 'NA', 'N/A', 'null'])
```

#### JSON (JavaScript Object Notation)
Excelente para datos estructurados y semi-estructurados, común en APIs.

**Ejemplo simple**:
```json
{
  "nombre": "Ana",
  "edad": 28,
  "habilidades": ["Python", "SQL", "ML"]
}
```

**JSON anidado complejo**:
```json
{
  "empresa": "TechCorp",
  "empleados": [
    {
      "nombre": "Carlos",
      "departamento": "Data Science",
      "proyectos": [
        {"nombre": "Predicción Churn", "estado": "activo"},
        {"nombre": "Segmentación", "estado": "completado"}
      ]
    }
  ]
}
```

**Lectura con Pandas**:
```python
# JSON simple
df = pd.read_json('datos.json')

# JSON anidado
import json
with open('datos.json', 'r') as f:
    data = json.load(f)
df = pd.json_normalize(data, record_path=['empleados'])
```

#### Excel
El formato preferido en el mundo corporativo, pero con muchas trampas.

**Desafíos comunes**:
- Múltiples hojas con diferentes estructuras
- Encabezados en fila 3 en lugar de fila 1
- Celdas combinadas
- Formato de fechas inconsistente
- Fórmulas en lugar de valores

**Lectura con Pandas**:
```python
# Leer hoja específica
df = pd.read_excel('datos.xlsx', sheet_name='Ventas')

# Leer todas las hojas
todas_hojas = pd.read_excel('datos.xlsx', sheet_name=None)

# Encabezado en fila diferente
df = pd.read_excel('datos.xlsx', header=2)

# Saltar filas
df = pd.read_excel('datos.xlsx', skiprows=3)

# Leer rango específico
df = pd.read_excel('datos.xlsx', usecols='A:D', nrows=100)
```

### Estrategias de Limpieza

#### 1. Inspección Inicial
```python
# Primeras filas
df.head()

# Información general
df.info()

# Estadísticas descriptivas
df.describe()

# Valores únicos por columna
df.nunique()

# Verificar nulos
df.isnull().sum()

# Tipos de datos
df.dtypes
```

#### 2. Manejo de Valores Nulos
```python
# Eliminar filas con nulos
df.dropna()

# Eliminar columnas con >50% nulos
df.dropna(axis=1, thresh=len(df)*0.5)

# Rellenar con valor
df.fillna(0)

# Rellenar con media/mediana
df['columna'].fillna(df['columna'].mean())

# Forward fill
df.ffill()
```

#### 3. Conversión de Tipos
```python
# String a numérico
df['edad'] = pd.to_numeric(df['edad'], errors='coerce')

# String a datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

# Categorical
df['categoria'] = df['categoria'].astype('category')
```

#### 4. Manejo de Duplicados
```python
# Detectar duplicados
df[df.duplicated()]

# Eliminar duplicados
df.drop_duplicates()

# Duplicados por columna específica
df.drop_duplicates(subset=['id'])
```

### Encodings Comunes
- **UTF-8**: Estándar moderno, soporta todos los caracteres
- **Latin-1 (ISO-8859-1)**: Común en datos antiguos europeos/latinoamericanos
- **Windows-1252**: Encoding por defecto en Windows
- **ASCII**: Solo caracteres ingleses básicos

**Tip**: Si ves caracteres extraños (�, Ã±), es un problema de encoding.

## 🏆 Reto: 3 Desafíos de Lectura de Datos

### Desafío 1: CSV Corrupto
Tienes un archivo CSV con múltiples problemas:
- Delimitadores inconsistentes (mezcla de comas y punto y comas)
- Encoding mixto (UTF-8 y Latin-1)
- Líneas con diferente número de columnas
- Valores faltantes representados como '', 'NA', 'null', '-'

**Objetivo**: Leer el archivo, limpiar los datos y generar un DataFrame válido.

### Desafío 2: JSON Anidado
Tienes un archivo JSON de una API con estructura profundamente anidada:
```json
{
  "data": {
    "users": [
      {
        "id": 1,
        "name": "Ana",
        "transactions": [
          {"date": "2023-01-15", "amount": 150.00, "category": "food"},
          {"date": "2023-01-16", "amount": 80.00, "category": "transport"}
        ]
      }
    ]
  }
}
```

**Objetivo**: Aplanar la estructura para crear un DataFrame donde cada fila sea una transacción.

### Desafío 3: Excel Complejo
Tienes un archivo Excel con:
- 5 hojas diferentes (Ventas, Inventario, Clientes, Productos, Resumen)
- Encabezados en fila 3 en algunas hojas
- Valores faltantes en diferentes formatos
- Fechas en formato texto
- Una hoja de "Resumen" con celdas combinadas que debes ignorar

**Objetivo**: Extraer y combinar datos de las 4 hojas principales en un solo DataFrame consolidado.

## 💡 Tips para el Éxito
1. **Inspecciona antes de cargar**: Abre el archivo en un editor de texto para ver su estructura real
2. **Lee en chunks**: Para archivos grandes, usa `chunksize` en `read_csv()`
3. **Valida después de importar**: Siempre verifica tipos de datos y valores nulos
4. **Documenta los problemas**: Anota los issues para reportarlos al proveedor de datos
5. **Automatiza**: Si lees el mismo tipo de archivo repetidamente, crea una función

## 📊 Criterios de Evaluación
- ✅ Datos importados correctamente sin pérdida de información crítica
- ✅ Tipos de datos apropiados (no todo como string)
- ✅ Valores nulos identificados y manejados adecuadamente
- ✅ Estructura de datos final útil para análisis
- ✅ Código robusto que maneja errores

## 📖 Recursos Complementarios
- [Pandas IO Tools Documentation](https://pandas.pydata.org/docs/user_guide/io.html)
- [JSON Normalize Guide](https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html)
- [Excel File Handling](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)
- [Character Encoding Guide](https://docs.python.org/3/library/codecs.html)

## 🔍 Conceptos Clave
- Lectura de CSV con parámetros avanzados
- JSON parsing y normalización
- Manejo de múltiples hojas Excel
- Detección y manejo de encoding
- Validación de datos importados
- Limpieza proactiva de datos

## 🚀 Siguiente Paso
Una vez que domines la importación de datos, en la Sesión 05 aprenderás a manipularlos con maestría usando Pandas: groupby, pivot tables, y transformaciones complejas.

---
**¡Los datos sucios son tu oportunidad de brillar!** 🧹📊
