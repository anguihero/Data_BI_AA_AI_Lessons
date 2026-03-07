
# FastAPI: Guía Completa para Full Stack

## ¿Qué es FastAPI?

[FastAPI](https://fastapi.tiangolo.com/) es un framework moderno de Python para construir APIs REST rápidas y eficientes. Basado en estándares web como OpenAPI y JSON Schema. Combina la velocidad de Starlette con la validación automática de Pydantic, permitiendo desarrollar APIs robustas con código limpio y productivo.

## Conceptos Fundamentales

### 1. Rutas y Métodos HTTP

Las rutas son los endpoints de tu API. Cada ruta se define mediante un decorador que especifica el método HTTP y la ruta URL.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

Los parámetros de ruta se capturan automáticamente. FastAPI valida el tipo de dato (`int` en este caso) y convierte el valor automáticamente.

[Documentación: Primeros pasos](https://fastapi.tiangolo.com/es/getting-started/)

### 2. Decoradores

Los decoradores definen el método HTTP y la ruta que maneja la función. Cada método HTTP cumple un propósito específico en el estándar REST:

- `@app.get()` - GET: Obtiene recursos sin modificarlos
- `@app.post()` - POST: Crea nuevos recursos
- `@app.put()` - PUT: Reemplaza recursos existentes
- `@app.delete()` - DELETE: Elimina recursos
- `@app.patch()` - PATCH: Actualiza parcialmente recursos

[Documentación: Métodos HTTP](https://fastapi.tiangolo.com/es/tutorial/first-steps/)

### 3. Pydantic Models

Define y valida datos automáticamente mediante clases de Pydantic. Estos modelos actúan como esquemas que validan solicitudes y respuestas:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_active: bool = True
```

Pydantic valida automáticamente tipos de datos, valores requeridos y opcionales. Si los datos no coinciden con el modelo, FastAPI devuelve automáticamente un error 422 con detalles de validación.

[Documentación: Cuerpos de solicitud](https://fastapi.tiangolo.com/es/tutorial/body/)

### 4. Swagger UI y ReDoc

FastAPI genera documentación interactiva automáticamente a partir de tus rutas y modelos:

- `/docs` - Swagger UI: Interfaz interactiva para probar endpoints
- `/redoc` - ReDoc: Documentación legible y organizada

Ambas se actualizan automáticamente cuando modificas tu código.

[Documentación: OpenAPI](https://fastapi.tiangolo.com/es/tutorial/metadata/)

## Workflow Full Stack

1. **Backend**: Definir rutas y modelos Pydantic
2. **Validación**: Pydantic valida automáticamente tipos y valores
3. **Documentación**: Swagger genera documentación interactiva
4. **Testing**: Interactúa via Swagger UI o cliente HTTP
5. **Frontend**: Consume la API mediante requests HTTP

## Referencias Oficiales

- [FastAPI Documentación](https://fastapi.tiangolo.com/)
- [Tutorial completo](https://fastapi.tiangolo.com/es/tutorial/)

