
# Fundamentos Docker, Contenedores y Cluster

## 1. Introducción a Docker y Contenedores

### ¿Qué es Docker?
Docker es una plataforma de virtualización ligera que permite empaquetar aplicaciones y sus dependencias en unidades aisladas llamadas contenedores. A diferencia de las máquinas virtuales tradicionales, los contenedores comparten el kernel del sistema operativo host, lo que los hace más eficientes en términos de recursos.

### Ventajas de Docker
- **Portabilidad**: Funciona consistentemente en cualquier entorno (desarrollo, testing, producción)
- **Eficiencia**: Menor consumo de recursos respecto a VMs tradicionales
- **Velocidad**: Inicio rápido de contenedores
- **Aislamiento**: Cada contenedor tiene su propio sistema de ficheros y dependencias

### Componentes clave
- **Docker Engine**: Motor que ejecuta los contenedores
- **Docker Image**: Plantilla inmutable para crear contenedores
- **Docker Container**: Instancia en ejecución de una imagen
- **Docker Registry**: Repositorio de imágenes (Docker Hub, etc.)

## 2. Contenedores: Conceptos Profundos

### Ciclo de vida de un contenedor
1. **Creación**: Se crea a partir de una imagen
2. **Ejecución**: El contenedor inicia y ejecuta su proceso principal
3. **Pausado**: Se detiene temporalmente
4. **Detenido**: Finaliza su ejecución
5. **Eliminado**: Se libera el espacio en disco

### Aislamiento y namespaces
Los contenedores utilizan Linux namespaces para aislar recursos:
- **PID namespace**: Procesos del contenedor
- **Network namespace**: Interfaz de red propia
- **Mount namespace**: Sistema de ficheros aislado
- **User namespace**: Usuarios y permisos
- **UTS namespace**: Hostname y dominio

### Capas de una imagen Docker
Las imágenes están compuestas por capas apiladas (layer-based architecture):
```
Capa N: Cambios de la aplicación
Capa N-1: Dependencias
Capa N-2: Runtime (Python, Node, etc.)
Capa Base: Sistema operativo base (Alpine, Debian, etc.)
```

## 3. Dockerfile y Construcción de Imágenes

### Ejemplo básico con FastAPI
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Optimizaciones en Dockerfile
- Usar imágenes base ligeras (`python:3.11-slim` vs `python:3.11`)
- Multi-stage builds para reducir tamaño final
- Ordenar comandos por frecuencia de cambio
- Usar `.dockerignore` para excluir archivos innecesarios

## 4. Microservicios con Docker y FastAPI

### Arquitectura de microservicios
Descomponer una aplicación monolítica en servicios independientes:
- **Servicio de Usuarios**: Gestiona autenticación y perfiles
- **Servicio de Datos**: API de datos
- **Servicio de Notificaciones**: Envío de mensajes
- **Gateway API**: Enrutador central (Kong, API Gateway)

### Ventajas en el contexto de FastAPI
- **Escalabilidad independiente**: Cada servicio se escala por separado
- **Despliegue autónomo**: Cambios en un servicio sin afectar otros
- **Equipos independientes**: Diferentes equipos pueden trabajar en paralelo
- **Tolerancia a fallos**: Un servicio caído no derriba la aplicación completa

### Estructura ejemplo con FastAPI

```
proyecto/
├── service-users/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── service-products/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
└── docker-compose.yml
```

### Comunicación entre microservicios
- **HTTP/REST**: FastAPI expone endpoints REST
- **Message Queue**: RabbitMQ, Kafka para comunicación asíncrona
- **gRPC**: Comunicación de alta performance

## 5. Docker Compose para Orquestación Local

### Archivo docker-compose.yml
```yaml
version: '3.8'

services:
    users-service:
        build: ./service-users
        ports:
            - "8001:8000"
        environment:
            - DATABASE_URL=postgresql://user:pass@postgres:5432/users_db
        depends_on:
            - postgres
        networks:
            - app-network

    products-service:
        build: ./service-products
        ports:
            - "8002:8000"
        environment:
            - DATABASE_URL=postgresql://user:pass@postgres:5432/products_db
        depends_on:
            - postgres
        networks:
            - app-network

    postgres:
        image: postgres:15
        environment:
            - POSTGRES_PASSWORD=password
        volumes:
            - postgres-data:/var/lib/postgresql/data
        networks:
            - app-network

networks:
    app-network:
        driver: bridge

volumes:
    postgres-data:
```

### Ventajas de Docker Compose
- Define múltiples servicios en un archivo YAML
- Gestiona networking automático entre contenedores
- Orquesta volúmenes persistentes
- Ideal para desarrollo local y testing

## 6. Clusters Kubernetes: Orquestación en Producción

### ¿Por qué Kubernetes?
- **Auto-scaling**: Escala automática basada en métricas
- **Auto-healing**: Reinicia contenedores que fall
- **Actualizaciones sin downtime**: Rolling updates
- **Gestión de recursos**: Distribución inteligente en nodos
- **Service Discovery**: DNS automático entre servicios

### Componentes principales

#### Control Plane (Master)
- **API Server**: Interfaz de Kubernetes
- **etcd**: Base de datos clave-valor distribuida
- **Scheduler**: Asigna pods a nodos
- **Controller Manager**: Mantiene el estado deseado
- **Cloud Controller Manager**: Integración con proveedores

#### Nodos (Workers)
- **kubelet**: Agente que ejecuta pods
- **Container Runtime**: Docker, containerd, CRI-O
- **kube-proxy**: Networking de servicios

### Conceptos de Kubernetes

#### Pod
Unidad mínima en Kubernetes, contenedor(s) con almacenamiento compartido:
```yaml
apiVersion: v1
kind: Pod
metadata:
    name: fastapi-pod
spec:
    containers:
    - name: fastapi-app
        image: myregistry/fastapi-users:1.0
        ports:
        - containerPort: 8000
```

#### Deployment
Gestiona replicas de pods con actualizaciones controladas:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: users-service
spec:
    replicas: 3
    selector:
        matchLabels:
            app: users-service
    template:
        metadata:
            labels:
                app: users-service
        spec:
            containers:
            - name: users
                image: myregistry/fastapi-users:1.0
                resources:
                    requests:
                        cpu: 100m
                        memory: 128Mi
                    limits:
                        cpu: 500m
                        memory: 512Mi
                livenessProbe:
                    httpGet:
                        path: /health
                        port: 8000
                    initialDelaySeconds: 10
                    periodSeconds: 10
```

#### Service
Expone pods con un endpoint estable:
```yaml
apiVersion: v1
kind: Service
metadata:
    name: users-service
spec:
    type: ClusterIP
    selector:
        app: users-service
    ports:
    - port: 80
        targetPort: 8000
```

#### ConfigMap y Secret
Gestión de configuración y credenciales:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
    name: fastapi-config
data:
    LOG_LEVEL: "INFO"
    API_VERSION: "1.0"

---
apiVersion: v1
kind: Secret
metadata:
    name: db-credentials
type: Opaque
stringData:
    DATABASE_URL: "postgresql://user:pass@postgres:5432/db"
```

## 7. Aplicación Completa: Microservicios FastAPI en Kubernetes

### Estructura de proyecto
```
proyecto/
├── service-users/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt
│   ├── Dockerfile
│   └── k8s/
│       ├── deployment.yaml
│       └── service.yaml
├── service-products/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── k8s/
│       ├── deployment.yaml
│       └── service.yaml
└── k8s/
        ├── namespace.yaml
        ├── configmap.yaml
        └── ingress.yaml
```

### FastAPI con health checks
```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
        return {"status": "healthy"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
        return {"user_id": user_id, "name": "John"}

@app.post("/users")
async def create_user(name: str, email: str):
        return {"id": 1, "name": name, "email": email}
```

### Ingress para enrutamiento
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
    name: api-gateway
spec:
    rules:
    - host: api.example.com
        http:
            paths:
            - path: /users
                pathType: Prefix
                backend:
                    service:
                        name: users-service
                        port:
                            number: 80
            - path: /products
                pathType: Prefix
                backend:
                    service:
                        name: products-service
                        port:
                            number: 80
```

## 8. Mejores Prácticas

- Usar **Alpine Linux** para imágenes menores
- Implementar **health checks** en FastAPI
- Definir **requests y limits** de recursos
- Usar **readiness y liveness probes**
- Segregar **secrets y configuración**
- Implementar **logging centralizado**
- Usar **private registries** para producción
- Aplicar **RBAC** en Kubernetes
