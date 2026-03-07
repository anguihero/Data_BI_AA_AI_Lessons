

# 🏗️ Guía Maestra de Gestión de Código, Arquitectura Git y Flujos MLOps

## 📋 Tabla de Contenidos
1. [Arquitectura de Git](#arquitectura-de-git)
2. [Estrategias de Branching](#estrategias-de-branching)
3. [Gestión de Repositorios Enterprise](#gestión-de-repositorios-enterprise)
4. [Ecosistema de Plataformas](#ecosistema-de-plataformas)
5. [MLOps y Versionamiento](#mops-y-versionamiento)
6. [Seguridad y DevOps](#seguridad-y-devops)

## 🔧 Arquitectura de Git

### Sistema Distribuido

Git es un **Sistema de Control de Versiones Distribuido (DVCS)** que almacena repositorios completos localmente. Cada cliente es un espejo completo del servidor central.

### Estructura de Objetos

Git utiliza **cuatro tipos de objetos** almacenados en `.git/objects/`:

| Objeto | Descripción | Hash SHA-1 |
|--------|-------------|-----------|
| **Blob** | Contenido de archivos (datos sin procesar) | Identificador único |
| **Tree** | Directorio que mapea nombres a blobs/trees | Captura estructura |
| **Commit** | Punto de referencia con metadatos | Padre + Author + Timestamp |
| **Tag** | Referencia nombrada a un commit específico | Anotada o ligera |

### Grafo Acíclico Dirigido (DAG)

```
    A (commit inicial)
    ↓
    B ← HEAD@main
    ↓
    C ← feature-branch
```

El **DAG** garantiza:
- ✅ Integridad criptográfica
- ✅ Imposibilidad de reescribir historial público
- ✅ Detección automática de conflictos

## 🌿 Estrategias de Branching

### Comparativa de Estrategias

| Estrategia | Equipos | Ciclo | Complejidad | Recomendación |
|------------|---------|-------|-------------|---------------|
| **GitFlow** | 5-20+ | Versiones planificadas | Alta | Productos grandes, releases regulares |
| **Trunk-Based** | 2-10 | Entrega continua | Baja | Startups, deploys frecuentes |
| **GitHub Flow** | 3-15 | Iterativo | Media | SaaS, equipos ágiles |

### GitFlow

```bash
# Rama main: producción
# Rama develop: integración
# Feature branches: feature/feature-name
# Release branches: release/version-number
# Hotfix branches: hotfix/issue-number

git flow feature start login-module
git flow feature finish login-module
```

**Ideal para:** Productos con versionamiento semántico y ciclos de release predefinidos.

### Trunk-Based Development

```bash
# Rama única: main
# Ramas de corta vida (<24 horas)
# Integración continua obligatoria

git checkout -b fix/critical-bug
# Cambios → Commit → PR → Test → Merge
git checkout main && git pull
git merge --ff-only fix/critical-bug
```

**Ideal para:** Equipos pequeños con fuerte cultura DevOps y testing automatizado.

### GitHub Flow

```bash
# main siempre está en producción
# Ramas por feature con PR
# Rebase o squash antes de merge

git checkout -b feature/dashboard-redesign
git push origin feature/dashboard-redesign
# Crear Pull Request en GitHub
```

**Ideal para:** Equipos ágiles con deploys frecuentes a producción.

## 🏢 Gestión de Repositorios Enterprise

### Mono-Repo vs Multi-Repo

| Aspecto | Mono-Repo | Multi-Repo |
|--------|-----------|-----------|
| **Escalabilidad** | Compleja en equipos >100 | Mejor escalado |
| **Testing** | Dependencias centralizadas | Testing independiente |
| **CI/CD** | Pipelines complejos | Pipelines simples |
| **Refactoring** | Más fácil | Requiere coordinación |
| **Permisos** | RBAC granular recomendado | RBAC natural |

### Protección de Ramas

```yaml
# .github/branch-protection.yml
branch_protection:
  main:
    require_status_checks: true
    require_code_reviews: 2
    require_signed_commits: true
    dismiss_stale_reviews: false
    enforce_admins: true
```

### Conventional Commits

```bash
# Formato: <tipo>(<scope>): <descripción>

feat(auth): agregar autenticación OAuth2
^--^  ^--^
│     └─ Scope (opcional)
└─ Tipo: feat, fix, docs, style, refactor, test, chore

# Con cuerpo detallado
feat(mlops): implementar versionamiento de modelos

Utiliza DVC para rastrear versiones de datos.
Integración con Git hooks para validación automática.

Closes #123
```

## 🌐 Ecosistema de Plataformas

### GitHub Enterprise

- ✅ Advanced Security (SAST, dependency scanning)
- ✅ GitHub Actions (CI/CD integrado)
- ✅ Codespaces (dev environments en la nube)
- ⚠️ Costo: Desde $21/usuario/mes

### GitLab

- ✅ CI/CD nativo robusto
- ✅ Auto DevOps (automático)
- ✅ Instancia auto-gestionada disponible
- ⚠️ Curva de aprendizaje mayor

### Soluciones Auto-Gestionadas

#### Gitea
```bash
# Instalación ligera
docker run -d --name gitea \
  -p 3000:3000 \
  -v gitea:/data \
  gitea/gitea:latest
```

#### Gitolite
- Acceso basado en SSH
- Control granular de permisos
- Minimal, eficiente

### Comparativa

| Plataforma | Auto-gestionado | CI/CD | SSO | Costo |
|------------|-----------------|-------|-----|-------|
| GitHub Enterprise | ❌ | ✅ Acciones | ✅ | Alto |
| GitLab Self-Hosted | ✅ | ✅ Nativo | ✅ | Medio |
| Gitea | ✅ | ❌ | Limitado | Bajo |
| Gitolite | ✅ | ❌ | ❌ | Muy bajo |

## 🤖 MLOps y Versionamiento

### DVC: Data Version Control

```bash
# Inicializar DVC
dvc init --no-scm

# Rastrear dataset
dvc add data/training_set.csv
git add data/training_set.csv.dvc .gitignore
git commit -m "feat(data): agregar dataset v1.2"

# Pipeline reproducible
dvc run -n train \
  -d data/training_set.csv \
  -d src/train.py \
  -o models/model.pkl \
  python src/train.py
```

### dvc.yaml

```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw/dataset.csv
    outs:
      - data/processed/

  train:
    cmd: python src/train.py
    deps:
      - data/processed/
      - src/train.py
    params:
      - train.epochs
      - train.lr
    outs:
      - models/model.pkl
      - metrics.json

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/
    metrics:
      - metrics.json:
          cache: false
```

### Versionamiento de Modelos

```bash
# Estructura recomendada
models/
├── v1.0/
│   ├── model.pkl
│   ├── metadata.json
│   └── requirements.txt
├── v1.1/
└── v1.2/ ← actual

# Git + DVC combinado
git tag -a model/v1.2 -m "Production model - Accuracy: 0.94"
dvc push  # Enviar a almacenamiento remoto (S3, GCS)
```

### Seguimiento de Experimentos

```python
# mlflow + Git
import mlflow
from git import Repo

repo = Repo(".")
mlflow.set_tag("git_commit", repo.head.commit.hexsha)
mlflow.set_tag("git_branch", repo.active_branch.name)
mlflow.log_metrics({"accuracy": 0.94, "loss": 0.12})
mlflow.log_artifacts("models/")
```

### Reproducibilidad

```yaml
# Reproducción garantizada
steps:
  1. git checkout <commit-hash>
  2. dvc checkout
  3. dvc repro
  # ↓ Genera mismo modelo exacto
```

## 🔐 Seguridad y DevOps

### Autenticación y Autorización

```yaml
# GitHub SAML + RBAC
saml_config:
  enabled: true
  idp_url: https://idp.company.com/saml
  
roles:
  admin: full access
  maintainer: merge + deploy
  developer: read + push
  viewer: read-only
```

### Auditoría de Logs

```bash
# Monitorear cambios críticos
git log --oneline --all --graph
git log -p -- sensitive_file.key  # Historial de archivos sensibles

# Herramienta: git-audit
gitleaks scan --source . --verbose
detect-secrets scan  # Detectar credenciales
```

### CI/CD con Despliegue Automatizado

```yaml
# GitHub Actions: deploy con rollback
name: Deploy MLOps Pipeline

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Model
        run: |
          dvc repro
          python -m pytest tests/
      
      - name: Push Artifacts
        run: |
          dvc push
          aws s3 cp models/ s3://ml-artifacts/
      
      - name: Deploy
        run: |
          ./scripts/deploy.sh production
      
      - name: Validate
        run: python scripts/validate_deployment.py
        
      - name: Rollback on Failure
        if: failure()
        run: ./scripts/rollback.sh
```

### Cumplimiento (Compliance)

| Estándar | Requerimiento | Implementación |
|----------|--------------|-----------------|
| **SOC2** | Auditoría completa | Git logs + SIEM |
| **GDPR** | Derecho al olvido | Purgar datos personales |
| **HIPAA** | Encriptación en tránsito | HTTPS + SSH obligatorio |
| **ISO 27001** | Control de acceso | MFA + RBAC |
