
# Reglas de Competencia de Clasificación

## 📋 Objetivo
Desarrollar el mejor modelo de clasificación bajo lineamientos definidos, comparar predicciones en dataset de validación y coronar al ganador por mejor desempeño.

## 🎯 Reglas Generales

### 1. Datos y Datasets
- **Dataset de entrenamiento**: Proporcionado por los organizadores
- **Dataset de validación**: Reservado para evaluación final
- **Prohibido**: Usar datos externos sin autorización previa
- **Prohibido**: Compartir datasets entre competidores antes del cierre

### 2. Modelos y Metodología
- Permitidos todos los algoritmos de clasificación (ML, Deep Learning, etc.)
- Máximo 5 submisiones por competidor
- Código debe ser reproducible y documentado
- Incluir explicación de features y preprocesamiento

### 3. Métrica de Evaluación
- **Métrica principal**: Accuracy / F1-Score / AUC-ROC (definir según contexto)
- **Desempate**: Matriz de confusión y precisión por clase
- **Penalización**: -5% por código no documentado

## 📐 Formato de Submisión

```
competidor_nombre/
├── modelo.pkl o .h5
├── predictions.csv (id, prediccion, probabilidad)
├── README.md (descripción del modelo)
└── notebook.ipynb (código reproducible)
```

## ✅ Buenas Prácticas

- **Validación cruzada**: K-fold mínimo 5
- **Feature engineering**: Documentar transformaciones
- **Manejo de desbalance**: Usar técnicas apropiadas
- **Versionado**: Git con commits descriptivos
- **Reproducibilidad**: Fijar random seeds

## 🏆 Criterios de Ganador

1. Mayor desempeño en métrica principal
2. Claridad y documentación del código
3. Estabilidad del modelo (bajo variance)

## ⏰ Fechas Clave
- Inicio: [FECHA]
- Cierre submisiones: [FECHA]
- Anuncio ganador: [FECHA]
