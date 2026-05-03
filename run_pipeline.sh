#!/bin/bash

# Orquestador del Proyecto de Detección de Anomalías (Estructura Organizada)

echo "-------------------------------------------------------"
echo "Iniciando Pipeline de Detección de Anomalías (CAE)"
echo "-------------------------------------------------------"

# 1. Instalación de dependencias
echo "[1/3] Verificando dependencias..."
pip install -r requirements.txt --quiet

# 2. Entrenamiento
echo "[2/3] Iniciando entrenamiento (Normalización por Frecuencia)..."
# Ejecutamos con PYTHONPATH=src para que los scripts encuentren sus módulos
PYTHONPATH=src python3 src/train.py

# 3. Evaluación
if [ -f "models/unsupervised/best_model.pth" ]; then
    echo "[3/3] Iniciando evaluación final..."
    PYTHONPATH=src python3 src/eval.py
    echo "-------------------------------------------------------"
    echo "PROCESO COMPLETADO CON ÉXITO"
    echo "Resultados en: results/figures/"
    echo "Modelo en: models/unsupervised/"
    echo "-------------------------------------------------------"
else
    echo "Error: No se encontró el modelo entrenado."
    exit 1
fi
