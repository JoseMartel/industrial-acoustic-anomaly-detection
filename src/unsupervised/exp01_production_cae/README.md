# Experimento 01: Produção CAE (Baseline ID 02)

## Descrição
Este experimento representa a primeira configuração estável e bem-sucedida do projeto. Foca na detecção de anomalias para o ventilador industrial utilizando uma abordagem de Autoencoder Convolucional (CAE).

## Configurações Principais
- **Modelo**: Convolutional Autoencoder (CAE) com 4 camadas.
- **Normalização**: Z-Score por Frequência Global.
- **Scoring**: Percentil 90 do MSE temporal.
- **Dataset**: MIMII Fan (Especializado na Máquina ID 02 para resultados ótimos).

## Resultados Alcançados
- **ROC-AUC**: 0.9880
- **p-AUC (0.1)**: 0.9469

## Como Executar
A partir da raiz do projeto:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/unsupervised/01_production_cae/train.py
```
