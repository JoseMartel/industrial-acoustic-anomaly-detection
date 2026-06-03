# Experimento 03: Multi-Scale Spectral Loss

## Descrição
Este experimento substitui a função de perda padrão (MSE) por uma combinação de MSE, **Spectral Convergence** e **Log-Magnitude Loss**. O objetivo é forçar o modelo a reconstruir com maior fidelidade os picos de frequência e as texturas finas do espectrograma, facilitando a detecção de anomalias sutilmente diferentes em frequência.

## Motivação
O MSE tende a suavizar as reconstruções e tratar todos os bins de frequência igualmente. Fallas mecânicas frequentemente aparecem como pequenas variações em bandas específicas que o MSE pode ignorar.

## Configurações
- **Loss**: `MultiScaleSpectralLoss` (alpha=0.5)
- **Modelo**: CAE (4 camadas)
- **Dados**: IDs 00, 02, 04 (Multi-ID)

## Como Executar
```bash
python3 src/unsupervised/exp03_spectral_loss/train.py
```
