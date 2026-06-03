# Experimento 02: Generalização Multi-ID

## Descrição
Este experimento foca em melhorar a generalização do modelo para múltiplos ventiladores (IDs 00, 02, 04) simultaneamente, tentando mitigar o efeito do "Domain Shift" entre máquinas físicas.

## Objetivos
- Superar o ROC-AUC de 0.81 atual em configurações multi-ID.
- Testar técnicas de invarianza de domínio.

## Técnicas Planejadas
- **DANN**: Domain Adversarial Neural Networks.
- **Instance Normalization**: Foco na textura do espectrograma.
- **Z-Score Adaptativo**: Normalização por amostra.

## Status
- Em desenvolvimento (Fase de Pesquisa).
