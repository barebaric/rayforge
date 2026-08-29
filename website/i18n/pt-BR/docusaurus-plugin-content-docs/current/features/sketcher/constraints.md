---
description: "Saiba mais sobre restrições geométricas e dimensionais no esboçador paramétrico 2D do Rayforge."
---

# Sistema de restrições paramétricas

O sistema de restrições é o núcleo do esboçador paramétrico, permitindo definir
relações geométricas precisas:

## Restrições geométricas

- **Coincidência**: Força dois pontos a ocupar a mesma posição
- **Vertical**: Restringe uma linha a ser perfeitamente vertical
- **Horizontal**: Restringe uma linha a ser perfeitamente horizontal
- **Tangente**: Torna uma linha tangente a um círculo ou arco
- **Perpendicular**: Força duas linhas, uma linha e um arco/círculo, ou dois
  arcos/círculos a se encontrarem em 90 graus
- **Ponto em linha/forma**: Restringe um ponto a ficar sobre uma linha, arco ou
  círculo
- **Colinear**: Força duas ou mais linhas a ficarem sobre a mesma linha
  infinita
- **Simetria**: Cria relações simétricas entre elementos. Suporta dois modos:
  - **Simetria de ponto**: Selecione 3 pontos (o primeiro é o centro)
  - **Simetria de linha**: Selecione 2 pontos e 1 linha (a linha é o eixo)

## Restrições dimensionais

- **Distância**: Define a distância exata entre dois pontos ou ao longo de uma
  linha
- **Diâmetro**: Define o diâmetro de um círculo
- **Raio**: Define o raio de um círculo ou arco
- **Ângulo**: Impõe um ângulo específico entre duas linhas
- **Proporção**: Força a razão entre duas distâncias a ser igual a um valor
  especificado
- **Comprimento/Raio igual**: Força múltiplos elementos (linhas, arcos,
  elipses ou círculos) a ter o mesmo comprimento ou raio
- **Distância igual**: Torna dois segmentos de linha do mesmo comprimento
  (diferente de Comprimento/Raio igual, que também pode se aplicar a arcos e
  círculos)
