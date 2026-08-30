---
description: "Desenhe retângulos e retângulos arredondados no esboçador do Rayforge, com pontos centrais, teclas modificadoras e entrada de dimensões."
---

# Retângulo e retângulo arredondado

O esboçador oferece duas ferramentas de retângulo que compartilham os
mesmos gestos e teclas modificadoras: a ferramenta de **retângulo**
(`G+R`) e a ferramenta de **retângulo arredondado** (`G+O`).

![Um retângulo e um retângulo arredondado](/screenshots/addons-sketcher-tool-rectangle.webp)

## Desenhando retângulos

Desenhe um retângulo especificando dois cantos opostos, ou pressione no
primeiro canto, arraste e solte no canto oposto. As teclas modificadoras
funcionam da mesma forma para ambas as ferramentas:

- Segure `Shift` para posicionar o retângulo simetricamente ao redor do
  ponto inicial.
- Segure `Ctrl` para restringi-lo a um quadrado.

Cada retângulo cria automaticamente um **ponto central** restrito ao
centro geométrico, para que você possa dimensionar ou snapar no meio da
forma.

Enquanto a pré-visualização estiver ativa, você pode digitar o tamanho
exato: a barra de status mostra os campos `W` e `H` (além de `R` para o
raio dos cantos de retângulos arredondados). Digite um valor, pressione
`Tab` para alternar entre os campos e `Enter` para aplicar. Ambas as
ferramentas aceitam os gestos de dois cliques e de clicar-e-arrastar de
forma intercambiável; `Esc` cancela a pré-visualização.

O raio dos cantos do retângulo arredondado também pode ser alterado
depois editando suas restrições — os cantos são totalmente restritos, de
modo que o raio permanece ajustável.
