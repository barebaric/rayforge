---
description:
  "Desenhe arcos e elipses (incluindo círculos) no esboçador do Rayforge, com teclas modificadoras e
  entrada de dimensões."
---

# Arco e elipse

O esboçador fornece duas ferramentas de formas curvas: a **ferramenta de arco** para arcos
circulares e a **ferramenta de elipse** para elipses e círculos.

![Um arco e uma elipse como criados por suas ferramentas](/screenshots/addons-sketcher-tool-arc-ellipse.webp)

## Ferramenta de arco

A ferramenta de arco (`G+A`) cria um arco em três cliques:

1. Clique no ponto **central**.
2. Clique no ponto **inicial** — sua distância do centro define o raio.
3. Mova o cursor para pré-visualizar o arco varrendo entre os dois pontos e clique na posição
   **final**.

Enquanto a pré-visualização estiver ativa, você pode digitar um número para fixar o raio exatamente;
pressione `Tab` ou `Enter` para aplicá-lo. `Tab` antes de digitar alterna o snap magnético.

## Ferramenta de elipse

A ferramenta de elipse (`G+C`) cria elipses e círculos com dois cliques: o primeiro define o centro,
o segundo define o ponto da borda. Você também pode pressionar no centro, arrastar e soltar na borda
— ambos os gestos funcionam de forma intercambiável.

- Segure `Ctrl` para restringir a forma a um círculo perfeito.
- Segure `Shift` para usar o ponto inicial como centro da elipse.

## Dois cliques ou arrastar

Como as ferramentas de [retângulo](rectangle.md), a ferramenta de elipse aceita dois gestos de forma
intercambiável: clique no primeiro ponto, mova e clique no segundo, ou pressione no primeiro ponto,
arraste e solte no segundo. Um clique rápido sem movimento apenas arma a ferramenta e aguarda o
segundo ponto, então cliques acidentais nunca deixam geometria degenerada para trás. Enquanto a
pré-visualização estiver ativa, a barra de status mostra as teclas modificadoras disponíveis, e
`Esc` cancela a pré-visualização.
