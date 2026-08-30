---
description: "Crie uma grade de construção de fileiras e colunas como estrutura de apoio para o desenho no esboçador do Rayforge."
---

# Grade

A ferramenta de grade (`G+G`) cria uma grade homogênea de linhas de
construção — fileiras e colunas de guias igualmente espaçadas que servem
como estrutura de apoio para o desenho, por exemplo para dispor um
padrão de furação ou alinhar elementos repetidos.

![Uma grade de construção 4x6](/screenshots/addons-sketcher-tool-grid.webp)

1. Selecione a ferramenta de grade no menu circular, no menu **Esboço**,
   ou com `G+G`.
2. Um diálogo pede o número de **fileiras** e **colunas**.
3. Confirme para criar a grade na origem do esboço com células de
   10 mm.

A grade consiste em geometria de construção: ela é desenhada tracejada,
atua como referência de snap e alinhamento como qualquer outra
geometria, e é excluída das trajetórias de ferramenta quando o esboço é
fabricado (veja [Geometria de construção](index.md#construction-geometry)).
Linhas individuais podem ser movidas ou excluídas como qualquer outra
geometria, e selecioná-las e alternar o modo de construção com `G+N`
transforma a estrutura de apoio em geometria real.
