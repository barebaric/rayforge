---
description:
  "Coloque texto gravado, etiquetas e números de série em um esboço com a ferramenta de caixa de
  texto do Rayforge."
---

# Caixa de texto

A ferramenta de caixa de texto (`G+T`) coloca texto no esboço como geometria editável — texto
gravado, etiquetas e números de série. As caixas de texto são totalmente paramétricas: os glifos
vivem dentro de um quadro com restrições, de modo que são resolvidos novamente sempre que o quadro é
movido ou dimensionado.

![Uma marca nominativa e uma etiqueta de peça](/screenshots/addons-sketcher-tool-text-box.webp)

## Criar e editar texto

1. Selecione a ferramenta de caixa de texto no menu circular, no menu **Esboço**, ou com `G+T`.
2. Clique onde deseja que o texto comece: uma caixa de texto aparece no ponto do clique e a
   ferramenta passa direto para o modo de edição.
3. Digite o texto — a caixa se redimensiona para caber enquanto você digita.
4. Pressione `Enter` ou `Esc` para terminar a edição.

Para editar uma caixa de texto existente, clique dentro dela. Um clique duplo seleciona uma palavra,
um clique triplo a linha inteira, e o texto pode ser selecionado e substituído como em qualquer
editor de texto, incluindo `Ctrl+C`/`Ctrl+V`, desfazer/refazer e colar no meio da edição.

## Propriedades da Fonte

![O painel de propriedades da fonte](/screenshots/addons-sketcher-tool-text-box-font-properties.webp)

O painel **Propriedades da Fonte** na barra lateral controla a aparência da caixa de texto
selecionada no canvas:

- **Família da Fonte** — escolha entre as fontes do sistema instaladas.
- **Tamanho da Fonte** — em pontos.
- Alternadores **Negrito** e **Itálico**.

## Um quadro paramétrico

Uma caixa de texto não é uma imagem raster: seus glifos são geometria de esboço real, disposta
dentro de um quadro definido por uma origem e pontos de largura e altura. O quadro é desenhado
tracejado como geometria de construção, portanto serve como referência de layout e nunca acaba nas
trajetórias de ferramenta quando o esboço é fabricado. Como tudo o mais no esboçador, o quadro tem
restrições, então pode ser dimensionado como qualquer outra geometria — altere a restrição de
largura e o texto é resolvido novamente para preencher a caixa.

Clicar dentro de uma caixa de texto com a [ferramenta de preenchimento](fill.md) alterna o
preenchimento dos glifos do texto em vez de criar um preenchimento de região.

## Expressões de modelo

As caixas de texto aceitam **expressões de modelo**: tudo o que estiver entre chaves é avaliado
quando o esboço é resolvido, de modo que rótulos podem exibir valores ao vivo, como dimensões, datas
ou números de série exclusivos. Consulte
[Expressões de modelo em caixas de texto](expressions.md#template-expressions-in-text-boxes) para
obter detalhes e as funções integradas.
