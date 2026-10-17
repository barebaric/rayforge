---
description:
  "Grave códigos QR e códigos de barras Code 128 e EAN-13 no Rayforge. O conteúdo pode ser um
  modelo, para que cada peça tenha seu próprio código."
---

# Códigos de barras

O complemento Códigos de barras adiciona códigos QR e códigos de barras como peças. Eles são
desenhados como contornos vetoriais com preenchimento, de modo que uma operação de Gravação os grava
de forma limpa e uma operação de Contorno ainda pode segui-los.

## Adicionar um código de barras

Escolha **Objeto - Adicionar código de barras**. Um código QR aparece no centro da área de trabalho,
já selecionado. Edite-o no grupo **Código de barras** do painel de propriedades:

- **Conteúdo** — o texto a codificar. Pressione `Enter` ou o botão de aplicar para atualizar o
  código.
- **Tipo** — Código QR, Code 128 ou EAN-13.
- **Correção de erros** (QR) — quanto do código pode estar danificado e ainda ser lido: Baixa,
  Média, Quartil ou Alta. Níveis mais altos deixam o código maior.
- **Tamanho do módulo** — a largura da barra mais estreita, ou de um quadrado do QR. O tamanho do
  módulo e a altura das barras são mostrados na unidade de comprimento escolhida. A peça assume o
  tamanho do código, então é essa configuração que define o tamanho.
- **Altura das barras** (Code 128, EAN-13) — a altura das barras.
- **Mostrar texto** (Code 128, EAN-13) — imprime o conteúdo abaixo das barras. O EAN-13 usa seu
  layout habitual, com as barras de guarda descendo entre os dígitos.
- **Inverter** — grava as partes claras em vez disso, para material escuro: o código fica claro
  sobre uma borda escura. **Zona de silêncio** define essa borda em módulos e só pode ser alterada
  enquanto **Inverter** estiver ativado; por padrão, é o mínimo que o tipo exige.

Deixe uma margem livre em volta de um código não invertido: 4 módulos para códigos QR, 10 para Code
128 e 9 para EAN-13.

## Tipos

- **Código QR** aceita qualquer texto, incluindo links.
- **Code 128** aceita ASCII imprimível. Sequências de dígitos são agrupadas em pares
  automaticamente.
- **EAN-13** recebe 12 dígitos e o dígito verificador é adicionado. Com 13 dígitos, o dígito
  verificador é conferido. O campo de conteúdo fica vermelho quando o conteúdo não se encaixa no
  tipo; passe o mouse sobre ele para ver o motivo.

## Modelos

O conteúdo pode conter expressões `{...}` como uma caixa de texto, por exemplo `{2 * 21}`. Com
[Texto variável](../features/variable-text.md), `SN-{serial:05d}` dá a cada peça seu próprio número
de série, e as cópias de uma matriz continuam a contagem.
