---
description:
  "Grave códigos QR, Code 128 e EAN-13 no Rayforge. Códigos de barras são ativos reutilizáveis cujo
  conteúdo pode ser um modelo, para que cada peça tenha seu próprio código."
---

# Códigos de barras

O complemento Códigos de barras adiciona códigos QR, Code 128 e EAN-13 como ativos reutilizáveis. Um
código é desenhado como contornos vetoriais com preenchimento, então uma operação de Gravação o
grava limpo e uma de Contorno ainda pode segui-lo.

## Adicionar um código de barras

Escolha **Objeto - Adicionar código de barras**. Um código QR aparece no centro da área de trabalho
e já fica selecionado. Edite-o no grupo **Código de barras** do painel de propriedades:

- **Nome** — o nome do ativo no navegador de ativos.
- **Conteúdo** — o texto a codificar. Pressione `Enter` ou o botão de aplicar.
- **Tipo** — Código QR, Code 128 ou EAN-13.
- **Correção de erros** (QR) — quanto do código pode ser danificado: Baixa, Média, Quartil ou Alta.
  Níveis mais altos aumentam o código.
- **Tamanho do módulo** — a largura da barra mais estreita ou de um quadrado QR. O tamanho do módulo
  e a altura das barras seguem sua unidade de comprimento.
- **Altura das barras** (Code 128, EAN-13) — a altura das barras.
- **Mostrar texto** (Code 128, EAN-13) — imprime o conteúdo sob as barras. EAN-13 usa o layout
  habitual, com as barras de guarda entrando nos dígitos.
- **Inverter** — grava as partes claras, para material escuro. A **zona de silêncio** define a borda
  escura em módulos e só pode ser alterada com Inverter ativado.

## O navegador de ativos

Cada código também é um ativo. Abra o navegador de ativos no painel inferior:

- **Adicionar** (`+`) cria um ativo sem colocá-lo. Use **Criar nova peça**, arraste-o para a tela ou
  dê um clique duplo no cartão.
- Um clique duplo em uma peça com código a seleciona e mostra suas propriedades.
- Duplicar, copiar, recortar e excluir funcionam como para outros ativos.

Alterar o ativo atualiza todas as cópias colocadas. Use modelos quando as cópias devem diferir.

Deixe uma margem livre ao redor de um código não invertido: 4 módulos para QR, 10 para Code 128 e 9
para EAN-13.

## Tipos

- **Código QR** aceita qualquer texto, incluindo links.
- **Code 128** aceita ASCII imprimível. Sequências de dígitos são empacotadas em pares.
- **EAN-13** recebe 12 dígitos e adiciona o dígito verificador. Com 13 dígitos ele é verificado. O
  campo fica vermelho quando o conteúdo não serve; passe o mouse para ver o motivo.

## Modelos

O conteúdo pode conter expressões `{...}` como uma caixa de texto, por exemplo `{2 * 21}`. Com o
[contador de trabalhos](../features/job-counter.md), `SN-{counter:05d}` dá a cada peça seu próprio
número, e as cópias de matriz continuam contando.
