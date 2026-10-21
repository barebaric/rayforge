---
description:
  "Encontre o melhor intervalo de linha para gravação e o ponto focal do seu laser com o teste de
  intervalo e o teste de foco."
---

# Testes de intervalo e de foco

Ao lado da [grade de teste de material](material-test-grid.md), o menu Ferramentas oferece mais dois
trabalhos de calibração. Os três funcionam da mesma forma: **Ferramentas → Criar …** adiciona o
teste com as configurações padrão em uma nova camada e o seleciona, e as configurações dele aparecem
em **Test Settings** no painel de propriedades à direita. Cada alteração ali reconstrói o teste na
hora, como um único passo que você pode desfazer, para que você veja o resultado enquanto o ajusta.
O teste é conteúdo comum: você pode movê-lo, enquadrá-lo e visualizá-lo como qualquer outro
conteúdo. Cada célula ou linha tem sua própria operação, que você também pode ajustar à mão, mas uma
alteração posterior em Test Settings reconstrói essas operações.

## Teste de intervalo

**Ferramentas → Criar teste de intervalo** grava uma fileira de quadrados preenchidos. Cada quadrado
tem sua própria operação **Gravar** com seu próprio intervalo de linha, distribuído uniformemente do
menor ao maior intervalo informado, enquanto a potência e a velocidade são as mesmas para todos. Os
rótulos abaixo de cada quadrado mostram o intervalo em milímetros e a densidade de linhas
correspondente em linhas por polegada (LPI).

Escolha o quadrado preenchido de forma uniforme, sem linhas visíveis e sem queimar demais, e use o
seu intervalo para gravações nesse material. Os rótulos são cortados antes dos quadrados com uma
operação separada de baixa potência.

## Teste de foco

**Ferramentas → Criar teste de foco** encontra a altura em que o feixe é mais nítido. Um
deslocamento positivo significa mais distância entre a cabeça e o material. A linha mais fina marca
o melhor foco.

| Método                          | Como a altura muda                                                                            |
| ------------------------------- | --------------------------------------------------------------------------------------------- |
| **Passos do eixo Z**            | A cabeça vai a cada deslocamento com movimentos Z relativos e volta à altura inicial no final |
| **Manual (pausa entre linhas)** | O trabalho pausa (`M0`) antes de cada linha; você move a cabeça à mão e pressiona Retomar     |
| **Rampa (material inclinado)**  | Uma linha longa com marcas de distância; você levanta uma extremidade de uma tira plana       |

Os passos do eixo Z só são oferecidos em máquinas com eixo Z, e tanto os passos do eixo Z quanto o
método manual precisam de um controlador G-code, porque usam operações de [Comando](command.md)
entre as linhas. Os deslocamentos são limitados a ±10 mm da altura inicial. Um novo teste de foco
usa o primeiro método que a máquina suporta, na ordem da tabela acima.

No método manual, primeiro foque o laser como de costume. Os rótulos são gravados nessa altura. Na
primeira pausa, coloque a cabeça no primeiro deslocamento; em cada pausa seguinte, mova-a um passo.
Verifique se o seu controlador para com `M0` e se o botão Retomar continua o trabalho antes de
confiar nisso, por exemplo com um teste a seco a 0 % de potência.

Na rampa, a altura sob qualquer ponto da linha resulta da subida da tira: a uma distância _d_ ao
longo de uma linha de comprimento _L_ sobre uma tira que sobe _h_, o material está _h_ × _d_ / _L_
mais alto que no início.
