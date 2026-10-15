---
description:
  "Desenhe polígonos regulares e estrelas no esboçador do Rayforge. Eles são feitos de pontos e
  linhas comuns, mantidos regulares por restrições."
---

# Polígono e estrela

A ferramenta de **polígono** (`G+X`) desenha um polígono regular, como um hexágono para um
porta-copos, ou uma estrela.

## Desenhar um polígono

Clique onde o centro deve ficar, mova o ponteiro e clique novamente, ou pressione no centro, arraste
e solte. O ponteiro define o primeiro vértice, portanto define tanto o tamanho quanto a rotação. Um
novo polígono tem 6 vértices.

- Segure `Ctrl` para desenhar uma estrela: um vértice interno é adicionado entre cada dois vértices
  externos, na metade do raio externo.

Enquanto a pré-visualização estiver ativa, você pode digitar valores exatos: a barra de status
mostra os campos `R`, `N` e `Star`. `R` é o raio do centro até os vértices, `N` o número de vértices
(de 3 a 100) e `Star` o raio interno como fração do externo (entre 0 e 1). Digitar um valor em
`Star` transforma o polígono em estrela. Digite um valor, pressione `Tab` para ir ao próximo campo e
`Enter` para criar a forma; `Esc` cancela a pré-visualização. A ferramenta lembra o número de
vértices para o próximo polígono.

## O que você obtém

Um polígono não é um objeto especial: ele consiste em pontos e linhas comuns, de modo que
arredondamento, chanfro, deslocamento e operações booleanas funcionam nele como em qualquer outra
forma.

Restrições o mantêm regular. Todas as arestas têm o mesmo comprimento e os vértices ficam sobre um
círculo de construção em torno do ponto central; uma estrela tem um segundo círculo de construção
para seus vértices internos e mantém a proporção entre os dois raios. Se você digitou um raio, uma
restrição de raio o fixa. Depois você pode arrastar o centro para mover o polígono, ou arrastar um
vértice para redimensioná-lo e girá-lo, e ele continua regular. Para alterar o tamanho depois, edite
a restrição de raio ou adicione uma ao círculo de construção.
