---
description:
  "Como as restrições funcionam no esboçador do Rayforge: como adicioná-las, editá-las,
  selecioná-las e excluí-las, e como resolver conflitos."
---

# Restrições

As restrições são as regras que mantêm um esboço unido. Cada uma é uma pequena afirmação sobre a
geometria — "estes dois pontos são um e o mesmo", "esta linha tem exatamente 80 mm de comprimento" —
e, após cada edição, o solver reorganiza o esboço para que todas as afirmações valham ao mesmo
tempo. Geometria sem restrições pode se deslocar livremente; cada restrição que você adiciona fixa
um grau de liberdade.

Existem duas famílias. As **restrições geométricas** capturam relações que não envolvem medida:
coincidência, horizontalidade, tangência, simetria. As **restrições dimensionais** atribuem um
número à geometria: uma distância, um raio, um ângulo. Valores dimensionais aceitam expressões (veja
[abaixo](#editing-dimensional-values)), e é aí que acontece o "paramétrico" do desenho paramétrico.

O solver informa seu estado por meio de cores. A geometria presa por restrições é desenhada em
verde, os pontos sem restrições em preto, e um esboço totalmente restrito torna o verde mais escuro.
Marcadores de restrições válidas são verdes, marcadores baseados em expressões são laranjas, e os
marcadores de restrições que o solver não consegue satisfazer ficam vermelhos (veja
[conflitos](#when-constraints-conflict)).

## Adicionando uma restrição

Selecione a geometria à qual a restrição deve se aplicar e, em seguida, pressione o atalho de
teclado ou escolha a restrição no menu circular — as restrições geométricas ficam no grupo
**Restringir**, e as dimensionais no grupo **Dimensão**. Cada restrição exige uma seleção
específica:

| Restrição                     | Selecionar                      | Atalho     |
| ----------------------------- | ------------------------------- | ---------- |
| Horizontal / Vertical         | 2 pontos, ou quaisquer linhas   | `H` / `V`  |
| Coincidência / Ponto em forma | 2 pontos, ou ponto + uma forma  | `O` ou `C` |
| Perpendicular                 | 2 formas                        | `N`        |
| Tangente                      | 1 linha + 1 arco ou círculo     | `T`        |
| Simetria                      | 3 pontos, ou 2 pontos + 1 linha | `S`        |
| Comprimento igual             | 2 ou mais formas                | `E`        |
| Distância                     | 2 pontos, ou 1 linha            | `K+D`      |
| Diâmetro                      | 1 círculo                       | `K+O`      |
| Raio                          | 1 arco ou círculo               | `K+R`      |
| Ângulo                        | 2 linhas                        | `K+A`      |
| Proporção                     | 2 linhas                        | `K+X`      |

A ordem da seleção nunca importa, com uma exceção: com três pontos selecionados, a Simetria usa o
**último** ponto como centro do espelhamento. Um atalho só é acionado quando a seleção atual se
adequa à restrição — todo o resto também é filtrado do menu circular.

Restrições também aparecem por conta própria enquanto você desenha: snapar em uma extremidade cria
uma restrição de coincidência, e guias de alinhamento se tornam restrições horizontais ou verticais
(veja [a visão geral do esboçador](index.md#grid-and-snapping)).

## Restrições geométricas

Uma restrição de **coincidência** mescla dois pontos distintos em uma única posição. Selecione os
dois pontos e ambos são puxados juntos; o marcador é um anel ao redor do ponto unido. Desenhar uma
linha que termina exatamente em uma extremidade existente cria essa restrição automaticamente.

![Duas linhas unidas por uma restrição de coincidência](/screenshots/addons-sketcher-constraint-coincident.webp)

**Horizontal** e **Vertical** giram a linha selecionada, ou o par de pontos selecionados, sobre um
eixo. Os marcadores são pequenas barras — horizontal e vertical, respectivamente — desenhadas ao
lado da geometria.

![Uma restrição horizontal](/screenshots/addons-sketcher-constraint-horizontal.webp)

![Uma restrição vertical](/screenshots/addons-sketcher-constraint-vertical.webp)

**Perpendicular** força duas formas a se encontrarem em ângulo reto. Funciona para duas linhas, uma
linha e um arco ou círculo, ou dois arcos e círculos. O marcador é um arco de ângulo reto na
interseção.

![Duas linhas se encontrando em ângulo reto](/screenshots/addons-sketcher-constraint-perpendicular.webp)

**Tangente** suaviza a transição onde uma linha encontra um arco ou círculo: a linha é girada para
tocar a curva sem cruzá-la. Seu marcador é um pequeno "T" no ponto de contato.

![Uma linha tangente a um círculo](/screenshots/addons-sketcher-constraint-tangent.webp)

**Ponto em forma** fixa um ponto sobre uma linha, arco ou círculo — sem mesclá-lo com nenhum ponto
específico, como faz a coincidência. Selecione um ponto e uma forma; o marcador é um anel ao redor
do ponto restringido. Quando a forma é uma curva (Bézier), o ponto é restringido a deslizar ao longo
dela.

![Uma extremidade de linha apoiada em outra linha](/screenshots/addons-sketcher-constraint-point-on-line.webp)

**Simetria** espelha dois pontos em relação a um centro ou a um eixo, e tem os dois modos já
mencionados: selecione três pontos e o último se torna o centro em torno do qual os dois primeiros
se espelham, ou selecione dois pontos e uma linha para espelhar através dessa linha. O marcador é um
par de pontas de seta opostas no ponto médio entre os pontos espelhados.

![Dois pontos espelhados em relação a uma linha](/screenshots/addons-sketcher-constraint-symmetry.webp)

Uma sétima restrição geométrica, **colinear**, força pontos a ficarem sobre uma mesma linha
infinita. Ela não tem marcador no canvas e não pode ser aplicada manualmente — as ferramentas de
chanfro e arredondamento a criam para manter o canto modificado alinhado.

## Restrições dimensionais

A restrição de **distância** fixa a distância entre dois pontos ou o comprimento de uma linha. Seu
rótulo mostra o valor atual no meio do intervalo medido; quando os dois pontos ainda não estão
unidos por uma linha, uma linha de chamada tracejada deixa claro o que está sendo medido.

![Uma restrição de distância de 80 mm](/screenshots/addons-sketcher-constraint-distance.webp)

Círculos e arcos têm suas próprias dimensões. **Diâmetro** rotula a largura total de um círculo com
o prefixo `Ø`, **raio** rotula a distância do centro de um arco ou círculo com o prefixo `R`, e
ambos posicionam o rótulo logo fora da forma, com uma linha de chamada curta.

![Uma restrição de diâmetro](/screenshots/addons-sketcher-constraint-diameter.webp)

![Uma restrição de raio](/screenshots/addons-sketcher-constraint-radius.webp)

A restrição de **ângulo** define o ângulo entre duas linhas selecionadas. Ela desenha um arco entre
as duas direções na interseção, rotulado com o valor em graus.

![Uma restrição de ângulo de 45 graus](/screenshots/addons-sketcher-constraint-angle.webp)

**Proporção** vincula os comprimentos de duas linhas: o comprimento da primeira dividido pelo
comprimento da segunda deve ser igual ao valor informado. Seu marcador, um par de colchetes de canto
opostos, fica na junção onde as linhas se encontram.

![Uma restrição de proporção entre duas linhas](/screenshots/addons-sketcher-constraint-aspect-ratio.webp)

Por fim, **comprimento igual** aplicado a duas ou mais linhas, arcos, círculos ou elipses faz com
que todos compartilhem um mesmo comprimento ou raio, marcando cada forma com um sinal `=`. O solver
também usa internamente uma variante de distância igual dessa restrição — por exemplo, para manter
um círculo redondo ou os dois lados de um chanfro simétricos — que carrega o mesmo marcador `=`, mas
não pode ser aplicada manualmente.

![Duas linhas de comprimento igual](/screenshots/addons-sketcher-constraint-equal-length.webp)

## Editando valores dimensionais {#editing-dimensional-values}

Dê um duplo clique no rótulo de uma restrição dimensional para editá-la. O diálogo aceita um número
simples ou uma expressão: parâmetros do esboço e variáveis de entrada podem ser referenciados pelo
nome, e funções matemáticas estão disponíveis — um raio de `width/2` acompanha o parâmetro de
largura para onde quer que ele vá. Quando uma restrição é guiada por uma expressão, seu marcador
fica laranja, como lembrete de que o número é calculado, e não digitado. A sintaxe completa, junto
com os parâmetros do esboço que ela pode referenciar, é descrita em [Expressões](expressions.md).

Dar um duplo clique em uma linha, arco ou círculo ainda sem dimensão oferece criar a dimensão
correspondente diretamente (distância, raio ou diâmetro).

## Selecionando e excluindo

Os marcadores de restrição participam da seleção como todo o resto: passar o cursor mostra um
destaque amarelo e uma dica com o nome da restrição, e um clique a seleciona, desenhando-a em azul.
Pressionar `Delete` remove a restrição selecionada e liberta a geometria que ela segurava. Excluir a
geometria leva suas restrições junto. Para restrições dimensionais, o diálogo de edição descrito
acima não tem botão de exclusão — remover uma dimensão é uma exclusão normal do marcador
selecionado.

## Quando as restrições conflitam {#when-constraints-conflict}

Restrições que se contradizem — um triângulo cujos lados não podem ser todos verdadeiros ao mesmo
tempo, por exemplo — não conseguem quebrar o esboço: o solver faz o seu melhor e sinaliza o que não
pôde satisfazer. As restrições em conflito ficam vermelhas, tanto seus marcadores quanto a geometria
que elas seguram, de modo que a área danificada fica visível de relance.

![Restrições de distância em conflito, sinalizadas na barra lateral](/screenshots/addons-sketcher-conflicts.webp)

A barra lateral lista cada conflito em **Restrições em Conflito**, cada linha nomeando a restrição e
os pontos que ela toca. As linhas são interativas: passar o cursor sobre uma destaca a restrição no
canvas, clicar em uma a seleciona, e o botão de exclusão à direita a remove. Normalmente, a forma
mais rápida de sair de um conflito é excluir ou alterar o valor da restrição que expressa a intenção
desatualizada — a lista existe justamente porque o solver não consegue adivinhar qual das regras em
contradição é a errada.

## Onde ir a seguir

Cada ferramenta de desenho está documentada em sua própria página — veja [Caminho](path.md),
[Arco e Elipse](arc-ellipse.md) e [Retângulo](rectangle.md) para saber como desenhar as formas às
quais essas restrições se aplicam.
