---
description: "O esboçador paramétrico 2D integrado do Rayforge permite desenhar designs baseados em restrições e guiados por dimensões, que permanecem editáveis e precisos."
---

# Esboçador paramétrico 2D

O Rayforge inclui um esboçador paramétrico 2D para desenhar peças diretamente
no aplicativo. Em vez de importar arte finalizada de outro programa, você
desenha linhas, curvas e formas em uma tela infinita e as vincula com
restrições. O resultado é um design que permanece preciso, não importa
quantas vezes você mude de ideia sobre suas dimensões.

![O editor de esboços](/screenshots/addons-sketcher-editor.webp)

## O que "paramétrico" significa aqui

Um esboço é mais do que um desenho — é um pequeno modelo com regras. As
regras são **restrições**: afirmações como "estas duas linhas são
paralelas", "este canto é um ângulo reto" ou "esta aresta tem exatamente
100 mm de comprimento". Após cada alteração, um solver reorganiza a
geometria de modo que todas as regras voltam a valer.

Isso tem uma consequência prática: você pode capturar sua intenção de design
uma única vez e depois continuar editando. Eleve a restrição de distância de
100 mm para 130 mm e a peça inteira acompanha. Restrições dimensionais
também aceitam expressões — um raio de `width/2` permanece metade da
largura, seja qual for a largura.

Quando cada grau de liberdade restante está fixado por uma restrição, o
esboço está *totalmente restrito*. O editor informa sua situação por meio de
cores: a geometria presa por restrições é desenhada em verde, os pontos sem
restrições em preto, e quando um esboço fica totalmente restrito o verde
fica mais escuro. Restrições que se contradizem são marcadas em vermelho e
listadas no painel de conflitos na barra lateral, onde você pode inspecioná-
las ou removê-las.

![Um esboço com dimensões](/screenshots/addons-sketcher-constraints.webp)

Um esboço sem restrições suficientes não é um erro — muitas vezes é
exatamente o que você quer enquanto experimenta. A página
[Restrições](constraints.md) explica em detalhes cada tipo de restrição
disponível.

## O editor de esboços

Os esboços vivem no documento como qualquer outra peça. Crie um com o botão
**Novo esboço** no painel inferior (ou clique com o botão direito no canvas
e escolha a mesma entrada no menu de contexto), e o editor de esboços assume
a janela: o canvas no meio, um painel de propriedades com o nome do esboço e
seus parâmetros à esquerda, e uma barra de ferramentas no topo.

A barra de ferramentas reúne as ferramentas de sessão — desfazer e refazer,
alternadores de visibilidade de restrições e geometria de construção, cores
de preenchimento e linha, espelhamento — e os botões **Finalizar** e
**Cancelar**. **Finalizar** salva o esboço de volta no documento;
**Cancelar** descarta as alterações feitas nesta sessão. Para editar
novamente um esboço existente mais tarde, dê um duplo clique nele no espaço
de trabalho principal, ou selecione-o e escolha **Editar esboço** no menu de
contexto.

O editor é orientado ao teclado. A barra de status na parte inferior sempre
lista os atalhos aplicáveis à ferramenta e à seleção atuais, de modo que as
teclas relevantes estejam na tela exatamente quando você precisar delas.
Desfazer e refazer completos estão disponíveis para cada operação.

## O menu circular

Clicar com o botão direito em qualquer lugar do editor de esboços abre o
menu circular — um menu radial que coloca todas as ferramentas de desenho e
modificação a um clique de distância. O menu reconhece o contexto: clicar
com o botão direito em espaço vazio oferece as ferramentas de desenho,
enquanto clicar com o botão direito em uma linha selecionada oferece as
restrições e modificações que fazem sentido para uma linha. Ferramentas
relacionadas são recolhidas em grupos; passe o cursor sobre um grupo para
expandir seus subitens. Clique com o botão direito novamente para fechar o
menu ou reabri-lo em outro lugar.

![O menu circular aberto sobre uma linha selecionada](/screenshots/addons-sketcher-pie-menu.webp)

## Grade e snap

O canvas exibe uma grade adaptativa cujo espaçamento se ajusta ao nível de
zoom e que é rotulada ao longo dos eixos nas suas unidades preferidas, de
modo que também funciona como régua: você pode ler tamanhos e posições
diretamente no canvas.

Enquanto você desenha ou arrasta, o *snap magnético* atrai o cursor para
pontos de referência próximos. O canvas indica o que atrai o cursor:

- um **círculo azul** marca um ponto existente (extremidade),
- **setas verdes** marcam um ponto médio,
- um **destaque rosa** significa que o cursor está sobre uma aresta,
- **linhas tracejadas** pelo canvas são guias de alinhamento, exibidas
  quando o cursor se alinha horizontal ou verticalmente com outro ponto,
- outros indicadores cobrem casos especiais, como espaçamentos
  equidistantes (laranja), tangência (roxo) e centros (vermelho).

O snap não é apenas um auxílio visual — soltar geometria sobre um alvo de
snap cria a restrição correspondente automaticamente. Terminar uma linha em
uma extremidade existente torna as duas coincidentes; snapar em um ponto
médio cria uma restrição de simetria; guias de alinhamento se tornam
restrições horizontais ou verticais. Se preferir posicionamento livre, `Tab`
desativa o snap magnético. Segurar `Shift` enquanto arrasta restringe o
movimento ao eixo mais próximo.

![Guias de alinhamento e o indicador de snap equidistante ao desenhar](/screenshots/addons-sketcher-snap.webp)

## Geometria de construção

Qualquer entidade pode ser marcada como geometria de construção. Entidades
de construção são desenhadas tracejadas, atuam como guias de layout para o
solver como qualquer outra geometria, e são excluídas das trajetórias de
ferramenta quando o esboço é fabricado. Elas são úteis para linhas de
centro, círculos de construção e a estrutura por trás de designs simétricos.
O alternador de construção na barra de ferramentas as oculta quando
atrapalham.

## Onde ir a seguir

[Criando geometria 2D](geometry.md) apresenta as ferramentas de desenho e
seus modificadores, [Ferramentas do esboçador](tools.md) é a referência de
atalhos de teclado e modificações como deslocamento, chanfro e
arredondamento, [Matrizes](arrays.md) cobre matrizes circulares e ao longo
de curvas, e [Expressões](expressions.md) explica parâmetros, expressões e
caixas de texto paramétricas. Esboços podem ser salvos e reimportados com
todas as restrições intactas — veja [Importação e exportação](import-export.md).
