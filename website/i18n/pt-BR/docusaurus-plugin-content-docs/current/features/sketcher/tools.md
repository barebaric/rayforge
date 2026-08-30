---
description: "Ferramentas do esboçador, atalhos de teclado, menu circular, modo construção, grade, snap, deslocamento, chanfro e arredondamento no Rayforge."
---

# Ferramentas do esboçador

## Atalhos de teclado

O esboçador fornece atalhos de teclado para um fluxo de trabalho eficiente:

### Atalhos de ferramentas

- `Space`: Ferramenta de seleção
- `G+P`: Ferramenta de caminho (linhas e curvas de Bézier)
- `G+A`: Ferramenta de arco
- `G+C`: Ferramenta de elipse
- `G+R`: Ferramenta de retângulo
- `G+O`: Ferramenta de retângulo arredondado
- `G+F`: Ferramenta de preenchimento de área
- `G+T`: Ferramenta de caixa de texto
- `G+Y`: Ferramenta de matriz circular
- `G+W`: Ferramenta de matriz ao longo de curva
- `G+G`: Ferramenta de grade (criar uma grade de cópias a partir da seleção)
- `G+N`: Alternar modo de construção na seleção

### Atalhos de ações

- `O+F`: Deslocar o contorno selecionado
- `C+H`: Adicionar chanfro no canto
- `C+F`: Adicionar arredondamento no canto
- `C+S`: Retificar curvas de Bézier selecionadas para linhas
- `M+V`: Espelhar seleção verticalmente
- `M+H`: Espelhar seleção horizontalmente
- `Ctrl+D`: Duplicar seleção no local

### Atalhos de restrições

- `H`: Aplicar restrição Horizontal
- `V`: Aplicar restrição Vertical
- `N`: Aplicar restrição Perpendicular
- `T`: Aplicar restrição Tangente
- `E`: Aplicar restrição Igual
- `O` ou `C`: Aplicar restrição de Alinhamento (Coincidência)
- `S`: Aplicar restrição de Simetria
- `K+D`: Aplicar restrição de Distância
- `K+R`: Aplicar restrição de Raio
- `K+O`: Aplicar restrição de Diâmetro
- `K+A`: Aplicar restrição de Ângulo
- `K+X`: Aplicar restrição de Proporção

### Atalhos gerais

- `Ctrl+Z`: Desfazer
- `Ctrl+Y` ou `Ctrl+Shift+Z`: Refazer
- `Ctrl+D`: Duplicar elementos selecionados
- `Delete`: Excluir elementos selecionados
- `Setas`: Mover entidades selecionadas (segure `Shift` para um passo maior)
- `Escape`: Cancelar operação atual ou deselecionar
- `F`: Ajustar visualização ao conteúdo

## Espelhar, Duplicar e Mover

Várias ferramentas de transformação funcionam na seleção atual:

- **Espelhar Verticalmente / Horizontalmente** (`M+V` / `M+H`): espelha a
  seleção no local através do centro de sua caixa delimitadora. Restrições
  que atravessam o limite da seleção são removidas; restrições internas
  são preservadas.
- **Duplicar** (`Ctrl+D`): copia a seleção no local. As cópias recebem
  IDs novos e restrições internas remapeadas; apenas as cópias permanecem
  selecionadas depois. Desfazer remove-as.
- **Mover**: com entidades selecionadas, as **setas** movem a seleção.
  Segure `Shift` para um passo de movimento maior.

Essas ferramentas estão disponíveis na barra de ferramentas e no menu
**Esboço**.

## Modo de construção

O modo de construção permite marcar entidades como "geometria de construção" —
elementos auxiliares usados para guiar seu design, mas que não fazem parte do
resultado final. As entidades de construção são exibidas de forma diferente
(geralmente como linhas tracejadas) e não são incluídas quando o esboço é usado
para corte ou gravação a laser.

Para alternar o modo de construção:

- Selecione uma ou mais entidades
- Pressione `N` ou `G+N`, ou use a opção Construção no menu circular

As entidades de construção são úteis para:

- Criar linhas e círculos de referência
- Definir geometria temporária para alinhamento
- Construir formas complexas a partir de uma estrutura de guias

## Controles de visibilidade

A grade se adapta ao nível de zoom e está sempre disponível como referência
de dimensionamento; como o snap funciona é descrito na [visão geral do
esboçador](index.md#grid-and-snapping).

A barra de ferramentas do esboçador inclui botões de alternância para
controlar a visibilidade:

- **Mostrar/ocultar geometria de construção**: Alterna a visibilidade das
  entidades de construção
- **Mostrar/ocultar restrições**: Alterna a visibilidade dos marcadores de
  restrições

Esses controles ajudam a reduzir a poluição visual ao trabalhar em esboços
complexos.

### Auto-restrição durante a criação

Muitas ferramentas de desenho aplicam restrições automaticamente ao criar
geometria. A ferramenta de caminho cria restrições horizontais e verticais
quando as guias de snap mostram alinhamento durante o desenho, o que ajuda
a manter seu esboço organizado desde o início, em vez de corrigir as coisas
depois.

### Movimento restrito ao eixo

Ao arrastar pontos ou geometria, segure `Shift` para restringir o movimento
ao eixo mais próximo (horizontal ou vertical). Isso é útil para manter o
alinhamento durante ajustes.

## Deslocar contorno

A ferramenta de deslocamento aumenta ou reduz o contorno selecionado em uma
distância informada, ou expande um trajeto aberto em uma ranhura. Selecione as
entidades que formam um contorno (ou use duplo clique para selecionar a
geometria conectada) e pressione `O+F`, ou use a entrada **Deslocar** no menu
circular.

![Diálogo de deslocar contorno](/screenshots/addons-sketcher-offset-dialog.webp)

O diálogo pede a distância de deslocamento e mostra uma pré-visualização ao
vivo do resultado na tela enquanto você digita:

- **Contornos fechados** crescem com distância positiva e encolhem com
  distância negativa. Um deslocamento que colapsaria o contorno é recusado.
- **Trajetos abertos** se tornam um contorno fechado em forma de ranhura da
  largura informada, com pontas arredondadas.

![Contorno Bézier](/screenshots/addons-sketcher-offset-before.webp)
![Bézier deslocado em uma ranhura](/screenshots/addons-sketcher-offset-after.webp)

Ao deslocar, o contorno selecionado é substituído pelo resultado:

- Círculos, arcos e elipses isolados mantêm seu tipo de entidade e são
  atualizados no lugar, permanecendo editáveis e restrigíveis como antes.
- Cadeias de segmentos conectados (incluindo Béziers) são substituídas por
  uma entidade polígono. O polígono é editado como um todo: arraste o ponto
  central para movê-lo e o ponto de alça para rotacioná-lo ou escalá-lo de
  forma uniforme.

Se a seleção contiver vários contornos desconectados, cada um é deslocado
independentemente em uma única etapa.

## Chanfro e arredondamento

O esboçador fornece ferramentas para modificar os cantos da sua geometria:

- **Chanfro**: Substitui um canto agudo por uma borda chanfrada. Selecione um
  ponto de junção (onde duas linhas se encontram) e aplique a ação de chanfro.
- **Arredondamento**: Substitui um canto agudo por uma borda arredondada.
  Selecione um ponto de junção (onde duas linhas se encontram) e aplique a ação
  de arredondamento.

Para usar chanfro ou arredondamento:

1. Selecione um ponto de junção onde duas linhas se encontram
2. Pressione `C+H` para chanfro ou `C+F` para arredondamento
3. Use o menu circular ou os atalhos de teclado para aplicar a modificação
