---
description: "Organize trabalhos a laser em camadas com configurações diferentes. Gerencie a ordem de corte, operações e materiais com o sistema multicamadas do Rayforge."
---

# Fluxo de trabalho multicamadas

![Painel de camadas](/screenshots/bottom-panel-layers.png)

O sistema multicamadas do Rayforge permite que você organize trabalhos
em estágios de processamento separados. Cada camada é um contêiner para
peças de trabalho e tem seu próprio fluxo de trabalho — uma sequência
de etapas, cada uma com configurações de laser independentes.

:::tip Quando você não precisa de múltiplas camadas
Em muitos casos, uma única camada é suficiente. Cada etapa dentro de
uma camada tem suas próprias configurações de laser, potência,
velocidade e outros parâmetros, então você pode gravar e contornar na
mesma camada. Camadas separadas são necessárias apenas quando você
quer contornar partes diferentes de uma imagem com configurações
diferentes, ou quando precisa de configurações diferentes de WCS ou
rotativo.
:::

## Criando e gerenciando camadas

### Adicionar uma camada

Clique no botão **+** no painel de camadas. Novos documentos começam
com três camadas vazias.

### Reordenar camadas

Arraste e solte camadas no painel para alterar a ordem de execução. As
camadas são processadas da esquerda para a direita. Você pode usar
**arrastar com clique do meio** para rolar dentro da lista de camadas.

### Reordenar peças de trabalho

Peças de trabalho dentro de uma camada podem ser reorganizadas
arrastando e soltando para controlar sua ordem Z. Você pode selecionar
múltiplas peças de trabalho com **Ctrl+clique** para alternar itens
individuais, ou **Shift+clique** para selecionar um intervalo. Arrastar uma
seleção move todos os itens selecionados de uma vez.

Peças de trabalho selecionadas são destacadas na coluna da camada e a
seleção permanece sincronizada com a tela.

### Excluir uma camada

Selecione a camada e clique no botão de exclusão. Todas as peças de
trabalho da camada são removidas. Você pode desfazer a exclusão se
necessário.

## Propriedades da camada

Cada camada possui as seguintes configurações, disponíveis através do
ícone de engrenagem na coluna da camada:

- **Nome** — exibido no cabeçalho da camada
- **Cor** — usada para renderizar as operações da camada na tela
- **Visibilidade** — o ícone de olho alterna se a camada é exibida na
  tela e nas visualizações. Camadas ocultas ainda são incluídas no
  G-code gerado.
- **Sistema de coordenadas (WCS)** — atribui um sistema de coordenadas
  de trabalho a esta camada. Quando definido para um WCS específico
  (ex.: G54, G55), a máquina muda para esse sistema de coordenadas no
  início da camada. Selecione **Padrão** para usar o WCS global.
- **Modo rotativo** — ativa o modo de acessório rotativo para esta
  camada, permitindo misturar trabalho plano e cilíndrico no mesmo
  projeto. Configure o módulo rotativo e o diâmetro do objeto nas
  configurações da camada.

## Fluxos de trabalho por camada

Cada camada possui um **fluxo de trabalho** — uma sequência de etapas
exibida como um pipeline de ícones na coluna da camada. Cada etapa
define uma única operação (ex.: contorno, gravação raster) com suas
próprias configurações de laser, potência, velocidade e outros
parâmetros.

Clique em uma etapa para configurá-la. Use o botão **+** no pipeline
para adicionar mais etapas a uma camada. As etapas podem ser
reordenadas arrastando e soltando.

## Importação de arquivos vetoriais

Ao importar arquivos vetoriais (SVG, DXF, PDF), o diálogo de importação
oferece três formas de lidar com as camadas do arquivo de origem:

- **Mapear para camadas existentes** — importa cada camada de origem
  para a camada do documento correspondente por posição
- **Novas camadas** — cria uma nova camada de documento para cada
  camada de origem
- **Achatar** — importa tudo para a camada ativa

Ao usar **Mapear para camadas existentes** ou **Novas camadas**, o
diálogo mostra uma lista das camadas do arquivo de origem com
interruptores para selecionar quais importar.

## Atribuir peças de trabalho a camadas

**Arrastar e soltar:** Selecione peça(s) de trabalho na tela ou no
painel de camadas e arraste-as para a camada de destino. Seleção múltipla
com Ctrl+clique e Shift+clique é suportada, e você pode arrastar itens
entre camadas.

**Recortar e colar:** Recorte uma peça de trabalho da camada atual
(Ctrl+X), selecione a camada de destino e cole (Ctrl+V).

**Menu de contexto:** Clique com o botão direito em uma peça de trabalho
na aba de camadas para abrir um menu de contexto com opções para movê-la
para outra camada, excluí-la ou abrir suas propriedades.

## Ordem de execução

Durante um trabalho, as camadas são processadas da esquerda para a
direita. Dentro de cada camada, todas as peças de trabalho são
processadas antes de passar para a próxima camada. O fluxo de trabalho
padrão é gravar primeiro e cortar por último, para que as peças
permaneçam no lugar durante a gravação.

## Páginas relacionadas

- [Operações](./operations/contour) - Tipos de operações para fluxos
  de trabalho por camada
- [Modo de simulação](./simulation-mode) - Visualizar execução
  multicamadas
- [Macros e Hooks](../machine/hooks-macros) - Hooks em nível de camada
  para automação
- [Visualização 3D](../ui/3d-preview) - Visualizar pilha de camadas
- [Navegador de Ativos](../ui/bottom-panel) - Gerenciando ativos com menus de contexto
