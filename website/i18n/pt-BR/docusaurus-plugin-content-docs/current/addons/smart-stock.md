# Smart Stock

O Smart Stock usa visão computacional para detectar material colocado na mesa do
laser e criar itens de material correspondentes no seu documento. Comparando uma
imagem de referência da mesa vazia com a visualização atual da câmera, o addon
identifica os contornos do material físico e gera itens de material
corretamente posicionados com a forma e o tamanho corretos.

## Pré-requisitos

Você precisa de uma câmera configurada e calibrada conectada à sua máquina. A
câmera deve estar configurada com correção de perspectiva para que a imagem
capturada se alinhe ao sistema de coordenadas físicas da máquina. Você também
precisa de uma máquina configurada para que o addon conheça as dimensões da área
de trabalho.

## Abrindo o diálogo de detecção

Abra o diálogo em **Ferramentas - Detectar Material da Câmera**. A janela
mostra uma prévia da câmera ao vivo à esquerda e as configurações de detecção à
direita.

## Capturando uma imagem de referência

Antes de detectar material, você precisa de uma imagem de referência da mesa do
laser vazia. Sem material na mesa, clique no botão **Capturar** ao lado de
**Capturar Referência**. O addon armazena esta imagem e a compara com a
transmissão da câmera ao vivo para encontrar novos objetos.

As imagens de referência são salvas por câmera. Quando você reabrir o diálogo
com a mesma câmera, a referência capturada anteriormente será carregada
automaticamente e a detecção será executada imediatamente se já houver material
na mesa.

## Detectando material

Coloque seu material na mesa do laser e clique em **Detectar Material** na parte
inferior do painel de configurações. O addon compara o quadro atual da câmera
com a imagem de referência e traça os contornos de quaisquer novos objetos. As
formas detectadas aparecem na prévia como contornos magenta com preenchimento
verde.

A linha de status na parte inferior do painel de configurações informa quantos
itens foram encontrados. Se nenhum material for detectado, ajuste o
posicionamento ou a iluminação e tente novamente.

## Configurações de detecção

**Câmera** mostra a câmera selecionada atualmente. Clique em **Alterar** para
mudar para uma câmera configurada diferente.

**Sensibilidade** controla quanto de mudança visual é necessária para registrar
como material. Com valores mais altos, diferenças menores ou mais sutis entre a
referência e o quadro atual são detectadas. Com valores mais baixos, apenas
mudanças grandes são captadas. Se o addon não detectar material que está
presente, aumente a sensibilidade. Se detectar sombras ou reflexos como
material, diminua-a.

**Suavização** controla a suavidade dos contornos detectados. Valores mais
altos produzem contornos mais arredondados e simples, filtrando pequenas bordas
irregulares da imagem da câmera. Valores mais baixos preservam mais detalhes da
forma real do material.

## Criando itens de material

Assim que a prévia mostrar os contornos detectados correspondendo ao seu
material, clique em **Criar Itens de Material** na barra de título. O addon
adiciona um ativo de material e um item de material ao seu documento para cada
forma detectada, posicionados nas coordenadas físicas corretas na tela. O
diálogo fecha após a criação dos itens.

## Tópicos relacionados

- [Configuração da câmera](../machine/camera) - Configurar e calibrar sua câmera
- [Manuseio de estoque](../features/stock-handling) - Trabalhar com itens de material no seu documento
