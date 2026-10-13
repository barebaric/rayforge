# Edição de imagens

O addon de edição de imagens altera como a imagem de uma peça fica sem tocar no arquivo importado.
Cada edição é armazenada junto com a peça, de modo que você pode desfazê-la, removê-la depois ou
colocar o mesmo arquivo várias vezes com edições diferentes.

A tela, a visualização 3D e cada operação usam a imagem editada. O que você vê na tela é o que o
laser grava.

## Inverter imagem

Selecione uma ou mais peças e escolha **Tools - Invert Image**. Áreas claras ficam escuras e áreas
escuras ficam claras; áreas transparentes permanecem transparentes, então o fundo continua sem ser
gravado.

![Uma foto em tons de cinza de montanhas e um lago, como importada](/images/image-edit-invert-before.webp)

![A mesma foto depois de Invert Image](/images/image-edit-invert-after.webp)

Escolha novamente para remover a inversão. Com uma seleção mista, as peças que ainda não estão
invertidas são invertidas, de modo que toda a seleção termine igual.

Use para litofanias e embossing em uma operação de gravação, onde áreas claras devem ser gravadas em
profundidade, ou para gravar arte clara em um fundo escuro.

## Ajustar imagem

Selecione uma ou mais peças e escolha **Tools - Adjust Image...** para corrigir uma foto antes de
gravá-la ou vetorizá-la:

- **Brightness** (brilho) clareia ou escurece a imagem.
- **Contrast** (contraste) afasta os tons do cinza médio ou os aproxima dele.
- **Gamma** acima de 1 clareia os tons médios e abaixo de 1 os escurece; o preto e o branco não
  mudam.
- **Sharpen** (nitidez) reforça bordas e detalhes finos (uma máscara de nitidez). **Sharpen Radius**
  define o tamanho desses detalhes em milímetros, então o resultado é o mesmo em qualquer intervalo
  de linha ou zoom.

A tela mostra as alterações enquanto você edita. **Apply** as mantém como uma única etapa de
desfazer; fechar a caixa de diálogo sem aplicar as descarta, e **Reset** volta todos os valores ao
neutro. As áreas transparentes continuam transparentes.

A caixa de diálogo abre com os valores da primeira peça selecionada, e Apply dá os mesmos valores a
todas as peças selecionadas. A nitidez é aplicada antes das correções de tom, e outras edições como
Invert Image mantêm seu lugar.

## Projetos mais antigos

Antes deste addon, Inverter era uma configuração da operação de gravação. Quando você abre um
projeto mais antigo que a usava, o Rayforge move a configuração para as peças dessa camada, para que
o resultado continue o mesmo.

## Quando o addon está desativado

Um projeto mantém suas edições de imagem quando o addon está desativado ou não está instalado. O
Rayforge informa que o projeto usa recursos indisponíveis e mostra as imagens sem essas edições até
que o addon seja ativado novamente.
