# Print & Cut

Alinhe cortes a laser em material pré-impresso registrando pontos de referência
no seu design e correspondendo-os às suas posições físicas no material. Isso é
útil para cortar adesivos, etiquetas ou qualquer item que precise se alinhar
com uma impressão existente.

## Pré-requisitos

O addon requer uma máquina configurada. Sua máquina precisa estar conectada
para a etapa de deslocamento. Você também precisa ter um workpiece ou grupo
selecionado na tela.

## Abrindo o assistente

Selecione um único workpiece ou grupo na tela e abra
**Ferramentas - Alinhar à posição física**. O assistente abre como um diálogo
de três etapas com uma prévia do seu workpiece à esquerda e os controles à
direita.

## Etapa 1: Selecionar pontos do design

![Selecionar pontos do design](/screenshots/addon-print-and-cut-pick.png)

O painel esquerdo mostra uma renderização do workpiece selecionado. Clique
diretamente na imagem renderizada para colocar o primeiro ponto de alinhamento,
marcado em verde, depois clique novamente para colocar o segundo ponto, marcado
em azul. Uma linha tracejada conecta os dois pontos.

Escolha dois pontos que correspondam a recursos identificáveis no seu material
físico — por exemplo, marcas de registro impressas ou cantos distintos. Os
pontos precisam estar suficientemente afastados para um alinhamento preciso.
Você pode arrastar qualquer ponto após posicioná-lo para ajustar a posição.

Use a roda do mouse para ampliar a prévia e clique do botão do meio com
arraste para navegar. O botão **Redefinir** na parte inferior limpa ambos os
pontos e permite recomeçar.

Assim que ambos os pontos estiverem posicionados, clique em **Avançar** para
continuar.

## Etapa 2: Registrar posições físicas

![Registrar posições físicas](/screenshots/addon-print-and-cut-jog.png)

Nesta página, você desloca o laser para as posições físicas que correspondem
aos dois pontos do design que você selecionou. O painel direito mostra um
teclado direcional para deslocamento e um controle de distância que define
quanto o laser se move por passo.

Desloque o laser para a posição física correspondente ao seu primeiro ponto do
design e clique em **Registrar** ao lado de Posição 1. As coordenadas
registradas aparecem na linha. Repita o processo para Posição 2. Você pode
voltar a uma posição registrada a qualquer momento clicando no botão
**Ir para** ao lado dela.

O interruptor **Foco do laser** liga o laser na potência de foco configurada,
criando um ponto visível no material para ajudá-lo a localizar posições com
precisão. Este interruptor requer um valor de potência de foco maior que zero
nas configurações do laser.

A posição atual do laser é exibida na parte inferior do painel. Quando ambas
as posições estiverem registradas, clique em **Avançar** para continuar.

## Etapa 3: Revisar e aplicar a transformação

![Revisar e aplicar a transformação](/screenshots/addon-print-and-cut-apply.png)

A última página mostra o alinhamento calculado como um deslocamento de
translação e um ângulo de rotação. Esses valores são derivados da diferença
entre seus pontos do design e as posições físicas registradas.

Por padrão, a escala está travada em 1.0. Se o seu material físico difere em
tamanho do design — por exemplo, devido ao escalonamento da impressora — ative
o interruptor **Permitir escalonamento**. O fator de escala é então calculado
a partir da proporção entre a distância física e a distância do design entre
seus dois pontos. Uma nota aparece quando a escala está travada, mas as
distâncias não correspondem, indicando que o segundo ponto pode não se alinhar
exatamente.

Clique em **Aplicar** para mover e girar o workpiece na tela para corresponder
às posições físicas. A transformação é aplicada como uma ação reversível.

## Tópicos relacionados

- [Posicionamento de workpieces](../features/workpiece-positioning) - Posicionar e transformar workpieces manualmente
- [Configurações do laser](../machine/laser) - Configurar a potência de foco do laser
