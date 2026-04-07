# Visualização 3D

A visualização 3D permite visualizar seus caminhos de ferramenta G-code e
simular a execução do trabalho antes de enviá-los para sua máquina.

![Visualização 3D](/screenshots/main-3d.png)

## Abrindo a Visualização 3D

Acesse a visualização 3D:

- **Menu**: Visualizar → Visualização 3D
- **Teclado**: <kbd>F12</kbd>

## Navegação

### Controles do Mouse

- **Rotacionar**: Clique esquerdo e arraste
- **Panorâmica**: Clique direito e arraste, ou clique do meio e arraste
- **Zoom**: Roda do mouse, ou <kbd>ctrl</kbd> + clique esquerdo e arraste

### Predefinições de Visualização

Ângulos rápidos de câmera:

- **Topo** (<kbd>1</kbd>): Vista de cima
- **Frente** (<kbd>2</kbd>): Elevação frontal
- **Direita** (<kbd>3</kbd>): Vista lateral direita
- **Esquerda** (<kbd>4</kbd>): Vista lateral esquerda
- **Traseira** (<kbd>5</kbd>): Elevação traseira
- **Isométrico** (<kbd>7</kbd>): Visualização isométrica 3D

## Exibição do Sistema de Coordenadas de Trabalho

A visualização 3D visualiza o Sistema de Coordenadas de Trabalho (WCS) ativo
de forma diferente da tela 2D:

### Grade e Eixos

- **Exibição isolada**: A grade e os eixos aparecem como se a origem WCS fosse
  a origem do mundo
- **Deslocamento aplicado**: A grade inteira é deslocada para alinhar com o
  deslocamento WCS selecionado
- **Rótulos relativos ao WCS**: Rótulos de coordenadas mostram posições relativas
  à origem WCS, não à origem da máquina

Esta exibição "em isolamento" facilita entender onde seu trabalho vai rodar em
relação ao sistema de coordenadas de trabalho selecionado, sem se confundir com
a posição absoluta da máquina.

### Mudando WCS

A visualização 3D atualiza automaticamente quando você muda o WCS ativo:
- Selecione um WCS diferente no menu suspenso da barra de ferramentas
- A grade e os eixos se deslocam para refletir a nova origem WCS
- Os rótulos atualizam para mostrar coordenadas relativas ao novo WCS

:::tip WCS na Visualização 3D
A visualização 3D mostra seus caminhos de ferramenta relativos ao WCS
selecionado. Quando você muda o WCS, verá os caminhos de ferramenta parecerem
se mover porque o ponto de referência (a grade) mudou, não porque os caminhos
de ferramenta em si se moveram.
:::


## Opções de Exibição

Botões de alternância de visibilidade estão localizados como botões de
sobreposição no canto superior direito da tela 3D:

- **Modelo**: Alternar visibilidade do modelo 3D da máquina
- **Movimentos de deslocamento**: Alternar visibilidade dos movimentos rápidos
- **Zonas de restrição**: Alternar visibilidade das zonas de restrição

### Visualização do Caminho da Ferramenta

Personalize o que você vê:

- **Mostrar Movimentos Rápidos**: Exibe movimentos de deslocamento (linhas
  tracejadas)
- **Mostrar Movimentos de Trabalho**: Exibe movimentos de corte/gravação (linhas
  sólidas)
- **Colorir por Operação**: Cores diferentes para cada operação

:::tip Cores por Laser
Ao usar máquinas com múltiplas cabeças de laser, cada laser pode ter suas
próprias cores de corte e raster configuradas nas
[Configurações do Laser](../machine/laser). Isso facilita identificar qual laser
realizará cada operação.
:::

### Modelo da Cabeça do Laser

A visualização 3D renderiza um modelo da sua cabeça de laser que se move ao
longo do caminho da ferramenta durante a simulação. Você pode atribuir um modelo
3D a cada cabeça de laser na página de [Configurações do
Laser](../machine/laser) nas Configurações da Máquina. A escala, rotação e
distância focal do modelo podem ser ajustadas para corresponder à sua
configuração física.

Durante a simulação, um feixe de laser brilhante é desenhado da cabeça para
baixo quando o laser está ativo.

## Simulação

A visualização 3D inclui um simulador integrado com controles de reprodução
sobrepostos na parte inferior da tela.

### Controles de Reprodução

- **Reproduzir/Pausar** (<kbd>espaço</kbd>): Anima execução do caminho da
  ferramenta
- **Avançar/Voltar**: Avança ou volta uma operação por vez
- **Velocidade**: Alterna entre velocidades de reprodução (1x, 2x, 4x, 8x, 16x)
- **Controle deslizante de linha do tempo**: Arraste para navegar pelo trabalho

### Visualizador G-code Sincronizado

A simulação permanece sincronizada com o visualizador G-code no painel
inferior. Percorrer a simulação destaca a linha correspondente no visualizador
G-code, e clicar em uma linha no visualizador G-code pula a simulação para
aquele ponto.

### Visibilidade de Camada

Alterne a visibilidade de camadas individuais:

- Clique no nome de uma camada para mostrar ou ocultá-la
- Foque em camadas específicas para inspeção

## Lista de Verificação

Antes de enviar para a máquina, verifique:

- [ ] O caminho da ferramenta está completo sem segmentos faltando
- [ ] Os caminhos permanecem dentro da área de trabalho da máquina
- [ ] Operações de gravação executam antes dos cortes
- [ ] Nenhum caminho de ferramenta entra em uma zona de restrição
- [ ] O trabalho começa na posição esperada
- [ ] Abas de fixação estão nos locais corretos

## Dicas de Desempenho

Para trabalhos grandes ou complexos:

1. Oculte movimentos rápidos para focar apenas nos movimentos de trabalho
2. Reduza o número de camadas visíveis
3. Feche outras aplicações para liberar recursos da GPU

## Solução de Problemas

### Pré-visualização está em branco ou preta

- Verifique se as operações estão ativadas
- Verifique se os objetos têm operações atribuídas

### Pré-visualização lenta ou travando

- Oculte movimentos rápidos
- Oculte modelos 3D
- Reduza o número de camadas visíveis

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems) - WCS
- [Janela Principal](main-window) - Visão geral da interface principal
- [Configurações](settings) - Preferências da aplicação
