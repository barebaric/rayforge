# Visualização 3D

A janela de visualização 3D permite visualizar seus caminhos de ferramenta G-code antes de
enviá-los para sua máquina. Este recurso poderoso ajuda a detectar erros
e verificar a configuração do seu trabalho.

![Visualização 3D](/screenshots/main-3d.png)

## Abrindo a Visualização 3D

Acesse a visualização 3D:

- **Menu**: Visualizar → Visualização 3D
- **Teclado**: <kbd>ctrl+3</kbd>
- **Após geração de G-code**: Abre automaticamente (configurável)

## Navegação

### Controles do Mouse

- **Rotacionar**: Clique esquerdo e arraste
- **Panorâmica**: Clique direito e arraste, ou clique do meio e arraste
- **Zoom**: Roda do mouse, ou <kbd>ctrl</kbd> + clique esquerdo e arraste

### Controles de Teclado

- <kbd>r</kbd>: Resetar câmera para visualização padrão
- <kbd>home</kbd>: Resetar zoom e posição
- <kbd>f</kbd>: Ajustar visualização ao caminho da ferramenta
- Teclas de seta: Rotacionar câmera

### Predefinições de Visualização

Ângulos rápidos de câmera:

- **Topo** (<kbd>1</kbd>): Vista de cima
- **Frente** (<kbd>2</kbd>): Elevação frontal
- **Direita** (<kbd>3</kbd>): Elevação lateral direita
- **Isométrico** (<kbd>4</kbd>): Visualização isométrica 3D

## Exibição do Sistema de Coordenadas de Trabalho

A visualização 3D visualiza o Sistema de Coordenadas de Trabalho (WCS) ativo
de forma diferente da tela 2D:

### Grade e Eixos

- **Exibição isolada**: A grade e os eixos aparecem como se a origem WCS fosse
  a origem do mundo
- **Deslocamento aplicado**: A grade inteira é deslocada para alinhar com o
  deslocamento WCS selecionado
- **Rótulos relativos ao WCS**: Rótulos de coordenadas mostram posições relativas à
  origem WCS, não à origem da máquina

Esta exibição "em isolamento" facilita entender onde seu trabalho vai
rodar em relação ao sistema de coordenadas de trabalho selecionado, sem se confundir
com a posição absoluta da máquina.

### Mudando WCS

A visualização 3D atualiza automaticamente quando você muda o WCS ativo:
- Selecione um WCS diferente no menu suspenso da barra de ferramentas
- A grade e os eixos se deslocam para refletir a nova origem WCS
- Os rótulos atualizam para mostrar coordenadas relativas ao novo WCS

:::tip WCS na Visualização 3D
A visualização 3D mostra seus caminhos de ferramenta relativos ao WCS selecionado. Quando você
muda o WCS, verá os caminhos de ferramenta parecerem se mover porque o ponto de referência
(a grade) mudou, não porque os caminhos de ferramenta em si se moveram.
:::


## Opções de Exibição

### Visualização do Caminho da Ferramenta

Personalize o que você vê:

- **Mostrar Movimentos Rápidos**: Exibe movimentos de deslocamento (linhas tracejadas)
- **Mostrar Movimentos de Trabalho**: Exibe movimentos de corte/gravação (linhas sólidas)
- **Colorir por Operação**: Cores diferentes para cada operação
- **Colorir por Potência**: Gradiente baseado na potência do laser
- **Colorir por Velocidade**: Gradiente baseado na taxa de avanço

### Visualização da Máquina

- **Mostrar Origem**: Exibe ponto de referência (0,0)
- **Mostrar Área de Trabalho**: Exibe limites da máquina
- **Mostrar Cabeça do Laser**: Exibe indicador de posição atual

### Configurações de Qualidade

- **Espessura da Linha**: Grossura das linhas do caminho da ferramenta
- **Anti-aliasing**: Renderização suave de linhas (pode impactar desempenho)
- **Fundo**: Cor clara, escura ou personalizada

## Controles de Reprodução

Simule a execução do trabalho:

- **Reproduzir/Pausar** (<kbd>espaço</kbd>): Anima execução do caminho da ferramenta
- **Velocidade**: Ajusta velocidade de reprodução (0.5x - 10x)
- **Avançar/Voltar**: Avança por comandos G-code individuais
- **Pular para Posição**: Clique na linha do tempo para pular para ponto específico

### Linha do Tempo

A linha do tempo mostra:

- Posição atual no trabalho
- Limites de operações (segmentos coloridos)
- Tempo estimado em qualquer ponto

## Ferramentas de Análise

### Medição de Distância

Meça distâncias em 3D:

1. Ative a ferramenta de medição
2. Clique em dois pontos no caminho da ferramenta
3. Veja a distância nas unidades atuais

### Painel de Estatísticas

Veja estatísticas do trabalho:

- **Distância Total**: Soma de todos os movimentos
- **Distância de Trabalho**: Apenas distância de corte/gravação
- **Distância Rápida**: Apenas movimentos de deslocamento
- **Tempo Estimado**: Estimativa de duração do trabalho
- **Caixa Delimitadora**: Dimensões gerais

### Visibilidade de Camada

Alterne visibilidade das operações:

- Clique no nome da operação para mostrar/ocultar
- Foque em operações específicas para inspeção
- Isole problemas sem regenerar G-code

## Lista de Verificação

Antes de enviar para a máquina, verifique:

- [ ] **Caminho da ferramenta está completo**: Sem segmentos faltando
- [ ] **Dentro da área de trabalho**: Permanece dentro dos limites da máquina
- [ ] **Ordem correta das operações**: Gravar antes de cortar
- [ ] **Sem colisões**: Cabeça não atinge grampos/fixações
- [ ] **Origem correta**: Começa na posição esperada
- [ ] **Posições das abas**: Abas de fixação nos locais corretos (se usadas)

## Dicas de Desempenho

Para trabalhos grandes ou complexos:

1. **Reduza detalhe da linha**: Menor qualidade de exibição para renderização mais rápida
2. **Oculte movimentos rápidos**: Foque apenas nos movimentos de trabalho
3. **Desative anti-aliasing**: Melhora taxa de quadros
4. **Feche outras aplicações**: Libere recursos da GPU

## Solução de Problemas

### Visualização está em branco ou preta

- Regenere o G-code (<kbd>ctrl+g</kbd>)
- Verifique se as operações estão ativadas
- Verifique se os objetos têm operações atribuídas

### Visualização lenta ou travando

- Reduza a espessura da linha
- Desative anti-aliasing
- Oculte movimentos rápidos
- Atualize drivers de vídeo

### Cores não aparecendo corretamente

- Verifique a configuração de colorir por (operação/potência/velocidade)
- Certifique-se que as operações têm cores diferentes atribuídas
- Reset as configurações de visualização para os padrões

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabalho](../general-info/work-coordinate-systems) - WCS
- [Janela Principal](main-window) - Visão geral da interface principal
- [Configurações](settings) - Preferências da aplicação
