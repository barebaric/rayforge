# Modo de Simulação

![Modo de Simulação](/screenshots/main-simulation.png)

O modo de simulação mostra como seu trabalho a laser será executado antes de
rodá-lo na máquina. Você pode percorrer o G-code e ver exatamente o que
acontecerá.

## Ativar o Modo de Simulação

- **Teclado**: Pressione <kbd>F11</kbd>
- **Menu**: Vá em **Visualizar → Simular Execução**
- **Barra de ferramentas**: Clique no botão de simulação

## Visualização

### Mapa de Calor de Velocidade

As operações são coloridas de acordo com a velocidade:

| Velocidade  | Cor      |
| ----------- | -------- |
| Mais lenta  | Azul     |
| Lenta       | Ciano    |
| Média       | Verde    |
| Rápida      | Amarelo  |
| Mais rápida | Vermelho |

As cores são relativas à faixa de velocidade do seu trabalho - azul é o mínimo,
vermelho é o máximo.

### Transparência de Potência

A opacidade das linhas mostra a potência do laser:

- **Linhas faintas** = Baixa potência (movimentos de deslocamento, gravação leve)
- **Linhas sólidas** = Alta potência (corte)

## Controles de Reprodução

Use os controles na parte inferior da tela:

- **Reproduzir/Pausar** (<kbd>Espaço</kbd>): Iniciar ou parar a reprodução
  automática
- **Controle deslizante de progresso**: Arraste para navegar pelo trabalho
- **Teclas de seta**: Percorrer as instruções uma por uma

A simulação e a visualização G-code permanecem sincronizadas — percorrer a
simulação destaca a linha G-code correspondente, e clicar em uma linha G-code
pula a simulação para aquele ponto.

A visualização 3D também tem sua própria simulação com reprodução sincronizada.
Percorrer o simulador 3D destaca a linha correspondente no visualizador G-code,
e vice-versa.

## Editar Durante a Simulação

Você pode editar as peças durante a simulação. Mova, escale ou gire objetos, e a
simulação é atualizada automaticamente.

## Tópicos Relacionados

- **[Visualização 3D](../ui/3d-preview)** - Visualização do trajeto da ferramenta 3D
- **[Grade de Teste de Material](operations/material-test-grid)** - Use a simulação para validar testes
