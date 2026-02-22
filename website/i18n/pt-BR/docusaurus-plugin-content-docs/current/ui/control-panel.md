# Painel de Controle

O Painel de Controle na parte inferior da janela do Rayforge fornece controle manual sobre a posição da sua cortadora a laser, status da máquina em tempo real e uma visão de log para monitorar operações.

## Visão Geral

O Painel de Controle combina várias funções em uma interface conveniente:

1. **Controles de Jog**: Movimento e posicionamento manual
2. **Status da Máquina**: Posição e estado de conexão em tempo real
3. **Console**: Terminal G-code interativo com destaque de sintaxe
4. **Sistema de Coordenadas de Trabalho (WCS)**: Seleção rápida de WCS

![Painel de Controle](/screenshots/control-panel.png)

## Acessando o Painel de Controle

O Painel de Controle está sempre visível na parte inferior da janela principal. Ele pode ser alternado via:

- **Menu**: Visualizar → Painel de Controle
- **Atalho de Teclado**: Ctrl+L

:::note Conexão Necessária
Os controles de jog estão disponíveis apenas quando conectado a uma máquina que suporta operações de jog.
:::


## Controles de Jog

Os controles de jog fornecem controle manual sobre a posição da sua cortadora a laser, permitindo mover a cabeça do laser precisamente para configuração, alinhamento e testes.

### Controles de Origem

Leve os eixos da sua máquina à origem para estabelecer uma posição de referência:

| Botão   | Função       | Descrição                       |
| -------- | -------------- | --------------------------------- |
| Origem X   | Origem eixo X   | Move eixo X para sua posição de origem |
| Origem Y   | Origem eixo Y   | Move eixo Y para sua posição de origem |
| Origem Z   | Origem eixo Z   | Move eixo Z para sua posição de origem |
| Origem Todos | Origem todos os eixos | Origina todos os eixos simultaneamente     |

:::tip Sequência de Homing
É recomendado levar todos os eixos à origem antes de iniciar qualquer trabalho para garantir posicionamento preciso.
:::


### Movimento Direcional

Os controles de jog fornecem botões para movimento direcional:

```
  ↖  ↑  ↗
  ←  •  →
  ↙  ↓  ↘
```

| Botão           | Movimento                        | Atalho de Teclado |
| ---------------- | ------------------------------- | ----------------- |
| ↑                | Y+ (Y- se máquina está Y-invertido) | Seta Para Cima          |
| ↓                | Y- (Y+ se máquina está Y-invertido) | Seta Para Baixo        |
| ←                | X- (esquerda)                       | Seta Para Esquerda          |
| →                | X+ (direita)                      | Seta Para Direita         |
| ↖ (superior-esquerdo)     | X- Y+/- (diagonal)              | -                 |
| ↗ (superior-direito)    | X+ Y+/- (diagonal)              | -                 |
| ↙ (inferior-esquerdo)  | X- Y-/+ (diagonal)              | -                 |
| ↘ (inferior-direito) | X+ Y-/+ (diagonal)              | -                 |
| Z+               | Eixo Z para cima                       | Page Up           |
| Z-               | Eixo Z para baixo                     | Page Down         |

:::note Foco Necessário
Atalhos de teclado só funcionam quando a janela principal tem foco.
:::


### Feedback Visual

Os botões de jog fornecem feedback visual:

- **Normal**: Botão está habilitado e seguro de usar
- **Aviso (laranja)**: Movimento se aproximaria ou excederia limites de software
- **Desabilitado**: Movimento não é suportado ou máquina não está conectada

### Configurações de Jog

Configure o comportamento das operações de jog:

**Velocidade de Jog:**
- **Faixa**: 1-10.000 mm/min
- **Padrão**: 1.000 mm/min
- **Propósito**: Controla quão rápido a cabeça do laser se move

:::tip Seleção de Velocidade
- Use velocidades mais baixas (100-500 mm/min) para posicionamento preciso
- Use velocidades mais altas (1.000-3.000 mm/min) para movimentos maiores
- Velocidades muito altas podem causar perda de passos em algumas máquinas
:::


**Distância de Jog:**
- **Faixa**: 0.1-1.000 mm
- **Padrão**: 10.0 mm
- **Propósito**: Controla quão longe a cabeça do laser se move por pressão de botão

:::tip Seleção de Distância
- Use distâncias pequenas (0.1-1.0 mm) para ajuste fino
- Use distâncias médias (5-20 mm) para posicionamento geral
- Use distâncias grandes (50-100 mm) para reposicionamento rápido
:::


## Display de Status da Máquina

O Painel de Controle exibe informação em tempo real sobre sua máquina:

### Posição Atual

Mostra a posição da cabeça do laser no sistema de coordenadas ativo:

- Coordenadas são relativas à origem WCS selecionada
- Atualiza em tempo real conforme você faz jog ou executa trabalhos
- Formato: Valores X, Y, Z em milímetros

### Status de Conexão

- **Conectado**: Indicador verde, máquina está respondendo
- **Desconectado**: Indicador cinza, sem conexão com máquina
- **Erro**: Indicador vermelho, problema de conexão ou comunicação

### Estado da Máquina

- **Ocioso**: Máquina está pronta para comandos
- **Executar**: Trabalho está atualmente em execução
- **Pausar**: Trabalho está pausado
- **Alarme**: Máquina está em estado de alarme
- **Origem**: Ciclo de homing está em progresso

## Sistema de Coordenadas de Trabalho (WCS)

O Painel de Controle fornece acesso rápido ao gerenciamento do Sistema de Coordenadas de Trabalho.

### Seleção de Sistema Ativo

Selecione qual sistema de coordenadas está atualmente ativo:

| Opção        | Tipo  | Descrição                                     |
| ------------- | ----- | ----------------------------------------------- |
| G53 (Máquina) | Fixo | Coordenadas absolutas da máquina, não podem ser alteradas |
| G54 (Trabalho 1)  | Usuário  | Primeiro sistema de coordenadas de trabalho                    |
| G55 (Trabalho 2)  | Usuário  | Segundo sistema de coordenadas de trabalho                   |
| G56 (Trabalho 3)  | Usuário  | Terceiro sistema de coordenadas de trabalho                    |
| G57 (Trabalho 4)  | Usuário  | Quarto sistema de coordenadas de trabalho                   |
| G58 (Trabalho 5)  | Usuário  | Quinto sistema de coordenadas de trabalho                    |
| G59 (Trabalho 6)  | Usuário  | Sexto sistema de coordenadas de trabalho                    |

### Deslocamentos Atuais

Exibe os valores de deslocamento para o WCS ativo:

- Mostrados como (X, Y, Z) em milímetros
- Representa a distância da origem da máquina até origem WCS
- Atualiza automaticamente quando deslocamentos WCS mudam

### Definindo Zero do WCS

Defina onde a origem do WCS ativo deve ser:

| Botão | Função | Descrição                                          |
| ------ | -------- | ---------------------------------------------------- |
| Zero X | Define X=0  | Torna posição X atual a origem X para WCS ativo |
| Zero Y | Define Y=0  | Torna posição Y atual a origem Y para WCS ativo |
| Zero Z | Define Z=0  | Torna posição Z atual a origem Z para WCS ativo |

:::note G53 Não Pode Ser Alterado
Botões de zero são desabilitados quando G53 (Coordenadas da Máquina) está selecionado, pois coordenadas da máquina são fixadas pelo hardware.
:::


:::tip Fluxo de Trabalho de Definição de WCS
1. Conecte à sua máquina e leve todos os eixos à origem
2. Selecione o WCS que deseja configurar (ex: G54)
3. Faça jog da cabeça do laser para a posição de origem desejada
4. Clique Zero X e Zero Y para definir esta posição como (0, 0)
5. O deslocamento é armazenado no controlador da sua máquina
:::


## Console

O Console fornece uma interface estilo terminal interativa para enviar comandos G-code e monitorar comunicação da máquina:

### Entrada de Comando

A caixa de entrada de comando permite enviar G-code bruto diretamente para a máquina:

- **Suporte Multi-linha**: Cole ou digite múltiplos comandos
- **Enter**: Envia todos comandos
- **Shift+Enter**: Insere nova linha (para edição antes de enviar)
- **Histórico**: Use setas Para Cima/Para Baixo para navegar comandos enviados anteriormente

### Display de Log

O log mostra comunicação entre Rayforge e sua máquina com destaque de sintaxe para leitura fácil:

- **Comandos do Usuário** (azul): Comandos que você digitou ou enviou durante trabalhos
- **Carimbos de Tempo** (cinza): Hora de cada mensagem
- **Erros** (vermelho): Mensagens de erro da máquina
- **Avisos** (laranja): Mensagens de aviso
- **Polls de Status** (escuro): Relatórios de posição/status em tempo real como
  `<Idle|WPos:0.000,0.000,0.000|...>`

### Modo Verboso

Clique no ícone de terminal no canto superior direito do console para alternar saída verbosa:

- **Desligado** (padrão): Oculta polls de status frequentes e respostas "ok"
- **Ligado**: Mostra toda comunicação da máquina

### Comportamento de Auto-Scroll

O console rola automaticamente para mostrar novas mensagens:

- Rolando para cima desabilita auto-scroll para que você possa revisar histórico
- Rolando para o fundo re-habilita auto-scroll
- Novas mensagens aparecem imediatamente quando auto-scroll está ativo

### Usando o Console para Solução de Problemas

O console é inestimável para diagnosticar problemas:

- Verifique se comandos estão sendo enviados corretamente
- Verifique mensagens de erro do controlador
- Monitore status de conexão e estabilidade
- Revise progresso de execução de trabalho em tempo real
- Envie comandos de diagnóstico (ex: `$$` para ver configurações GRBL)

## Compatibilidade de Máquina

O Painel de Controle adapta-se às capacidades da sua máquina:

### Suporte de Eixos

- **Eixo X/Y**: Suportado por virtualmente todas cortadoras a laser
- **Eixo Z**: Disponível apenas em máquinas com controle de eixo Z
- **Movimento Diagonal**: Requer suporte para ambos eixos X e Y

### Tipos de Máquina

| Tipo de Máquina       | Suporte de Jog | Notas                     |
| ------------------ | ----------- | ------------------------- |
| GRBL (v1.1+)       | Completo        | Suporta todos recursos de jog |
| Smoothieware       | Completo        | Suporta todos recursos de jog |
| Controladores Personalizados | Variável    | Depende da implementação |

## Recursos de Segurança

### Limites de Software

Quando limites de software estão habilitados no seu perfil de máquina:

- Botões mostram aviso laranja ao se aproximar dos limites
- Movimento é automaticamente limitado para prevenir exceder limites
- Fornece feedback visual para prevenir colisões

### Status de Conexão

- Todos controles são desabilitados quando não conectado a uma máquina
- Botões atualizam sensibilidade com base no estado da máquina
- Previne movimento acidental durante operação

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabalho (WCS)](../general-info/work-coordinate-systems) - Gerenciando WCS
- [Configuração da Máquina](../machine/general) - Configure sua máquina
- [Atalhos de Teclado](../reference/shortcuts) - Referência completa de atalhos
- [Janela Principal](main-window) - Visão geral da interface principal
- [Configurações Gerais](../machine/general) - Configuração do dispositivo
