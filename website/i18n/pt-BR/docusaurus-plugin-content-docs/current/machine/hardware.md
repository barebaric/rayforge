# Configurações de Hardware

A página de Hardware nas Configurações da Máquina configura as dimensões físicas, sistema de coordenadas e limites de movimento da sua máquina.

![Configurações de Hardware](/screenshots/machine-settings-hardware.png)

## Eixos

Configure a extensão dos eixos e o sistema de coordenadas para sua máquina.

### Extensão X/Y

A faixa completa de deslocamento de cada eixo nas unidades da máquina.

- Meça a área de corte real, não o exterior da máquina
- Considere quaisquer obstruções ou limites
- Exemplo: 400 para um laser K40 típico

### Origem das Coordenadas

Selecione onde a origem das coordenadas da sua máquina (0,0) está localizada. Isso determina como as coordenadas são interpretadas.

- **Inferior Esquerdo**: Mais comum para dispositivos GRBL. X aumenta para a direita, Y aumenta para cima.
- **Superior Esquerdo**: Comum para algumas máquinas estilo CNC. X aumenta para a direita, Y aumenta para baixo.
- **Superior Direito**: X aumenta para a esquerda, Y aumenta para baixo.
- **Inferior Direito**: X aumenta para a esquerda, Y aumenta para cima.

#### Encontrando Sua Origem

1. Leve sua máquina à origem usando o botão Home
2. Observe para onde a cabeça do laser se move
3. Essa posição é sua origem (0,0)

:::info
A configuração de origem das coordenadas afeta como o G-code é gerado. Certifique-se de que corresponde à configuração de homing do seu firmware.
:::

### Direção do Eixo

Inverta a direção de qualquer eixo se necessário:

- **Inverter Direção do Eixo X**: Torna os valores de coordenada X negativos
- **Inverter Direção do Eixo Y**: Torna os valores de coordenada Y negativos
- **Inverter Direção do Eixo Z**: Habilite se um comando Z positivo (ex: G0 Z10) move a cabeça para baixo

### Eixo Z

Nem todo laser tem um eixo Z — muitas máquinas de diodo simples de 2 eixos
não têm. Use estas configurações para informar ao Rayforge se sua máquina
pode se mover em Z:

- **Tem eixo Z**: Habilite se esta máquina tiver um eixo Z (motor de foco,
  mesa móvel ou pórtico que pode se mover verticalmente). Quando
  desabilitado, o eixo Z é tratado como ausente: nenhum movimento Z é
  gerado, e o canvas 3D distribui o conteúdo no plano de gravação em vez
  de mostrar uma cabeça se movendo ao longo de Z.
- **Inverter Direção do Eixo Z**: Só é mostrado quando **Tem eixo Z** está
  habilitado. Habilite se um comando Z positivo (ex: G0 Z10) move a cabeça
  para baixo.

### Orientação do painel

A orientação do painel gira a área de trabalho plana como apresentada na
tela e reatribui a saída de volta à máquina. Isso é útil quando uma cama
fisicamente orientada em retrato é mais fácil de editar em paisagem.

- **Nativo**: A área de trabalho é mostrada exatamente como a máquina está
  orientada
- **Girar para a esquerda**: A área de trabalho é girada 90 graus no sentido
  anti-horário
- **Girar para a direita**: A área de trabalho é girada 90 graus no sentido
  horário

Camadas rotativas exigem a orientação **Nativo**.

## Área de Trabalho

As margens definem o espaço inutilizável ao redor das bordas da extensão dos seus eixos. Isso é útil quando sua máquina tem áreas onde o laser não pode alcançar (ex: devido à montagem da cabeça do laser, correntes de cabos ou outras obstruções).

- **Margem Esquerda/Superior/Direita/Inferior**: O espaço inutilizável de cada borda nas unidades da máquina

Quando as margens são definidas, a área de trabalho (espaço utilizável) é calculada como a extensão dos eixos menos as margens.

## Limites de Software

Limites de segurança configuráveis para mover a cabeça da máquina. Quando habilitados, os controles de jog impedirão movimento fora desses limites.

- **Habilitar Limites de Software Personalizados**: Alternar para usar limites personalizados em vez dos limites da superfície de trabalho
- **X/Y Mín**: Coordenada mínima para cada eixo
- **X/Y Máx**: Coordenada máxima para cada eixo

Os limites de software são automaticamente restringidos para ficar dentro da extensão dos eixos (0 ao valor de extensão).

:::tip
Use limites de software para proteger áreas da sua superfície de trabalho que nunca devem ser alcançadas durante o jog, como áreas com fixadores ou equipamentos sensíveis.
:::

## Veja Também

- [Configurações Gerais](general) - Nome da máquina e configurações de velocidade
- [Configurações do Dispositivo](device) - Homing GRBL e configurações de eixos
