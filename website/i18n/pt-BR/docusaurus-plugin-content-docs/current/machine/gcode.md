# Configurações de G-code

A página de G-code nas Configurações da Máquina configura como o Rayforge gera G-code para sua máquina.

![Configurações de G-code](/screenshots/machine-gcode.png)

## Dialeto de G-code

Selecione o dialeto de G-code que corresponde ao firmware do seu controlador. Diferentes controladores usam comandos e formatos ligeiramente diferentes.

### Dialetos Disponíveis

- **Grbl (Compat)**: Dialeto GRBL padrão para cortadoras a laser de hobby. Usa M3/M5 para controle do laser.
- **Grbl (Compat, no Z axis)**: Igual ao Grbl (Compat) mas sem comandos de eixo Z. Para máquinas apenas 2D.
- **GRBL Dynamic**: Usa o modo de potência laser dinâmica do GRBL para gravação de potência variável.
- **GRBL Dynamic (no Z axis)**: Modo dinâmico sem comandos de eixo Z.
- **Smoothieware**: Para Smoothieboard e controladores similares.
- **Marlin**: Para controladores baseados em Marlin.

:::info
O dialeto afeta como a potência do laser, movimentos e outros comandos são formatados no G-code de saída.
:::

## Preâmbulo e Postscript do Dialeto

Cada dialeto inclui preâmbulo e postscript G-code personalizáveis que executam no início e no fim dos trabalhos.

### Preâmbulo

Comandos G-code executados no início de cada trabalho, antes de qualquer operação de corte. Usos comuns incluem definir unidades (G21 para mm), modo de posicionamento (G90 para absoluto) e inicializar o estado da máquina.

### Postscript

Comandos G-code executados no final de cada trabalho, após todas as operações de corte. Usos comuns incluem desligar o laser (M5), retornar à origem (G0 X0 Y0) e estacionar a cabeça.

## Veja Também

- [Básico de G-code](../general-info/gcode-basics) - Entendendo G-code
- [Dialetos de G-code](../reference/gcode-dialects) - Diferenças detalhadas de dialeto
- [Hooks & Macros](hooks-macros) - Pontos de injeção de G-code personalizado
