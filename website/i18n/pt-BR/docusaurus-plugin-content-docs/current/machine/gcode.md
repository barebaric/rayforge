# Configurações de G-code

A página de G-code nas Configurações da Máquina configura como o Rayforge gera G-code para sua máquina.

![Configurações de G-code](/screenshots/machine-gcode.png)

## Dialato de G-code

Selecione o dialeto de G-code que corresponde ao firmware do seu controlador. Diferentes controladores usam comandos e formatos ligeiramente diferentes.

### Dialetos Disponíveis

- **GRBL**: Mais comum para cortadoras a laser de hobby. Usa M3/M5 para controle do laser.
- **Smoothieware**: Para Smoothieboard e controladores similares.
- **Marlin**: Para controladores baseados em Marlin.
- **GRBL-compatible**: Para controladores que seguem principalmente a sintaxe GRBL.

:::info
O dialeto afeta como a potência do laser, movimentos e outros comandos são formatados no G-code de saída.
:::

## G-code Personalizado

Você pode personalizar o G-code que o Rayforge gera em pontos específicos do trabalho.

### Início do Programa

Comandos G-code executados no início de cada trabalho, antes de qualquer operação de corte.

Usos comuns:
- Definir unidades (G21 para mm)
- Definir modo de posicionamento (G90 para absoluto)
- Inicializar o estado da máquina

### Fim do Programa

Comandos G-code executados no final de cada trabalho, após todas as operações de corte.

Usos comuns:
- Desligar laser (M5)
- Retornar à origem (G0 X0 Y0)
- Estacionar a cabeça

### Troca de Ferramenta

Comandos G-code executados ao alternar entre cabeças de laser (para máquinas multi-laser).

## Veja Também

- [Básico de G-code](../general-info/gcode-basics) - Entendendo G-code
- [Dialetos de G-code](../reference/gcode-dialects) - Diferenças detalhadas de dialeto
- [Hooks & Macros](hooks-macros) - Pontos de injeção de G-code personalizado
