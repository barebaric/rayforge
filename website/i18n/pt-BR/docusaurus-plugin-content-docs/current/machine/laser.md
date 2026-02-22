# Configurações do Laser

A página Laser nas Configurações da Máquina configura sua(s) cabeça(s) de laser e suas propriedades.

![Configurações do Laser](/screenshots/machine-laser.png)

## Cabeças de Laser

O Rayforge suporta máquinas com múltiplas cabeças de laser. Cada cabeça de laser tem sua própria configuração.

### Adicionando uma Cabeça de Laser

Clique no botão **Adicionar Laser** para criar uma nova configuração de cabeça de laser.

### Propriedades da Cabeça de Laser

Cada cabeça de laser tem as seguintes configurações:

#### Nome

Um nome descritivo para esta cabeça de laser.

Exemplos:
- "Diodo 10W"
- "Tubo CO2"
- "Laser Infravermelho"

#### Número da Ferramenta

O índice da ferramenta para esta cabeça de laser. Usado no G-code com o comando T.

- Máquinas de cabeça única: Use 0
- Máquinas multi-cabeça: Atribua números únicos (0, 1, 2, etc.)

#### Potência Máxima

O valor de potência máxima para seu laser.

- **GRBL típico**: 1000 (faixa S0-S1000)
- **Alguns controladores**: 255 (faixa S0-S255)
- **Modo porcentagem**: 100 (faixa S0-S100)

Este valor deve corresponder à configuração $30 do seu firmware.

#### Potência de Enquadramento

O nível de potência usado para operações de enquadramento (delinear sem cortar).

- Defina como 0 para desabilitar enquadramento
- Valores típicos: 5-20 (apenas visível, não marcará o material)
- Ajuste com base no seu laser e material

#### Tamanho do Ponto

O tamanho físico do seu feixe de laser focalizado em milímetros.

- Digite ambas as dimensões X e Y
- A maioria dos lasers tem um ponto circular (ex: 0.1 x 0.1)
- Afeta cálculos de qualidade de gravação

:::tip Medindo o Tamanho do Ponto
Para medir o tamanho do seu ponto:
1. Dispare um pulso curto com baixa potência em um material de teste
2. Meça a marca resultante com paquímetro
3. Use a média de múltiplas medições
:::

## Veja Também

- [Configurações do Dispositivo](device) - Configurações de modo laser GRBL
