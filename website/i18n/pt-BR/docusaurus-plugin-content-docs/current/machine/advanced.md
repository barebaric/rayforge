# Configurações Avançadas

A página Avançado nas Configurações da Máquina contém opções de configuração adicionais para casos de uso especializados.

![Configurações Avançadas](/screenshots/machine-advanced.png)

## Comportamento de Conexão

Configurações que controlam como o Rayforge interage com sua máquina durante a conexão.

### Origem ao Conectar

Quando habilitado, o Rayforge envia automaticamente um comando de homing ($H) ao conectar à máquina.

- **Habilite se**: Sua máquina tem chaves de limite confiáveis
- **Desabilite se**: Sua máquina não tem chaves de limite ou homing não é confiável

### Limpar Alarmes ao Conectar

Quando habilitado, o Rayforge limpa automaticamente qualquer estado de alarme ao conectar.

- **Habilite se**: Sua máquina frequentemente inicia em estado de alarme
- **Desabilite se**: Você quer investigar alarmes manualmente antes de limpar

## Inverter Eixos

Estas configurações invertem a direção dos movimentos dos eixos.

### Inverter Eixo X

Inverte a direção do eixo X. Quando habilitado, X positivo move para a esquerda em vez de direita.

### Inverter Eixo Y

Inverte a direção do eixo Y. Quando habilitado, Y positivo move para baixo em vez de cima.

:::info
Inverter eixos é útil quando:
- O sistema de coordenadas da sua máquina não corresponde ao comportamento esperado
- Você conectou seus motores ao contrário
- Você quer corresponder ao comportamento de outra máquina
:::

## Veja Também

- [Configurações de Hardware](hardware) - Configuração de origem dos eixos
- [Configurações do Dispositivo](device) - Configurações de direção de eixos GRBL
