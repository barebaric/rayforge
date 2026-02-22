# Configurações Gerais

A página Geral nas Configurações da Máquina contém informações básicas da máquina e configurações de velocidade.

![Configurações Gerais](/screenshots/machine-general.png)

## Nome da Máquina

Dê à sua máquina um nome descritivo. Isso ajuda a identificar a máquina no menu suspenso seletor de máquina quando você tem múltiplas máquinas configuradas.

Exemplos:
- "K40 da Oficina"
- "Laser de Diodo da Garagem"
- "Ortur LM2 Pro"

## Velocidades e Aceleração

Estas configurações controlam as velocidades máximas e aceleração para planejamento de movimento e estimativa de tempo.

### Velocidade Máxima de Deslocamento

A velocidade máxima para movimentos rápidos (sem corte). Isso é usado quando o laser está desligado e a cabeça está se movendo para uma nova posição.

- **Faixa típica**: 2000-5000 mm/min
- **Propósito**: Planejamento de movimento e estimativa de tempo
- **Nota**: A velocidade real também é limitada pelas configurações do seu firmware

### Velocidade Máxima de Corte

A velocidade máxima permitida durante operações de corte ou gravação.

- **Faixa típica**: 500-2000 mm/min
- **Propósito**: Limita velocidades de operação para segurança
- **Nota**: Operações individuais podem usar velocidades menores

### Aceleração

A taxa na qual a máquina acelera e desacelera.

- **Faixa típica**: 500-2000 mm/s²
- **Propósito**: Estimativa de tempo e planejamento de movimento
- **Nota**: Deve corresponder ou ser menor que as configurações de aceleração do firmware

:::tip
Comece com valores de velocidade conservadores e aumente gradualmente. Observe sua máquina para pular correias, travamento de motores ou perda de precisão de posicionamento.
:::

## Veja Também

- [Configurações de Hardware](hardware) - Dimensões da máquina e configuração de eixos
- [Configurações do Dispositivo](device) - Conexão e configurações GRBL
