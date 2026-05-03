---
description: "Configure as definições gerais da máquina no Rayforge — defina o nome, selecione um driver e configure velocidades e aceleração."
---

# Definições gerais

A página Geral nas Definições da máquina contém o nome da máquina, a seleção
do driver e as definições de conexão, além dos parâmetros de velocidade.

![Definições gerais](/screenshots/machine-general.png)

## Nome da máquina

Dê à sua máquina um nome descritivo. Isso ajuda a identificá-la no menu
suspenso de seleção quando você tem várias máquinas configuradas.

## Driver

Selecione o driver correspondente ao controlador da sua máquina. O driver
gerencia a comunicação entre o Rayforge e o hardware.

Após selecionar um driver, as definições específicas de conexão aparecem
abaixo do seletor (ex.: porta serial, baud rate). Elas variam conforme o
driver escolhido.

:::tip
Um banner de erro no topo da página avisa você se o driver não estiver
configurado ou encontrar um problema.
:::

## Velocidades e aceleração

Essas definições controlam as velocidades máximas e a aceleração. Elas são
usadas para a estimativa de tempo de trabalho e otimização de trajetórias.

### Velocidade máxima de deslocamento

A velocidade máxima para movimentos rápidos (sem corte) quando o laser está
desligado e o cabeçote se move para uma nova posição.

- **Faixa típica**: 2000–5000 mm/min
- **Nota**: A velocidade real também é limitada pelas definições do seu
  firmware. Este campo está desativado se o dialeto de G-code selecionado
  não suportar a especificação de uma velocidade de deslocamento.

### Velocidade máxima de corte

A velocidade máxima permitida durante operações de corte ou gravação.

- **Faixa típica**: 500–2000 mm/min
- **Nota**: Operações individuais podem usar velocidades menores

### Aceleração

A taxa na qual a máquina acelera e desacelera, usada para estimativas de
tempo e cálculo da distância de overscan padrão.

- **Faixa típica**: 500–2000 mm/s²
- **Nota**: Deve corresponder ou ser inferior às definições de aceleração
  do firmware

:::tip
Comece com valores de velocidade conservadores e aumente gradualmente.
Observe sua máquina quanto a saltos de correia, travamento do motor ou
perda de precisão de posicionamento.
:::

## Exportar um perfil de máquina

Clique no ícone de compartilhamento na barra de cabeçalho do diálogo de
definições para exportar a configuração atual da máquina. Escolha uma pasta
para salvar. Um arquivo ZIP é criado contendo as definições da máquina e seu
dialeto de G-code, que pode ser compartilhado com outros usuários ou
importado em outro sistema.

## Veja também

- [Assistente de Configuração](config-wizard) - Detectar e configurar
  automaticamente um dispositivo conectado
- [Definições de hardware](hardware) - Dimensões da área de trabalho e
  configuração dos eixos
- [Definições do dispositivo](device) - Ler e escrever definições do
  firmware no controlador
