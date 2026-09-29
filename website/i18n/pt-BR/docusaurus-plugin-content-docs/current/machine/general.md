---
description:
  "Configure as definições gerais da máquina no Rayforge — defina o nome, selecione um driver e
  configure velocidades e aceleração."
---

# Definições gerais

A página Geral nas Definições da máquina contém o nome da máquina, a seleção do driver e as
definições de conexão, além dos parâmetros de velocidade.

![Definições gerais](/screenshots/machine-settings-general.webp)

## Nome da máquina

Dê à sua máquina um nome descritivo. Isso ajuda a identificá-la no menu suspenso de seleção quando
você tem várias máquinas configuradas.

## Driver

Selecione o driver correspondente ao controlador da sua máquina. O driver gerencia a comunicação
entre o Rayforge e o hardware.

Dispositivos GRBL têm três opções de driver serial:

- **GRBL (Serial)** — Driver com contagem de buffer, detecção de deadlock e recuperação de parada.
  Recomendado para a maioria dos dispositivos GRBL
- **GRBL (Serial Simple)** — Driver de protocolo ping-pong. Envia uma linha, aguarda "ok", envia a
  próxima. Sem gerenciamento de buffer ou detecção de deadlock. Útil quando o driver padrão causa
  falsos alarmes
- **GRBL (Rust)** — Driver experimental cuja pilha completa do protocolo serial GRBL (controle de
  fluxo, streaming de trabalhos, detecção de paradas, recuperação de deadlock, configurações e
  probing) roda em Rust. Pode ser selecionado como alternativa direta ao GRBL (Serial)

Controladores baseados em Ruida são suportados pelo driver **Ruida RPA**, que conecta diretamente
via USB ou UDP, ou via TUI RPC por meio do Ruida Protocol Analyzer.

### Vínculo da Porta Serial

Em vez de um caminho de dispositivo (ex.: `/dev/ttyUSB0` ou `COM3`), o campo da porta serial também
aceita um identificador USB `VID:PID` como `0403:6001`. Quando uma máquina é vinculada por VID:PID,
a reconexão automática a segue para a nova porta depois que o sistema operacional reenumera os
dispositivos USB — por exemplo, após uma reinicialização ou ao desconectar e reconectar. Você pode
encontrar o VID:PID de um dispositivo na saída do `lsusb` (Linux) ou no Gerenciador de Dispositivos
→ IDs de Hardware (Windows).

Após selecionar um driver, as definições específicas de conexão aparecem abaixo do seletor (ex.:
porta serial, baud rate). Elas variam conforme o driver escolhido.

<!-- prettier-ignore-start -->
:::tip
Um banner de erro no topo da página avisa você se o driver não estiver configurado ou
encontrar um problema.
:::
<!-- prettier-ignore-end -->

## Velocidades e aceleração

Essas definições controlam as velocidades máximas e a aceleração. Elas são usadas para a estimativa
de tempo de trabalho e otimização de trajetórias.

### Velocidade máxima de deslocamento

A velocidade máxima para movimentos rápidos (sem corte) quando o laser está desligado e o cabeçote
se move para uma nova posição.

- **Faixa típica**: 2000–5000 mm/min
- **Nota**: A velocidade real também é limitada pelas definições do seu firmware. Este campo está
  desativado se o dialeto de G-code selecionado não suportar a especificação de uma velocidade de
  deslocamento.

### Velocidade máxima de corte

A velocidade máxima permitida durante operações de corte ou gravação.

- **Faixa típica**: 500–2000 mm/min
- **Nota**: Operações individuais podem usar velocidades menores

### Aceleração

A taxa na qual a máquina acelera e desacelera, usada para estimativas de tempo e cálculo da
distância de overscan padrão.

- **Faixa típica**: 500–2000 mm/s²
- **Nota**: Deve corresponder ou ser inferior às definições de aceleração do firmware

<!-- prettier-ignore-start -->
:::tip
Comece com valores de velocidade conservadores e aumente gradualmente. Observe sua máquina
quanto a saltos de correia, travamento do motor ou perda de precisão de posicionamento.
:::
<!-- prettier-ignore-end -->

## Exportar um perfil de máquina

Clique no ícone de compartilhamento na barra de cabeçalho do diálogo de definições para exportar a
configuração atual da máquina. Escolha uma pasta para salvar. Um arquivo ZIP é criado contendo as
definições da máquina e seu dialeto de G-code, que pode ser compartilhado com outros usuários ou
importado em outro sistema.

## Veja também

- [Configuração Inicial](../getting-started/first-time-setup.md) - Criar uma máquina passo a passo
  com o assistente de configuração
- [Definições de hardware](hardware) - Dimensões da área de trabalho e configuração dos eixos
- [Definições do dispositivo](device) - Ler e escrever definições do firmware no controlador
