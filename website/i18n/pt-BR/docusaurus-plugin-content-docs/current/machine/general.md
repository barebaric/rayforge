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

Dispositivos GRBL podem se conectar via USB/serial ou pela rede:

- **GRBL (USB/Serial)** — Driver com contagem de buffer, detecção de deadlock e recuperação de
  parada. Sua pilha do protocolo GRBL roda em Rust. Recomendado para a maioria dos dispositivos GRBL
- **GRBL (USB/Serial — compatibility mode)** — Driver de protocolo ping-pong. Envia uma linha,
  aguarda "ok", envia a próxima. Sem gerenciamento de buffer ou detecção de deadlock. Necessário
  apenas quando o driver padrão causa falsos alarmes ou erros de comunicação
- **GRBL (Bluetooth)** — Conexão Bluetooth Classic SPP/RFCOMM para controladores pareados no sistema
  operacional
- **grblHAL (Telnet)** — Conexão TCP bruta (telnet) para controladores grblHAL em rede e para a
  ponte telnet do ESP3D
- **FluidNC / ESP3D (Web API)** — Driver HTTP/WebSocket para controladores FluidNC e ESP3D v3, com
  upload de arquivos e suporte a configurações do dispositivo

Máquinas xTool por Wi-Fi usam drivers experimentais próprios; veja [xTool por Wi-Fi](xtool) para
D1/D1 Pro e M2.

Controladores baseados em Ruida são suportados pelo driver **Ruida RPA**, que conecta diretamente
via USB ou UDP, ou via TUI RPC por meio do Ruida Protocol Analyzer. O campo USB dele oferece um menu
suspenso de dispositivos conectados com correspondência VID:PID, e ele respeita a definição de Modo
de Potência de cada etapa — Dinâmico ativa o escalonamento de potência do controlador e Constante o
desativa. O comportamento da potência é ajustado com as opções do driver **Power Scaling**, **VECTOR
power floor** e **IMAGE power bias**: o escalonamento de potência eleva a potência mínima emitida
conforme a velocidade de corte da camada diminui, o piso de potência vector compensa a queima
excessiva nas extremidades das linhas, e o viés de imagem compensa tubos CO2 que não disparam com
potências muito baixas.

### Epilog

Lasers Epilog com porta Ethernet (Zing 16, Zing 24/6030) usam o driver **Epilog**. Os trabalhos são
codificados como trabalhos de impressão Epilog e enviados à fila de impressão da máquina; inicie-os
na lista de trabalhos do laser. O driver pede o endereço IP, a porta de impressão (515 por padrão),
a resolução de raster (100–1000 DPI) e a frequência vetorial (10–5000 Hz).

O protocolo de impressão não oferece controle remoto, então o Rayforge não pode referenciar, mover,
consultar o status nem movimentar o cabeçote nessas máquinas, e o foco e o ar assistido são
ajustados no laser. **Enviar** envia o trabalho e **Enquadrar** envia um trabalho de contorno com o
feixe desligado que você inicia da mesma forma. A velocidade é convertida para a escala percentual
da máquina: o Rayforge usa 6000 mm/min como referência de 100%, então 600 mm/min são 10%. Gravações
com pontilhado e limiar são enviadas como linhas de raster de 1 bit nativas; gravações com modulação
de potência (escala de cinza) usam o modo de potência por ponto da máquina. Trabalhos rotativos não
são suportados.

### Vínculo da Porta Serial

Em vez de um caminho de dispositivo (ex.: `/dev/ttyUSB0` ou `COM3`), o campo da porta serial também
aceita um identificador USB `VID:PID` como `0403:6001`. Quando uma máquina é vinculada por VID:PID,
a reconexão automática a segue para a nova porta depois que o sistema operacional reenumera os
dispositivos USB — por exemplo, após uma reinicialização ou ao desconectar e reconectar. Você pode
encontrar o VID:PID de um dispositivo na saída do `lsusb` (Linux) ou no Gerenciador de Dispositivos
→ IDs de Hardware (Windows).

Após selecionar um driver, as definições específicas de conexão aparecem abaixo do seletor (ex.:
porta serial, baud rate). Elas variam conforme o driver escolhido.

### Taxas de Transmissão Personalizadas

O seletor de taxa de transmissão lista as taxas comuns, mas aceita qualquer valor entre 300 e
4000000 bits por segundo. Escolha **Personalizado...** no final da lista e insira a taxa usada pelo
seu firmware — por exemplo `500000`. Uma taxa personalizada configurada permanece visível no
seletor.

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
