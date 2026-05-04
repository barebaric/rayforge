---
description: "Use o assistente de configuração para detectar e configurar automaticamente um dispositivo conectado consultando suas configurações de firmware."
---

# Assistente de Configuração

O assistente de configuração detecta automaticamente seu dispositivo
conectando-se a ele e lendo suas configurações de firmware. Ele cria um
perfil de máquina totalmente configurado em segundos, eliminando a
configuração manual.

## Iniciando o Assistente

1. Abra **Configurações → Máquinas** e clique em **Add Machine**
2. No seletor de perfil, clique em **Other Device…** na parte inferior

Isso abre o assistente. Ele **não** requer um perfil de dispositivo
existente — o assistente cria um do zero consultando o hardware conectado.

## Conectar

A primeira página pede que você selecione um driver e forneça os parâmetros
de conexão.

![Assistente Conectar](/screenshots/app-settings-machines-wizard-connect.png)

### Seleção de Driver

Escolha o driver que corresponde ao controlador do seu dispositivo. Apenas
drivers que suportam detecção são listados. Tipicamente:

- **GRBL (Serial)** — Dispositivos GRBL conectados via USB
- **GRBL (Network)** — Dispositivos GRBL WiFi/Ethernet

### Parâmetros de Conexão

Após selecionar um driver, preencha os detalhes de conexão (porta serial,
baud rate, host, etc.). Estes são os mesmos parâmetros usados nas
[Configurações Gerais](general).

Clique em **Next** para iniciar a detecção.

## Descobrir

O assistente se conecta ao dispositivo e consulta seu firmware para obter
dados de configuração. Isso inclui:

- Versão do firmware e informações de compilação (`$I`)
- Todas as configurações do firmware (`$$`)
- Courses des eixos, velocidades, aceleração e faixa de potência do laser

Esta etapa geralmente é concluída em poucos segundos.

## Revisar

Após uma detecção bem-sucedida, a página de revisão mostra todas as
configurações descobertas. Tudo está pré-preenchido, mas pode ser ajustado
antes de criar a máquina.

![Assistente Revisar](/screenshots/app-settings-machines-wizard-review.png)

### Informações do Dispositivo

Informações somente leitura detectadas do firmware:

- **Nome do Dispositivo** — extraído das informações de compilação do firmware
- **Versão do Firmware** — p.ex. `1.1h`
- **Tamanho do Buffer RX** — tamanho do buffer de recepção serial
- **Tolerância de Arco** — tolerância de interpolação de arco do firmware

### Área de Trabalho

- **Curso X** / **Curso Y** — curso máximo dos eixos em unidades de
  máquina, lido das configurações de firmware `$130` e `$131`

### Velocidade

- **Velocidade máx. de deslocamento** — o menor valor entre `$110` e `$111`
- **Velocidade máx. de corte** — padrão igual à velocidade de deslocamento;
  ajuste conforme necessário

### Aceleração

- **Aceleração** — o menor valor entre `$120` e `$121`, em unidades de
  máquina por segundo ao quadrado

### Laser

- **Potência máx. (valor S)** — da configuração de firmware `$30`
  (spindle máx.)

### Comportamento

- **Homing ao iniciar** — ativado se o homing do firmware (`$22`) estiver
  ativado
- **Homing de eixo único** — ativado se o firmware foi compilado com o
  flag `H`

### Avisos

O assistente pode exibir avisos sobre problemas potenciais, como:

- Modo laser não ativado (`$32=0`)
- Dispositivo reportando em polegadas (`$13=1`)

## Criar a Máquina

Clique em **Create Machine** para finalizar. O assistente emite o perfil
configurado e o processo normal de criação de máquina continua — o
[diálogo de configurações da máquina](general) abre para ajustes
adicionais.

## Limitações

- O assistente funciona apenas com drivers que suportam detecção. Se o seu
  driver não está listado, use um perfil predefinido do seletor.
- A detecção requer que o dispositivo esteja ligado e conectado.
- Algumas configurações de firmware podem não ser legíveis em todos os
  dispositivos.

## Veja também

- [Configurações Gerais](general) — configuração manual da máquina
- [Definições do dispositivo](device) — ler e escrever definições do
  firmware em uma máquina já configurada
- [Adicionar uma Máquina](../application-settings/machines) — visão geral
  do processo de criação de máquinas
