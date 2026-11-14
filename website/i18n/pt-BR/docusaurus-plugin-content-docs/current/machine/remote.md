---
description:
  "Execute sua máquina a partir de um pequeno computador junto ao hardware enquanto o Rayforge roda
  em outro lugar. Os jobs sobrevivem a desconexões e a reconexão sincroniza o progresso exatamente."
---

# Dispositivos remotos

Execute sua máquina a partir de um pequeno computador junto ao hardware (por exemplo, um Raspberry
Pi na oficina) enquanto o Rayforge roda no seu notebook ou desktop. Os jobs rodam no host do
dispositivo e continuam mesmo se o Rayforge desconectar; ao reconectar, o progresso é sincronizado
exatamente.

## Como funciona

- O **daemon do raydriver** hospeda a conexão da máquina junto ao hardware.
- O Rayforge conecta-se a um _dispositivo no daemon_ — ele não precisa saber qual protocolo o
  dispositivo fala. O daemon resolve isso e informa ao Rayforge o que o dispositivo pode fazer.
- Máquinas remotas usam um único driver **Remote machine (raydriver)**. Os drivers específicos de
  protocolo (GRBL, Smoothie, Marlin) são para hardware conectado diretamente ao seu computador.

## Configurar o host do dispositivo

1. Instale o raydriver no host do dispositivo: `pip install raydriver`. A descoberta em rede está
   incluída.
2. Execute `raydriver daemon add /dev/ttyUSB0` (use a porta serial da sua máquina; `raydriver scan`
   as lista). Isso cria a configuração com um **token** gerado e registra o dispositivo. Adicione
   `--protocol`/`--baudrate` para firmware não padrão.
3. Execute `raydriver daemon install` para instalar e iniciar o serviço em segundo plano (systemd no
   Linux, launchd no macOS, uma tarefa de logon no Windows), ou execute `raydriver daemon` em
   primeiro plano.
4. Para acesso LAN, crie um certificado com o comando `openssl` de uma linha impresso por
   `raydriver daemon --setup` e defina `tls_cert` e `tls_key` na configuração. Um daemon em um
   endereço que não seja loopback não inicia sem token e TLS.

Consulte o
[guia do daemon do raydriver](https://github.com/barebaric/raydriver/blob/main/docs/daemon.md) para
a referência completa.

### Obter o token

O token é o segredo compartilhado que o Rayforge usa para conectar. Ele é gerado ao adicionar o
primeiro dispositivo e impresso uma vez nesse momento. Para imprimi-lo novamente, execute isto **no
host do dispositivo**:

```bash
raydriver daemon status
```

Copie o valor de `token:` — você o colará no Rayforge abaixo.

## Conectar o Rayforge

O assistente de máquina descobre os daemons na rede e lista seus dispositivos; escolha um para
adicioná-lo. Alternativamente, adicione uma máquina manualmente com o driver **Remote machine
(raydriver)** e insira o nome do host, a porta e o token do daemon; depois escolha o dispositivo na
lista lida do daemon:

![Assistente — conexão remota](/screenshots/config-wizard-remote.webp)

- **Nome do host / porta do daemon** — o computador que executa o daemon (porta padrão `8765`).
- **Token** — o valor de `raydriver daemon status` no host do dispositivo.
- **Dispositivo** — a máquina no daemon, lida da lista de dispositivos dele assim que o nome do host
  e o token são definidos.
- Os campos TLS só aparecem para um daemon acessado pela rede; um daemon no mesmo computador
  (`127.0.0.1`) não precisa de certificado.

A sondagem funciona pela conexão remota, então o assistente pode identificar hardware que não está
conectado ao seu computador.

## Configurações da máquina

As configurações de uma máquina remota mostram o endpoint do daemon e o dispositivo em vez de uma
porta serial local:

![Configurações da máquina — máquina remota](/screenshots/machine-settings-remote.webp)

## Jobs e reconexões

- Um job enviado roda até o fim no host do dispositivo mesmo se o Rayforge fechar ou a rede cair.
- Reconecte para ver o progresso exato. Um cliente por vez detém o controle; os outros podem
  observar.

## Câmeras no host do dispositivo

O vídeo não trafega pela conexão de controle. Use as [câmeras de rede](camera.md) do Rayforge com um
stream acessível no host do dispositivo:

- **Câmera IP**: adicione a URL do stream diretamente.
- **Câmera USB no host do dispositivo**: execute um streamer padrão como `ustreamer` e adicione a
  URL MJPEG dele como uma câmera de stream HTTP.

O daemon anuncia os endpoints de câmera configurados para que o assistente possa pré-preencher a
URL.

## Notas de segurança

- Apenas LAN (ou túnel) — sem nuvem, sem contas.
- Token e TLS são obrigatórios para qualquer daemon que não seja loopback.
- Mantenha os streams de câmera apenas na LAN ou atrás de um firewall; eles são separados do canal
  de controle TLS.
