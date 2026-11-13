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
- O Rayforge mantém a configuração da sua máquina e envia os jobs pela LAN.
- Todos os drivers de G-code (GRBL serial, GRBL telnet, GRBL simples, Smoothie, Marlin) funcionam
  local ou remotamente — remoto é uma configuração de conexão, não um driver separado.

## Configurar o host do dispositivo

1. Instale o raydriver no host do dispositivo: `pip install raydriver` (adicione `[daemon]` para a
   descoberta na rede).
2. Execute `raydriver daemon --setup`. Isso grava `~/.config/raydriver/daemon.json` com um token
   gerado e imprime o comando do certificado.
3. Para acesso LAN, crie um certificado com o comando `openssl` impresso e defina `tls_cert` e
   `tls_key` na configuração. Um daemon em um endereço que não seja loopback não inicia sem token e
   TLS.
4. Inicie `raydriver daemon`, ou execute-o como serviço systemd (uma unidade de exemplo acompanha a
   documentação do raydriver).

## Conectar o Rayforge

- O assistente de máquina descobre daemons na rede e lista seus dispositivos.
- Como alternativa, ative **Dispositivo remoto** nas configurações de conexão do driver e informe o
  nome do host, a porta, o token e o certificado TLS do daemon.
- A sondagem funciona pela conexão remota, então o assistente pode identificar hardware que não está
  conectado ao seu computador.

## Jobs e reconexões

- Um job enviado roda até o fim no host do dispositivo, mesmo se o Rayforge fechar ou a rede cair.
- Reconecte para ver o progresso exato. Um cliente por vez detém o controle; os outros podem
  observar.

## Câmeras no host do dispositivo

O vídeo não trafega pela conexão de controle. Use as [câmeras de rede](camera.md) do Rayforge com um
stream acessível no host do dispositivo:

- **Câmera IP**: adicione a URL do stream diretamente.
- **Câmera USB no host do dispositivo**: execute um streamer padrão como o `ustreamer` e adicione a
  URL MJPEG dele como câmera de stream HTTP.

O daemon anuncia os endpoints de câmera configurados para que o assistente possa preencher a URL.

## Notas de segurança

- Somente LAN (ou túnel) — sem nuvem, sem contas.
- Token e TLS são obrigatórios para qualquer daemon que não seja loopback.
- Mantenha os streams de câmera somente na LAN ou atrás de um firewall; eles são separados do canal
  de controle TLS.
