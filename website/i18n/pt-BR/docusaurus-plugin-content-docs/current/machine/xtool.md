---
description:
  "Conecte as cortadoras a laser xTool D1 e D1 Pro e a xTool M2 por Wi-Fi com os drivers xTool
  experimentais do Rayforge."
---

# xTool por Wi-Fi

O Rayforge inclui drivers experimentais para máquinas xTool conectadas por Wi-Fi. Eles usam a API de
rede proprietária da xTool em vez das interfaces GRBL Telnet ou FluidNC / ESP3D, então a máquina
precisa ser configurada com seu próprio driver xTool.

## Máquinas compatíveis

| Perfil           | Driver           | O que pode fazer                                                                            |
| ---------------- | ---------------- | ------------------------------------------------------------------------------------------- |
| **xTool D1 Pro** | xTool (Wi-Fi)    | Enviar e iniciar trabalhos, referenciar, mover, pausar, retomar e cancelar                  |
| **xTool M2**     | xTool M2 (Wi-Fi) | Monitorar a máquina, referenciar, mover e pausar, retomar ou cancelar um trabalho preparado |

Os dois drivers estão marcados como **experimentais**. Teste-os em um retalho antes de confiar neles
para produção e relate problemas com um log de depuração anexado.

## Configurar uma máquina xTool

1. Conecte a máquina à mesma rede do seu computador e descubra o endereço IP dela (por exemplo, no
   aplicativo da xTool ou na lista de dispositivos do roteador).
2. Adicione uma máquina e escolha o perfil **xTool D1 Pro** ou **xTool M2**. O perfil já configura o
   driver e as dimensões da máquina.
3. Informe o **Hostname ou endereço IP** da máquina.
4. Deixe as portas nos valores padrão, a menos que sua rede exija outra coisa:
   - D1 Pro: porta HTTP `8080`, porta WebSocket `8081`
   - M2: porta `28900`
5. Conecte.

Se o Rayforge informar que nenhuma máquina xTool respondeu, confira o endereço IP, se a máquina está
ligada e se o computador e a máquina estão na mesma rede.

## D1 e D1 Pro

O driver **xTool (Wi-Fi)** envia o trabalho para a máquina, inicia-o e acompanha o progresso pelos
eventos de trabalho do WebSocket. Referenciamento, movimentos manuais e absolutos, pausa, retomada e
cancelamento funcionam normalmente.

A API de rede da máquina não informa a posição, então o Rayforge deriva a posição do estado do
trabalho. O referenciamento está disponível para todos os eixos ao mesmo tempo; o referenciamento
por eixo está desativado. Este driver não oferece gravação de configurações de firmware,
deslocamentos de coordenadas de trabalho nem sondagem.

## xTool M2

O driver **xTool M2 (Wi-Fi)** conecta ao serviço TLS do M2 na porta `28900`. Ele monitora o estado
da máquina e oferece referenciamento, movimentos manuais e pausa, retomada ou cancelamento de um
trabalho já preparado na máquina.

A API de rede do M2 ainda não recebe G-code, então o Rayforge não pode enviar seus próprios
trabalhos para ela. Prepare o trabalho no xTool Studio, inicie-o lá e use o Rayforge para monitorar
e controlar. Movimentos Z, trocas de ferramenta, controle remoto da potência do laser, configurações
de firmware e deslocamentos de coordenadas de trabalho não são suportados.

## Certificados TLS

O driver do M2 valida o certificado TLS da máquina. Se a máquina usar um certificado autoassinado,
ative **Permitir certificados TLS inválidos** nas configurações do driver. Faça isso apenas em uma
rede confiável.

## Veja também

- [Configurações gerais](general) - Seleção do driver, hostname e portas
- [Referência de firmware](../reference/firmware.md) - Controladores e drivers compatíveis
- [Dispositivos remotos](remote) - Conectar a uma máquina através de um daemon raydriver
