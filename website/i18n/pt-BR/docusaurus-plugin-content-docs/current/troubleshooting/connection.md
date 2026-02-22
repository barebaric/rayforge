# Problemas de Conexão

Esta página ajuda você a diagnosticar e resolver problemas ao conectar o Rayforge à sua máquina de laser via conexão serial.

## Diagnóstico Rápido

### Sintomas

Problemas comuns de conexão incluem:

- Erro "Porta deve ser configurada" ao tentar conectar
- Conexão falhando e reconectando repetidamente
- Porta serial não aparecendo na lista de portas
- Erros "Permissão negada" ao tentar abrir porta serial
- Dispositivo parece conectar mas não responde a comandos

---

## Problemas Comuns e Soluções

### Nenhuma Porta Serial Detectada

**Problema:** O menu suspenso de porta serial está vazio ou não mostra seu dispositivo.

**Diagnóstico:**

1. Verifique se seu dispositivo está ligado e conectado via USB
2. Tente desconectar e reconectar o cabo USB
3. Teste o cabo USB com outro dispositivo (cabos podem falhar)
4. Tente uma porta USB diferente no seu computador

**Soluções:**

**Linux:**
Se você está usando a versão Snap, você precisa conceder permissões de porta serial:

```bash
sudo snap connect rayforge:serial-port
```

Veja [Permissões Snap](snap-permissions) para configuração detalhada do Linux.

Para instalações não-Snap, adicione seu usuário ao grupo `dialout`:

```bash
sudo usermod -a -G dialout $USER
```

Então saia e entre novamente para a mudança ter efeito.

**Windows:**
1. Abra Gerenciador de Dispositivos (Win+X, depois selecione Gerenciador de Dispositivos)
2. Procure em "Portas (COM e LPT)" pelo seu dispositivo
3. Se você vir um ícone de aviso amarelo, atualize ou reinstale o driver
4. Anote o número da porta COM (ex: COM3)
5. Se o dispositivo não está listado, o cabo USB ou driver pode estar com defeito

**macOS:**
1. Verifique Informação do Sistema → USB para verificar se o dispositivo é reconhecido
2. Instale drivers CH340/CH341 se seu controlador usa este chipset
3. Verifique dispositivos `/dev/tty.usbserial*` ou `/dev/cu.usbserial*`

### Erros de Permissão Negada

**Problema:** Você recebe erros "Permissão negada" ou similares ao tentar conectar.

**No Linux (não-Snap):**

Seu usuário precisa estar no grupo `dialout` (ou `uucp` em algumas distribuições):

```bash
# Adicione você mesmo ao grupo dialout
sudo usermod -a -G dialout $USER

# Verifique se você está no grupo (após sair/entrar)
groups | grep dialout
```

**Importante:** Você deve sair e entrar novamente (ou reiniciar) para mudanças de grupo terem efeito.

**No Linux (Snap):**

Conceda acesso à porta serial ao snap:

```bash
sudo snap connect rayforge:serial-port
```

Veja o guia [Permissões Snap](snap-permissions) para mais detalhes.

**No Windows:**

Feche quaisquer outras aplicações que possam estar usando a porta serial, incluindo:
- Instâncias anteriores do Rayforge
- Ferramentas de monitor serial
- Outro software de laser
- Arduino IDE ou ferramentas similares

### Porta Serial Errada Selecionada

**Problema:** O Rayforge conecta mas a máquina não responde.

**Diagnóstico:**

Você pode ter selecionado a porta errada, especialmente se tem múltiplos dispositivos USB conectados.

**Solução:**

1. Desconecte todos os outros dispositivos seriais USB
2. Note quais portas estão disponíveis no Rayforge
3. Conecte seu controlador de laser
4. Atualize a lista de portas - a nova porta é seu laser
5. No Linux, controladores de laser tipicamente aparecem como:
   - `/dev/ttyUSB0` (comum para chipsets CH340)
   - `/dev/ttyACM0` (comum para controladores USB nativos)
6. No Windows, anote a porta COM do Gerenciador de Dispositivos
7. Evite portas nomeadas `/dev/ttyS*` no Linux - estas são portas seriais de hardware, não USB

:::warning Portas Seriais de Hardware
O Rayforge avisará se você selecionar portas `/dev/ttyS*` no Linux, pois estas tipicamente não são dispositivos GRBL baseados em USB. Portas seriais USB usam `/dev/ttyUSB*` ou `/dev/ttyACM*`.
:::


### Taxa de Transmissão Incorreta

**Problema:** Conexão estabelece mas comandos não funcionam ou produzem respostas truncadas.

**Solução:**

Controladores GRBL tipicamente usam uma destas taxas de transmissão:

- **115200** (mais comum, GRBL 1.1+)
- **9600** (versões GRBL mais antigas)
- **250000** (menos comum, alguns firmwares personalizados)

Tente diferentes taxas de transmissão nas configurações de dispositivo do Rayforge. A mais comum é **115200**.

### Conexão Cai Continuamente

**Problema:** O Rayforge conecta com sucesso mas continua desconectando e reconectando.

**Possíveis Causas:**

1. **Cabo USB com defeito** - Substitua por cabo conhecido-bom (preferencialmente curto, <2m)
2. **Problemas de energia USB** - Tente uma porta USB diferente, preferencialmente no próprio computador em vez de um hub
3. **EMI/Interferência** - Mantenha cabos USB longe de fios de motores e fontes de alta tensão
4. **Problemas de firmware** - Atualize seu firmware GRBL se possível
5. **Conflitos de porta USB** - No Windows, tente portas USB diferentes

**Passos de Solução de Problemas:**

```bash
# No Linux, monitore logs do sistema enquanto conecta:
sudo dmesg -w
```

Procure mensagens como:
- "USB disconnect" - indica problemas físicos/cabo
- "device descriptor read error" - frequentemente problema de energia ou cabo

### Dispositivo Não Responde Após Conexão

**Problema:** Status de conexão mostra "Conectado" mas a máquina não responde a comandos.

**Diagnóstico:**

1. Verifique se o tipo de firmware correto está selecionado (GRBL vs outro)
2. Verifique se a máquina está ligada (controlador e fonte de alimentação)
3. Verifique se a máquina está em estado de alarme (requer homing ou limpar alarme)

**Solução:**

Tente enviar um comando manual no Console:

- `?` - Solicitar relatório de status
- `$X` - Limpar alarme
- `$H` - Levar a máquina à origem

Se não há resposta, verifique novamente taxa de transmissão e seleção de porta.

---

## Mensagens de Status de Conexão

O Rayforge mostra diferentes estados de conexão:

| Status | Significado | Ação |
|--------|---------|--------|
| **Desconectado** | Não conectado a nenhum dispositivo | Configure porta e conecte |
| **Conectando** | Tentando estabelecer conexão | Espere, ou verifique configuração se travado |
| **Conectado** | Conectado com sucesso e recebendo status | Pronto para usar |
| **Erro** | Conexão falhou com um erro | Verifique mensagem de erro para detalhes |
| **Hibernando** | Esperando antes de tentar reconexão | Conexão anterior falhou, tentando novamente em 5s |

---

## Testando Sua Conexão

### Teste de Conexão Passo a Passo

1. **Configure a máquina:**
   - Abra Configurações  Máquina
   - Selecione ou crie um perfil de máquina
   - Escolha o driver correto (GRBL Serial)
   - Selecione a porta serial
   - Defina taxa de transmissão (tipicamente 115200)

2. **Tente conexão:**
   - Clique "Conectar" no painel de controle da máquina
   - Observe o indicador de status de conexão

3. **Verifique comunicação:**
   - Se conectado, tente enviar uma consulta de status
   - A máquina deve reportar sua posição e estado

4. **Teste comandos básicos:**
   - Tente homing (`$H`) se sua máquina tem chaves de limite
   - Ou limpe alarmes (`$X`) se necessário

### Usando Logs de Depuração

O Rayforge inclui log detalhado para problemas de conexão. Para habilitar log de depuração:

```bash
# Execute Rayforge do terminal com log de depuração
rayforge --loglevel DEBUG
```

Verifique os logs para:
- Tentativas de conexão e falhas
- Dados seriais transmitidos (TX) e recebidos (RX)
- Mensagens de erro com rastreamento de pilha

---

## Solução de Problemas Avançada

### Verificando Disponibilidade de Porta Manualmente

**Linux:**
```bash
# Liste todos dispositivos seriais USB
ls -l /dev/ttyUSB* /dev/ttyACM*

# Verifique permissões
ls -l /dev/ttyUSB0  # Substitua pela sua porta

# Deve mostrar: crw-rw---- 1 root dialout
# Você precisa estar no grupo 'dialout'

# Teste porta manualmente
sudo minicom -D /dev/ttyUSB0 -b 115200
```

**Windows:**
```powershell
# Liste portas COM no PowerShell
[System.IO.Ports.SerialPort]::getportnames()

# Ou use Gerenciador de Dispositivos:
# Win + X → Gerenciador de Dispositivos → Portas (COM e LPT)
```

### Compatibilidade de Firmware

O Rayforge é projetado para firmware compatível com GRBL. Certifique-se de que seu controlador executa:

- **GRBL 1.1** (mais comum, recomendado)
- **GRBL 0.9** (mais antigo, pode ter recursos limitados)
- **grblHAL** (fork GRBL moderno, suportado)

Outros tipos de firmware (Marlin, Smoothieware) não são atualmente suportados via driver GRBL.

### Chipsets USB-para-Serial

Chipsets comuns e seus drivers:

| Chipset | Linux | Windows | macOS |
|---------|-------|---------|-------|
| **CH340/CH341** | Driver de kernel integrado | [Driver CH341SER](http://www.wch.cn/downloads/) | Requer driver |
| **FTDI FT232** | Driver de kernel integrado | Integrado (Windows 10+) | Integrado |
| **CP2102 (SiLabs)** | Driver de kernel integrado | Integrado (Windows 10+) | Integrado |

---

## Ainda Tendo Problemas?

Se você tentou tudo acima e ainda não consegue conectar:

1. **Verifique as issues do GitHub** - Alguém pode ter relatado o mesmo problema
2. **Crie um relatório de issue detalhado** com:
   - Sistema operacional e versão
   - Versão do Rayforge (Snap/Flatpak/AppImage/fonte)
   - Modelo da placa controladora e versão do firmware
   - Chipset USB (verifique Gerenciador de Dispositivos no Windows ou `lsusb` no Linux)
   - Mensagens de erro completas e logs de depuração
3. **Teste com outra aplicação** - Tente conectar com um terminal serial (minicom, PuTTY, Monitor Serial Arduino) para verificar se o hardware funciona

---

## Páginas Relacionadas

- [Permissões Snap](snap-permissions) - Configuração de permissão Snap Linux
- [Modo de Depuração](debug) - Habilitar log de diagnóstico
- [Configurações Gerais](../machine/general) - Guia de configuração da máquina
- [Configurações do Dispositivo](../machine/device) - Referência de configuração GRBL
