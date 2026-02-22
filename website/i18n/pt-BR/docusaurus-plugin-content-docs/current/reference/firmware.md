# Compatibilidade de Firmware

Esta página documenta a compatibilidade de firmware para controladores de laser usados com o Rayforge.

## Visão Geral

O Rayforge é projetado principalmente para **controladores baseados em GRBL** mas tem suporte experimental para outros tipos de firmware.

### Matriz de Compatibilidade

| Firmware         | Versão | Status          | Driver                 | Notas                  |
| ---------------- | ------- | --------------- | ---------------------- | ---------------------- |
| **GRBL**         | 1.1+    | Totalmente suportado | GRBL Serial            | Recomendado            |
| **grblHAL**      | 2023+   | Compatível      | GRBL Serial            | Fork moderno do GRBL       |
| **GRBL**         | 0.9     | Limitado         | GRBL Serial            | Antigo, pode ter problemas |
| **Smoothieware** | Todos     | Experimental    | Nenhum (use driver GRBL) | Não testado               |
| **Marlin**       | 2.0+    | Experimental    | Nenhum (use driver GRBL) | Modo laser necessário    |
| **Outro**        | -       | Não suportado   | -                      | Solicite suporte        |

---

## Firmware GRBL

**Status:**  Totalmente Suportado
**Versões:** 1.1+
**Driver:** GRBL Serial

### GRBL 1.1 (Recomendado)

**O que é GRBL 1.1?**

GRBL 1.1 é o firmware mais comum para máquinas CNC e laser de hobby. Lançado em 2017, é estável, bem documentado e amplamente suportado.

**Recursos suportados pelo Rayforge:**

- Comunicação serial (USB)
- Relatório de status em tempo real
- Modo laser (M4 potência constante)
- Leitura/escrita de configurações ($$, $X=valor)
- Ciclos de homing ($H)
- Sistemas de coordenadas de trabalho (G54)
- Comandos de jogging ($J=)
- Override de taxa de avanço
- Limites suaves
- Limites físicos (endstops)

**Limitações conhecidas:**

- Intervalo de potência: 0-1000 (parâmetro S)
- Sem conectividade de rede (apenas USB)
- Memória onboard limitada (buffer de G-code pequeno)

### Verificando a Versão do GRBL

**Consulte a versão:**

Conecte ao seu controlador e envie:

```
$I
```

**Exemplos de resposta:**

```
[VER:1.1h.20190825:]
[OPT:V,15,128]
```

- `1.1h` = versão GRBL 1.1h
- Data indica a compilação

### GRBL 0.9 (Antigo)

**Status:** Suporte Limitado

GRBL 0.9 é uma versão antiga com alguns problemas de compatibilidade:

**Diferenças:**

- Formato de relatório de status diferente
- Sem modo laser (M4) - usa apenas M3
- Menos configurações
- Sintaxe de jogging diferente

**Se você tem GRBL 0.9:**

1. **Atualize para GRBL 1.1** se possível (recomendado)
2. **Use M3 em vez de M4** (potência menos previsível)
3. **Teste completamente** - alguns recursos podem não funcionar

**Instruções de atualização:** Veja [GRBL Wiki](https://github.com/gnea/grbl/wiki)

---

## grblHAL

**Status:** Compatível
**Versões:** 2023+
**Driver:** GRBL Serial

### O que é grblHAL?

grblHAL é um fork moderno do GRBL com recursos aprimorados:

- Suporte a múltiplos hardwares de controlador (STM32, ESP32, etc.)
- Rede Ethernet/WiFi
- Suporte a cartão SD
- Mais pinos de I/O
- Suporte a laser aprimorado

**Compatibilidade com Rayforge:**

- **Totalmente compatível** - grblHAL mantém o protocolo GRBL 1.1
- Todos os recursos GRBL funcionam
- Recursos adicionais (rede, SD) ainda não suportados pelo Rayforge
- Relatório de status idêntico ao GRBL

**Usando grblHAL:**

1. Selecione o driver "GRBL Serial" no Rayforge
2. Conecte via serial USB (assim como GRBL)
3. Todos os recursos funcionam como documentado para GRBL

**Futuro:** Rayforge pode adicionar suporte para recursos específicos do grblHAL (rede, etc.)

---

## Smoothieware

**Versões:** Todos
**Driver:** GRBL Serial (modo de compatibilidade)

### Notas de Compatibilidade

Smoothieware usa sintaxe G-code diferente:

**Principais diferenças:**

| Recurso         | GRBL           | Smoothieware     |
| --------------- | -------------- | ---------------- |
| **Laser Ligado**    | `M4 S<valor>`  | `M3 S<valor>`    |
| **Intervalo de Potência** | 0-1000         | 0.0-1.0 (float)  |
| **Status**      | formato `<...>` | Formato diferente |

**Usando Smoothieware com Rayforge:**

1. **Selecione o dialeto Smoothieware** nas configurações da máquina > G-code > Dialeto
2. **Teste com baixa potência** primeiro
3. **Verifique se o intervalo de potência** corresponde à sua configuração
4. **Sem status em tempo real** - feedback limitado

**Limitações:**

- Relatório de status não totalmente compatível
- Escala de potência pode diferir
- Comandos de configurações ($$) não suportados
- Não testado em hardware real

**Recomendação:** Se possível, use firmware compatível com GRBL em vez disso.

---

## Marlin

**Versões:** 2.0+ com suporte a laser
**Driver:** GRBL Serial

### Marlin para Laser

Marlin 2.0+ pode controlar lasers quando configurado corretamente.

**Requisitos:**

1. **Firmware Marlin 2.0 ou posterior**
2. **Recursos de laser ativados:**
   ```cpp
   #define LASER_FEATURE
   #define LASER_POWER_INLINE
   ```
3. **Intervalo de potência correto** configurado:
   ```cpp
   #define SPEED_POWER_MAX 1000
   ```

**Compatibilidade:**

- Modo laser M4 suportado
- G-code básico (G0, G1, G2, G3)
- Relatório de status difere
- Comandos de configurações diferentes
- Assistente de ar (M8/M9) pode não funcionar

**Usando Marlin com Rayforge:**

1. **Selecione o dialeto Marlin** nas configurações da máquina > G-code > Dialeto
2. **Configure o Marlin** para uso com laser
3. **Teste o intervalo de potência** corresponde (0-1000 ou 0-255)
4. **Teste limitado** - use com cautela

**Alternativa melhor:** Use firmware GRBL em máquinas a laser.

---

## Guia de Atualização de Firmware

### Atualizando para GRBL 1.1

**Por que atualizar?**

- Modo laser (M4) para potência constante
- Melhor relatório de status
- Mais confiável
- Melhor suporte do Rayforge

**Como atualizar:**

1. **Identifique sua placa controladora:**
   - Arduino Nano/Uno (ATmega328P)
   - Arduino Mega (ATmega2560)
   - Placa personalizada

2. **Baixe GRBL 1.1:**
   - [GRBL Releases](https://github.com/gnea/grbl/releases)
   - Obtenha a versão 1.1 mais recente (1.1h recomendado)

3. **Grave o firmware:**

   **Usando Arduino IDE:**

   ```
   1. Instale Arduino IDE
   2. Abra o sketch GRBL (grbl.ino)
   3. Selecione a placa e porta corretas
   4. Faça upload
   ```

   **Usando avrdude:**

   ```bash
   avrdude -c arduino -p m328p -P /dev/ttyUSB0 \
           -U flash:w:grbl.hex:i
   ```

4. **Configure o GRBL:**
   - Conecte via serial
   - Envie `$$` para ver as configurações
   - Configure para sua máquina

### Backup Antes de Atualizar

**Salve suas configurações:**

1. Conecte ao controlador
2. Envie o comando `$$`
3. Copie toda a saída de configurações
4. Salve em um arquivo

**Após a atualização:**

- Restaure as configurações uma por uma: `$0=10`, `$1=25`, etc.
- Ou use os padrões e reconfigure

---

## Hardware do Controlador

### Controladores Comuns

| Placa                  | Firmware Típico | Suporte Rayforge |
| ---------------------- | ---------------- | ---------------- |
| **Arduino CNC Shield** | GRBL 1.1         | Excelente        |
| **MKS DLC32**          | grblHAL          | Excelente        |
| **Cohesion3D**         | Smoothieware     | Limitado          |
| **Placas SKR**         | Marlin/grblHAL   | Varia           |
| **Ruida**              | Proprietário      | Não suportado    |
| **Trocen**             | Proprietário      | Não suportado    |
| **TopWisdom**          | Proprietário      | Não suportado    |

### Controladores Recomendados

Para melhor compatibilidade com Rayforge:

1. **Arduino Nano + CNC Shield** (GRBL 1.1)
   - Barato (~$10-20)
   - Fácil de gravar
   - Bem documentado

2. **MKS DLC32** (grblHAL)
   - Moderno (baseado em ESP32)
   - Capacidade WiFi
   - Desenvolvimento ativo

3. **Placas GRBL personalizadas**
   - Muitas disponíveis no mercado
   - Verifique suporte a GRBL 1.1+

---

## Configuração de Firmware

### Configurações GRBL para Laser

**Configurações essenciais:**

```
$30=1000    ; Potência máxima spindle/laser (1000 = 100%)
$31=0       ; Potência mínima spindle/laser
$32=1       ; Modo laser ativado (1 = ligado)
```

**Configurações da máquina:**

```
$100=80     ; Passos/mm X (calibre para sua máquina)
$101=80     ; Passos/mm Y
$110=3000   ; Taxa máxima X (mm/min)
$111=3000   ; Taxa máxima Y
$120=100    ; Aceleração X (mm/seg)
$121=100    ; Aceleração Y
$130=300    ; Deslocamento máximo X (mm)
$131=200    ; Deslocamento máximo Y (mm)
```

**Configurações de segurança:**

```
$20=1       ; Limites suaves ativados
$21=1       ; Limites físicos ativados (se você tem endstops)
$22=1       ; Homing ativado
```

### Testando o Firmware

**Sequência básica de teste:**

1. **Teste de conexão:**

   ```
   Envia: ?
   Espera: &lt;Idle|...&gt;
   ```

2. **Verificação de versão:**

   ```
   Envia: $I
   Espera: [VER:1.1...]
   ```

3. **Verificação de configurações:**

   ```
   Envia: $$
   Espera: $0=..., $1=..., etc.
   ```

4. **Teste de movimento:**

   ```
   Envia: G91 G0 X10
   Espera: Máquina move 10mm em X
   ```

5. **Teste de laser (potência muito baixa):**
   ```
   Envia: M4 S10
   Espera: Laser liga (fraco)
   Envia: M5
   Espera: Laser desliga
   ```

---

## Solução de Problemas de Firmware

### Firmware Não Responde

**Sintomas:**

- Sem resposta aos comandos
- Conexão falha
- Status não reportado

**Diagnóstico:**

1. **Verifique a taxa de transmissão (baud rate):**
   - Padrão GRBL 1.1: 115200
   - GRBL 0.9: 9600
   - Tente ambos

2. **Verifique o cabo USB:**
   - Cabo de dados, não apenas de carga
   - Substitua por um cabo conhecido bom

3. **Verifique a porta:**
   - Linux: `/dev/ttyUSB0` ou `/dev/ttyACM0`
   - Windows: COM3, COM4, etc.
   - Porta correta selecionada no Rayforge

4. **Teste com terminal:**
   - Use screen, minicom ou PuTTY
   - Envie `?` e veja se obtém resposta

### Falhas do Firmware

**Sintomas:**

- Controlador trava durante o trabalho
- Desconexões aleatórias
- Comportamento inconsistente

**Possíveis causas:**

1. **Estouro de buffer** - Arquivo G-code muito complexo
2. **Ruído elétrico** - Aterramento ruim ou EMI
3. **Bug do firmware** - Atualize para a versão mais recente
4. **Problema de hardware** - Controlador com defeito

**Soluções:**

- Atualize o firmware para a versão estável mais recente
- Simplifique o G-code (reduza precisão, menos segmentos)
- Adicione contas de ferrite ao cabo USB
- Melhore o aterramento e roteamento de cabos

### Firmware Errado

**Sintomas:**

- Comandos rejeitados
- Comportamento inesperado
- Mensagens de erro

**Solução:**

1. Consulte a versão do firmware: `$I`
2. Compare com as expectativas do Rayforge
3. Atualize ou selecione o dialeto correto

---

## Suporte Futuro de Firmware

### Recursos Solicitados

Usuários solicitaram suporte para:

- **Controladores Ruida** - Controladores de laser chineses
- **Trocen/AWC** - Controladores de laser comerciais
- **ESP32 WiFi** - Conectividade de rede para grblHAL
- **API Laser** - API direta da máquina (sem G-code)

**Status:** Atualmente não suportado. Solicitações de recursos são bem-vindas no GitHub.

### Contribuindo

Para adicionar suporte a firmware:

1. Implemente o driver em `rayforge/machine/driver/`
2. Defina o dialeto G-code em `rayforge/machine/models/dialect.py`
3. Teste completamente em hardware real
4. Envie um pull request com documentação

---

## Páginas Relacionadas

- [Dialeto G-code](gcode-dialects) - Detalhes do dialeto
- [Configurações de Dispositivo](../machine/device) - Configuração GRBL
- [Problemas de Conexão](../troubleshooting/connection) - Solução de problemas de conexão
- [Configurações Gerais](../machine/general) - Configuração da máquina
