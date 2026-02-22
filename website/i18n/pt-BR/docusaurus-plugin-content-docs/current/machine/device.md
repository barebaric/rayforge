# Configurações do Dispositivo

A página do Dispositivo nas Configurações da Máquina permite ler e aplicar configurações diretamente no seu dispositivo conectado (controlador). Estas também são conhecidas como configurações "dollar" ou `$$` no GRBL.

![Configurações do Dispositivo](/screenshots/machine-device.png)

:::warning Cuidado ao Alterar Configurações
Configurações incorretas de firmware podem fazer sua máquina se comportar de forma imprevisível, perder posição ou até danificar o hardware. Sempre anote os valores originais antes de fazer alterações e modifique uma configuração de cada vez.
:::

## Visão Geral

A página do Dispositivo fornece acesso direto às configurações de firmware do seu controlador. Aqui você pode:

- Ler configurações atuais do dispositivo
- Modificar configurações individuais
- Aplicar alterações ao dispositivo

Configurações de firmware controlam:

- **Parâmetros de movimento**: Limites de velocidade, aceleração, calibração
- **Chaves de limite**: Comportamento de homing, limites de software/hardware
- **Controle do laser**: Faixa de potência, habilitação do modo laser
- **Configuração elétrica**: Inversões de pinos, pullups
- **Relatórios**: Formato e frequência das mensagens de status

Essas configurações são armazenadas no seu controlador (não no Rayforge) e persistem entre ciclos de energia.

## Lendo Configurações

Clique no botão **Ler do Dispositivo** para buscar as configurações atuais do seu controlador conectado. Isso requer:

- A máquina estar conectada
- O driver suportar leitura de configurações do dispositivo

## Aplicando Configurações

Após modificar configurações, as alterações são aplicadas ao dispositivo. O dispositivo pode:

- Reiniciar temporariamente
- Desconectar e reconectar
- Exigir um ciclo de energia para algumas alterações

## Acesso via Console

Você também pode visualizar/modificar configurações via console de G-code:

**Ver todas as configurações:**
```
$$
```

**Ver configuração única:**
```
$100
```

**Modificar configuração:**
```
$100=80.0
```

**Restaurar padrões:**
```
$RST=$
```

:::danger Restaurar Padrões Apaga Todas as Configurações
O comando `$RST=$` reseta todas as configurações GRBL para os padrões de fábrica. Você perderá qualquer calibração e ajuste. Faça backup das suas configurações primeiro!
:::

---

## Configurações Críticas para Lasers

Estas configurações são mais importantes para operação a laser:

### $32 - Modo Laser

**Valor:** 0 = Desabilitado, 1 = Habilitado

**Propósito:** Habilita recursos específicos de laser no GRBL

**Quando habilitado (1):**
- Laser desliga automaticamente durante movimentos G0 (rápidos)
- Potência ajusta dinamicamente durante aceleração/desaceleração
- Previne queimaduras acidentais durante posicionamento

**Quando desabilitado (0):**
- Laser se comporta como spindle (modo CNC)
- Não desliga durante rápidos
- **Perigoso para uso com laser!**

:::warning Sempre Habilite o Modo Laser
$32 deve **sempre** ser definido como 1 para cortadoras a laser. Modo laser desabilitado pode causar queimaduras não intencionais e riscos de incêndio.
:::

### $30 & $31 - Faixa de Potência do Laser

**$30 - Potência Máxima do Laser (RPM)**
**$31 - Potência Mínima do Laser (RPM)**

**Propósito:** Define a faixa de potência para comandos S

**Valores típicos:**
- $30=1000, $31=0 (faixa S0-S1000, mais comum)
- $30=255, $31=0 (faixa S0-S255, alguns controladores)

:::tip Correspondência com Configuração do Rayforge
A configuração "Potência Máxima" nas suas [Configurações do Laser](laser) deve corresponder ao seu valor $30. Se $30=1000, defina potência máxima para 1000 no Rayforge.
:::

### $130 & $131 - Deslocamento Máximo

**$130 - Deslocamento Máximo X (mm)**
**$131 - Deslocamento Máximo Y (mm)**

**Propósito:** Define a área de trabalho da sua máquina

**Por que importa:**
- Limites de software ($20) usam esses valores para prevenir colisões
- Define os limites do sistema de coordenadas
- Deve corresponder ao tamanho físico da sua máquina

---

## Referência de Configurações

### Configuração dos Motores de Passo ($0-$6)

Controla sinais elétricos e temporização dos motores de passo.

| Configuração | Descrição | Valor Típico |
|---------|-------------|---------------|
| $0 | Tempo do pulso do passo (μs) | 10 |
| $1 | Atraso de passo ocioso (ms) | 25 |
| $2 | Máscara de inversão de pulso de passo | 0 |
| $3 | Máscara de inversão de direção do passo | 0 |
| $4 | Inverter pino de habilitação de passo | 0 |
| $5 | Inverter pinos de limite | 0 |
| $6 | Inverter pino de sonda | 0 |

### Limites & Homing ($20-$27)

Controla chaves de limite e comportamento de homing.

| Configuração | Descrição | Valor Típico |
|---------|-------------|---------------|
| $20 | Habilitar limites de software | 0 ou 1 |
| $21 | Habilitar limites físicos | 0 |
| $22 | Habilitar ciclo de homing | 0 ou 1 |
| $23 | Inverter direção de homing | 0 |
| $24 | Taxa de alimentação de localização de homing (mm/min) | 25 |
| $25 | Taxa de busca de homing (mm/min) | 500 |
| $26 | Atraso de debounce de homing (ms) | 250 |
| $27 | Distância de recuo de homing (mm) | 1.0 |

### Spindle & Laser ($30-$32)

| Configuração | Descrição | Valor do Laser |
|---------|-------------|-------------|
| $30 | Velocidade máxima do spindle | 1000.0 |
| $31 | Velocidade mínima do spindle | 0.0 |
| $32 | Habilitar modo laser | 1 |

### Calibração dos Eixos ($100-$102)

Define quantos passos do motor de passo equivalem a um milímetro de movimento.

| Configuração | Descrição | Notas |
|---------|-------------|-------|
| $100 | passos/mm X | Depende da relação polia/correia |
| $101 | passos/mm Y | Geralmente igual ao X |
| $102 | passos/mm Z | Não usado na maioria dos lasers |

**Calculando passos/mm:**
```
passos/mm = (passos_motor_por_volta × microstepping) / (dentes_polia × passo_correia)
```

**Exemplo:** 200 passos/volta, 16 microstepping, polia de 20 dentes, correia GT2:
```
passos/mm = (200 × 16) / (20 × 2) = 3200 / 40 = 80
```

### Velocidade & Aceleração dos Eixos ($110-$122)

| Configuração | Descrição | Valor Típico |
|---------|-------------|---------------|
| $110 | Taxa máxima X (mm/min) | 5000.0 |
| $111 | Taxa máxima Y (mm/min) | 5000.0 |
| $112 | Taxa máxima Z (mm/min) | 500.0 |
| $120 | Aceleração X (mm/s²) | 500.0 |
| $121 | Aceleração Y (mm/s²) | 500.0 |
| $122 | Aceleração Z (mm/s²) | 100.0 |

### Deslocamento dos Eixos ($130-$132)

| Configuração | Descrição | Notas |
|---------|-------------|-------|
| $130 | Deslocamento máximo X (mm) | Largura da área de trabalho |
| $131 | Deslocamento máximo Y (mm) | Profundidade da área de trabalho |
| $132 | Deslocamento máximo Z (mm) | Deslocamento Z (se aplicável) |

---

## Exemplo de Configuração Comum

### Laser de Diodo Típico (300×400mm)

```gcode
$0=10          ; Pulso de passo 10μs
$1=255         ; Atraso de passo ocioso 255ms
$2=0           ; Sem inversão de passo
$3=0           ; Sem inversão de direção
$4=0           ; Sem inversão de habilitação
$5=0           ; Sem inversão de limite
$10=1          ; Reportar WPos
$11=0.010      ; Desvio de junção 0.01mm
$12=0.002      ; Tolerância de arco 0.002mm
$13=0          ; Reportar mm
$20=1          ; Limites de software habilitados
$21=0          ; Limites físicos desabilitados
$22=1          ; Homing habilitado
$23=0          ; Home para mínimo
$24=50.0       ; Alimentação de homing 50mm/min
$25=1000.0     ; Busca de homing 1000mm/min
$26=250        ; Debounce de homing 250ms
$27=2.0        ; Recuo de homing 2mm
$30=1000.0     ; Potência máxima S1000
$31=0.0        ; Potência mínima S0
$32=1          ; Modo laser LIGADO
$100=80.0      ; passos/mm X
$101=80.0      ; passos/mm Y
$102=80.0      ; passos/mm Z
$110=5000.0    ; Taxa máxima X
$111=5000.0    ; Taxa máxima Y
$112=500.0     ; Taxa máxima Z
$120=500.0     ; Acel X
$121=500.0     ; Acel Y
$122=100.0     ; Acel Z
$130=400.0     ; Deslocamento máximo X
$131=300.0     ; Deslocamento máximo Y
$132=0.0       ; Deslocamento máximo Z
```

---

## Backup de Configurações

### Procedimento de Backup

1. **Via Rayforge:**
   - Abra o painel de Configurações do Dispositivo
   - Clique em "Exportar Configurações"
   - Salve o arquivo como `grbl-backup-YYYY-MM-DD.txt`

2. **Via console:**
   - Envie comando `$$`
   - Copie toda a saída para arquivo de texto
   - Salve com data

### Procedimento de Restauração

1. Abra arquivo de backup
2. Envie cada linha (`$100=80.0`, etc.) via console
3. Verifique com comando `$$`

:::tip Backups Regulares
Faça backup das suas configurações após qualquer calibração ou ajuste. Armazene backups em um local seguro.
:::

---

## Veja Também

- [Configurações Gerais](general) - Nome da máquina e configurações de velocidade
- [Configurações do Laser](laser) - Configuração da cabeça do laser
- [Solução de Problemas de Conexão](../troubleshooting/connection) - Corrigindo problemas de conexão

## Recursos Externos

- [Configuração GRBL v1.1](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration)
- [Comandos GRBL v1.1](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands)
- [Documentação Grbl_ESP32](https://github.com/bdring/Grbl_Esp32/wiki)
