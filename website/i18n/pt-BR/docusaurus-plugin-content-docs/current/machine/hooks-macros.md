# Macros & Hooks

O Rayforge fornece dois poderosos recursos de automação para personalizar seu fluxo de trabalho: **Macros** e **Hooks**. Ambos permitem injetar G-code personalizado em seus trabalhos, mas servem a propósitos diferentes.

![Configurações de Hooks & Macros](/screenshots/machine-hooks-macros.png)

---

## Visão Geral

| Recurso    | Propósito                    | Gatilho              | Caso de Uso                                    |
| ---------- | ---------------------------- | -------------------- | ---------------------------------------------- |
| **Macros** | Snippets de G-code reutilizáveis   | Execução manual     | Comandos rápidos, padrões de teste, rotinas personalizadas |
| **Hooks**  | Injeção automática de G-code | Eventos do ciclo de vida do trabalho | Sequências de inicialização, mudanças de camada, limpeza      |

---

## Macros

Macros são **scripts de G-code nomeados e reutilizáveis** que você pode executar manualmente a qualquer momento.

### Para Que Servem as Macros?

Casos de uso comuns de macros:

- **Levar a máquina à origem** - Enviar `$H` rapidamente
- **Definir deslocamentos de trabalho** - Armazenar e recuperar posições G54/G55
- **Controle de assistência de ar** - Ligar/desligar assistência de ar
- **Teste de foco** - Executar um padrão rápido de teste de foco
- **Trocas de ferramenta personalizadas** - Para configurações multi-laser
- **Rotinas de emergência** - Desligamento rápido ou limpar alarme
- **Sondagem de material** - Auto-foco ou medição de altura

### Criando uma Macro

1. **Abrir Configurações da Máquina:**
   - Navegue até **Configurações Máquina Macros**

2. **Adicionar uma nova macro:**
   - Clique no botão **"+"**
   - Digite um nome descritivo (ex: "Home Machine", "Enable Air Assist")

3. **Escreva seu G-code:**
   - Cada linha é um comando G-code separado
   - Comentários começam com `;` ou `(`
   - Variáveis podem ser usadas (veja Substituição de Variáveis abaixo)

4. **Salve a macro**

5. **Execute a macro:**
   - Na lista de macros, clique na macro
   - Ou atribua um atalho de teclado (se suportado)

### Exemplos de Macros

#### Simples: Levar a Máquina à Origem

**Nome:** Home Machine
**Código:**

```gcode
$H
; Espera o homing completar
```

**Uso:** Rapidamente leva a máquina à origem antes de começar o trabalho.

---

#### Médio: Definir Deslocamento de Trabalho

**Nome:** Set G54 to Current Position
**Código:**

```gcode
G10 L20 P1 X0 Y0
; Define a origem do sistema de coordenadas de trabalho G54 para a posição atual
```

**Uso:** Marca a posição atual do laser como a origem do trabalho.

---

#### Avançado: Grade de Teste de Foco

**Nome:** 9-Point Focus Test
**Código:**

```gcode
; Grade de 9 pontos para encontrar foco ideal
G21  ; Milímetros
G90  ; Posicionamento absoluto
G0 X10 Y10
M3 S1000
G4 P0.1
M5
G0 X20 Y10
M3 S1000
G4 P0.1
M5
; ... (repetir para os pontos restantes)
```

**Uso:** Testa rapidamente o foco em diferentes posições da mesa.

---

---

### Exemplos de Macro

Hooks são **injeções automáticas de G-code** acionadas por eventos específicos durante a execução do trabalho.

### Gatilhos de Hook

O Rayforge suporta estes gatilhos de hook:

| Gatilho             | Quando Executa                     | Usos Comuns                                 |
| ------------------- | ---------------------------------- | ------------------------------------------- |
| **Início do Trabalho**       | Início do trabalho        | Homing, deslocamento de trabalho, assistência de ar ligada, pré-aquecimento |
| **Fim do Trabalho**         | Fim do trabalho              | Retornar à origem, assistência de ar desligada, bip, resfriamento |
| **Início da Camada**     | Antes de processar cada camada     | Troca de ferramenta, ajuste de potência, comentários         |
| **Fim da Camada**       | Após processar cada camada      | Notificação de progresso, pausa                |
| **Início da Peça** | Antes de processar cada peça | Numeração de peças, marcas de alinhamento             |
| **Fim da Peça**   | Após processar cada peça  | Resfriamento, pausa de inspeção                  |

### Criando um Hook

1. **Abrir Configurações da Máquina:**
   - Navegue até **Configurações Máquina Hooks**

2. **Selecionar um gatilho:**
   - Escolha o evento quando este hook deve executar

3. **Escreva seu G-code:**
   - O código do hook é injetado no ponto do gatilho
   - Use variáveis para valores dinâmicos (veja abaixo)

4. **Habilitar/desabilitar:**
   - Ative/desative hooks sem excluí-los

### Exemplos de Hooks

#### Início do Trabalho: Inicializar Máquina

**Gatilho:** Início do Trabalho
**Código:**

```gcode
G21         ; Milímetros
G90         ; Posicionamento absoluto
$H          ; Levar a máquina à origem
G0 X0 Y0    ; Mover para origem
M3 S0       ; Laser ligado mas potência 0 (alguns controladores precisam disso)
M8          ; Assistência de ar LIGADA
```

**Propósito:** Garante que a máquina está em um estado conhecido antes de cada trabalho.

---

#### Fim do Trabalho: Retornar à Origem e Bipar

**Gatilho:** Fim do Trabalho
**Código:**

```gcode
M5          ; Laser DESLIGADO
M9          ; Assistência de ar DESLIGADA
G0 X0 Y0    ; Retornar à origem
M300 S800 P200  ; Bipe (se suportado)
```

**Propósito:** Finaliza o trabalho com segurança e sinaliza conclusão.

---

#### Início da Camada: Adicionar Comentário

**Gatilho:** Início da Camada
**Código:**

```gcode
; Iniciando camada: {layer_name}
; Índice da camada: {layer_index}
```

**Propósito:** Torna o G-code mais legível para depuração.

---

#### Início da Peça: Numeração de Peças

**Gatilho:** Início da Peça
**Código:**

```gcode
; Peça: {workpiece_name}
; Peça {workpiece_index} de {total_workpieces}
```

**Propósito:** Rastreia o progresso em trabalhos com múltiplas peças.

---

### Ordem de Execução dos Hooks

Para um trabalho com 2 camadas, cada uma com 2 peças:

```
[Hook Início do Trabalho]
  [Hook Início da Camada] (Camada 1)
    [Hook Início da Peça] (Peça 1)
       ... G-code da peça 1 ...
    [Hook Fim da Peça] (Peça 1)
    [Hook Início da Peça] (Peça 2)
       ... G-code da peça 2 ...
    [Hook Fim da Peça] (Peça 2)
  [Hook Fim da Camada] (Camada 1)
  [Hook Início da Camada] (Camada 2)
    [Hook Início da Peça] (Peça 3)
       ... G-code da peça 3 ...
    [Hook Fim da Peça] (Peça 3)
    [Hook Início da Peça] (Peça 4)
       ... G-code da peça 4 ...
    [Hook Fim da Peça] (Peça 4)
  [Hook Fim da Camada] (Camada 2)
[Hook Fim do Trabalho]
```

---

## Substituição de Variáveis

Tanto macros quanto hooks suportam **substituição de variáveis** para injetar valores dinâmicos.

### Variáveis Disponíveis

Variáveis usam a sintaxe `{variable_name}` e são substituídas durante a geração do G-code.

**Variáveis de nível de trabalho:**

| Variável     | Descrição                      | Valor de Exemplo |
| ------------ | -------------------------------- | ------------- |
| `{job_name}` | Nome do trabalho/documento atual | "test-job"    |
| `{date}`     | Data atual                     | "2025-10-03"  |
| `{time}`     | Hora atual                     | "14:30:25"    |

**Variáveis de nível de camada:**

| Variável         | Descrição                       | Valor de Exemplo |
| ---------------- | --------------------------------- | ------------- |
| `{layer_name}`   | Nome da camada atual         | "Cut Layer"   |
| `{layer_index}`  | Índice baseado em zero da camada atual | 0, 1, 2...    |
| `{total_layers}` | Número total de camadas no trabalho     | 3             |

**Variáveis de nível de peça:**

| Variável             | Descrição                           | Valor de Exemplo |
| -------------------- | ------------------------------------- | ------------- |
| `{workpiece_name}`   | Nome da peça                 | "Circle 1"    |
| `{workpiece_index}`  | Índice baseado em zero da peça atual | 0, 1, 2...    |
| `{total_workpieces}` | Número total de peças            | 5             |

**Variáveis da máquina:**

| Variável         | Descrição                    | Valor de Exemplo |
| ---------------- | ------------------------------ | ------------- |
| `{machine_name}` | Nome do perfil da máquina    | "My K40"      |
| `{max_speed}`    | Velocidade máxima de corte (mm/min) | 1000          |
| `{work_width}`   | Largura da área de trabalho (mm)           | 300           |
| `{work_height}`  | Altura da área de trabalho (mm)          | 200           |

### Exemplo: Notificação de Progresso

**Hook:** Início da Camada
**Código:**

```gcode
; ========================================
; Camada {layer_index} de {total_layers}: {layer_name}
; Trabalho: {job_name}
; Hora: {time}
; ========================================
```

**Resultado no G-code:**

```gcode
; ========================================
; Camada 0 de 3: Cut Layer
; Trabalho: test-project
; Hora: 14:30:25
; ========================================
```

---

## Casos de Uso Avançados

### Configuração Multi-Ferramenta

Para máquinas com múltiplos lasers ou ferramentas:

**Hook:** Início da Peça
**Código:**

```gcode
; Selecionar ferramenta para peça {workpiece_name}
T{tool_number}  ; Comando de troca de ferramenta (se suportado)
G4 P1           ; Esperar pela troca de ferramenta
```

### Pausas Condicionais

Adicione pausas opcionais para inspeção:

**Hook:** Fim da Camada
**Código:**

```gcode
; M0  ; Descomente para pausar após cada camada para inspeção
```

### Assistência de Ar Por Camada

Controle a assistência de ar em uma base por camada:

**Hook:** Início da Camada (para camadas de corte)
**Código:**

```gcode
M8  ; Assistência de ar LIGADA
```

**Hook:** Início da Camada (para camadas de gravação)
**Código:**

```gcode
M9  ; Assistência de ar DESLIGADA (previne dispersão de poeira para gravação)
```

:::note Hooks Específicos por Camada
O Rayforge atualmente não suporta personalização de hooks por camada. Para conseguir isso, use G-code condicional ou perfis de máquina separados.
:::

---

## Considerações de Segurança

:::danger Teste Antes da Produção
Sempre teste macros e hooks no **modo de simulação** ou com o laser **desabilitado** antes de executar em trabalhos reais. G-code configurado incorretamente pode:

- Colidir a máquina contra os limites
- Disparar o laser inesperadamente
- Danificar materiais ou equipamentos
  :::

**Checklist de segurança:**

- [ ] Macros incluem limites de avanço (parâmetro `F`)
- [ ] Macros verificam limites de posição
- [ ] Hooks de início de trabalho incluem `M5` ou comando de laser desligado
- [ ] Hooks de fim de trabalho desligam o laser (`M5`) e assistência de ar (`M9`)
- [ ] Nenhum comando destrutivo sem confirmação
- [ ] Testado em simulação ou com laser desabilitado

---

## Páginas Relacionadas

- [Configurações do Dispositivo](device) - Referência de comandos GRBL
- [Dialetos de G-code](../reference/gcode-dialects) - Compatibilidade de G-code
- [Configurações Gerais](general) - Configuração da máquina
- [Fluxo de Trabalho Multi-Camadas](../features/multi-layer) - Usando hooks com camadas
