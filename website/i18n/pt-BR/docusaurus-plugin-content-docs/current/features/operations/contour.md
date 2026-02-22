# Corte de Contorno

O corte de contorno traça o contorno de formas vetoriais para cortá-las livres do material. É a operação de laser mais comum para criar peças, sinais e peças decorativas.

## Visão Geral

Operações de contorno:

- Seguem caminhos vetoriais (linhas, curvas, formas)
- Cortam ao longo do perímetro dos objetos
- Suportam passagem única ou múltipla para materiais espessos
- Podem usar caminhos de corte internos, externos ou na linha
- Funcionam com qualquer forma vetorial fechada ou aberta


## Quando Usar Contorno

Use corte de contorno para:

- Cortar peças livres do material de estoque
- Criar contornos e bordas
- Cortar formas de madeira, acrílico, papelão
- Perfurar ou marcar (com potência reduzida)
- Criar estênceis e modelos

**Não use contorno para:**

- Preencher áreas (use [Gravação](engrave) em vez disso)
- Imagens bitmap (converta para vetores primeiro)

## Criando uma Operação de Contorno

### Passo 1: Selecionar Objetos

1. Importe ou desenhe formas vetoriais na tela
2. Selecione os objetos que deseja cortar
3. Certifique-se de que as formas são caminhos fechados para cortes completos

### Passo 2: Adicionar Operação de Contorno

- **Menu:** Operações Adicionar Contorno
- **Atalho:** <kbd>ctrl+shift+c</kbd>
- **Clique direito:** Menu de contexto Adicionar Operação Contorno

### Passo 3: Configurar Definições

![Configurações de etapa de contorno](/screenshots/step-settings-contour-general.png)

## Configurações Principais

### Potência e Velocidade

**Potência (%):**

- Intensidade do laser de 0-100%
- Maior potência para materiais mais espessos
- Menor potência para marcação ou pontilhado

**Velocidade (mm/min):**

- Quão rápido o laser se move
- Mais lento = mais energia = corte mais profundo
- Mais rápido = menos energia = corte mais leve

### Corte Multi-Passagem

Para materiais mais espessos que uma única passagem pode cortar:

**Passagens:**

- Número de vezes para repetir o corte
- Cada passagem corta mais fundo

**Profundidade de Passagem (degrau-Z):**

- Quanto baixar o eixo Z por passagem (se suportado)
- Requer controle de eixo Z na sua máquina
- Cria corte verdadeiro 2.5D
- Defina como 0 para múltiplas passagens na mesma profundidade

:::warning Eixo Z Necessário
:::

Profundidade de passagem só funciona se sua máquina tem controle de eixo Z. Para máquinas sem eixo Z, use múltiplas passagens na mesma profundidade.

### Deslocamento de Caminho

Controla onde o laser corta relativo ao caminho vetorial:

| Deslocamento      | Descrição               | Usar Para                        |
| ----------- | ------------------------- | ------------------------------ |
| **Na Linha** | Corta diretamente no caminho | Cortes de linha central, marcação       |
| **Interno**  | Corta dentro da forma     | Peças que devem caber no tamanho exato |
| **Externo** | Corta fora da forma    | Furos que peças encaixam      |

**Distância de Deslocamento:**

- Quão longe dentro/fora deslocar (mm)
- Tipicamente definido como metade da sua largura de kerf
- Kerf = largura do material removido pelo laser
- Exemplo: deslocamento de 0.15mm para kerf de 0.3mm

### Direção de Corte

**Horário vs Anti-Horário:**

- Afeta qual lado do corte recebe mais calor
- Geralmente horário para regra da mão direita
- Mude se um lado queima mais que o outro

**Otimizar Ordem:**

- Classifica automaticamente caminhos para deslocamento mínimo
- Reduz tempo de trabalho
- Previne cortes perdidos

## Recursos Avançados

![Configurações de pós-processamento de contorno](/screenshots/step-settings-contour-post.png)

### Abas de Fixação

Abas mantêm peças cortadas anexadas ao material de estoque durante o corte:

- Adicione abas para prevenir peças de caírem
- Abas são pequenas seções não cortadas
- Quebre as abas após o trabalho completar
- Veja [Abas de Fixação](../holding-tabs) para detalhes

### Compensação de Kerf

Kerf é a largura do material removido pelo feixe do laser:

**Por que importa:**

- Um círculo cortado "na linha" será ligeiramente menor que o projetado
- O laser remove ~0.2-0.4mm de material (dependendo da largura do feixe)

**Como compensar:**

1. Meça seu kerf em cortes de teste
2. Use deslocamento de caminho = kerf/2
3. Para peças: desloque **dentro** por kerf/2
4. Para furos: desloque **fora** por kerf/2

Veja [Kerf](../kerf) para guia detalhado.

### Entrada/Saída

Entradas e saídas controlam onde cortes começam e terminam:

**Entrada:**

- Entrada gradual no corte
- Previne marcas de queimadura no ponto inicial
- Move laser para velocidade total antes de atingir a borda do material

**Saída:**

- Saída gradual do corte
- Previne dano no ponto final
- Comum para metais e acrílicos

**Configuração:**

- Comprimento: Quão longe a entrada se estende (mm)
- Ângulo: Direção do caminho de entrada
- Tipo: Linha reta, arco ou espiral

## Dicas e Melhores Práticas

### Teste de Material

**Sempre teste primeiro:**

1. Corte pequenas formas de teste em sucata
2. Comece com configurações conservadoras (menor potência, velocidade mais lenta)
3. Aumente gradualmente a potência ou diminua a velocidade
4. Registre configurações bem-sucedidas

### Ordem de Corte

**Melhores práticas:**

- Grave antes de cortar (mantém material fixado)
- Corte recursos internos antes do perímetro externo
- Use abas de fixação para peças que podem se mover
- Corte peças menores primeiro (menos vibração)

## Solução de Problemas

### Cortes não atravessam o material

- **Aumente:** Configuração de potência
- **Diminua:** Configuração de velocidade
- **Adicione:** Mais passagens
- **Verifique:** Foco está correto
- **Verifique:** Feixe está limpo (lente suja)

### Carbonização ou queima excessiva

- **Diminua:** Configuração de potência
- **Aumente:** Configuração de velocidade
- **Use:** Assistência de ar
- **Tente:** Múltiplas passagens mais rápidas em vez de uma lenta
- **Verifique:** Material é apropriado para corte a laser

### Peças caem durante o corte

- **Adicione:** [Abas de fixação](../holding-tabs)
- **Use:** Otimização de ordem de corte
- **Corte:** Recursos internos antes dos externos
- **Certifique-se:** Material está plano e fixado

### Profundidade de corte inconsistente

- **Verifique:** Espessura do material é uniforme
- **Verifique:** Material está plano (não empenado)
- **Verifique:** Distância do foco é consistente
- **Verifique:** Potência do laser está estável

### Cantos ou curvas perdidos

- **Diminua:** Velocidade (especialmente em cantos)
- **Verifique:** Configurações de aceleração da máquina
- **Verifique:** Correias estão esticadas
- **Reduza:** Complexidade do caminho (simplifique curvas)

## Detalhes Técnicos

### Sistema de Coordenadas

Operações de contorno funcionam em:

- **Unidades:** Milímetros (mm)
- **Origem:** Depende da máquina e configuração do trabalho
- **Coordenadas:** Plano X/Y (Z para profundidade multi-passagem)

### Geração de Caminho

O Rayforge converte formas vetoriais para G-code:

1. Desloca caminho (se corte interno/externo)
2. Otimiza ordem do caminho (minimiza deslocamento)
3. Insere entrada/saída (se configurado)
4. Adiciona abas de fixação (se configurado)
5. Gera comandos G-code

### Comandos G-code

G-code de contorno típico:

```gcode
G0 X10 Y10          ; Movimento rápido para início
M3 S204             ; Laser ligado a 80% de potência
G1 X50 Y10 F500     ; Corta para ponto a 500 mm/min
G1 X50 Y50 F500     ; Corta para próximo ponto
G1 X10 Y50 F500     ; Continua cortando
G1 X10 Y10 F500     ; Completa o quadrado
M5                  ; Laser desligado
```

## Tópicos Relacionados

- **[Gravação](engrave)** - Preenchendo áreas com padrões de gravação
- **[Abas de Fixação](../holding-tabs)** - Mantendo peças fixadas durante o corte
- **[Kerf](../kerf)** - Melhorando precisão do corte
- **[Grade de Teste de Material](material-test-grid)** - Encontrando configurações ideais de potência/velocidade
