# Corte de Contorno

O corte de contorno traça o contorno de formas vetoriais para cortá-las livres do material. É a
operação de laser mais comum para criar peças, sinais e peças decorativas.

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

![Configurações de etapa de contorno](/screenshots/step-settings-contour-general.webp)

## Configurações Principais

O diálogo de configurações de etapa tem três abas: **Configurações de Etapa**, **Laser** e
**Pós-Processamento**. As configurações são descritas em ordem de aba abaixo.

### Configurações de Contorno

![Configurações de etapa de contorno](/screenshots/step-settings-contour-general.webp)

O grupo **Configurações de Contorno** na aba _Configurações de Etapa_ controla como o contorno é
traçado.

#### Lado de Corte e Deslocamento de Caminho

Controla onde o laser corta relativo ao caminho vetorial:

| Deslocamento      | Descrição                    | Usar Para                              |
| ----------------- | ---------------------------- | -------------------------------------- |
| **Linha Central** | Corta diretamente no caminho | Cortes na linha central, marcação      |
| **Interno**       | Corta dentro da forma        | Peças que devem caber no tamanho exato |
| **Externo**       | Corta fora da forma          | Furos em que as peças se encaixam      |

**Distância de Deslocamento:**

- Quão longe dentro/fora deslocar (mm)
- Tipicamente definido como metade da sua largura de kerf
- Kerf = largura do material removido pelo laser
- Exemplo: deslocamento de 0.15mm para kerf de 0.3mm

#### Ordem de Corte

Controla a ordem em que caminhos aninhados são processados:

**Dentro-Fora:**

- Corta recursos internos primeiro, depois trabalha para fora
- Mantém as partes externas do material intactas por mais tempo

**Fora-Dentro:**

- Corta o perímetro externo primeiro, depois se move para dentro
- Mantém a peça de trabalho fixada ao estoque por mais tempo

**Recomendado:** Dentro-Fora (padrão)

#### Remover Caminhos Internos

Para designs com furos ou recortes internos, você pode optar por traçar apenas o limite mais
externo:

- **Remover Caminhos Internos**: Quando habilitado, apenas o contorno mais externo é traçado
- Furos e recortes internos são ignorados

Isso é útil quando você quer cortar uma forma mas preservar o interior, como criar uma moldura ou
contorno sem cortar detalhes internos.

#### Recorte Excessivo (Overcut)

Estende caminhos de corte fechados além do ponto inicial para que o feixe de laser se sobreponha ao
início do corte:

**Recorte Excessivo:**

- Distância em unidades de máquina para estender o corte além da junção início/fim
- Defina como **0** para desativar (padrão)
- Valores típicos: 1–5 para a maioria dos materiais
- Máximo: 100

**Por que usar recorte excessivo:**

No início e no fim de um contorno fechado, o laser pode não penetrar completamente devido à
aceleração e desaceleração. O recorte excessivo garante que o feixe se sobreponha na junção, criando
um corte limpo e completamente separado. Isso é especialmente útil para:

- Materiais espessos onde a penetração completa é marginal
- Cortes em alta velocidade onde os efeitos de aceleração são mais pronunciados
- Peças que devem cair livres sem pós-processamento

O recorte excessivo se aplica tanto a contornos externos quanto a furos internos.

<!-- prettier-ignore-start -->
:::tip[Entrada/Saída vs Recorte Excessivo]
[Entrada/Saída](../lead-in-out.md) adiciona movimentos de
aproximação e saída com potência zero antes e depois do trajeto de corte. O recorte excessivo
estende o próprio trajeto de corte além da junção. Eles podem ser usados juntos para qualidade de
corte ideal.
:::
<!-- prettier-ignore-end -->

#### Retraçamento com Limiar Personalizado

Ao trabalhar com imagens bitmap que foram convertidas em vetores, você pode controlar quais partes
são traçadas:

- **Reverificar Conteúdo**: Habilita um limiar de brilho personalizado para o traçado
- **Limiar de Traçado (0.0-1.0)**: Valor de corte de brilho quando a reverificação está habilitada
  - Valores menores traçam apenas áreas mais escuras
  - Valores maiores incluem áreas mais claras

Isso é útil quando o traçado padrão não captura o nível de detalhe que você precisa.

### Configurações do Laser

![Configurações do laser](/screenshots/step-settings-contour-laser.webp)

Potência, velocidade e seleção da cabeça do laser ficam na página **Laser** do diálogo de
configurações de etapa.

#### Potência e Velocidade

**Potência (%):**

- Intensidade do laser de 0-100%
- Maior potência para materiais mais espessos
- Menor potência para marcação ou pontilhado

**Velocidade (mm/min):**

- Quão rápido o laser se move
- Mais lento = mais energia = corte mais profundo
- Mais rápido = menos energia = corte mais leve

#### Compensação de Kerf

Kerf é a largura do material removido pelo feixe do laser:

**Por que importa:**

- Um círculo cortado "na linha" será ligeiramente menor que o projetado
- O laser remove ~0.2-0.4mm de material (dependendo da largura do feixe)

**Como compensar:**

1. Meça seu kerf em cortes de teste
2. Use deslocamento de caminho = kerf/2
3. Para peças: desloque **dentro** por kerf/2
4. Para furos: desloque **fora** por kerf/2

Veja [Kerf](../kerf.md) para um guia detalhado.

## Pós-Processamento

![Configurações de pós-processamento de contorno](/screenshots/step-settings-contour-post.webp)

Operações de contorno suportam várias opções de pós-processamento:

- **[Suavização de Caminho](../smooth.md)** - Reduz bordas irregulares em caminhos de corte
- **[Abas de Fixação](../holding-tabs.md)** - Mantém peças cortadas anexadas ao material de estoque
- **[Cortar para Estoque](../crop-to-stock.md)** - Limita cortes ao limite do material
- **[Otimização de Caminho](../path-optimization.md)** - Reduz distância de deslocamento entre
  cortes
- **[Multi-Passagem](../multi-pass.md)** - Repete cortes para materiais espessos
- **[Entrada/Saída](../lead-in-out.md)** - Adiciona movimentos de aproximação e saída sem potência
  para extremidades de corte mais limpas

### Corte Multi-Passagem

Para materiais mais espessos do que uma única passagem pode cortar:

**Passagens:**

- Número de vezes para repetir o corte
- Cada passagem corta mais fundo

**Profundidade de Passagem (degrau-Z):**

- Quanto baixar o eixo Z por passagem (se suportado)
- Requer controle de eixo Z na sua máquina
- Cria corte verdadeiro 2.5D
- Defina como 0 para múltiplas passagens na mesma profundidade

<!-- prettier-ignore-start -->
:::warning[Eixo Z Necessário]
A profundidade de passagem só funciona se sua máquina tem controle de eixo Z. Para máquinas sem eixo
Z, use múltiplas passagens na mesma profundidade.
:::
<!-- prettier-ignore-end -->

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

- **Adicione:** [Abas de fixação](../holding-tabs.md)
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
3. Adiciona abas de fixação (se configurado)
4. Gera comandos G-code

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
- **[Abas de Fixação](../holding-tabs.md)** - Mantendo peças fixadas durante o corte
- **[Kerf](../kerf.md)** - Melhorando a precisão do corte
- **[Grade de Teste de Material](material-test-grid)** - Encontrando configurações ideais de
  potência/velocidade
