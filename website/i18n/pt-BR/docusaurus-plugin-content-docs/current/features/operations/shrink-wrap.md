# Shrink Wrap

Shrink Wrap cria um caminho de corte eficiente ao redor de múltiplos objetos gerando um limite que "encolhe" ao redor deles. É útil para cortar múltiplas peças de uma chapa com desperdício mínimo.

## Visão Geral

Operações Shrink Wrap:

- Criam caminhos de limite ao redor de grupos de objetos
- Minimizam desperdício de material
- Reduzem tempo de corte combinando caminhos
- Suportam distâncias de deslocamento para folga
- Funcionam com qualquer combinação de formas vetoriais

## Quando Usar Shrink Wrap

Use shrink wrap para:

- Cortar múltiplas peças pequenas de uma chapa
- Minimizar desperdício de material
- Criar limites de aninhamento eficientes
- Separar grupos de peças
- Reduzir tempo total de corte

**Não use shrink wrap para:**

- Objetos únicos (use [Contorno](contour) em vez disso)
- Peças que precisam de limites individuais
- Cortes retangulares precisos

## Como Shrink Wrap Funciona

Shrink wrap cria um limite usando um algoritmo de geometria computacional:

1. **Começa** com um feixe convexo ao redor de todos os objetos
2. **Encolhe** o limite em direção aos objetos
3. **Envolve** firmemente ao redor do grupo de objetos
4. **Desloca** para fora pela distância especificada

O resultado é um caminho de corte eficiente que segue a forma geral das suas peças enquanto mantém folga.

## Criando uma Operação Shrink Wrap

### Passo 1: Organizar Objetos

1. Coloque todas as peças que deseja envolver na tela
2. Posicione-as com espaçamento desejado
3. Múltiplos grupos separados podem ser shrink-wrapped juntos

### Passo 2: Selecionar Objetos

1. Selecione todos os objetos para incluir no shrink wrap
2. Podem ser formas, tamanhos e tipos diferentes
3. Todos os objetos selecionados serão envolvidos juntos

### Passo 3: Adicionar Operação Shrink Wrap

- **Menu:** Operações Adicionar Shrink Wrap
- **Clique direito:** Menu de contexto Adicionar Operação Shrink Wrap

### Passo 4: Configurar Definições

![Configurações de etapa Shrink Wrap](/screenshots/step-settings-shrink-wrap-general.png)

## Configurações Principais

### Potência e Velocidade

Como outras operações de corte:

**Potência (%):**

- Intensidade do laser para corte
- Mesma que você usaria para corte de [Contorno](contour)

**Velocidade (mm/min):**

- Quão rápido o laser se move
- Corresponda à velocidade de corte do seu material

**Passagens:**

- Número de vezes para cortar o limite
- Geralmente 1-2 passagens
- Mesmo que corte de contorno para seu material

### Distância de Deslocamento

**Deslocamento (mm):**

- Quanta folga ao redor das peças
- Distância dos objetos até o limite shrink-wrap
- Maior deslocamento = mais material deixado ao redor das peças

**Valores típicos:**

- **2-3mm:** Envolvimento apertado, desperdício mínimo
- **5mm:** Folga confortável
- **10mm+:** Material extra para manuseio

**Por que o deslocamento importa:**

- Muito pequeno: Risco de cortar nas peças
- Muito grande: Desperdiça material
- Considere: Largura do kerf, precisão de corte

### Suavidade

Controla quão de perto o limite segue as formas dos objetos:

**Alta suavidade:**

- Segue objetos mais de perto
- Caminho mais complexo
- Tempo de corte mais longo
- Menos desperdício de material

**Baixa suavidade:**

- Caminho mais simples e arredondado
- Tempo de corte mais curto
- Ligeiramente mais desperdício de material

**Recomendado:** Suavidade média para a maioria dos casos

## Casos de Uso

### Produção de Peças em Lote

**Cenário:** Cortando 20 peças pequenas de uma chapa grande

**Sem shrink wrap:**

- Corta limite da chapa inteira
- Desperdiça todo material ao redor das peças
- Tempo de corte longo

**Com shrink wrap:**

- Corta limite apertado ao redor do grupo de peças
- Salva material para outros projetos
- Corte mais rápido (perímetro menor)

### Otimização de Aninhamento

**Fluxo de trabalho:**

1. Aninhe peças eficientemente na chapa
2. Agrupe peças em seções
3. Shrink wrap cada seção
4. Corte seções separadamente

**Benefícios:**

- Pode remover seções finalizadas enquanto continua
- Manuseio mais fácil de peças cortadas
- Risco reduzido de movimento de peças

### Conservação de Material

**Exemplo:** Peças pequenas em material caro

**Processo:**

1. Organize peças firmemente
2. Shrink wrap com deslocamento de 3mm
3. Corte livre da chapa
4. Salve material restante

**Resultado:** Máxima eficiência de material

## Combinando com Outras Operações

### Shrink Wrap + Contorno

Fluxo de trabalho comum:

1. Operações de **Contorno** em peças individuais (corta detalhes)
2. **Shrink wrap** ao redor do grupo (corta livre da chapa)

**Ordem de execução:**

- Primeiro: Corta detalhes nas peças (enquanto fixadas)
- Por último: Shrink wrap corta grupo livre

Veja [Fluxo de Trabalho Multi-Camadas](../multi-layer) para detalhes.

### Shrink Wrap + Raster

**Exemplo:** Peças gravadas e cortadas

1. **Raster** grava logos nas peças
2. **Contorno** corta contornos das peças
3. **Shrink wrap** ao redor de todo o grupo

**Benefícios:**

- Toda gravação acontece enquanto material está fixado
- Shrink wrap final corta todo lote livre

## Dicas e Melhores Práticas

![Configurações de pós-processamento Shrink Wrap](/screenshots/step-settings-shrink-wrap-post.png)

### Espaçamento de Peças

**Espaçamento ideal:**

- 5-10mm entre peças
- Suficiente para shrink wrap distinguir objetos separados
- Não tanto que você desperdiça material

**Muito perto:**

- Peças podem ser envolvidas juntas
- Shrink wrap pode preencher lacunas
- Difícil separar após corte

**Muito longe:**

- Desperdiça material
- Tempo de corte mais longo
- Uso ineficiente da chapa

### Considerações de Material

**Melhor para:**

- Execuções de produção (muitas peças idênticas)
- Peças pequenas de chapas grandes
- Materiais caros (minimizar desperdício)
- Trabalhos de corte em lote

**Não ideal para:**

- Peças grandes únicas
- Peças preenchendo chapa inteira
- Quando você precisa de corte de chapa inteira

### Segurança

**Sempre:**

- Verifique se o limite não sobrepõe peças
- Verifique se o deslocamento é suficiente
- Pré-visualize no [Modo Simulação](../simulation-mode)
- Teste em sucata primeiro

**Fique atento a:**

- Shrink wrap cortando nas peças (aumente deslocamento)
- Peças movendo antes de shrink wrap completar
- Material empenando puxando peças fora de posição

## Técnicas Avançadas

### Múltiplos Shrink Wraps

Crie limites separados para grupos diferentes:

**Processo:**

1. Organize peças em grupos lógicos
2. Shrink wrap Grupo 1 (peças superiores)
3. Shrink wrap Grupo 2 (peças inferiores)
4. Corte grupos separadamente

**Benefícios:**

- Remover grupos finalizados durante o trabalho
- Melhor organização
- Recuperação de peças mais fácil

### Shrink Wraps Aninhados

Shrink wrap dentro de um limite maior:

**Exemplo:**

1. Shrink wrap interno: Peças detalhadas pequenas
2. Shrink wrap externo: Inclui peças maiores
3. Contorno: Limite de chapa inteira

**Use para:** Layouts complexos de múltiplas peças

### Teste de Folga

Antes da execução de produção:

1. Crie shrink wrap
2. Pré-visualize com [Modo Simulação](../simulation-mode)
3. Verifique se a folga é adequada
4. Verifique se nenhuma peça é intersectada
5. Execute teste em material de sucata

## Solução de Problemas

### Shrink wrap corta nas peças

- **Aumente:** Distância de deslocamento
- **Verifique:** Peças não estão muito próximas
- **Verifique:** Caminho shrink wrap na pré-visualização
- **Conte com:** Largura do kerf (largura do feixe do laser)

### Limite não segue formas

- **Aumente:** Configuração de suavidade
- **Verifique:** Peças estão propriamente selecionadas
- **Tente:** Deslocamento menor (pode estar envolvendo muito longe)

### Peças são envolvidas juntas

- **Aumente:** Espaçamento entre peças
- **Adicione:** Contornos manuais ao redor de peças individuais
- **Divida:** Em múltiplas operações shrink wrap

### Corte demora muito

- **Diminua:** Suavidade (caminho mais simples)
- **Aumente:** Deslocamento (limites mais retos)
- **Considere:** Múltiplos shrink wraps menores

### Peças se movem durante o corte

- **Adicione:** Pequenas abas para segurar peças (veja [Abas de Fixação](../holding-tabs))
- **Use:** Ordem de corte: dentro para fora
- **Certifique-se:** Material está plano e fixado
- **Verifique:** Chapa não está empenada

## Detalhes Técnicos

### Algoritmo

Shrink wrap usa geometria computacional:

1. **Feixe convexo** - Encontra limite externo
2. **Forma alfa** - Encolhe em direção aos objetos
3. **Deslocamento** - Expande pela distância de deslocamento
4. **Simplificação** - Com base na configuração de suavidade

### Otimização de Caminho

O caminho do limite é otimizado para:

- Comprimento total mínimo
- Curvas suaves (com base na suavidade)
- Pontos de início/fim eficientes

### Sistema de Coordenadas

- **Unidades:** Milímetros (mm)
- **Precisão:** 0.01mm típico
- **Coordenadas:** Mesmo que área de trabalho

## Tópicos Relacionados

- **[Corte de Contorno](contour)** - Cortando contornos de objetos individuais
- **[Fluxo de Trabalho Multi-Camadas](../multi-layer)** - Combinando operações efetivamente
- **[Abas de Fixação](../holding-tabs)** - Mantendo peças seguras durante o corte
- **[Modo Simulação](../simulation-mode)** - Pré-visualizando caminhos de corte
- **[Grade de Teste de Material](material-test-grid)** - Encontrando configurações de corte ideais
