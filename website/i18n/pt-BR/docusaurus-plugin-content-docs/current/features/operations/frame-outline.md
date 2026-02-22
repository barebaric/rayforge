# Contorno de Moldura

O Contorno de Moldura cria um caminho de corte retangular simples ao redor de todo o seu design. É a maneira mais rápida de adicionar uma borda limpa ou cortar seu trabalho livre da chapa de material.

## Visão Geral

Operações de Contorno de Moldura:

- Criam um limite retangular ao redor de todo o conteúdo
- Adicionam deslocamento/margem configurável do design
- Suportam compensação de kerf para dimensionamento preciso
- Funcionam com qualquer combinação de objetos na tela

![Configurações de etapa de Contorno de Moldura](/screenshots/step-settings-frame-outline-general.png)

## Quando Usar Contorno de Moldura

Use contorno de moldura para:

- Adicionar uma borda decorativa ao redor do seu design
- Cortar seu trabalho livre da chapa de material
- Criar um limite retangular simples
- Enquadramento rápido sem cálculos de caminho complexos

**Não use contorno de moldura para:**

- Formas irregulares ao redor de múltiplos objetos (use [Shrink Wrap](shrink-wrap) em vez disso)
- Cortar peças individuais (use [Contorno](contour) em vez disso)
- Seguir a forma exata do seu design

## Criando uma Operação de Contorno de Moldura

### Passo 1: Organizar Seu Design

1. Coloque todos os objetos na tela
2. Posicione-os onde você quer eles relativos à moldura
3. A moldura será calculada ao redor da caixa delimitadora de todo o conteúdo

### Passo 2: Adicionar Operação de Contorno de Moldura

- **Menu:** Operações → Adicionar Contorno de Moldura
- **Clique direito:** Menu de contexto → Adicionar Operação → Contorno de Moldura

### Passo 3: Configurar Definições

Configure os parâmetros da moldura:

- **Potência e Velocidade:** Corresponda aos requisitos de corte do seu material
- **Deslocamento:** Distância da borda do conteúdo até a moldura
- **Deslocamento de Caminho:** Corte interno, externo ou linha central

## Configurações Principais

### Potência e Velocidade

**Potência (%):**

- Intensidade do laser para cortar a moldura
- Corresponda aos requisitos de corte do seu material

**Velocidade (mm/min):**

- Quão rápido o laser se move
- Mais lento para materiais mais espessos

**Passagens:**

- Número de vezes para cortar a moldura
- Geralmente 1-2 passagens
- Adicione passagens para materiais mais espessos

### Distância de Deslocamento

**Deslocamento (mm):**

- Distância da caixa delimitadora do design até a moldura
- Cria uma margem/borda ao redor do seu trabalho

**Valores típicos:**

- **0mm:** Moldura toca a borda do design
- **2-5mm:** Pequena margem para aparência limpa
- **10mm+::** Borda grande para montagem ou manuseio

### Deslocamento de Caminho (Lado de Corte)

Controla onde o laser corta relativo ao caminho da moldura:

| Lado de Corte       | Descrição                 | Usar Para                           |
| -------------- | --------------------------- | --------------------------------- |
| **Linha Central** | Corta diretamente no caminho  | Corte padrão                  |
| **Externo**    | Corta fora do caminho da moldura | Fazendo a moldura ligeiramente maior  |
| **Interno**     | Corta dentro do caminho da moldura  | Fazendo a moldura ligeiramente menor |

### Compensação de Kerf

O contorno de moldura suporta compensação de kerf:

- Ajusta automaticamente para a largura do feixe do laser
- Garante dimensões finais precisas
- Usa o valor de kerf das configurações da sua cabeça de laser

## Opções de Pós-Processamento

![Configurações de pós-processamento de Contorno de Moldura](/screenshots/step-settings-frame-outline-post.png)

### Multi-Passagem

Corta a moldura múltiplas vezes:

- **Passagens:** Número de repetições
- **Degrau Z:** Baixa Z entre passagens (requer eixo Z)
- Útil para materiais espessos

### Abas de Fixação

Adiciona abas para manter a peça emoldurada anexada:

- Previne peças de caírem durante o corte
- Configure largura, altura e espaçamento das abas
- Veja [Abas de Fixação](../holding-tabs) para detalhes

## Casos de Uso

### Borda Decorativa

**Cenário:** Adicionar uma borda retangular limpa ao redor de uma placa ou sinal

**Processo:**

1. Design seu conteúdo (texto, logos, etc.)
2. Adicione Contorno de Moldura com deslocamento de 3-5mm
3. Corte em configurações de marcação decorativa (baixa potência)

**Resultado:** Peça emoldurada de aparência profissional

### Cortar Livre da Chapa

**Cenário:** Remover seu trabalho finalizado da chapa de material

**Processo:**

1. Complete todas as outras operações (grave, cortes de contorno)
2. Adicione Contorno de Moldura como última operação
3. Defina deslocamento para incluir uma pequena margem

**Benefícios:**

- Separação limpa da chapa
- Qualidade de borda consistente
- Fácil de executar como passo final

### Limite de Processamento em Lote

**Cenário:** Criar um limite de corte para múltiplas peças aninhadas

**Processo:**

1. Organize todas as peças na tela
2. Adicione operações de contorno individuais para peças
3. Adicione Contorno de Moldura ao redor de tudo
4. Moldura corta por último (em camada separada)

**Ordem:** Grave → Contornos de peças → Contorno de moldura

## Dicas e Melhores Práticas

### Ordem de Camadas

**Melhor prática:**

- Coloque Contorno de Moldura em sua própria camada
- Execute moldura como a **última** operação
- Isso garante que todo outro trabalho completa primeiro

**Por que por último?**

- Material permanece fixado durante outras operações
- Previne peças de deslocarem
- Resultado final mais limpo

### Seleção de Deslocamento

**Escolhendo deslocamento:**

- **0-2mm:** Encaixe apertado, desperdício de material mínimo
- **3-5mm:** Margem padrão, parece profissional
- **10mm+:** Material extra para manuseio/montagem

**Considere:**

- Uso final da peça
- Se bordas serão visíveis
- Custo e disponibilidade do material

### Configurações de Qualidade

**Para cortes de moldura limpos:**

- Use assistência de ar
- Certifique-se de foco adequado
- Múltiplas passagens mais rápidas frequentemente melhor que uma passagem lenta
- Mantenha material plano e fixado

## Combinando com Outras Operações

### Moldura + Gravação + Contorno

Fluxo de trabalho típico para uma peça finalizada:

1. **Camada 1:** Grave detalhes (texto, imagens)
2. **Camada 2:** Contorno corta peças individuais
3. **Camada 3:** Contorno de moldura (corta livre)

**Ordem de execução garante:**

- Gravação acontece enquanto material está plano e fixado
- Detalhes das peças são cortados antes da separação final
- Moldura corta tudo livre no final

### Moldura vs Shrink Wrap

| Recurso         | Contorno de Moldura                | Shrink Wrap             |
| --------------- | ---------------------------- | ----------------------- |
| **Forma**       | Sempre retangular           | Segue contornos do objeto |
| **Velocidade**       | Muito rápido (4 linhas)          | Depende da complexidade   |
| **Caso de uso**    | Bordas simples, cortar livre | Uso eficiente de material  |
| **Flexibilidade** | Retângulo fixo              | Adapta ao design        |

**Escolha Contorno de Moldura quando:**

- Você quer uma borda retangular
- Simplicidade é preferida
- Cortar livre da chapa

**Escolha Shrink Wrap quando:**

- Você quer minimizar desperdício de material
- Design tem forma irregular
- Eficiência é importante

## Solução de Problemas

### Moldura muito apertada/solta

- **Ajuste:** Configuração de distância de deslocamento
- **Verifique:** Deslocamento de caminho (interno/externo/linha central)
- **Verifique:** Compensação de kerf está correta

### Moldura não aparece

- **Verifique:** Objetos estão na tela
- **Verifique:** Operação está habilitada
- **Olhe:** Moldura pode estar fora da área visível (diminua zoom)

### Moldura corta no design

- **Aumente:** Distância de deslocamento
- **Verifique:** Objetos estão propriamente posicionados
- **Verifique:** Cálculo da caixa delimitadora inclui todos os objetos

### Profundidade de corte inconsistente

- **Verifique:** Material está plano
- **Verifique:** Distância do foco está correta
- **Tente:** Múltiplas passagens em potência menor

## Detalhes Técnicos

### Cálculo da Caixa Delimitadora

O contorno de moldura usa a caixa delimitadora combinada de:

- Todas as peças na tela
- Suas posições transformadas finais
- Incluindo quaisquer rotações/escalonamentos aplicados

### Geração de Caminho

1. Calcula caixa delimitadora combinada
2. Aplica distância de deslocamento
3. Aplica deslocamento de caminho (interno/externo/linha central)
4. Aplica compensação de kerf
5. Gera caminho G-code retangular

### Exemplo de G-code

```gcode
G0 X5 Y5           ; Move para início da moldura (com deslocamento)
M3 S200            ; Laser ligado a 80% de potência
G1 X95 Y5 F500     ; Corta borda inferior
G1 X95 Y95         ; Corta borda direita
G1 X5 Y95          ; Corta borda superior
G1 X5 Y5           ; Corta borda esquerda (completa)
M5                 ; Laser desligado
```

## Tópicos Relacionados

- **[Corte de Contorno](contour)** - Cortando contornos de objetos individuais
- **[Shrink Wrap](shrink-wrap)** - Limites irregulares eficientes
- **[Abas de Fixação](../holding-tabs)** - Mantendo peças seguras durante o corte
- **[Fluxo de Trabalho Multi-Camadas](../multi-layer)** - Organizando operações efetivamente
- **[Compensação de Kerf](../kerf)** - Melhorando precisão dimensional
