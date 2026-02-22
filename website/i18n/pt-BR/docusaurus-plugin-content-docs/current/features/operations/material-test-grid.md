# Grade de Teste de Material

O gerador de Grade de Teste de Material cria padrões de teste paramétricos para ajudá-lo a encontrar configurações ideais de laser para diferentes materiais.

## Visão Geral

Teste de material é essencial para trabalho a laser - diferentes materiais requerem diferentes configurações de potência e velocidade. A Grade de Teste de Material automatiza este processo:

- Gerando grades de teste com faixas de velocidade/potência configuráveis
- Fornecendo predefinições para tipos comuns de laser (Diodo, CO2)
- Otimizando ordem de execução para segurança (velocidades mais rápidas primeiro)
- Adicionando rótulos para identificar as configurações de cada célula de teste

## Criando uma Grade de Teste de Material

### Passo 1: Abrir o Gerador

Acesse o gerador de Grade de Teste de Material:

- Menu: **Ferramentas → Grade de Teste de Material**
- Isso cria uma peça especial que gera o padrão de teste

### Passo 2: Escolher uma Predefinição (Opcional)

O Rayforge inclui predefinições para cenários comuns:

| Predefinição            | Faixa de Velocidade       | Faixa de Potência | Usar Para               |
| ----------------- | ----------------- | ----------- | --------------------- |
| **Gravação Diodo** | 1000-10000 mm/min | 10-100%     | Gravação com laser de diodo |
| **Corte Diodo**     | 100-5000 mm/min   | 50-100%     | Corte com laser de diodo   |
| **Gravação CO2**   | 3000-20000 mm/min | 10-50%      | Gravação com laser CO2   |
| **Corte CO2**       | 1000-20000 mm/min | 30-100%     | Corte com laser CO2     |

Predefinições são pontos de partida - você pode ajustar todos os parâmetros após selecionar uma.

### Passo 3: Configurar Parâmetros

Ajuste os parâmetros da grade de teste no diálogo de configurações:

![Configurações da Grade de Teste de Material](/screenshots/material-test-grid.png)

#### Tipo de Teste

- **Gravação**: Preenche quadrados com padrão raster
- **Corte**: Corta contorno dos quadrados

#### Faixa de Velocidade

- **Velocidade Mín**: Velocidade mais lenta para testar (mm/min)
- **Velocidade Máx**: Velocidade mais rápida para testar (mm/min)
- Colunas na grade representam velocidades diferentes

#### Faixa de Potência

- **Potência Mín**: Potência mais baixa para testar (%)
- **Potência Máx**: Potência mais alta para testar (%)
- Linhas na grade representam níveis de potência diferentes

#### Dimensões da Grade

- **Colunas**: Número de variações de velocidade (tipicamente 3-7)
- **Linhas**: Número de variações de potência (tipicamente 3-7)

#### Tamanho e Espaçamento

- **Tamanho da Forma**: Tamanho de cada quadrado de teste em mm (padrão: 20mm)
- **Espaçamento**: Lacuna entre quadrados em mm (padrão: 5mm)

#### Rótulos

- **Incluir Rótulos**: Habilitar/desabilitar rótulos de eixo mostrando valores de velocidade e potência
- Rótulos aparecem nas bordas esquerda e superior
- Rótulos são gravados a 10% de potência, 1000 mm/min

### Passo 4: Gerar a Grade

Clique em **Gerar** para criar o padrão de teste. A grade aparece na sua tela como uma peça especial.

## Entendendo o Layout da Grade

### Organização da Grade

```
Potência (%)     Velocidade (mm/min) →
    ↓      1000   2500   5000   7500   10000
  100%     [  ]   [  ]   [  ]   [  ]   [  ]
   75%     [  ]   [  ]   [  ]   [  ]   [  ]
   50%     [  ]   [  ]   [  ]   [  ]   [  ]
   25%     [  ]   [  ]   [  ]   [  ]   [  ]
   10%     [  ]   [  ]   [  ]   [  ]   [  ]
```

- **Colunas**: Velocidade aumenta da esquerda para direita
- **Linhas**: Potência aumenta de baixo para cima
- **Rótulos**: Mostram valores exatos para cada linha/coluna

### Cálculo do Tamanho da Grade

**Sem rótulos:**

- Largura = colunas × (tamanho_forma + espaçamento) - espaçamento
- Altura = linhas × (tamanho_forma + espaçamento) - espaçamento

**Com rótulos:**

- Adiciona margem de 15mm à esquerda e superior para espaço de rótulos

**Exemplo:** Grade 5×5 com quadrados de 20mm e espaçamento de 5mm:

- Sem rótulos: 120mm × 120mm
- Com rótulos: 135mm × 135mm

## Ordem de Execução (Otimização de Risco)

O Rayforge executa células de teste em uma **ordem otimizada por risco** para prevenir dano ao material:

1. **Velocidade mais alta primeiro**: Velocidades rápidas são mais seguras (menor acúmulo de calor)
2. **Potência mais baixa dentro da velocidade**: Minimiza risco em cada nível de velocidade

Isso previne carbonização ou fogo de começar com combinações lentas e de alta potência.

**Exemplo de ordem de execução para grade 3×3:**

```
Ordem:  1  2  3
        4  5  6  ← Velocidade mais alta, potência aumentando
        7  8  9

(Velocidade mais rápida/potência mais baixa executado primeiro)
```

## Usando Resultados de Teste de Material

### Passo 1: Executar o Teste

1. Carregue seu material no laser
2. Focalize o laser propriamente
3. Execute o trabalho de grade de teste de material
4. Monitore o teste - pare se alguma célula causar problemas

### Passo 2: Avaliar Resultados

Após o teste completar, examine cada célula:

- **Muito claro**: Aumente potência ou diminua velocidade
- **Muito escuro/carbonizado**: Diminua potência ou aumente velocidade
- **Perfeito**: Anote a combinação de velocidade/potência

### Passo 3: Registrar Configurações

Documente suas configurações bem-sucedidas para referência futura:

- Tipo e espessura do material
- Tipo de operação (grave ou corte)
- Combinação de velocidade e potência
- Número de passagens
- Quaisquer notas especiais

:::tip Banco de Dados de Materiais
Considere criar um documento de referência com seus resultados de teste de material para consulta rápida em projetos futuros.
:::

## Uso Avançado

### Combinando com Outras Operações

Grades de teste de material são peças regulares - você pode combiná-las com outras operações:

**Exemplo de fluxo de trabalho:**

1. Crie grade de teste de material
2. Adicione corte de contorno ao redor de toda a grade
3. Execute teste, corte livre, avalie resultados

Isso é útil para cortar a peça de teste livre do material de estoque.

### Faixas de Teste Personalizadas

Para ajuste fino, crie testes de faixa estreita:

**Teste grosseiro** (encontre faixa aproximada):

- Velocidade: 1000-10000 mm/min (5 colunas)
- Potência: 10-100% (5 linhas)

**Teste fino** (otimize):

- Velocidade: 4000-6000 mm/min (5 colunas)
- Potência: 35-45% (5 linhas)

### Diferentes Materiais, Mesma Grade

Execute a mesma configuração de grade em materiais diferentes para construir sua biblioteca de materiais mais rápido.

## Dicas e Melhores Práticas

### Design da Grade

✅ **Comece com predefinições** - Bons pontos de partida para cenários comuns
✅ **Use grades 5×5** - Bom equilíbrio de detalhe e tempo de teste
✅ **Habilite rótulos** - Essencial para identificar resultados
✅ **Mantenha quadrados ≥20mm** - Mais fácil de ver e medir resultados

### Estratégia de Teste

✅ **Teste sucata primeiro** - Nunca teste em material final
✅ **Uma variável de cada vez** - Teste faixa de velocidade OU potência, não ambos extremos
✅ **Permita resfriamento** - Espere entre testes no mesmo material
✅ **Foco consistente** - Mesma distância de foco para todos os testes

### Segurança

⚠️ **Monitore testes** - Nunca deixe testes rodando sem supervisão
⚠️ **Comece conservador** - Comece com faixas de potência mais baixas
⚠️ **Verifique ventilação** - Certifique-se de extração de fumaça adequada
⚠️ **Vigia de incêndio** - Tenha extintor de incêndio pronto

## Solução de Problemas

### Células de teste executam em ordem errada

- O Rayforge usa ordem otimizada por risco (velocidades mais rápidas primeiro)
- Isso é intencional e não pode ser alterado
- Veja [Ordem de Execução](#ordem-de-execucao-otimização-de-risco) acima

### Resultados são inconsistentes

- **Verifique**: Material está plano e propriamente fixado
- **Verifique**: Foco é consistente através de toda área de teste
- **Verifique**: Potência do laser está estável (verifique fonte de alimentação)
- **Tente**: Grade menor para reduzir área de teste

## Tópicos Relacionados

- **[Modo Simulação](../simulation-mode)** - Pré-visualize execução do teste antes de rodar
- **[Gravação](engrave)** - Entendendo operações de gravação
- **[Corte de Contorno](contour)** - Entendendo operações de corte
