# Receitas e Configurações

![Configurações de Receitas](/screenshots/app-settings-recipes.png)

O Rayforge fornece um poderoso sistema de receitas que permite criar,
gerenciar e aplicar configurações consistentes em seus projetos de corte a laser.
Este guia cobre a jornada completa do usuário desde criar receitas nas
configurações gerais até aplicá-las em operações e gerenciar configurações no
nível de etapa.

## Visão Geral

O sistema de receitas consiste em três componentes principais:

1. **Gerenciamento de Receitas**: Cria e gerencia predefinições de configurações reutilizáveis
2. **Gerenciamento de Material Base**: Define propriedades e espessura do material
3. **Configurações de Etapa**: Aplica e ajusta configurações para operações individuais

## Gerenciamento de Receitas

### Criando Receitas

Receitas são predefinições nomeadas que contêm todas as configurações necessárias para operações específicas.
Você pode criar receitas através da interface principal de configurações:

#### 1. Acesse o Gerenciador de Receitas

Menu: Editar → Configurações, depois selecione Receitas

#### 2. Crie uma Nova Receita

Clique em "Adicionar Nova Receita" para abrir o diálogo do editor de receitas.

**Aba Geral** - Defina o nome e descrição da receita:

![Editor de Receitas - Aba Geral](/screenshots/recipe-editor-general.png)

Preencha as informações básicas:

- **Nome**: Nome descritivo (ex., "Corte Compensado 3mm")
- **Descrição**: Descrição detalhada opcional

#### 3. Defina Critérios de Aplicabilidade

**Aba Aplicabilidade** - Defina quando esta receita deve ser sugerida:

![Editor de Receitas - Aba Aplicabilidade](/screenshots/recipe-editor-applicability.png)

Todos os critérios são opcionais - deixe qualquer campo em seu valor "Qualquer"
para corresponder a tudo:

- **Máquina**: Escolha uma máquina específica ou deixe como "Qualquer"
- **Tipo de Tarefa**: Selecione a categoria de operação à qual esta receita se
  aplica (Corte, Gravação, etc.), ou deixe como "Qualquer" para aplicar a todos
  os tipos de tarefa
- **Tipo de Etapa**: Restrinja a receita a um tipo de operação específico
  (ex. "Contorno" ou "Raster"). A lista é filtrada pelos tipos de etapa que
  suportam o tipo de tarefa selecionado. Deixe como "Qualquer Tipo" para
  corresponder a cada tipo de etapa dentro da tarefa
- **Material**: Selecione um tipo de material ou deixe aberto para qualquer material
- **Espessura Mín/Máx**: Defina valores mínimo e máximo de espessura do material base

#### 4. Configure as Definições

**Aba Configurações** - Ajuste potência, velocidade e outros parâmetros.
Quando a receita tem como alvo um **tipo de etapa** específico, o editor
mostra duas páginas de configurações: uma página "Laser" com as
configurações de processo compartilhadas (potência, assistência de ar, etc.)
e uma página "Configurações de Etapa" com os atributos específicos daquele
tipo de etapa (ex. lado de corte, ordem de corte):

![Editor de Receitas - Aba Laser](/screenshots/recipe-editor-laser.png)

![Editor de Receitas - Aba Configurações de Etapa](/screenshots/recipe-editor-step-settings.png)

- Selecionar apenas um **tipo de tarefa** (com "Qualquer Tipo" como tipo de
  etapa) mostra uma única página "Configurações" com as configurações de
  processo para aquela tarefa
- Deixar ambos como "Qualquer" mostra apenas as configurações básicas de
  movimento (velocidade de corte e velocidade de deslocamento) que são
  compartilhadas por todas as etapas

**Aba Pós-Processamento** - Armazene configurações de pós-processamento
(entrada/saída, multipasse, overscan e outros transformadores) na receita
para que sejam aplicadas às etapas que ela tem como alvo:

![Editor de Receitas - Aba Pós-Processamento](/screenshots/recipe-editor-post-processing.png)

Cada transformador é mostrado com um botão de três estados:

- **Deixar inalterado**: a receita não toca neste transformador ao ser
  aplicada
- **Habilitado**: a receita ativa o transformador e aplica seus parâmetros
  à etapa
- **Desabilitado**: a receita desativa explicitamente o transformador

Quando a receita tem como alvo vários tipos de etapa, apenas os
transformadores comuns a todos são mostrados.

### Sistema de Correspondência de Receitas

O Rayforge sugere e aplica automaticamente as receitas mais apropriadas com
base em:

- **Compatibilidade de máquina**: Receitas podem ser específicas para máquina
- **Compatibilidade de cabeça de laser**: Receitas podem forçar uma cabeça específica na
  máquina
- **Correspondência de material**: Receitas podem direcionar materiais específicos
- **Intervalos de espessura**: Receitas se aplicam dentro dos limites de espessura definidos
- **Correspondência de tipo de tarefa**: Receitas estão vinculadas a categorias
  de operação específicas
- **Correspondência de tipo de etapa**: Receitas podem ter como alvo um tipo de
  operação específico (ex. apenas etapas "Contorno")

Uma receita só corresponde quando todos os seus critérios são satisfeitos. Quando
uma nova etapa é criada, o Rayforge pesquisa a biblioteca de receitas em busca de
receitas correspondentes e aplica automaticamente a melhor. O sistema usa um
algoritmo de pontuação de especificidade para priorizar as receitas mais relevantes:

1. Receitas específicas de máquina têm classificação mais alta que genéricas
2. Receitas específicas de cabeça de laser têm classificação mais alta
3. Receitas específicas de material têm classificação mais alta
4. Receitas específicas de espessura têm classificação mais alta
5. Receitas específicas de tipo de etapa têm classificação mais alta

### Aplicando Receitas a Etapas

As receitas são aplicadas por etapa. Abra as configurações de qualquer etapa e
encontre a linha "Receita" na seção "Geral":

- **Escolher...**: Abre uma lista filtrável de receitas. Use o campo de busca
  ou o alternador "Mostrar apenas receitas compatíveis" para estreitar a lista;
  receitas compatíveis correspondem ao tipo de tarefa, tipo de etapa, máquina e
  aos materiais base da etapa no documento. Selecionar uma receita aplica todas
  as suas configurações à etapa.
- **Salvar Como...**: Abre o editor de receitas pré-preenchido com as
  configurações, máquina, material e espessura atuais da etapa. Salvar a nova
  receita a aplica à etapa imediatamente.
- **Atualizar**: Aparece quando as configurações da etapa divergiram da receita
  que foi aplicada a ela (ex. depois que você alterou um valor
  manualmente). Clicar nele sobrescreve a receita salva com as configurações
  atuais da etapa.

O nome da receita atualmente aplicada é mostrado na linha. Etapas
sem uma receita aplicada são rotuladas como "Configurações Manuais".

---

**Tópicos Relacionados**:

- [Materiais](materials) - Gerenciando propriedades de materiais
- [Manuseio de Material](../features/stock-handling.md) - Trabalhando com materiais base
- [Configuração de Máquina](../machine/general.md) - Configurando máquinas e cabeças de laser
- [Visão Geral de Operações](../features/operations/contour.md) - Entendendo diferentes tipos de operação
- [Regras de Cor](color-rules) - Mapeia cores SVG para tipos de etapa na importação
