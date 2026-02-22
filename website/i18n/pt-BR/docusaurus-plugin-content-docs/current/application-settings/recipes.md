# Receitas e Configurações

![Configurações de Receitas](/screenshots/application-recipes.png)

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

Menu: Editar → Preferências → Receitas

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

- **Tipo de Tarefa**: Selecione o tipo de operação (Corte, Gravação, etc.)
- **Máquina**: Escolha uma máquina específica ou deixe como "Qualquer Máquina"
- **Material**: Selecione um tipo de material ou deixe aberto para qualquer material
- **Intervalo de Espessura**: Defina valores mínimo e máximo de espessura

#### 4. Configure as Definições

**Aba Configurações** - Ajuste potência, velocidade e outros parâmetros:

![Editor de Receitas - Aba Configurações](/screenshots/recipe-editor-settings.png)

- Ajuste potência, velocidade e outros parâmetros
- As configurações se adaptam automaticamente com base no tipo de tarefa selecionado

### Sistema de Correspondência de Receitas

O Rayforge sugere automaticamente as receitas mais apropriadas com base em:

- **Compatibilidade de máquina**: Receitas podem ser específicas para máquina
- **Correspondência de material**: Receitas podem direcionar materiais específicos
- **Intervalos de espessura**: Receitas se aplicam dentro dos limites de espessura definidos
- **Correspondência de capacidade**: Receitas estão vinculadas a tipos de operação específicos

O sistema usa um algoritmo de pontuação de especificidade para priorizar as receitas mais relevantes:

1. Receitas específicas de máquina têm classificação mais alta que genéricas
2. Receitas específicas de cabeça de laser têm classificação mais alta
3. Receitas específicas de material têm classificação mais alta
4. Receitas específicas de espessura têm classificação mais alta

---

**Tópicos Relacionados**:

- [Materiais](materials) - Gerenciando propriedades de materiais
- [Manuseio de Material](../features/stock-handling) - Trabalhando com materiais base
- [Configuração de Máquina](../machine/general) - Configurando máquinas e cabeças de laser
- [Visão Geral de Operações](../features/operations/contour) - Entendendo diferentes tipos de operação
