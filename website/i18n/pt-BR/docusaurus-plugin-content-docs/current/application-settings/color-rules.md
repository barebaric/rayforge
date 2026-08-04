# Regras de Cor

As regras de cor permitem atribuir um tipo de etapa a uma cor específica para
que a operação correta seja escolhida automaticamente ao importar um SVG, PDF
ou outro arquivo vetorial. Em vez de criar etapas manualmente para cada camada
importada, o Rayforge lê a cor de cada forma e aplica a regra correspondente.

## Como Funciona

Ao importar um arquivo vetorial, o Rayforge pode agrupar as formas recebidas
pela sua cor. Cada cor distinta se torna uma camada. Se existir uma regra de
cor para aquela cor, a camada recebe o tipo de etapa da regra automaticamente.
Cores sem regra recebem o comportamento padrão (Contorno para contornos, além
de Gravação se as formas tiverem preenchimentos).

Depois que o tipo de etapa é atribuído, o sistema normal de
[correspondência de receitas](recipes) atua por cima — então as regras de cor
determinam *qual* operação é executada, e as receitas determinam *como* ela é
executada (potência, velocidade, passadas, etc.).

## Criando Regras de Cor

### 1. Abra a Página de Regras de Cor

Menu: **Editar → Configurações**, depois selecione **Regras de Cor** na barra
lateral.

### 2. Adicione uma Regra

Clique em **Adicionar Regra de Cor** para abrir o diálogo do editor:

- **Cor** — Escolha a cor SVG que deve acionar esta regra. Use o seletor de
  cores para combinar com a cor do traço ou do preenchimento do seu software
  de design.
- **Rótulo** *(opcional)* — Um nome amigável exibido na lista de regras (por
  exemplo, "Cortar Vermelho", "Gravar Azul"). Se deixado em branco, o valor
  hexadecimal é usado.
- **Tipo de Etapa** — A operação a ser criada quando esta cor é importada.
  Qualquer tipo de etapa registrado está disponível, incluindo os fornecidos
  por [addons](addons) (por exemplo, Shrink Wrap, Material Test Grid).

### 3. Salve

Clique em **Adicionar** para salvar a regra. Ela entra em vigor imediatamente
na próxima importação. As regras são armazenadas na sua configuração de
usuário e persistem entre sessões.

:::tip Combinando Cores Exatamente
As regras de cor combinam pelo valor hexadecimal exato. Ao escolher uma cor no
seu software de design (Inkscape, Illustrator, etc.), anote o código
hexadecimal exato e digite o mesmo valor no Rayforge. Por exemplo, `#e34c4c`
no seu SVG deve ser `#e34c4c` na regra — mesmo uma diferença de um dígito
impedirá a correspondência.
:::

## Gerenciando Regras

Cada regra na lista mostra uma amostra de cor, o rótulo, o tipo de etapa e
botões de editar/excluir.

- **Editar** — Altere a cor, o rótulo ou o tipo de etapa. Alterar a cor de uma
  regra existente a substitui (a cor antiga é removida).
- **Excluir** — Remove a regra permanentemente.
- **Tipos de etapa indisponíveis** — Se o addon do tipo de etapa foi
  desinstalado, um ícone de aviso aparece ao lado da regra. A regra é
  preservada para que você possa corrigi-la ou reinstalar o addon. Durante a
  importação, camadas que correspondem a uma regra com um tipo de etapa
  indisponível voltam ao comportamento padrão.

## Comportamento de Importação

### Agrupamento Automático de Cores

Quando existem regras de cor, o diálogo de importação muda automaticamente
para **Cores** como a fonte de camadas para arquivos que contêm cores
distintas. Isso garante que cada cor se torne sua própria camada para que as
regras possam ser aplicadas. Você ainda pode voltar para **Camadas SVG** ou
outras fontes no diálogo, se preferir.

### O Que Aciona uma Regra

Uma regra de cor é aplicada quando:

1. O arquivo é importado com **Cores** como a fonte de camadas.
2. A cor do traço ou do preenchimento de uma forma corresponde exatamente à
   cor da regra.
3. O tipo de etapa da regra está registrado atualmente.

As regras **não** se aplicam a arquivos importados com as fontes de camadas
**Camadas SVG** ou **Achatar**, porque essas fontes não agrupam por cor.

## Fluxo de Trabalho de Exemplo

Uma configuração comum para designs SVG multicoloridos:

1. **No seu software de design**, atribua cores distintas a diferentes
   operações:
   - Vermelho (`#ff0000`) para contornos de corte
   - Azul (`#0000ff`) para gravação
   - Verde (`#00ff00`) para escarificação

2. **No Rayforge**, crie três regras de cor:
   - `#ff0000` → Contorno
   - `#0000ff` → Gravação
   - `#00ff00` → Contorno (com configurações de receita diferentes)

3. **Importe o SVG.** O diálogo de importação seleciona automaticamente Cores,
   e cada grupo de cores recebe seu tipo de etapa automaticamente.

4. **Ajuste fino** com [receitas](recipes) para definir potência, velocidade e
   outros parâmetros por tipo de etapa.

## Regras de Cor e Receitas

As regras de cor e as receitas são complementares:

| Recurso       | O que define                            | Quando se aplica    |
| ------------- | --------------------------------------- | ------------------- |
| Regras de Cor | Tipo de etapa (Contorno, etc.)          | Na importação       |
| Receitas      | Configurações de etapa (potência, etc.) | Na criação da etapa |

Uma configuração típica é usar regras de cor para escolher a operação e
receitas para configurar os parâmetros. Por exemplo, uma regra de cor vermelha
corresponde a Contorno, e uma receita limitada ao tipo de etapa Contorno no seu
material atual aplica a velocidade e a potência de corte corretas.

---

**Tópicos Relacionados**:

- [Receitas](recipes) - Aplicar predefinições de potência, velocidade e
  parâmetros
- [Importando Arquivos](../files/importing.md) - Opções de importação SVG e
  vetorial
- [Fluxo de Trabalho Multi-Camadas](../features/multi-layer.md) - Organização
  de camadas
- [Operações](../features/operations/contour.md) - Referência de tipos de etapa
