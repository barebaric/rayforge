# Fluxo de Trabalho de Manuseio de Material

O manuseio de material no Rayforge é um processo sequencial que permite definir o material físico com o qual você vai trabalhar, atribuir propriedades a ele e depois organizar seus elementos de design nele. Este guia percorre o fluxo de trabalho completo desde adicionar material até o layout automático do seu design.

## 1. Adicionando Material

O material representa a peça física de material que você vai cortar ou gravar. Para adicionar material ao seu documento:

1. No painel **Material de Base** na barra lateral, clique no botão **Adicionar Material**
2. Um novo item de material será criado com dimensões padrão (80% da área de trabalho da sua máquina)
3. O material aparecerá como um retângulo no espaço de trabalho, centralizado na mesa da máquina

### Propriedades do Material

Cada item de material tem as seguintes propriedades:
- **Nome**: Um nome descritivo para identificação (numerado automaticamente como "Material 1", "Material 2", etc.)
- **Dimensões**: Largura e altura do material base
- **Espessura**: A espessura do material (opcional mas recomendado para visualização 3D precisa)
- **Material**: O tipo de material (atribuído no próximo passo)
- **Visibilidade**: Alternar para mostrar/ocultar o material no espaço de trabalho

### Gerenciando Itens de Material

- **Renomear**: Abra o diálogo de Propriedades do Material e edite o campo de nome
- **Redimensionar**: Selecione o item de material no espaço de trabalho e arraste as alças de canto para redimensionar
- **Mover**: Selecione o item de material no espaço de trabalho e arraste para reposicionar
- **Excluir**: Clique no botão excluir (ícone de lixeira) ao lado do item de material no painel Material de Base
- **Editar propriedades**: Clique no botão de propriedades (ícone de documento) para abrir o diálogo de Propriedades do Material
- **Alternar visibilidade**: Clique no botão de visibilidade (ícone de olho) para mostrar/ocultar o item de material

## 2. Atribuindo Material

Uma vez que você tem o material base definido, pode atribuir um material a ele:

1. No painel **Material de Base**, clique no botão de propriedades (ícone de documento) no item de material
2. No diálogo de Propriedades do Material, clique no botão **Selecionar** ao lado do campo Material
3. Navegue pelas suas bibliotecas de materiais e selecione o material apropriado
4. O material base será atualizado para mostrar a aparência visual do material

### Propriedades do Material

Materiais definem as propriedades visuais do seu material base:
- **Aparência visual**: Cor e padrão para visualização
- **Categoria**: Agrupamento (ex., "Madeira", "Acrílico", "Metal")
- **Descrição**: Informações adicionais sobre o material

Nota: As propriedades do material são definidas em bibliotecas de materiais e não podem ser editadas através do diálogo de propriedades do material base. As propriedades do material base apenas permitem atribuir um material a um item de material.

## 3. Atribuindo Material a Camadas

Após definir seu material base e atribuir materiais, você pode associar camadas a itens de material específicos:

1. No painel **Camadas**, localize a camada que deseja atribuir ao material
2. Clique no botão de atribuição de material (mostra "Superfície Inteira" por padrão)
3. No menu suspenso, selecione o item de material que deseja associar a esta camada
4. O conteúdo dessa camada agora será restrito aos limites do material atribuído

Você também pode escolher "Superfície Inteira" para usar todo o espaço de trabalho da máquina em vez de um item de material específico.

### Por Que Atribuir Material a Camadas?

- **Limites de layout**: Fornece limites para o algoritmo de layout automático trabalhar dentro
- **Organização visual**: Ajuda a organizar seu design associando camadas com materiais físicos
- **Visualização de material**: Mostra a aparência visual do material atribuído no material base

## 4. Layout Automático

O recurso de layout automático ajuda a organizar eficientemente seus elementos de design:

1. Selecione os itens que deseja organizar (ou deixe nada selecionado para organizar todos os itens na camada ativa)
2. Clique no botão **Organizar** na barra de ferramentas e selecione **Layout Automático (empacotar peças)**
3. O Rayforge organizará automaticamente os itens para otimizar o uso do material

### Comportamento do Layout Automático

O algoritmo de layout automático funciona de forma diferente dependendo da sua configuração de camada:

- **Se um item de material está atribuído à camada**: Os itens são organizados dentro dos limites daquele item de material específico
- **Se "Superfície Inteira" está selecionado**: Os itens são organizados em todo o espaço de trabalho da máquina

O algoritmo considera:
- **Limites dos itens**: Respeita as dimensões de cada elemento de design
- **Rotação**: Pode rotacionar itens em incrementos de 90 graus para melhor encaixe
- **Espaçamento**: Mantém uma margem entre itens (padrão 0.5mm)
- **Limites do material**: Mantém todos os itens dentro dos limites definidos

### Alternativas de Layout Manual

Se você prefere mais controle, o Rayforge também oferece ferramentas de layout manual:
- **Ferramentas de alinhamento**: Alinhar esquerda, direita, centro, topo, inferior
- **Ferramentas de distribuição**: Distribuir itens horizontal ou verticalmente
- **Posicionamento individual**: Clique e arraste itens para colocá-los manualmente

## Dicas para Manuseio Eficaz de Material

1. **Comece com dimensões precisas do material** - Meça seu material precisamente para melhores resultados
2. **Use nomes descritivos** - Nomeie seus itens de material claramente (ex., "Madeira Bétula 3mm")
3. **Defina a espessura do material** - Isso pode ser útil para cálculos futuros e referência
4. **Atribua materiais cedo** - Isso garante representação visual adequada desde o início
5. **Use camadas para organização** - Separe diferentes partes do seu design em camadas antes de atribuir ao material
6. **Verifique o encaixe antes de cortar** - Use a visualização 2D para verificar se tudo cabe no seu material base

## Solução de Problemas

### Layout automático não funciona como esperado
- Verifique se sua camada tem um material atribuído
- Certifique-se de que os itens não estão agrupados (desagrupe-os primeiro)
- Tente reduzir o número de itens selecionados de uma vez
- Verifique se os itens cabem dentro dos limites (material ou superfície inteira)
