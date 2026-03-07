# Manuseio de Material

Material no Rayforge representa o material físico que você vai cortar ou gravar. O material é um conceito **global do documento**—seu documento pode ter um ou mais itens de material, e eles existem independentemente de camadas.

## Adicionando Material

O material representa a peça física de material com a qual você vai trabalhar. Para adicionar material ao seu documento:

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

## Atribuindo Material

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

## Convertendo Peças em Material

Você pode converter qualquer peça em um item de material. Isso é útil quando você tem uma peça de material com formato irregular e quer usar seu contorno exato como limite do material.

Para converter uma peça em material:

1. Clique com o botão direito na peça na tela ou no painel Documento
2. Selecione **Converter para Material** no menu de contexto
3. A peça será substituída por um novo item de material com o mesmo formato e posição

O novo item de material:

- Usa a geometria da peça como seu limite
- Herda o nome da peça
- Pode ter um material atribuído como qualquer outro item de material

## Layout Automático

O recurso de layout automático ajuda a organizar eficientemente seus elementos de design dentro dos limites do material:

1. Selecione os itens que deseja organizar (ou deixe nada selecionado para organizar todos os itens na camada ativa)
2. Clique no botão **Organizar** na barra de ferramentas e selecione **Layout Automático (empacotar peças)**
3. O Rayforge organizará automaticamente os itens para otimizar o uso do material

### Comportamento do Layout Automático

O algoritmo de layout automático organiza itens dentro dos itens de material visíveis no seu documento:

- **Se itens de material estão definidos**: Os itens são organizados dentro dos limites dos itens de material visíveis
- **Se nenhum material está definido**: Os itens são organizados em todo o espaço de trabalho da máquina

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
5. **Use material irregular para sobras** - Converta peças em material quando usar material sobrante com formatos personalizados
6. **Verifique o encaixe antes de cortar** - Use a visualização 2D para verificar se tudo cabe no seu material base

## Solução de Problemas

### Layout automático não funciona como esperado

- Certifique-se de que pelo menos um item de material está visível
- Certifique-se de que os itens não estão agrupados (desagrupe-os primeiro)
- Tente reduzir o número de itens selecionados de uma vez
- Verifique se os itens cabem dentro dos limites do material
