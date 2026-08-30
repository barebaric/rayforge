# Materiais

![Configurações de Materiais](/screenshots/app-settings-materials.webp)

Bibliotecas de materiais no Rayforge permitem organizar e gerenciar coleções de materiais para seus
projetos de corte e gravação a laser. Este guia explica a diferença entre bibliotecas do sistema e
do usuário, e como criar suas próprias bibliotecas e adicionar materiais a elas.

:::note Atribuir um material a um item de material afeta tanto sua aparência visual no canvas 2D e
3D quanto quais [receitas](recipes.md) se aplicam a ele: receitas específicas de material
correspondem ao material atribuído. Em versões futuras, materiais serão usados para derivar mais
parâmetros funcionais. :::

## Criando uma Nova Biblioteca

Para criar sua própria biblioteca de materiais:

1. Abra o menu **Configurações** e selecione **Materiais**
2. Clique no botão **Adicionar Nova Biblioteca** para criar uma nova biblioteca
3. Digite um nome descritivo para sua biblioteca (ex., "Materiais da Minha Oficina")
4. Clique em **Criar** para finalizar

Sua nova biblioteca será criada no diretório de dados do usuário e estará disponível imediatamente.

## Adicionando Materiais às Bibliotecas

### Criando um Novo Material

1. Selecione a biblioteca onde deseja adicionar o material
2. Clique no botão **Adicionar Novo Material** na lista de materiais
3. Preencha as propriedades do material:
   - **Nome**: Nome legível para humanos
   - **Categoria**: Categoria de agrupamento (ex., "Madeira", "Acrílico")
   - **Aparência**: Propriedades visuais (veja abaixo)
4. Clique em **Salvar** para adicionar o material à biblioteca

### Propriedades do Material Explicadas

#### Nome

- Nome legível para humanos exibido na interface
- Pode conter espaços e caracteres especiais

#### Categoria

- Usada para organizar materiais dentro da biblioteca
- Categorias comuns incluem: Madeira, Acrílico, Metal, Papel, Couro
- Você pode criar categorias personalizadas conforme necessário

#### Textura

Uma imagem de textura (WebP ou PNG) que é repetida em mosaico sobre a superfície do material. Quando
definida, o material é renderizado com a textura em vez de uma cor sólida. As texturas podem ser
otimizadas para WebP com o script `scripts/optimize_material_textures.py` para manter os arquivos de
material pequenos.

#### Escala da textura

O tamanho (em mm) que uma peça de textura cobre sobre o material. Valores menores repetem a textura
com mais frequência na mesma superfície.

#### Cor

Uma cor de tonalidade opcional. Quando definida, a textura do material é tingida com essa cor;
quando não, a textura é exibida como está. Isso permite que um único material texturizado (ex.,
"Acrílico") cubra múltiplas variantes de cor: a cor é aplicada por item de material no diálogo
[Propriedades do Material](../features/stock-handling.md). A cor é usada apenas para aparência
visual na superfície de trabalho - não afeta o caminho do laser de nenhuma forma.

#### Rugosidade

Um valor de 0-1 que descreve quão rugosa ou polida a superfície aparece na visualização 3D. Valores
mais baixos parecem brilhantes, valores mais altos parecem foscos.

#### Metálico

Um valor de 0-1 que descreve se a superfície reflete luz como um metal na visualização 3D. Defina 1
para materiais metálicos, 0 para não metálicos.

#### Absorção {#absorption}

:::note Novidade na 1.11 Dados de absorção alimentam o
[modelo de queimadura física](../ui/3d-preview.md#physical-burn-model) na pré-visualização 3D. :::

Coeficientes de absorção por comprimento de onda (0–1) descrevem quanta energia do laser um material
absorve em determinado comprimento de onda. A pré-visualização 3D usa esses dados, junto com o
comprimento de onda, a potência óptica e o tamanho do ponto da cabeça de laser, para calcular a
fluência (J/cm²) entregue e renderizar um efeito de queima fisicamente fundamentado no material.

Adicione um bloco `absorption` sob `appearance` no YAML do material:

```yaml
appearance:
  absorption:
    blue: 0.7 # ~445 nm lasers de diodo
    ir: 0.25 # ~1064 nm lasers de fibra / IR
    co2: 0.9 # ~10600 nm lasers CO2
  # ...outras propriedades de aparência
```

| Faixa  | Comprimento de onda representativo | Lasers típicos       |
| ------ | ---------------------------------- | -------------------- |
| `blue` | 445 nm                             | Lasers de diodo azul |
| `ir`   | 1064 nm                            | Lasers de fibra      |
| `co2`  | 10600 nm                           | Lasers de tubo CO2   |

Quando uma faixa está ausente, um padrão conservador é usado. A biblioteca de materiais incluída
possui valores de absorção pesquisados para todos os materiais incluídos; o modelo de queimadura
ainda não está totalmente calibrado, então contribuições de dados de teste reais são bem-vindas.

## Gerenciando Materiais Existentes

### Editando Materiais

1. Selecione o material que deseja editar
2. Clique no botão **Editar**
3. Modifique as propriedades desejadas
4. Clique em **Salvar** para aplicar as alterações

### Excluindo Materiais

1. Selecione o material que deseja excluir
2. Clique no botão **Excluir**
3. Confirme a exclusão no diálogo

:::warning Excluir um material é permanente e não pode ser desfeito. :::
