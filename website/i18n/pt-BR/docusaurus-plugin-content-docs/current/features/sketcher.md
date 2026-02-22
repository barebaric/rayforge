# Desenhista 2D Paramétrico

O Desenhista 2D Paramétrico é um recurso poderoso no Rayforge que permite criar
e editar designs 2D precisos baseados em restrições diretamente na aplicação. Este
recurso permite projetar peças personalizadas do zero sem precisar de software CAD
externo.

## Visão Geral

O desenhista fornece um conjunto completo de ferramentas para criar formas geométricas e aplicar
restrições paramétricas para definir relações precisas entre elementos. Esta abordagem
garante que seus designs mantenham sua geometria pretendida mesmo quando as dimensões são modificadas.

## Criando e Editando Esboços

### Criando um Novo Esboço

1. Clique no botão "Novo Esboço" na barra de ferramentas ou use o menu principal
2. Um novo espaço de trabalho de esboço vazio será aberto com a interface do editor de esboço
3. Comece a criar geometria usando as ferramentas de desenho do menu circular ou atalhos
   de teclado
4. Aplique restrições para definir relações entre elementos
5. Clique em "Finalizar Esboço" para salvar seu trabalho e retornar ao espaço de trabalho principal

### Editando Esboços Existentes

1. Dê um duplo clique em uma peça de trabalho baseada em esboço no espaço de trabalho principal
2. Alternativamente, selecione um esboço e escolha "Editar Esboço" no menu de contexto
3. Faça suas modificações usando as mesmas ferramentas e restrições
4. Clique em "Finalizar Esboço" para salvar as alterações ou "Cancelar Esboço" para descartá-las

## Criando Geometria 2D

O desenhista suporta a criação dos seguintes elementos geométricos básicos:

- **Linhas**: Desenhe segmentos de linha reta entre pontos
- **Círculos**: Crie círculos definindo um ponto central e raio
- **Arcos**: Desenhe arcos especificando um ponto central, ponto inicial e ponto final
- **Retângulos**: Desenhe retângulos especificando dois cantos opostos
- **Retângulos Arredondados**: Desenhe retângulos com cantos arredondados
- **Caixas de Texto**: Adicione elementos de texto ao seu esboço
- **Preenchimentos**: Preencha regiões fechadas para criar áreas sólidas

Esses elementos formam a base dos seus designs 2D e podem ser combinados para criar
formas complexas. Preenchimentos são particularmente úteis para criar regiões sólidas que
serão gravadas ou cortadas como uma única peça.

## Sistema de Restrições Paramétricas

O sistema de restrições é o núcleo do desenhista paramétrico, permitindo definir
relações geométricas precisas:

### Restrições Geométricas

- **Coincidente**: Força dois pontos a ocupar a mesma localização
- **Vertical**: Restringe uma linha a ser perfeitamente vertical
- **Horizontal**: Restringe uma linha a ser perfeitamente horizontal
- **Tangente**: Torna uma linha tangente a um círculo ou arco
- **Perpendicular**: Força duas linhas, uma linha e um arco/círculo, ou dois arcos/círculos
  a se encontrarem em 90 graus
- **Ponto em Linha/Forma**: Restringe um ponto a ficar em uma linha, arco ou círculo
- **Simetria**: Cria relações simétricas entre elementos. Suporta dois modos:
  - **Simetria de Ponto**: Selecione 3 pontos (o primeiro é o centro)
  - **Simetria de Linha**: Selecione 2 pontos e 1 linha (a linha é o eixo)

### Restrições Dimensionais

- **Distância**: Define a distância exata entre dois pontos ou ao longo de uma linha
- **Diâmetro**: Define o diâmetro de um círculo
- **Raio**: Define o raio de um círculo ou arco
- **Ângulo**: Impõe um ângulo específico entre duas linhas
- **Proporção**: Força a razão entre duas distâncias a ser igual a um valor
  especificado
- **Comprimento/Raio Igual**: Força múltiplos elementos (linhas, arcos ou círculos) a ter
  o mesmo comprimento ou raio
- **Distância Igual**: Força a distância entre dois pares de pontos a ser igual

## Interface do Menu Circular

O desenhista apresenta um menu circular sensível ao contexto que fornece acesso rápido a todas as ferramentas
de desenho e restrição. Este menu radial aparece quando você clica com o botão direito no espaço de
trabalho do esboço e se adapta com base no seu contexto atual e seleção.

Os itens do menu circular mostram dinamicamente as opções disponíveis com base no que você selecionou.
Por exemplo, ao clicar em um espaço vazio, você verá ferramentas de desenho. Ao clicar em
geometria selecionada, você verá restrições aplicáveis.

![Menu Circular do Desenhista](/screenshots/sketcher-pie-menu.png)

## Atalhos de Teclado

O desenhista fornece atalhos de teclado para um fluxo de trabalho eficiente:

### Atalhos de Ferramentas
- `Espaço`: Ferramenta Seleção
- `G+L`: Ferramenta Linha
- `G+A`: Ferramenta Arco
- `G+C`: Ferramenta Círculo
- `G+R`: Ferramenta Retângulo
- `G+O`: Ferramenta Retângulo Arredondado
- `G+F`: Ferramenta Preencher Área
- `G+T`: Ferramenta Caixa de Texto
- `G+N`: Alternar modo de construção na seleção

### Atalhos de Ação
- `C+H`: Adicionar canto Chanfro
- `C+F`: Adicionar canto Filete

### Atalhos de Restrição
- `H`: Aplicar restrição Horizontal
- `V`: Aplicar restrição Vertical
- `N`: Aplicar restrição Perpendicular
- `T`: Aplicar restrição Tangente
- `E`: Aplicar restrição Igual
- `O` ou `C`: Aplicar restrição de Alinhamento (Coincidente)
- `S`: Aplicar restrição Simetria
- `K+D`: Aplicar restrição Distância
- `K+R`: Aplicar restrição Raio
- `K+O`: Aplicar restrição Diâmetro
- `K+A`: Aplicar restrição Ângulo
- `K+X`: Aplicar restrição Proporção

### Atalhos Gerais
- `Ctrl+Z`: Desfazer
- `Ctrl+Y` ou `Ctrl+Shift+Z`: Refazer
- `Delete`: Excluir elementos selecionados
- `Escape`: Cancelar operação atual ou desselecionar
- `F`: Ajustar visualização ao conteúdo

## Modo de Construção

O modo de construção permite marcar entidades como "geometria de construção" - elementos
auxiliares usados para guiar seu design, mas não fazem parte da saída final. Entidades de
construção são exibidas de forma diferente (tipicamente como linhas tracejadas) e não são incluídas quando
o esboço é usado para corte ou gravação a laser.

Para alternar o modo de construção:
- Selecione uma ou mais entidades
- Pressione `N` ou `G+N`, ou use a opção Construção no menu circular

Entidades de construção são úteis para:
- Criar linhas e círculos de referência
- Definir geometria temporária para alinhamento
- Construir formas complexas a partir de uma estrutura de guias

## Chanfro e Filete

O desenhista fornece ferramentas para modificar cantos da sua geometria:

- **Chanfro**: Substitui um canto agudo por uma borda chanfrada. Selecione um ponto de junção
  (onde duas linhas se encontram) e aplique a ação de chanfro.
- **Filete**: Substitui um canto agudo por uma borda arredondada. Selecione um ponto de junção
  (onde duas linhas se encontram) e aplique a ação de filete.

Para usar chanfro ou filete:
1. Selecione um ponto de junção onde duas linhas se encontram
2. Pressione `C+H` para chanfro ou `C+F` para filete
3. Use o menu circular ou atalhos de teclado para aplicar a modificação

## Importar e Exportar

### Exportando Objetos

Você pode exportar qualquer peça de trabalho selecionada para vários formatos vetoriais:

1. Selecione uma peça de trabalho na tela
2. Escolha **Objeto → Exportar Objeto...** (ou clique com o botão direito e selecione no menu de contexto)
3. Escolha o formato de exportação:
   - **RFS (.rfs)**: Formato de esboço paramétrico nativo do Rayforge - preserva todas
     as restrições e pode ser reimportado para edição
   - **SVG (.svg)**: Formato vetorial padrão - amplamente compatível com software de design
   - **DXF (.dxf)**: Formato de intercâmbio CAD - compatível com a maioria das aplicações CAD

### Salvando Esboços

Você pode salvar seus esboços 2D em arquivos para reutilização em outros projetos. Todas as restrições
paramétricas são preservadas ao salvar, garantindo que seus designs mantenham suas relações
geométricas.

### Importando Esboços

Esboços salvos podem ser importados para qualquer espaço de trabalho, permitindo criar uma biblioteca de
elementos de design comumente usados. O processo de importação mantém todas as restrições e
relações dimensionais.

## Dicas de Fluxo de Trabalho

1. **Comece com Geometria Aproximada**: Crie formas básicas primeiro, depois refine com restrições
2. **Use Restrições Cedo**: Aplique restrições enquanto constrói para manter a intenção do design
3. **Verifique o Status das Restrições**: O sistema indica quando esboços estão totalmente restringidos
4. **Observe Conflitos**: Restrições que conflitam entre si são destacadas em vermelho
5. **Utilize Simetria**: Restrições de simetria podem acelerar significativamente designs complexos
6. **Itere e Refine**: Não hesite em modificar restrições para alcançar o resultado
   desejado

## Recursos de Edição

- **Suporte Completo a Desfazer/Refazer**: O estado inteiro do esboço é salvo com cada operação
- **Cursor Dinâmico**: O cursor muda para refletir a ferramenta de desenho ativa
- **Visualização de Restrições**: Restrições aplicadas são claramente indicadas na interface
- **Atualizações em Tempo Real**: Mudanças nas restrições atualizam imediatamente a geometria
- **Edição por Duplo Clique**: Duplo clique em restrições dimensionais (Distância, Raio, Diâmetro,
  Ângulo, Proporção) abre um diálogo para editar seus valores
- **Expressões Paramétricas**: Restrições dimensionais suportam expressões, permitindo valores serem
  calculados a partir de outros parâmetros (ex., `largura/2` para um raio que é metade da
  largura)
