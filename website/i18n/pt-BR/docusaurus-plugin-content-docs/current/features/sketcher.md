# Esboçador paramétrico 2D

O Esboçador paramétrico 2D é um recurso poderoso do Rayforge que permite criar
e editar designs 2D precisos baseados em restrições diretamente no aplicativo.
Esse recurso permite projetar peças personalizadas do zero sem precisar de
software CAD externo.

## Visão geral

O esboçador fornece um conjunto completo de ferramentas para criar formas
geométricas e aplicar restrições paramétricas para definir relações precisas
entre os elementos. Essa abordagem garante que seus designs mantenham a
geometria pretendida mesmo quando as dimensões são modificadas.

## Criando e editando esboços

### Criando um novo esboço

1. Clique no botão "Novo esboço" na barra de ferramentas ou use o menu
   principal
2. Um novo espaço de trabalho vazio será aberto com a interface do editor de
   esboços
3. Comece a criar geometria usando as ferramentas de desenho do menu circular
   ou os atalhos de teclado
4. Aplique restrições para definir as relações entre os elementos
5. Clique em "Finalizar esboço" para salvar seu trabalho e retornar ao espaço
   de trabalho principal

### Editando esboços existentes

1. Dê um duplo clique em uma peça baseada em esboço no espaço de trabalho
   principal
2. Alternativamente, selecione um esboço e escolha "Editar esboço" no menu de
   contexto
3. Faça suas modificações usando as mesmas ferramentas e restrições
4. Clique em "Finalizar esboço" para salvar as alterações ou em "Cancelar
   esboço" para descartá-las

## Criando geometria 2D

O esboçador suporta a criação dos seguintes elementos geométricos básicos:

- **Caminhos (linhas e curvas de Bézier)**: Desenhe linhas retas e curvas de
  Bézier suaves usando a ferramenta de caminho unificada. Clique para colocar
  pontos, arraste para criar alças de Bézier.
- **Arcos**: Desenhe arcos especificando um ponto central, um ponto inicial e um
  ponto final
- **Elipses**: Crie elipses (e círculos) definindo um ponto central e
  arrastando para definir o tamanho e a proporção. Segure `Ctrl` enquanto
  arrasta para restringir a um círculo perfeito.
- **Retângulos**: Desenhe retângulos especificando dois cantos opostos
- **Retângulos arredondados**: Desenhe retângulos com cantos arredondados
- **Caixas de texto**: Adicione elementos de texto ao seu esboço
- **Preenchimentos**: Preencha regiões fechadas para criar áreas sólidas

Esses elementos formam a base dos seus designs 2D e podem ser combinados para
criar formas complexas. Os preenchimentos são particularmente úteis para criar
regiões sólidas que serão gravadas ou cortadas como uma única peça.

## Trabalhando com curvas de Bézier

A ferramenta de caminho suporta curvas de Bézier para criar formas suaves e
orgânicas:

### Desenhando curvas de Bézier

1. Selecione a ferramenta de caminho no menu circular ou use o atalho de
   teclado
2. Clique para colocar pontos — cada clique cria um novo ponto
3. Arraste após clicar para criar alças de Bézier para curvas suaves
4. Continue adicionando pontos para construir seu caminho
5. Pressione Escape ou dê um duplo clique para finalizar o caminho

### Editando curvas de Bézier

- **Mover pontos**: Clique e arraste qualquer ponto para reposicioná-lo
- **Ajustar alças**: Arraste as extremidades das alças para modificar a forma
  da curva
- **Conectar a pontos existentes**: Ao editar um caminho, você pode snapar para
  pontos existentes no seu esboço
- **Tornar suave/simétrico**: Pontos conectados por uma restrição de
  coincidência podem ser suavizados (tangente contínua) ou simetrizados (alças
  espelhadas)

### Convertendo curvas em linhas

Use a **ferramenta de retificação** para converter curvas de Bézier de volta em
linhas retas. Isso é útil quando você precisa de geometria limpa e simples.
Selecione os segmentos de Bézier que deseja converter e aplique a ação de
retificação.

## Sistema de restrições paramétricas

O sistema de restrições é o núcleo do esboçador paramétrico, permitindo definir
relações geométricas precisas:

### Restrições geométricas

- **Coincidência**: Força dois pontos a ocupar a mesma posição
- **Vertical**: Restringe uma linha a ser perfeitamente vertical
- **Horizontal**: Restringe uma linha a ser perfeitamente horizontal
- **Tangente**: Torna uma linha tangente a um círculo ou arco
- **Perpendicular**: Força duas linhas, uma linha e um arco/círculo, ou dois
  arcos/círculos a se encontrarem em 90 graus
- **Ponto em linha/forma**: Restringe um ponto a ficar sobre uma linha, arco ou
  círculo
- **Colinear**: Força duas ou mais linhas a ficarem sobre a mesma linha
  infinita
- **Simetria**: Cria relações simétricas entre elementos. Suporta dois modos:
  - **Simetria de ponto**: Selecione 3 pontos (o primeiro é o centro)
  - **Simetria de linha**: Selecione 2 pontos e 1 linha (a linha é o eixo)

### Restrições dimensionais

- **Distância**: Define a distância exata entre dois pontos ou ao longo de uma
  linha
- **Diâmetro**: Define o diâmetro de um círculo
- **Raio**: Define o raio de um círculo ou arco
- **Ângulo**: Impõe um ângulo específico entre duas linhas
- **Proporção**: Força a razão entre duas distâncias a ser igual a um valor
  especificado
- **Comprimento/Raio igual**: Força múltiplos elementos (linhas, arcos,
  elipses ou círculos) a ter o mesmo comprimento ou raio
- **Distância igual**: Torna dois segmentos de linha do mesmo comprimento
  (diferente de Comprimento/Raio igual, que também pode se aplicar a arcos e
  círculos)

## Interface do menu circular

O esboçador apresenta um menu circular contextual que fornece acesso rápido a
todas as ferramentas de desenho e restrição. Esse menu radial aparece quando
você clica com o botão direito no espaço de trabalho do esboço e se adapta com
base no seu contexto e seleção atuais.

Os itens do menu circular mostram dinamicamente as opções disponíveis com base
no que você selecionou. Por exemplo, ao clicar em um espaço vazio, você verá
ferramentas de desenho. Ao clicar em geometria selecionada, verá as restrições
aplicáveis.

![Menu circular do esboçador](/screenshots/sketcher-pie-menu.png)

## Atalhos de teclado

O esboçador fornece atalhos de teclado para um fluxo de trabalho eficiente:

### Atalhos de ferramentas

- `Space`: Ferramenta de seleção
- `G+P`: Ferramenta de caminho (linhas e curvas de Bézier)
- `G+A`: Ferramenta de arco
- `G+C`: Ferramenta de elipse
- `G+R`: Ferramenta de retângulo
- `G+O`: Ferramenta de retângulo arredondado
- `G+F`: Ferramenta de preenchimento de área
- `G+T`: Ferramenta de caixa de texto
- `G+G`: Ferramenta de grade (alternar visibilidade da grade)
- `G+N`: Alternar modo de construção na seleção

### Atalhos de ações

- `C+H`: Adicionar chanfro no canto
- `C+F`: Adicionar arredondamento no canto
- `C+S`: Retificar curvas de Bézier selecionadas para linhas

### Atalhos de restrições

- `H`: Aplicar restrição Horizontal
- `V`: Aplicar restrição Vertical
- `N`: Aplicar restrição Perpendicular
- `T`: Aplicar restrição Tangente
- `E`: Aplicar restrição Igual
- `O` ou `C`: Aplicar restrição de Alinhamento (Coincidência)
- `S`: Aplicar restrição de Simetria
- `K+D`: Aplicar restrição de Distância
- `K+R`: Aplicar restrição de Raio
- `K+O`: Aplicar restrição de Diâmetro
- `K+A`: Aplicar restrição de Ângulo
- `K+X`: Aplicar restrição de Proporção

### Atalhos gerais

- `Ctrl+Z`: Desfazer
- `Ctrl+Y` ou `Ctrl+Shift+Z`: Refazer
- `Delete`: Excluir elementos selecionados
- `Escape`: Cancelar operação atual ou deselecionar
- `F`: Ajustar visualização ao conteúdo

## Modo de construção

O modo de construção permite marcar entidades como "geometria de construção" —
elementos auxiliares usados para guiar seu design, mas que não fazem parte do
resultado final. As entidades de construção são exibidas de forma diferente
(geralmente como linhas tracejadas) e não são incluídas quando o esboço é usado
para corte ou gravação a laser.

Para alternar o modo de construção:

- Selecione uma ou mais entidades
- Pressione `N` ou `G+N`, ou use a opção Construção no menu circular

As entidades de construção são úteis para:

- Criar linhas e círculos de referência
- Definir geometria temporária para alinhamento
- Construir formas complexas a partir de uma estrutura de guias

## Grade, snap e controles de visibilidade

### Ferramenta de grade

A ferramenta de grade fornece uma referência visual para alinhamento e
dimensionamento:

- Ative/desative a grade usando o botão da ferramenta ou `G+G`
- A grade se adapta ao seu nível de zoom para manter um espaçamento
  consistente

### Snap magnético

Ao criar ou mover geometria, o Rayforge automaticamente atrai seu cursor para
elementos próximos — extremidades, pontos médios de linhas, interseções e outros
pontos de referência. Isso facilita a conexão precisa de formas sem precisar
colocar cada ponto manualmente. O indicador de snap destaca quando seu cursor
está próximo de um alvo de snap.

### Auto-restrição durante a criação

Muitas ferramentas de desenho aplicam restrições automaticamente ao criar
geometria. Por exemplo, ao desenhar uma linha próxima à horizontal ou vertical,
o esboçador oferecerá para travá-la no lugar. Isso ajuda a manter seu esboço
organizado desde o início, em vez de corrigir as coisas depois.

### Controles mostrar/ocultar

A barra de ferramentas do esboçador inclui botões de alternância para controlar
a visibilidade:

- **Mostrar/ocultar geometria de construção**: Alterna a visibilidade das
  entidades de construção
- **Mostrar/ocultar restrições**: Alterna a visibilidade dos marcadores de
  restrições

Esses controles ajudam a reduzir a poluição visual ao trabalhar em esboços
complexos.

### Movimento restrito ao eixo

Ao arrastar pontos ou geometria, segure `Shift` para restringir o movimento ao
eixo mais próximo (horizontal ou vertical). Isso é útil para manter o
alinhamento durante ajustes.

## Chanfro e arredondamento

O esboçador fornece ferramentas para modificar os cantos da sua geometria:

- **Chanfro**: Substitui um canto agudo por uma borda chanfrada. Selecione um
  ponto de junção (onde duas linhas se encontram) e aplique a ação de chanfro.
- **Arredondamento**: Substitui um canto agudo por uma borda arredondada.
  Selecione um ponto de junção (onde duas linhas se encontram) e aplique a ação
  de arredondamento.

Para usar chanfro ou arredondamento:

1. Selecione um ponto de junção onde duas linhas se encontram
2. Pressione `C+H` para chanfro ou `C+F` para arredondamento
3. Use o menu circular ou os atalhos de teclado para aplicar a modificação

## Importação e exportação

### Exportando objetos

Você pode exportar qualquer peça selecionada para vários formatos vetoriais:

1. Selecione uma peça na tela
2. Escolha **Objeto → Exportar objeto...** (ou clique com o botão direito e
   selecione no menu de contexto)
3. Escolha o formato de exportação:
   - **RFS (.rfs)**: Formato nativo de esboço paramétrico do Rayforge —
     preserva todas as restrições e pode ser reimportado para edição
   - **SVG (.svg)**: Formato vetorial padrão — amplamente compatível com
     softwares de design
   - **DXF (.dxf)**: Formato de intercâmbio CAD — compatível com a maioria dos
     aplicativos CAD

### Salvando esboços

Você pode salvar seus esboços 2D em arquivos para reutilização em outros
projetos. Todas as restrições paramétricas são preservadas ao salvar,
garantindo que seus designs mantenham suas relações geométricas.

### Importando esboços

Esboços salvos podem ser importados em qualquer espaço de trabalho, permitindo
que você crie uma biblioteca de elementos de design comumente usados. O processo
de importação mantém todas as restrições e relações dimensionais.

## Dicas de fluxo de trabalho

1. **Comece com geometria aproximada**: Crie formas básicas primeiro e depois
   refine com restrições
2. **Use restrições cedo**: Aplique restrições enquanto constrói para manter a
   intenção do design
3. **Verifique o status das restrições**: O sistema indica quando os esboços
   estão totalmente restritos
4. **Fique atento a conflitos**: Restrições que conflitam entre si são
   destacadas em vermelho e mostradas no painel de restrições para fácil
   identificação
5. **Utilize a simetria**: Restrições de simetria podem acelerar
   significativamente designs complexos
6. **Use a grade**: Ative a grade para alinhamento preciso e use Ctrl para
   snapar na grade
7. **Itere e refine**: Não hesite em modificar restrições para obter o
   resultado desejado

## Recursos de edição

- **Suporte completo a desfazer/refazer**: O estado completo do esboço é salvo
  com cada operação
- **Cursor dinâmico**: O cursor muda para refletir a ferramenta de desenho
  ativa
- **Visualização de restrições**: As restrições aplicadas são claramente
  indicadas na interface
- **Atualizações em tempo real**: Alterações nas restrições atualizam
  imediatamente a geometria
- **Edição por duplo clique**: Dar um duplo clique em restrições dimensionais
  (Distância, Raio, Diâmetro, Ângulo, Proporção) abre um diálogo para editar
  seus valores
- **Expressões paramétricas**: Restrições dimensionais suportam expressões,
  permitindo que valores sejam calculados a partir de outros parâmetros (por
  ex., `width/2` para um raio que seja metade da largura)
