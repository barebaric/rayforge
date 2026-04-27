# Deepnest

O Deepnest organiza automaticamente seus workpieces em um layout compacto no
seu material de estoque ou área de trabalho da máquina. Ele usa um algoritmo
genético para encontrar um empacotamento eficiente de formas, minimizando o
desperdício e encaixando mais peças em cada chapa.

![Diálogo de configuração do Deepnest](/screenshots/addon-deepnest.png)

## Pré-requisitos

Selecione um ou mais workpieces na tela antes de executar o nesting. Você
também pode selecionar itens de estoque para definir os limites da chapa. Se
nenhum estoque for selecionado, o addon utiliza o estoque do documento ou
recorre à área de trabalho da máquina.

## Executando o layout de nesting

Inicie o layout de nesting a partir do menu **Organizar**, do botão da barra
de ferramentas ou do atalho de teclado **Ctrl+Alt+N**. Um diálogo de
configurações abre antes que o algoritmo seja executado.

## Configurações de nesting

O diálogo de configurações oferece as seguintes opções antes que o algoritmo
de nesting comece.

**Espaçamento** define a distância entre as formas aninhadas, em milímetros.
O valor padrão é obtido do tamanho do spot do laser da sua máquina. Aumente
este valor para adicionar uma margem de segurança entre as peças.

**Restringir rotação** mantém todas as peças em sua orientação original.
Quando esta opção está desativada, o algoritmo gira as peças em incrementos
de 10 graus para encontrar um encaixe mais justo. Deixar a rotação livre
produz melhor uso do material, mas leva mais tempo para calcular.

**Permitir inversão horizontal** espelha as peças horizontalmente durante o
nesting. Isso pode ajudar a encaixar as peças mais justamente, porém os cortes
resultantes estarão espelhados.

**Permitir inversão vertical** espelha as peças verticalmente durante o
nesting. A mesma consideração sobre saída espelhada se aplica.

Clique em **Iniciar nesting** para começar. O diálogo se fecha e o algoritmo
roda em segundo plano. Um indicador de progresso aparece no painel inferior
enquanto o nesting está em andamento.

## Após o nesting

Quando o algoritmo termina, todos os workpieces na tela são reposicionados
para suas localizações aninhadas. As posições são aplicadas como uma única
ação reversível, então você pode desfazer o layout com um passo se o
resultado não for o que você precisa.

Se o algoritmo não conseguiu colocar todos os workpieces no estoque
disponível, os itens não colocados são movidos para a direita da área de
estoque para que permaneçam visíveis e fáceis de identificar.

Se o resultado do nesting for pior que o layout original — por exemplo, se as
peças já se encaixavam bem — os workpieces permanecem em suas posições
originais.

## Tópicos relacionados

- [Manuseio de estoque](../features/stock-handling) - Definir material de estoque para o nesting
- [Posicionamento de workpieces](../features/workpiece-positioning) - Posicionar workpieces manualmente
