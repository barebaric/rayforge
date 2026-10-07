---
description:
  "Gere uma grade de teste de material para encontrar as configurações ideais de potência e
  velocidade do laser para qualquer material. Calibre sua máquina de corte de forma sistemática."
---

# Grade de Teste de Material

Cada material — e frequentemente cada cor e espessura do mesmo material — responde de forma
diferente à potência e à velocidade do laser. A Grade de Teste de Material elimina as suposições na
busca pela combinação certa: ela gera um padrão de células de teste em que cada célula é gravada ou
cortada com uma configuração ligeiramente diferente, tudo em um único trabalho. Depois de uma
execução, você pode ver de relance qual combinação produz o resultado desejado.

Crie uma via **Ferramentas → Criar Grade de Teste de Material**. O Rayforge adiciona uma peça
especial à tela, junto com uma operação correspondente, e você configura a grade no diálogo de
configurações dela.

![Configurações da Grade de Teste de Material](/screenshots/material-test.webp)

## Predefinições

O diálogo de configurações oferece predefinições para tipos comuns de laser. Elas preenchem uma
faixa de velocidade, uma faixa de potência e um tipo de teste razoáveis, para que você comece com
uma base sensata:

| Predefinição       | Faixa de Velocidade | Faixa de Potência | Tipo de Teste |
| ------------------ | ------------------- | ----------------- | ------------- |
| **Gravação Diodo** | 1000-10000 mm/min   | 10-100%           | Gravação      |
| **Corte Diodo**    | 100-5000 mm/min     | 50-100%           | Corte         |
| **Gravação CO2**   | 3000-20000 mm/min   | 10-50%            | Gravação      |
| **Corte CO2**      | 1000-20000 mm/min   | 30-100%           | Corte         |

Uma predefinição é apenas um ponto de partida — todos os valores permanecem ajustáveis depois, e as
faixas de velocidade são limitadas automaticamente ao que a sua máquina é capaz de fazer.

## Modos de Grade

Uma grade de teste varia dois parâmetros ao mesmo tempo: um ao longo das colunas e outro ao longo
das linhas. O modo de grade decide quais são esses dois. **Potência vs Velocidade** é o padrão e
cobre a pergunta mais comum — potência ao longo das colunas, velocidade ao longo das linhas.

**Potência vs Passagens** e **Velocidade vs Passagens** mantêm um dos dois fixo e variam o número de
passagens, o que é útil para cortar materiais mais espessos. **Velocidade vs Deslocamento** é um
modo de calibração especial para gravação bidirecional: ele varia o deslocamento horizontal da
varredura para que você possa corrigir desalinhamentos entre linhas. Como isso só faz sentido para
trabalhos raster, selecioná-lo alterna a grade para Gravação e amplia o espaçamento entre linhas
para que qualquer desalinhamento seja fácil de ver. Dentro de cada linha, a potência é escalada
junto com a velocidade, de modo que todas as células permaneçam visualmente comparáveis.

## Configurando a Grade

O diálogo de configurações agrupa os parâmetros em três seções.

A seção **Grade** controla o teste em si. O tipo de teste determina se cada célula corta o contorno
de um quadrado ou o preenche com linhas raster. As dimensões da grade definem quantas colunas e
linhas testar — cada coluna representa um passo do primeiro parâmetro do modo e cada linha um passo
do segundo, do mínimo ao máximo da faixa que você inserir. São permitidas entre 2 e 20 etapas por
eixo; 5×5 é um bom padrão. O tamanho da forma (10 mm por padrão) e o espaçamento (2 mm por padrão)
determinam o tamanho da grade. Para o tipo de teste Gravação, o intervalo entre linhas controla a
distância entre as linhas de varredura — valores menores preenchem de forma mais densa, mas levam
mais tempo. Deixe em zero para usar o tamanho do ponto do seu laser, o que funciona bem para a
maioria das gravações.

A seção **Rótulos** controla as anotações gravadas ao lado da grade. Os rótulos ficam ativados por
padrão e são gravados primeiro, para que o padrão de teste não os obscureça. Eles têm potência
própria (10% por padrão) e velocidade própria (1000 mm/min por padrão), e os valores de velocidade
são exibidos na sua unidade de exibição preferida.

A seção **Parâmetros** contém as faixas que a grade varia — velocidade, potência, passagens ou
deslocamento, dependendo do modo selecionado. Os modos que mantêm um parâmetro fixo (por exemplo, a
velocidade em Potência vs Passagens) permitem definir essa constante aqui também.

## Entendendo o Layout

No modo padrão Potência vs Velocidade, a potência aumenta da esquerda para a direita e a velocidade
de cima para baixo:

```
                   Potência (%)
                 10       55       100
Velocidade 100  [  ]     [  ]     [  ]
(mm/min)   300  [  ]     [  ]     [  ]
           500  [  ]     [  ]     [  ]
```

Os rótulos nas bordas esquerda e superior mostram o valor exato de cada linha e coluna, para que
você nunca precise contar as células.

O tamanho total decorre diretamente das dimensões da grade: cada eixo mede _etapas × tamanho da
forma + (etapas − 1) × espaçamento_, mais espaço para os rótulos à esquerda e no topo (no máximo 15
mm, e apenas quando os rótulos estão ativados). Uma grade 5×5 de quadrados de 20 mm com espaçamento
de 5 mm tem 120 mm de lado sem rótulos e 135 mm com eles.

## Como a Grade É Executada {#how-the-grid-runs}

As células deliberadamente **não** são executadas em ordem de leitura. O Rayforge as executa em uma
ordem otimizada por risco: primeiro a velocidade mais alta, a potência mais baixa dentro de cada
velocidade e o menor número de passagens dentro de cada potência. As combinações lentas e de alta
potência são as mais propensas a carbonizar o material ou a iniciar um incêndio, por isso são
executadas por último. Essa ordenação é intencional e não pode ser alterada.

## Executando o Teste

Carregue o material que você quer caracterizar — sucata, não a sua peça final — e focalize o laser
como faria em um trabalho real, já que a distância de foco altera o resultado. Inicie o trabalho e
permaneça junto à máquina: se uma célula começar a carbonizar fortemente ou a soltar fumaça em
excesso, interrompa o trabalho em vez de deixá-lo terminar.

Quando o teste terminar, examine cada célula. Se a gravação sair clara demais, vá na direção de mais
potência ou de velocidade mais lenta; se sair escura ou carbonizada, vá na direção de menos potência
ou de velocidade mais alta. Para testes de corte, procure a célula que atravessa completamente com a
menor carbonização possível. Para chegar ao ponto ideal, execute uma segunda grade mais fina: se um
teste grosseiro 5×5 encontrou a melhor célula em torno de 40% de potência e 4000 mm/min, uma grade
de acompanhamento cobrindo 35-45% e 3000-5000 mm/min vai identificá-la com precisão.

<!-- prettier-ignore-start -->
:::tip[Salve como receita]
Em vez de manter um caderno com as configurações vencedoras, armazene-as como uma
[receita](../../application-settings/recipes.md): dê um nome a ela (por exemplo "Corte de Madeira
Compensada 3 mm"), vincule-a à máquina, à operação, ao material e à espessura testados, e o Rayforge
sugerirá exatamente essas configurações na próxima vez que você cortar o mesmo material.
:::
<!-- prettier-ignore-end -->

## Uso Avançado

Grades de teste de material são peças comuns, por isso se combinam livremente com outras operações.
Um padrão comum é adicionar uma operação de contorno ao redor da grade pronta e recortar a peça de
teste do material de estoque depois que a gravação terminar.

Executar a mesma configuração de grade em materiais diferentes é uma maneira rápida de construir uma
biblioteca de configurações comprovadamente boas — e as receitas tornam essa biblioteca pesquisável
por material e espessura mais tarde.

## Dicas e Melhores Práticas

Alguns hábitos tornam os resultados de teste mais confiáveis:

- Comece a partir de uma predefinição e ajuste a partir dela, em vez de configurar do zero.
- Dê espaço às células: quadrados de 15-20 mm são muito mais fáceis de avaliar do que minúsculos.
- Mude uma variável por vez ao restringir as opções — uma grade fina que varia os dois eixos em
  faixas amplas é difícil de interpretar.
- Deixe o material esfriar entre testes consecutivos na mesma peça.
- Use a mesma distância de foco em todos os testes, incluindo o trabalho final.

E as regras usuais de segurança com laser se aplicam em dobro às grades de teste, que exploram
intencionalmente território desconhecido:

- Nunca deixe um teste em execução sem supervisão.
- Comece com faixas de potência conservadoras e aumente a partir daí.
- Verifique se a extração de fumos está funcionando antes de começar.
- Mantenha um extintor de incêndio ao alcance.

## Solução de Problemas

**As células são executadas em uma ordem estranha.** Essa é a ordem de execução otimizada por risco
descrita em [Como a Grade É Executada](#how-the-grid-runs) — primeiro as combinações mais rápidas e
mais fracas. É intencional.

**Os resultados variam entre execuções.** Certifique-se de que o material está plano e fixado, de
que o foco é idêntico em toda a grade e de que a sua fonte de alimentação entrega potência estável.
Se apenas uma região da grade parecer errada, o próprio material pode ser irregular.

## Tópicos Relacionados

- **[Visualização 3D](../../ui/3d-preview.md)** - Pré-visualize a execução do teste antes de rodar
- **[Receitas](../../application-settings/recipes.md)** - Reutilize seus resultados de teste
  automaticamente
- **[Gravação](engrave)** - Entendendo operações de gravação
- **[Corte de Contorno](contour)** - Entendendo operações de corte
