# Preenchimento por Deslocamento

O preenchimento por deslocamento preenche formas vetoriais fechadas com anéis paralelos ao seu
contorno. O primeiro anel segue o próprio contorno; cada anel seguinte fica um intervalo de linha
mais para dentro, até que a forma esteja preenchida. Furos em uma forma permanecem livres: os anéis
também seguem suas bordas.

## Visão Geral

Enquanto a [Gravação](engrave) varre para frente e para trás em linhas retas e o
[Frente de Onda](wavefront) se expande a partir do centro do bolsão, o preenchimento por
deslocamento trabalha do contorno para dentro. Cada anel é um caminho fechado cortado ao longo de
sua linha central, de modo que o preenchimento assume a forma do design.

As operações de preenchimento por deslocamento:

- Preenchem formas vetoriais fechadas com anéis paralelos ao contorno
- Mantêm livres os furos dentro de uma forma
- Ignoram caminhos abertos
- Cortam os anéis de fora para dentro ou de dentro para fora

## Quando Usar Preenchimento por Deslocamento

Use o preenchimento por deslocamento para:

- Preencher texto, logotipos e outras formas vetoriais com um preenchimento que segue seus contornos
- Esvaziar bolsões onde os anéis devem acompanhar a borda da forma
- Designs em que o padrão de preenchimento é visível na peça acabada

**Não use o preenchimento por deslocamento para:**

- Cortar apenas ao longo de contornos (use [Contorno](contour) em vez disso)
- Preencher imagens bitmap (use [Gravação](engrave) em vez disso)
- Caminhos abertos, que não delimitam uma área a preencher

## Criando uma Operação de Preenchimento por Deslocamento

### Passo 1: Preparar as Formas

1. Importe ou desenhe formas vetoriais fechadas na tela
2. Certifique-se de que as formas são caminhos fechados; caminhos abertos são ignorados

### Passo 2: Adicionar uma Etapa de Preenchimento por Deslocamento

1. Abra o fluxo de trabalho da camada no painel direito.
2. Clique no botão **Adicionar etapa** e escolha **Preenchimento por Deslocamento**.

### Passo 3: Configurar Ajustes

Ajuste o intervalo de linha, o deslocamento e a direção de preenchimento conforme seu material e o
acabamento desejado.

## Configurações Principais

O diálogo de configurações da etapa tem três abas: **Configurações da Etapa**, **Laser** e
**Pós-processamento**. As configurações são descritas abaixo na ordem das abas.

### Configurações de Preenchimento por Deslocamento

O grupo **Preenchimento por Deslocamento** na aba _Configurações da Etapa_ controla o padrão de
preenchimento.

#### Intervalo de Linha (Line Interval)

A distância entre anéis de preenchimento vizinhos. Valores menores dão um preenchimento mais denso
com mais anéis e trabalhos mais longos; valores maiores são mais rápidos, mas deixam mais espaço
entre os anéis.

**O intervalo de linha usa por padrão o tamanho do ponto do laser** e tem uma faixa de 0,01–50,0 mm.

#### Deslocamento (Offset)

A distância do primeiro anel para dentro do contorno. Com o padrão de 0,0 o primeiro anel fica sobre
o contorno; um valor maior deixa uma margem entre o contorno e o preenchimento. Isso é útil quando
uma passagem de [Contorno](contour) separada vai finalizar a borda.

Faixa: 0,0–20,0 mm. O padrão é 0,0.

#### Direção de Preenchimento (Fill Direction)

A ordem em que os anéis são cortados:

- **Externo-Interno** (padrão): o anel mais externo é cortado primeiro e o preenchimento avança em
  direção ao centro
- **Interno-Externo**: o anel mais interno é cortado primeiro e o anel do contorno por último

### Configurações do Laser

Potência, velocidade e seleção do cabeçote laser ficam na página **Laser** do diálogo de
configurações da etapa. Uma nova etapa de preenchimento por deslocamento começa com 80% de potência.

**Potência (%):**

- Intensidade do laser para o preenchimento
- Ajuste aos requisitos do seu material

**Velocidade (mm/min):**

- Quão rápido o laser se move
- Ajuste à velocidade de gravação do seu material

## Como o Preenchimento por Deslocamento Funciona

1. **Formas fechadas** — Caminhos abertos são descartados; cada forma fechada é tratada junto com
   seus furos
2. **Primeiro anel** — O contorno, movido para dentro pelo deslocamento configurado
3. **Anéis seguintes** — Cada anel seguinte fica um intervalo de linha dentro do anterior; os anéis
   ao redor dos furos crescem ao mesmo tempo para fora a partir dos furos
4. **Conclusão** — Anéis são adicionados até que a forma esteja preenchida
5. **Corte** — Os anéis são cortados ao longo de sua linha central na direção de preenchimento
   escolhida

## Pós-Processamento

As operações de preenchimento por deslocamento suportam:

- **[Suavização de Caminho](../smooth.md)** — Reduz bordas irregulares nas trajetórias (desativado
  por padrão)
- **[Cortar para Estoque](../crop-to-stock.md)** — Limita as trajetórias ao limite do material
  (desativado por padrão)
- **[Otimização de Caminho](../path-optimization.md)** — Minimiza a distância de deslocamento entre
  os anéis
- **[Multi-Passagem](../multi-pass.md)** — Repete o preenchimento várias vezes

## Dicas e Melhores Práticas

### Escolhendo o Intervalo de Linha

- Comece pelo padrão (o tamanho do ponto do laser) para um preenchimento fechado e uniforme
- Aumente o intervalo para um padrão de anéis visível ou um trabalho mais rápido

### Combinando com Contorno

Esvazie o interior de uma forma com o preenchimento por deslocamento e depois finalize a borda com
uma passagem de [Contorno](contour). Ajuste o deslocamento para que o preenchimento deixe espaço
suficiente para o corte de contorno.

## Tópicos Relacionados

- **[Frente de Onda](wavefront)** — Preenchimento concêntrico que se expande a partir do centro do
  bolsão
- **[Contorno](contour)** — Corte ao longo de contornos vetoriais
- **[Gravação](engrave)** — Preenchimento de áreas com padrões de gravação raster
- **[Suavização de Caminho](../smooth.md)** — Refinamento de bordas de trajetória de ferramenta
