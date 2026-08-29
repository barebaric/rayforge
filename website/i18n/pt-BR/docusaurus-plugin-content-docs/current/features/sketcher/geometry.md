---
description: "Aprenda a criar linhas, curvas de Bézier, arcos, elipses, retângulos e outra geometria 2D no esboçador Rayforge."
---

# Criando geometria 2D

O esboçador suporta a criação dos seguintes elementos geométricos básicos:

- **Caminhos (linhas e curvas de Bézier)**: Desenhe linhas retas e curvas de
  Bézier suaves usando a ferramenta de caminho unificada. Clique para colocar
  pontos, arraste para criar alças de Bézier.
- **Arcos**: Desenhe arcos especificando um ponto central, um ponto inicial e um
  ponto final
- **Elipses**: Crie elipses (e círculos) definindo um ponto central e
  arrastando para definir o tamanho e a proporção. Segure `Ctrl` enquanto
  arrasta para restringir a um círculo perfeito.
- **Retângulos**: Desenhe retângulos especificando dois cantos opostos.
  Cada retângulo cria automaticamente um ponto central (restringido ao
  centro geométrico) para que você possa dimensionar ou fazer snap nele.
  Segure `Shift` ao desenhar para posicionar o retângulo simetricamente
  ao redor do ponto inicial, semelhante à ferramenta de elipse.
- **Retângulos arredondados**: Desenhe retângulos com cantos arredondados
- **Caixas de texto**: Adicione elementos de texto ao seu esboço. O conteúdo
  do texto suporta expressões de modelo paramétricas (veja
  [Modelos de Texto](../text.md)).
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
