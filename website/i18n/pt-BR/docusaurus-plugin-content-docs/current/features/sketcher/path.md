---
description:
  "Desenhe linhas retas e curvas de Bézier suaves com a ferramenta de caminho no esboçador do
  Rayforge."
---

# Ferramenta de caminho

A ferramenta de caminho (`G+P` ou `G+L`) desenha cadeias conectadas de linhas retas e curvas de
Bézier suaves em um fluxo de trabalho unificado. É a ferramenta de desenho mais versátil do
esboçador: clique para colocar pontos, arraste para curvar o segmento.

![Um caminho de duas linhas unidas por um segmento de Bézier, com seus waypoints e alças](/screenshots/addons-sketcher-tool-path.webp)

## Desenhando caminhos

1. Selecione a ferramenta de caminho no menu circular, no menu **Esboço**, ou com `G+P`.
2. Clique para colocar o primeiro ponto. Uma pré-visualização ao vivo segue o cursor.
3. Clique novamente sem arrastar para finalizar um segmento reto — o próximo segmento começa
   imediatamente a partir desse ponto.
4. Pressione em um ponto e arraste antes de soltar para transformar o segmento em uma curva de
   Bézier. O arrasto controla a "curvatura" da curva.
5. Continue adicionando pontos para construir seu caminho.
6. Pressione `Escape` ou dê um duplo clique para finalizar o caminho.

Enquanto uma pré-visualização estiver ativa, a barra de status lista as teclas modificadoras
aplicáveis, e `Esc` a cancela.

## Trabalhando com curvas de Bézier

Curvas de Bézier criam formas suaves e orgânicas:

- **Ajustar alças**: selecione uma Bézier e arraste as extremidades das alças redondas para
  modificar a forma da curva. Cada alça curva a curva em seu lado do waypoint.
- **Conectar a pontos existentes**: ao desenhar, o snap magnético anexa novos segmentos a pontos
  existentes no seu esboço, e a restrição correspondente é criada automaticamente.

### Tipos de waypoint

O ponto onde dois segmentos de um caminho se encontram é um _waypoint_. O tipo do waypoint controla
como a curva flui através dele:

- **Sharp**: as alças em ambos os lados são independentes, produzindo um canto.
- **Smooth**: as alças compartilham uma tangente, produzindo uma transição contínua e arredondada.
- **Symmetric**: como Smooth, mas as alças também são espelhadas, de modo que ambos os lados se
  curvam igualmente.

Para alterar o tipo de um waypoint, clique com o botão direito nele (ou no segmento de Bézier
adjacente) e escolha o tipo no menu circular. Waypoints de Bézier recém-desenhados são simétricos
(Symmetric).

![O menu circular sobre um waypoint de Bézier selecionado, com as
ferramentas Straighten, Sharp, Smooth e Symmetric](/screenshots/addons-sketcher-tool-path-pie-menu.webp)

### Convertendo curvas em linhas

A ferramenta **Straighten** do mesmo menu circular converte curvas de Bézier de volta em linhas
retas, o que é útil quando você precisa de geometria limpa e simples. Selecione os segmentos de
Bézier que deseja converter e aplique a ação de retificação. Os segmentos colapsam para a conexão
reta entre suas extremidades.

## Restrições automáticas

A ferramenta de caminho participa do snap magnético como todas as outras ferramentas de desenho.
Quando as guias de snap mostram alinhamento durante o desenho, restrições horizontais e verticais
correspondentes são criadas automaticamente, o que mantém seu esboço organizado desde o início, em
vez de corrigir as coisas depois. Segure `Shift` para restringir o novo segmento ao eixo mais
próximo. Veja [Grade e snap](index.md#grid-and-snapping) para a lista completa de indicadores de
snap.
