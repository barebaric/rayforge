# Frente de Onda

A limpeza adaptativa por frente de onda preenche formas vetoriais
fechadas com trajetórias de ferramenta concêntricas que se expandem
para fora a partir do centro do bolsão como ondulações em um lago. Os
anéis em expansão lidam automaticamente com ilhas interiores e produzem
trajetórias suaves e contínuas sem as inversões bruscas da varredura
raster.

## Visão Geral

Diferente da gravação raster tradicional, que varre para frente e para
trás em linhas paralelas, o frente de onda gera passagens concêntricas
que irradiam do centro de cada bolsão. Isso produz um acabamento
uniforme, semelhante a ondulações, adequado para aplicações onde o
próprio padrão de preenchimento contribui para o resultado visual.

As operações de frente de onda:

- Preenchem formas vetoriais fechadas (bolsões) com passagens concêntricas
- Expandem-se para fora a partir do centro do bolsão
- Contornam automaticamente ilhas interiores (furos dentro do bolsão)
- Produzem trajetórias suaves sem inversões de direção

## Quando Usar Frente de Onda

O frente de onda é um padrão de preenchimento alternativo para áreas de
bolsão. Seus anéis concêntricos podem ser visualmente mais agradáveis que
linhas raster paralelas, e o padrão em expansão complementa naturalmente
formas circulares ou orgânicas.

Use limpeza adaptativa por frente de onda para:

- Preencher bolsões em designs vetoriais
- Fabricação de carimbos e matrizes — o frente de onda limpa o bolsão
  de fundo preservando características em relevo como ilhas interiores
- Aplicações onde a textura de preenchimento é visível na peça acabada

**Não use frente de onda para:**

- Cortar ao longo de contornos (use [Contorno](contour) em vez disso)
- Preencher imagens bitmap (use [Gravação](engrave) em vez disso)
- Seções de parede fina onde não existe um bolsão

## Criando uma Operação de Frente de Onda

### Passo 1: Selecionar Objetos

1. Importe ou desenhe formas vetoriais fechadas na tela
2. Selecione os objetos que definem o limite do bolsão
3. Certifique-se de que as formas sejam caminhos fechados

### Passo 2: Adicionar Operação de Frente de Onda

- **Menu:** Operações → Adicionar Frente de Onda
- **Clique direito:** Menu de contexto → Adicionar Operação → Frente de Onda

### Passo 3: Configurar Ajustes

Ajuste o passo e o deslocamento para combinar com seu material e
acabamento desejado.

![Resultado da operação de frente de onda](/screenshots/operations-wavefront.png)

## Ajustes Principais

### Passo (Step Over)

A distância entre passagens consecutivas do frente de onda (mm). Valores
menores fornecem cobertura mais densa com mais passagens e tempos de
trabalho mais longos. Valores maiores espaçam mais as passagens para
conclusão mais rápida.

**O Passo padrão é o tamanho do ponto do laser** e tem uma faixa de
0,05–50,0 mm.

| Passo   | Densidade de linha    | Tempo de trabalho |
| ------- | --------------------- | ----------------- |
| 0,1 mm  | Densa, muitas linhas  | Mais lento        |
| 0,3 mm  | Moderada              | Médio             |
| 1,0 mm+ | Esparsa, menos linhas | Rápido            |

Valores típicos são de 0,1–0,5 mm para a maioria das aplicações.

### Deslocamento (Offset)

Folga adicional da parede do bolsão (mm). Cria uma margem entre a
passagem de frente de onda mais externa e o contorno do limite. Isso é
útil quando uma passagem de [Contorno](contour) separada finalizará a
borda, ou quando você deseja deixar uma borda deliberada ao redor do
bolsão.

Faixa: 0,0–20,0 mm. O padrão é 0,0 (as passagens de frente de onda se
estendem até o limite).

## Como o Frente de Onda Funciona

1. **Passagem de entrada** — Uma entrada helicoidal mergulha no centro
   do bolsão para estabelecer uma área limpa inicial
2. **Expansão do frente de onda** — Começando do centro limpo, anéis
   concêntricos se expandem para fora. Cada anel se estende além do
   anterior pela distância de passo configurada
3. **Tratamento de ilhas** — À medida que o frente de onda cresce, ele
   encontra e contorna quaisquer ilhas interiores, deixando-as em pé
4. **Conclusão** — A expansão continua até que toda a área do bolsão
   esteja coberta

## Pós-Processamento

As operações de frente de onda suportam:

- **[Suavização de Caminho](../smooth)** — Reduz bordas irregulares nas
  trajetórias de ferramenta
- **[Otimização de Caminho](../path-optimization)** — Minimiza a distância
  de deslocamento entre passagens

## Dicas e Melhores Práticas

### Escolhendo o Passo

- Cobertura mais densa (passo pequeno) significa mais passagens e tempos
  de trabalho mais longos
- Cobertura esparsa (passo grande) é mais rápida, mas deixa mais material
  entre as passagens
- Equilibre a densidade com o tempo de trabalho para sua aplicação

### Fabricação de Carimbos e Matrizes

O frente de onda é bem adequado para fabricação de carimbos. Os anéis
concêntricos em expansão limpam naturalmente o bolsão de fundo enquanto
navegam ao redor de características em relevo tratadas como ilhas
interiores.

### Combinando com Contorno

Um fluxo de trabalho comum é limpar o interior do bolsão com frente de
onda e depois finalizar o limite com uma passagem de [Contorno](contour)
para uma borda limpa. Ajuste o deslocamento para deixar margem suficiente
para o corte de contorno.

## Tópicos Relacionados

- **[Contorno](contour)** — Corte ao longo de contornos vetoriais
- **[Gravação](engrave)** — Preenchimento de áreas com padrões de gravação
  raster
- **[Envelopamento Ajustado](shrink-wrap)** — Corte de limite ao redor de
  objetos
- **[Suavização de Caminho](../smooth)** — Refinamento de bordas de
  trajetória de ferramenta
