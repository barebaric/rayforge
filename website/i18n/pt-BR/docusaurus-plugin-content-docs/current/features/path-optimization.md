# Otimização de Caminho

A otimização de caminho reordena segmentos de corte para minimizar a distância de deslocamento. O
laser se move eficientemente entre cortes em vez de saltar aleatoriamente pela área de trabalho.

## Como Funciona

Sem otimização, caminhos são cortados na ordem em que aparecem no seu arquivo de design. A
otimização analisa todos os segmentos de caminho e os reorganiza para que o laser viaje a menor
distância total entre cortes.

**Antes da otimização** — o laser salta de um lado para outro pelo material:

![Os deslocamentos saltam entre cortes na ordem desenhada](/images/path-optimization-before.webp)

**Depois da otimização** — o laser se move sequencialmente de corte para corte:

![Os deslocamentos conectam cortes vizinhos](/images/path-optimization-after.webp)

## Configurações

![As configurações de Otimizar na página de pós-processamento de uma etapa de Contorno](/screenshots/step-settings-contour-post-optimize.webp)

### Habilitar Otimização

Ativa ou desativa a otimização de caminho. Habilitada por padrão para a maioria das operações.

### Escolher o Melhor Ponto de Início

Inicia cada caminho fechado no vértice mais próximo da cabeça do laser, em vez de onde o caminho foi
desenhado. Isso encurta os deslocamentos. Caminhos abertos, e caminhos com entradas, sobrecorte ou
abas de fixação, mantêm seu ponto de início. Desativado por padrão.

### Preferir Cantos

Disponível apenas quando **Escolher o Melhor Ponto de Início** está ativado. Caminhos fechados
começam em um canto vivo quando o caminho tem um, o que deixa a emenda menos visível. Caminhos sem
cantos, como círculos, começam no vértice mais próximo. Desativado por padrão.

O mesmo caminho fechado iniciado onde foi desenhado (1), no vértice mais próximo da cabeça do laser
com **Escolher o Melhor Ponto de Início** (2), e no canto vivo mais próximo com **Preferir Cantos**
(3):

![Pontos de início de um caminho fechado: como desenhado, no vértice mais próximo, no canto mais
próximo](/images/path-optimization-start-point.webp)

## Quando Usar Otimização

**Habilitar para:**

- Designs com muitas formas separadas
- Reduzir tempo total do trabalho
- Minimizar desgaste no sistema de movimento
- Layouts aninhados complexos

**Desabilitar para:**

- Designs onde a ordem de corte importa (ex.: recursos internos antes do externo)
- Depurar problemas de caminho
- Quando você precisa de ordem de execução previsível e repetível

## Como Afeta Seu Trabalho

**Economia de tempo:** Pode reduzir o tempo do trabalho em 20-50% para designs com muitos cortes
separados.

**Eficiência de movimento:** Menos movimento rápido significa menos desgaste em correias, motores e
rolamentos.

**Distribuição de calor:** Caminhos otimizados podem concentrar calor em uma área. Para materiais
sensíveis ao calor, considere se a ordem importa.

<!-- prettier-ignore-start -->
:::tip
A otimização roda automaticamente. Apenas habilite-a e o software cuida do resto.
:::
<!-- prettier-ignore-end -->

---

## Tópicos Relacionados

- [Corte de Contorno](operations/contour) - Operação de corte principal
- [Abas de Fixação](holding-tabs) - Mantendo peças seguras durante o corte
