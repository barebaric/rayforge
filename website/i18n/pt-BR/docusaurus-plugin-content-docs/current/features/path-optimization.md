# Otimização de Caminho

A otimização de caminho reordena segmentos de corte para minimizar a distância de deslocamento. O laser se move eficientemente entre cortes em vez de saltar aleatoriamente pela área de trabalho.

## Como Funciona

Sem otimização, caminhos são cortados na ordem em que aparecem no seu arquivo de design. A otimização analisa todos os segmentos de caminho e os reorganiza para que o laser viaje a menor distância total entre cortes.

**Antes da otimização:** O laser salta de um lado para outro pelo material
**Depois da otimização:** O laser se move sequencialmente de corte para corte

## Configurações

### Habilitar Otimização

Ativa ou desativa a otimização de caminho. Habilitada por padrão para a maioria das operações.

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

**Economia de tempo:** Pode reduzir o tempo do trabalho em 20-50% para designs com muitos cortes separados.

**Eficiência de movimento:** Menos movimento rápido significa menos desgaste em correias, motores e rolamentos.

**Distribuição de calor:** Caminhos otimizados podem concentrar calor em uma área. Para materiais sensíveis ao calor, considere se a ordem importa.

:::tip
A otimização roda automaticamente. Apenas habilite-a e o software cuida do resto.
:::

---

## Tópicos Relacionados

- [Corte de Contorno](operations/contour) - Operação de corte principal
- [Abas de Fixação](holding-tabs) - Mantendo peças seguras durante o corte
