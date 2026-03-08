# Multi-Passagem

Multi-passagem repete o caminho de corte ou gravação múltiplas vezes, opcionalmente descendo em Z entre passagens. Isso é útil para materiais espessos ou criar gravações mais profundas.

## Como Funciona

Cada passagem traça o mesmo caminho novamente. Com o degrau-Z habilitado, o laser se move mais próximo do material entre passagens, cortando progressivamente mais fundo.

## Configurações

### Número de Passagens

Quantas vezes repetir toda a etapa (1-100). Cada passagem segue o mesmo caminho.

- **1 passagem:** Corte único (padrão)
- **2-3 passagens:** Comum para materiais de espessura média
- **4+ passagens:** Materiais muito espessos ou duros

### Degrau-Z por Passagem

Distância para baixar o eixo Z entre passagens (0-50 mm). Só funciona se sua máquina tiver controle de eixo Z.

- **0 mm:** Todas as passagens na mesma profundidade (padrão)
- **Espessura do material ÷ passagens:** Corte de profundidade progressiva
- **Incrementos pequenos (0.1-0.5mm):** Controle fino para gravação profunda

:::warning Eixo Z Necessário
Degrau-Z só funciona com máquinas que têm controle de eixo Z motorizado. Para máquinas sem eixo Z, todas as passagens ocorrem na mesma altura de foco.
:::

## Quando Usar Multi-Passagem

**Cortando materiais espessos:**

Múltiplas passagens na mesma profundidade frequentemente cortam mais limpo que uma única passagem lenta. A primeira passagem cria um kerf, e passagens subsequentes seguem o mesmo caminho mais eficientemente.

**Gravação profunda:**

Com degrau-Z, você pode esculpir padrões de relevo ou gravações que seriam impossíveis em uma única passagem.

**Qualidade de borda melhorada:**

Múltiplas passagens mais rápidas frequentemente produzem bordas mais limpas que uma passagem lenta, especialmente em materiais que carbonizam facilmente.

## Dicas

- Comece com 2-3 passagens na sua velocidade de corte normal
- Para materiais espessos, aumente passagens em vez de diminuir velocidade
- Habilite degrau-Z só se sua máquina suportar
- Teste em material de sucata para encontrar contagem ideal de passagens

---

## Tópicos Relacionados

- [Corte de Contorno](operations/contour) - Operação de corte principal
- [Gravação](operations/engrave) - Operações de gravação
