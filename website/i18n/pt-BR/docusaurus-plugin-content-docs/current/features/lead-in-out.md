# Aproximação / Saída

Os movimentos de aproximação e saída estendem cada caminho de contorno com segmentos curtos sem potência antes do início do corte e após o seu término. Isso dá tempo para a cabeça do laser atingir uma velocidade estável antes do início real do corte e desacelerar gradualmente após o fim do corte, o que produz resultados mais limpos nos pontos de início e fim de cada corte.

## Como Funciona

Quando a aproximação/saída está habilitada, o Rayforge observa a direção tangente de cada caminho de contorno nos seus pontos de início e fim. Em seguida, insere um movimento retilíneo curto sem potência do laser ao longo dessa tangente antes do primeiro ponto de corte e outro após o último ponto de corte. O laser está desligado durante esses segmentos adicionais, então nenhum material é removido fora do caminho pretendido.

## Configurações

### Habilitar Aproximação/Saída

Ativa ou desativa a função para a operação. Quando desabilitada, o corte começa e termina exatamente nos pontos finais do caminho sem movimentos adicionais de aproximação ou saída.

### Distância Automática

Quando esta opção está habilitada, o Rayforge calcula automaticamente a distância de aproximação e saída com base na velocidade de corte e na configuração de aceleração da máquina. A fórmula usa um fator de segurança de dois para garantir que a cabeça do laser tenha espaço suficiente para atingir a velocidade máxima. Sempre que você alterar a velocidade de corte ou a aceleração da máquina for atualizada, a distância é recalculada.

### Distância de Aproximação

O comprimento do movimento de aproximação sem potência antes do início do corte, em milímetros. O padrão é 2 mm. Este campo só é editável quando a distância automática está desabilitada.

### Distância de Saída

O comprimento do movimento de saída sem potência após o fim do corte, em milímetros. O padrão é 2 mm. Este campo só é editável quando a distância automática está desabilitada.

## Quando Usar Aproximação/Saída

A aproximação/saída é mais útil quando você nota marcas de queima, queima excessiva ou qualidade de corte inconsistente nos pontos de início e fim dos seus contornos. A aproximação sem potência dá à máquina tempo para acelerar até a velocidade de corte para que o laser atinja o material em velocidade máxima, e a saída sem potência permite uma desaceleração suave em vez de permanecer com potência total no último ponto.

Está disponível como opção de pós-processamento em operações de contorno, contorno de moldura e shrink wrap.

---

## Páginas Relacionadas

- [Corte de Contorno](operations/contour) - Operação de corte principal
- [Contorno de Moldura](operations/frame-outline) - Corte de limite retangular
- [Shrink Wrap](operations/shrink-wrap) - Corte de limite eficiente
- [Abas de Fixação](holding-tabs) - Manter peças seguras durante o corte
