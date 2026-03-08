# Suavização de Caminho

A suavização de caminho reduz bordas irregulares e transições abruptas em seus caminhos de corte, resultando em curvas mais limpas e movimento mais suave da máquina.

## Como Funciona

A suavização aplica um filtro à geometria do seu caminho que arredonda cantos angulares e suaviza bordas ásperas. O laser segue uma trajetória mais suave em vez de fazer mudanças abruptas de direção.

## Configurações

### Habilitar Suavização

Ativa ou desativa a suavização para esta operação. A suavização está desabilitada por padrão.

### Suavidade

Controla quanto o caminho é suavizado (0-100). Valores mais altos produzem curvas mais arredondadas, mas podem desviar mais do caminho original.

- **Baixa (0-30):** Suavização mínima, preserva detalhes nítidos
- **Média (30-60):** Suavização equilibrada para a maioria dos designs
- **Alta (60-100):** Suavização agressiva, melhor para formas orgânicas

### Limiar de Ângulo de Canto

Ângulos mais agudos que este valor são preservados como cantos em vez de suavizados (0-179 graus). Isso evita que características afiadas importantes sejam arredondadas.

- **Valores menores:** Mais cantos são suavizados, resultado mais arredondado
- **Valores maiores:** Mais cantos são preservados, resultado mais nítido

## Quando Usar Suavização

**Bom para:**

- Designs importados de fontes baseadas em pixels com degraus
- Reduzir estresse mecânico em mudanças rápidas de direção
- Melhorar qualidade de corte em curvas
- Designs com muitos pequenos segmentos de linha

**Não necessário para:**

- Arte vetorial limpa com curvas bezier suaves
- Designs onde cantos afiados devem ser preservados exatamente
- Desenhos técnicos que requerem geometria precisa

---

## Tópicos Relacionados

- [Corte de Contorno](operations/contour) - Operação de corte principal
- [Otimização de Caminho](path-optimization) - Reduzindo distância de deslocamento
