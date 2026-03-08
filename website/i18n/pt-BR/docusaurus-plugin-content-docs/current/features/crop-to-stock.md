# Cortar para Estoque

Cortar para Estoque limita os caminhos de corte ao limite do seu material. Quaisquer cortes que se estendem além da área do estoque são aparados, prevenindo que o laser corte fora do seu material.

## Como Funciona

O transformador compara seus caminhos de corte contra o limite do estoque definido. Segmentos de caminho fora deste limite são removidos ou cortados até a borda do estoque.

## Configurações

### Habilitar Cortar para Estoque

Ativa ou desativa o corte. Desabilitado por padrão.

### Deslocamento

Ajusta o limite efetivo do estoque antes de cortar (-100 a +100 mm).

- **Valores positivos:** Encolhe o limite (corta mais conservadoramente)
- **Valores negativos:** Expande o limite (permite cortes mais próximos da borda)
- **0 mm:** Usa o limite exato do estoque

Use deslocamento quando você quer uma margem de segurança da borda do estoque, ou quando o posicionamento do seu material não está perfeitamente alinhado.

## Quando Usar Cortar para Estoque

**Designs parciais:** Seu design é maior que seu material, mas você quer cortar apenas a porção que cabe.

**Margem de segurança:** Previne cortes acidentais além das bordas do material.

**Chapas aninhadas:** Corta apenas as peças que cabem no seu pedaço atual de material.

**Cortes de teste:** Limita um teste a uma área específica do seu material.

## Exemplo

Você tem um design grande mas apenas um pedaço pequeno de material:

1. Defina o tamanho do seu estoque para corresponder ao seu material
2. Habilite Cortar para Estoque
3. Defina deslocamento para 2mm para margem de segurança
4. Apenas as porções dentro do limite do seu material serão cortadas

---

## Tópicos Relacionados

- [Manuseio de Estoque](stock-handling) - Configurando limites de material
- [Corte de Contorno](operations/contour) - Operação de corte principal
