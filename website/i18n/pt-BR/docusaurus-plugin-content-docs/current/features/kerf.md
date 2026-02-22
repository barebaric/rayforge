# Compensação de Kerf

Kerf é o material removido pelo feixe do laser durante o corte. A compensação de kerf ajusta os caminhos da ferramenta para considerar isso, garantindo que as peças cortadas correspondam às suas dimensões projetadas.

## O Que é Kerf?

**Kerf** = a largura do material removido pelo processo de corte.

**Exemplo:**
- Tamanho do ponto do laser: 0.2mm
- Interação com material: adiciona ~0.1mm em cada lado
- **Kerf total:** ~0.4mm

---

## Como Funciona a Compensação de Kerf

A compensação de kerf **desloca o caminho da ferramenta** para dentro ou para fora para considerar a remoção de material:

**Para cortes externos (cortando uma peça):**
- Desloca o caminho **para fora** pela metade da largura do kerf
- Resultado: A peça final tem o tamanho correto

**Para cortes internos (cortando um furo):**
- Desloca o caminho **para dentro** pela metade da largura do kerf
- Resultado: O furo final tem o tamanho correto

**Exemplo com kerf de 0.4mm:**

```
Caminho original:  quadrado de 50mm
Compensação:       Deslocamento para fora de 0.2mm (metade do kerf)
Laser segue:       quadrado de 50.4mm
Após o corte:      Peça mede 50.0mm (perfeito!)
```

---

## Medindo o Kerf

**Procedimento preciso de medição de kerf:**

1. **Crie um arquivo de teste:**
   - Desenhe um quadrado de 50mm x 50mm
   - Desenhe um círculo (qualquer tamanho, para teste de corte interno)

2. **Corte o teste:**
   - Use suas configurações normais de corte
   - Corte completamente
   - Deixe o material esfriar

3. **Meça:**
   - **Quadrado externo (peça):** Meça com paquímetro
     - Se < 50mm, o kerf foi removido para fora
     - Kerf = (50 - medido) x 2
   - **Círculo interno (furo):** Meça o diâmetro
     - Se > diâmetro projetado, o kerf foi removido para dentro
     - Kerf = (medido - projetado) / 2

4. **Média:** Use a média de múltiplas medições

**Variáveis que afetam o kerf:**
- Potência do laser (maior = mais largo)
- Velocidade de corte (mais lento = mais largo)
- Tipo e densidade do material
- Distância de foco
- Pressão do assistente de ar

---

## Compensação Manual de Kerf

Se a compensação automática de kerf não estiver disponível, compense no seu software de design:

**Inkscape:**

1. **Selecione o caminho**
2. **Caminho → Deslocamento Dinâmico** (Ctrl+J)
3. **Arraste para deslocar** pela metade da sua medição de kerf
   - Para fora para peças (para aumentar o caminho)
   - Para dentro para furos (para diminuir o caminho)
4. **Caminho → Objeto para Caminho** para finalizar

**Illustrator:**

1. **Selecione o caminho**
2. **Objeto → Caminho → Deslocar Caminho**
3. **Insira o valor de deslocamento:** (kerf / 2)
   - Positivo para fora, negativo para dentro
4. **OK** para aplicar

**Fusion 360 / CAD:**

- Desloque as entidades do esboço antes de exportar
- Use a dimensão kerf/deslocamento

---

## Páginas Relacionadas

- [Operação de Contorno](operations/contour) - Operações de corte
- [Grade de Teste de Material](operations/material-test-grid) - Encontre configurações ideais
