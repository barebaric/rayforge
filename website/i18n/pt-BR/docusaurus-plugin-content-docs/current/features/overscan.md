# Overscan

Overscan estende linhas de gravação raster além da área de conteúdo real para garantir que o laser atinja velocidade constante durante a gravação, eliminando artefatos de aceleração.

## O Problema: Marcas de Aceleração

Sem overscan, gravação raster sofre de **artefatos de aceleração**:

- **Bordas claras** onde aceleração começa (laser se movendo muito rápido para nível de potência)
- **Bordas escuras** onde desaceleração ocorre (laser permanecendo mais tempo)
- **Profundidade/escuridão de gravação inconsistente** através de cada linha
- Faixas ou listras visíveis nas bordas das linhas

## Como Overscan Funciona

Overscan **estende o caminho da ferramenta** antes e depois de cada linha raster:

**Processo:**

1. **Entrada:** Laser move para uma posição _antes_ da linha começar
2. **Acelera:** Laser acelera para velocidade alvo (laser DESLIGADO)
3. **Grava:** Laser liga e grava em velocidade constante
4. **Desacelera:** Laser desliga e desacelera _após_ a linha terminar

**Resultado:** Toda a área gravada recebe potência consistente em velocidade constante.

**Benefícios:**

- Profundidade de gravação uniforme através de toda linha raster
- Sem bordas claras/escuras
- Gravação de foto de maior qualidade
- Resultados de aparência profissional

## Configurando Overscan

Overscan é um **transformador** no fluxo de trabalho de pipeline do Rayforge.

**Para habilitar:**

1. **Selecione a camada** com gravação raster
2. **Abra configurações de fluxo de trabalho** (ou configurações de operação)
3. **Adicione transformador Overscan** se ainda não presente
4. **Configure distância**

**Configurações:**

| Configuração           | Descrição             | Valor Típico   |
| ----------------- | ----------------------- | --------------- |
| **Habilitado**       | Alterna overscan ligado/desligado  | LIGADO (para raster) |
| **Distância (mm)** | Quão longe estender linhas | 2-5 mm          |

## Escolhendo Distância de Overscan

A distância de overscan deve permitir que a máquina **acelere completamente** para velocidade alvo.

**Diretrizes práticas:**

| Velocidade Máx              | Aceleração | Overscan Recomendado |
| ---------------------- | ------------ | -------------------- |
| 3000 mm/min (50 mm/s)  | Baixa          | 5 mm                 |
| 3000 mm/min (50 mm/s)  | Média       | 3 mm                 |
| 3000 mm/min (50 mm/s)  | Alta        | 2 mm                 |
| 6000 mm/min (100 mm/s) | Baixa          | 10 mm                |
| 6000 mm/min (100 mm/s) | Média       | 6 mm                 |
| 6000 mm/min (100 mm/s) | Alta        | 4 mm                 |

**Fatores afetando distância necessária:**

- **Velocidade:** Velocidade mais alta = precisa de mais distância para acelerar
- **Aceleração:** Aceleração mais baixa = precisa de mais distância
- **Mecânica da máquina:** Transmissão por correia vs direta afeta aceleração

**Ajuste fino:**

- **Muito pouco:** Marcas de aceleração ainda visíveis nas bordas
- **Muito:** Perde tempo, pode atingir limites da máquina
- **Comece com 3mm** e ajuste com base nos resultados

## Testando Configurações de Overscan

**Procedimento de teste:**

1. **Crie uma gravação de teste:**
   - Retângulo preenchido sólido (50mm x 20mm)
   - Use suas configurações típicas de gravação
   - Habilite overscan em 3mm

2. **Grave o teste:**
   - Execute o trabalho
   - Permita completar

3. **Examine as bordas:**
   - Olhe para bordas esquerda e direita do retângulo
   - Verifique variação de escuridão nas bordas
   - Compare escuridão da borda com escuridão do centro

4. **Ajuste:**
   - **Se bordas são mais claras/escuras:** Aumente overscan
   - **Se bordas correspondem ao centro:** Overscan é suficiente
   - **Se bordas são perfeitas:** Tente reduzir overscan ligeiramente para economizar tempo

## Quando Usar Overscan

**Sempre use para:**

- Gravação de fotos (raster)
- Padrões de preenchimento
- Qualquer trabalho raster de alto detalhe
- Gravação de imagem em escala de cinza
- Gravação de texto (modo raster)

**Opcional para:**

- Corte vetorial (não necessário)
- Gravação muito lenta (aceleração menos perceptível)
- Formas simples grandes (bordas menos críticas)

**Desabilite para:**

- Operações vetoriais
- Áreas de trabalho muito pequenas (pode exceder limites)
- Quando qualidade de borda não é importante

---

## Tópicos Relacionados

- [Operações de Gravação](./operations/engrave) - Configure configurações de gravação
- [Grade de Teste de Material](./operations/material-test-grid) - Encontre potência/velocidade ideal
