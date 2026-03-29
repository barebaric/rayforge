# Fluxo de trabalho com múltiplas camadas

O sistema de múltiplas camadas do Rayforge permite organizar trabalhos complexos
em estágios de processamento separados, cada um com suas próprias operações e
configurações. Isso é essencial para combinar diferentes processos como gravação
e corte, ou para trabalhar com múltiplos materiais.

## O que são camadas?

Uma **camada** no Rayforge é:

- **Um contêiner** para peças de trabalho (formas importadas, imagens, texto)
- **Um fluxo de trabalho** que define como essas peças são processadas
- **Uma etapa** processada sequencialmente durante os trabalhos

**Conceito-chave:** As camadas são processadas em ordem, uma após a outra,
permitindo que você controle a sequência de operações.

:::note Camadas e peças de trabalho
Uma camada contém uma ou mais peças de trabalho. Ao importar arquivos SVG com
camadas, cada camada do seu design se torna uma camada separada no Rayforge.
Isso permite manter seu design organizado exatamente como você o criou.
:::

---

## Por que usar múltiplas camadas?

### Casos de uso comuns

**1. Gravar e depois cortar**

O fluxo de trabalho com múltiplas camadas mais comum:

- **Camada 1:** Gravação raster do design
- **Camada 2:** Corte de contorno do perfil

**Por que camadas separadas?**

- Gravar primeiro garante que a peça não se mova durante a gravação
- Cortar por último evita que as peças caiam antes de completar a gravação
- Diferentes configurações de potência/velocidade para cada operação

**2. Corte em múltiplas passagens**

Para materiais espessos:

- **Camada 1:** Primeira passagem com potência moderada
- **Camada 2:** Segunda passagem com potência máxima (mesma geometria)
- **Camada 3:** Terceira passagem opcional, se necessário

**Benefícios:**

- Reduz o carbonizado em comparação com uma única passagem de alta potência
- Cada camada pode ter diferentes configurações de velocidade/potência

**3. Projetos com múltiplos materiais**

Diferentes materiais em um único trabalho:

- **Camada 1:** Cortar peças de acrílico
- **Camada 2:** Gravar peças de madeira
- **Camada 3:** Marcar peças de metal

**Requisitos:**

- Cada camada direciona áreas diferentes da mesa de trabalho
- Diferentes velocidade/potência/foco para cada material

**4. Importação de camadas SVG**

Importar arquivos SVG com estrutura de camadas existente:

- **Camada 1:** Elementos de gravação do SVG
- **Camada 2:** Elementos de corte do SVG
- **Camada 3:** Elementos de vinco do SVG

**Fluxo de trabalho:**

- Importar um arquivo SVG que possua camadas
- Ativar "Usar vetores originais" no diálogo de importação
- Selecionar quais camadas importar da lista de camadas detectadas
- Cada camada se torna uma camada separada no Rayforge

**Requisitos:**

- Seu arquivo SVG deve usar camadas (criadas no Inkscape ou software similar)
- Ativar "Usar vetores originais" ao importar
- Os nomes das camadas são preservados do seu software de design

---

## Criando e gerenciando camadas

### Adicionar uma nova camada

1. **Clique no botão "+"** no painel de Camadas
2. **Nomeie a camada** de forma descritiva (ex.: "Camada de gravação",
   "Camada de corte")
3. **A camada aparece** na lista de camadas

**Padrão:** Novos documentos começam com uma camada.

### Propriedades das camadas

Cada camada possui:

| Propriedade           | Descrição                                           |
| --------------------- | --------------------------------------------------- |
| **Nome**              | O nome exibido na lista de camadas                  |
| **Visível**           | Alternar visibilidade na tela e na pré-visualização |
| **Fluxo de trabalho** | As operações aplicadas às peças nesta camada        |
| **Rotativo**          | Se esta camada é executada em modo rotativo         |
| **Peças de trabalho** | As formas/imagens contidas nesta camada             |

### Modo rotativo por camada

Se você tem um [acessório rotativo](../machine/rotary) configurado, pode ativar
o modo rotativo para camadas individuais. Isso permite combinar trabalho em
superfície plana e cilíndrica no mesmo projeto — por exemplo, gravar um design
na tampa plana de uma caixa em uma camada e envolver texto ao redor do corpo
cilíndrico em outra.

Camadas com o modo rotativo ativo exibem um pequeno ícone rotativo na lista de
camadas. Cada camada lembra sua própria configuração rotativa, permitindo
misturá-las livremente.

:::note Camadas como contêineres
Camadas são contêineres para suas peças de trabalho. Ao importar arquivos SVG
com camadas, cada camada do seu design se torna uma camada separada no Rayforge.
:::

### Reordenar camadas

**Ordem de execução = ordem das camadas na lista (de cima para baixo)**

Para reordenar:

1. **Arraste e solte** as camadas no painel de Camadas
2. **A ordem importa** - as camadas são executadas de cima para baixo

**Exemplo:**

```
Painel de Camadas:
1. Camada de gravação     Executa primeiro
2. Camada de vinco        Executa segundo
3. Camada de corte        Executa por último (recomendado)
```

### Excluir camadas

1. **Selecione a camada** no painel de Camadas
2. **Clique no botão de exclusão** ou pressione Delete
3. **Confirme a exclusão** (todas as peças de trabalho da camada são removidas)

:::warning A exclusão é permanente
Excluir uma camada remove todas as suas peças de trabalho e configurações de
fluxo de trabalho. Use Desfazer se excluir acidentalmente.
:::

---

## Atribuir peças de trabalho às camadas

### Atribuição manual

1. **Importe ou crie** uma peça de trabalho
2. **Arraste a peça de trabalho** para a camada desejada no painel de Camadas
3. **Ou use o painel de propriedades** para alterar a camada da peça

### Importação de camadas SVG

Ao importar arquivos SVG com "Usar vetores originais" ativado:

1. **Ative "Usar vetores originais"** no diálogo de importação
2. **O Rayforge detecta as camadas** do seu arquivo SVG
3. **Selecione quais camadas** importar usando os interruptores de camada
4. **Cada camada selecionada** se torna uma camada separada com sua própria peça

:::note Detecção de camadas
O Rayforge detecta automaticamente as camadas do seu arquivo SVG. Cada camada
que você criou no seu software de design aparecerá como uma camada separada no
Rayforge.
:::

:::note Somente importação de vetores
A seleção de camadas só está disponível ao usar a importação direta de vetores.
Ao usar o modo de traço, o SVG inteiro é processado como uma única peça de
trabalho.
:::

### Mover peças de trabalho entre camadas

**Arrastar e soltar:**

- Selecione a(s) peça(s) de trabalho na tela ou no painel de Documento
- Arraste para a camada de destino no painel de Camadas

**Recortar e colar:**

- Recorte a peça de trabalho da camada atual (Ctrl+X)
- Selecione a camada de destino
- Cole (Ctrl+V)

### Diálogo de importação SVG

Ao importar arquivos SVG, o diálogo de importação oferece opções que afetam o
gerenciamento de camadas:

**Modo de importação:**

- **Usar vetores originais:** Preserva seus caminhos vetoriais e a estrutura de
  camadas. Quando ativado, uma seção "Camadas" aparece mostrando todas as
  camadas do seu arquivo.
- **Modo de traço:** Converte o SVG em um bitmap e traça os contornos. A
  seleção de camadas é desativada neste modo.

**Seção de camadas (somente importação de vetores):**

- Mostra todas as camadas do seu arquivo SVG
- Cada camada possui um interruptor para ativar/desativar a importação
- Os nomes das camadas do seu software de design são preservados
- Apenas as camadas selecionadas são importadas como camadas separadas

:::tip Preparando arquivos SVG para importação de camadas
Para usar a importação de camadas SVG, crie seu design com camadas em software
como o Inkscape. Use o painel de Camadas para organizar seu design, e o
Rayforge preservará essa estrutura.
:::

---

## Fluxos de trabalho das camadas

Cada camada possui um **fluxo de trabalho** que define como suas peças de
trabalho são processadas.

### Configurar fluxos de trabalho das camadas

Para cada camada, você escolhe um tipo de operação e configura seus parâmetros:

**Tipos de operação:**

- **Contorno** - Segue os contornos (para corte ou vinco)
- **Gravação raster** - Grava imagens e preenche áreas
- **Gravação em profundidade** - Cria gravações com profundidade variável

**Aprimoramentos opcionais:**

- **Abas** - Pequenas pontes para manter as peças no lugar durante o corte
- **Overscan** - Estende os cortes além da forma para bordas mais limpas
- **Ajuste de kerf** - Compensa a largura de corte do laser

### Configurações comuns de camadas

**Camada de gravação:**

- Operação: Gravação raster
- Configurações: 300-500 DPI, velocidade moderada
- Geralmente não precisa de opções adicionais

**Camada de corte:**

- Operação: Corte de contorno
- Opções: Abas (para segurar as peças), Overscan (para bordas limpias)
- Configurações: Velocidade mais lenta, potência mais alta

**Camada de vinco:**

- Operação: Contorno (potência leve, não corta completamente)
- Configurações: Baixa potência, velocidade rápida
- Finalidade: Linhas de dobra, linhas decorativas

---

## Visibilidade das camadas

Controle quais camadas são exibidas na tela e nas pré-visualizações:

### Visibilidade na tela

- **Ícone de olho** no painel de Camadas alterna a visibilidade
- **Camadas ocultas:**
  - Não exibidas na tela 2D
  - Não exibidas na pré-visualização 3D
  - **Ainda incluídas no G-code gerado**

**Casos de uso:**

- Ocultar camadas de gravação complexas enquanto posiciona camadas de corte
- Desobstruir a tela ao trabalhar em camadas específicas
- Focar em uma camada por vez

### Visível vs. Ativado

| Estado                   | Tela | Pré-visualização | G-code |
| ------------------------ | ---- | ---------------- | ------ |
| **Visível e ativado**    | Sim  | Sim              | Sim    |
| **Oculto e ativado**     | Não  | Não              | Sim    |
| **Visível e desativado** | Sim  | Sim              | Não    |
| **Oculto e desativado**  | Não  | Não              | Não    |

:::note Desativando camadas
:::

Para excluir temporariamente uma camada dos trabalhos sem excluí-la, desative a
operação da camada ou desative-a nas configurações da camada.

---

## Ordem de execução das camadas

### Como as camadas são processadas

Durante a execução do trabalho, o Rayforge processa cada camada em ordem, de
cima para baixo. Dentro de cada camada, todas as peças de trabalho são
processadas antes de passar para a próxima camada.

### A ordem importa

**Ordem incorreta:**

```
1. Camada de corte
2. Camada de gravação
```

**Problema:** As peças cortadas podem cair ou se mover antes da gravação!

**Ordem correta:**

```
1. Camada de gravação
2. Camada de corte
```

**Por quê:** A gravação ocorre enquanto a peça ainda está fixa, depois o corte
a liberta.

### Múltiplas passagens

Para materiais espessos, crie múltiplas camadas de corte:

```
1. Camada de gravação
2. Camada de corte (Passagem 1) - 50% potência
3. Camada de corte (Passagem 2) - 75% potência
4. Camada de corte (Passagem 3) - 100% potência
```

**Dica:** Use a mesma geometria para todas as passagens de corte (duplique a
camada).

---

## Técnicas avançadas

### Agrupamento de camadas por material

Use camadas para organizar por material ao executar trabalhos mistos:

```
Material 1 (Acrílico 3mm):
  - Camada de gravação do acrílico
  - Camada de corte do acrílico

Material 2 (Compensado 3mm):
  - Camada de gravação da madeira
  - Camada de corte da madeira
```

**Fluxo de trabalho:**

1. Processar todas as camadas do Material 1
2. Trocar os materiais
3. Processar todas as camadas do Material 2

**Alternativa:** Use documentos separados para diferentes materiais.

### Pausar entre camadas

Você pode configurar o Rayforge para fazer uma pausa entre as camadas. Isso é
útil quando você precisa:

- Trocar de material no meio do trabalho
- Inspecionar o progresso antes de continuar
- Ajustar o foco para diferentes operações

Para configurar pausas entre camadas, use o recurso de hooks nas configurações
da sua máquina.

### Configurações específicas por camada

O fluxo de trabalho de cada camada pode ter configurações exclusivas:

| Camada   | Operação | Velocidade | Potência | Passagens |
| -------- | -------- | ---------- | -------- | --------- |
| Gravação | Raster   | 300 mm/min | 20%      | 1         |
| Vinco    | Contorno | 500 mm/min | 10%      | 1         |
| Corte    | Contorno | 100 mm/min | 90%      | 2         |

---

## Melhores práticas

### Convenções de nomenclatura

**Bons nomes de camadas:**

- "Gravação - Logo"
- "Corte - Contorno externo"
- "Vinco - Linhas de dobra"
- "Passagem 1 - Corte bruto"
- "Passagem 2 - Corte final"

**Nomes de camadas ruins:**

- "Camada 1", "Camada 2" (não descritivos)
- Descrições longas (mantenha conciso)

### Organização das camadas

1. **De cima para baixo = ordem de execução**
2. **Gravação antes do corte** (regra geral)
3. **Agrupe operações relacionadas** (todo o corte, toda a gravação)
4. **Use a visibilidade** para focar no trabalho atual
5. **Exclua camadas não usadas** para manter os projetos organizados

### Preparando arquivos SVG para importação de camadas

**Para melhores resultados ao importar camadas SVG:**

1. **Use o painel de Camadas** no seu software de design para organizar seu
   design
2. **Atribua nomes significativos** a cada camada (ex.: "Gravação", "Corte")
3. **Mantenha as camadas planas** - evite colocar camadas dentro de outras
   camadas
4. **Salve seu arquivo** e importe no Rayforge
5. **Verifique a detecção de camadas** conferindo o diálogo de importação

O Rayforge funciona melhor com arquivos SVG criados no Inkscape ou software
similar de design vetorial que suporte camadas.

### Desempenho

**Muitas camadas:**

- Sem impacto significativo no desempenho
- 10 a 20 camadas é comum para trabalhos complexos
- Organize logicamente, não para minimizar a quantidade de camadas

**Simplifique se necessário:**

- Combine operações similares em uma camada quando possível
- Use menos gravações raster (as que mais consomem recursos)

---

## Solução de problemas

### A camada não gera G-code

**Problema:** A camada aparece no documento mas não no G-code gerado.

**Soluções:**

1. **Verifique se a camada tem peças de trabalho** - Camadas vazias são
   ignoradas
2. **Verifique se o fluxo de trabalho está configurado** - A camada precisa de
   uma operação
3. **Verifique as configurações da operação** - Potência > 0, velocidade
   válida, etc.
4. **Verifique a visibilidade das peças** - Peças ocultas podem não ser
   processadas
5. **Regenere o G-code** - Faça uma pequena alteração para forçar a regeneração

### Ordem incorreta das camadas

**Problema:** As operações são executadas em uma ordem inesperada.

**Solução:** Reordene as camadas no painel de Camadas. Lembre-se: cima =
primeiro.

### Camadas sobrepostas na pré-visualização

**Problema:** Múltiplas camadas mostram conteúdo sobreposto na pré-visualização.

**Esclarecimento:** Isso é normal se as camadas compartilham a mesma área XY.

**Soluções:**

- Use a visibilidade de camadas para ocultar outras camadas temporariamente
- Verifique a pré-visualização 3D para ver a profundidade/ordem
- Verifique se isso é intencional (ex.: gravar e depois cortar a mesma forma)

### Peça de trabalho na camada errada

**Problema:** A peça de trabalho foi atribuída a uma camada incorreta.

**Solução:** Arraste a peça de trabalho para a camada correta no painel de
Camadas ou na árvore de Documento.

### Camadas SVG não detectadas

**Problema:** Importando um arquivo SVG mas nenhuma camada aparece no diálogo de
importação.

**Soluções:**

1. **Verifique a estrutura do SVG** - Abra seu arquivo no Inkscape ou software
   similar para verificar se ele possui camadas
2. **Ative "Usar vetores originais"** - A seleção de camadas só está disponível
   neste modo de importação
3. **Verifique se seu design tem camadas** - Certifique-se de ter criado
   camadas no seu software de design, não apenas grupos
4. **Verifique camadas aninhadas** - Camadas dentro de outras camadas podem não
   ser detectadas corretamente
5. **Salve novamente seu arquivo** - Às vezes, salvar novamente com uma versão
   atualizada do seu software de design ajuda

### A importação de camadas SVG mostra conteúdo incorreto

**Problema:** A camada importada mostra conteúdo de outras camadas ou está vazia.

**Soluções:**

1. **Verifique a seleção de camadas** - Confirme se as camadas corretas estão
   ativadas no diálogo de importação
2. **Verifique seu design** - Abra o arquivo original no seu software de design
   para confirmar que cada camada contém o conteúdo correto
3. **Verifique elementos compartilhados** - Elementos que aparecem em múltiplas
   camadas podem causar confusão
4. **Experimente o modo de traço** - Use o modo de traço como alternativa se a
   importação vetorial tiver problemas

---

## Páginas relacionadas

- [Operações](./operations/contour) - Tipos de operações para fluxos de
  trabalho de camadas
- [Modo de simulação](./simulation-mode) - Pré-visualização da execução com
  múltiplas camadas
- [Macros e Hooks](../machine/hooks-macros) - Hooks em nível de camada para
  automação
- [Pré-visualização 3D](../ui/3d-preview) - Visualizar a pilha de camadas
