# Abas de Fixação

Abas de fixação (também chamadas de pontes ou abas) são pequenas seções não cortadas deixadas ao longo dos caminhos de corte que mantêm as peças anexadas ao material circundante. Isso previne que peças cortadas se movam durante o trabalho, o que poderia causar desalinhamento, dano ou riscos de incêndio.

## Por Que Usar Abas de Fixação?

Ao cortar através do material, a peça cortada pode:

- **Mudar de posição** no meio do trabalho, fazendo operações subsequentes desalinharem
- **Cair através** da grade da mesa ou tombar se apoiada apenas nas bordas
- **Colidir com** a cabeça do laser enquanto se move
- **Pegar fogo** se cair em sucata quente abaixo
- **Ser danificada** por queda ou vibração

Abas de fixação resolvem esses problemas mantendo a peça anexada até você estar pronto para removê-la.

---

## Como Abas de Fixação Funcionam

O Rayforge implementa abas criando **pequenas lacunas no caminho de corte**:

1. Você marca posições ao longo do caminho de corte onde abas devem estar
2. Durante geração de G-code, o Rayforge interrompe o corte em cada aba
3. O laser levanta (ou desliga), pula a largura da aba, então retoma o corte
4. Após o trabalho completar, você manualmente quebra ou corta as abas para libertar a peça

---

## Adicionando Abas de Fixação

### Adição Rápida

1. **Selecione a peça** à qual deseja adicionar abas (deve ser uma operação de corte/contorno)
2. **Clique na ferramenta de aba** na barra de ferramentas ou pressione o atalho de aba
3. **Clique no caminho** onde deseja abas:
   - Abas aparecem como pequenas alças no contorno do caminho
   - Clique múltiplas vezes para adicionar mais abas
   - Tipicamente 3-4 abas para peças pequenas, mais para peças maiores
4. **Habilite abas** se ainda não habilitadas (alterne no painel de propriedades)

### Usando o Popover Adicionar Abas

Para mais controle:

1. **Clique direito** na peça ou use **Editar → Adicionar Abas**
2. **Escolha método de colocação de aba:**
   - **Manual:** Clique em localizações individuais
   - **Equidistante:** Espaça abas automaticamente de forma uniforme ao redor do caminho
3. **Configure configurações de aba:**
   - **Número de abas:** Quantas abas criar (para equidistante)
   - **Largura da aba:** Comprimento de cada seção não cortada (tipicamente 2-5mm)
4. **Clique em Aplicar**

---

## Propriedades da Aba

### Largura da Aba

A **largura** é o comprimento da seção não cortada ao longo do caminho.

**Larguras recomendadas:**

| Material | Espessura | Largura da Aba |
|----------|-----------|-----------|
| **Papelão** | 1-3mm | 2-3mm |
| **Compensado** | 3mm | 3-4mm |
| **Compensado** | 6mm | 4-6mm |
| **Acrílico** | 3mm | 2-3mm |
| **Acrílico** | 6mm | 3-5mm |
| **MDF** | 3mm | 3-4mm |
| **MDF** | 6mm | 5-7mm |

**Diretrizes:**
- **Materiais mais espessos** precisam de abas mais largas para força
- **Peças mais pesadas** precisam de mais e/ou abas mais largas
- **Materiais frágeis** (acrílico) podem usar abas menores (mais fáceis de quebrar)
- **Materiais fibrosos** (madeira) podem precisar de abas mais largas

:::warning Largura da Aba vs Espessura do Material
Abas devem ser largas o suficiente para suportar a peça mas pequenas o suficiente para remover limpo. Muito estreita = peça pode quebrar livre; muito larga = difícil de remover ou dana a peça.
:::

### Posição da Aba

Abas são posicionadas usando dois parâmetros:

- **Índice do segmento:** Qual segmento de linha/arco do caminho
- **Posição (0.0 - 1.0):** Onde ao longo daquele segmento (0 = início, 1 = fim)

**Dicas de colocação manual:**
- Coloque abas em **seções retas** quando possível (mais fácil de remover)
- Evite abas em **curvas fechas** (concentração de tensão)
- Distribua abas **uniformemente** ao redor da peça
- Coloque abas em **cantos** para suporte máximo se necessário

### Abas Equidistantes

O recurso **equidistante** coloca abas automaticamente em intervalos uniformes:

**Benefícios:**
- Distribuição de peso uniforme
- Padrão de quebra previsível
- Configuração rápida para formas regulares

---

## Trabalhando com Abas

### Editando Abas

**Mover uma aba:**
1. Selecione a peça
2. Arraste a alça da aba ao longo do caminho
3. Solte para definir nova posição

**Redimensionar uma aba:**
- Use o painel de propriedades para ajustar a largura
- Todas as abas em uma peça compartilham a mesma largura

**Excluir uma aba:**
1. Clique na alça da aba para selecioná-la
2. Pressione Delete ou use o menu de contexto
3. Ou limpe todas as abas e comece novamente

### Habilitando/Desabilitando Abas

Alterne abas ligado/desligado sem excluí-las:

- **Painel de propriedades da peça:** Caixa de seleção "Habilitar Abas"
- **Barra de ferramentas:** Ícone de alternância de visibilidade de aba

**Quando desabilitado:**
- Abas não são geradas no G-code
- Alças de aba estão ocultas na tela
- O caminho corta completamente através

**Caso de uso:** Desabilite temporariamente abas para testar o corte, então reabilite para produção.

---

## Removendo Abas Após o Corte

**Ferramentas:**
- Estilete ou cortador
- Alicates de corte diagonal
- Formão (para madeira)
- Serra fina para materiais espessos

**Técnica:**

1. **Marque a aba** de ambos os lados se acessível
2. **Dobre suavemente** a peça para estressar a aba
3. **Corte através** do material restante
4. **Lixar ou arquivar** o remanescente da aba nivelado com a borda

**Para materiais frágeis (acrílico):**
- Use abas mínimas (elas quebram facilmente)
- Marque profundamente antes de quebrar
- Apoie a peça enquanto quebra abas para evitar rachaduras

**Para madeira:**
- Abas podem requerer corte (não quebram limpo)
- Use uma faca afiada ou formão
- Corte nivelado, então lixe suave

---

## Páginas Relacionadas

- [Corte de Contorno](operations/contour) - Operação primária que usa abas
- [Fluxo de Trabalho Multi-Camadas](multi-layer) - Gerenciando abas através de múltiplas camadas
- [Visualização 3D](../ui/3d-preview) - Visualizando abas na pré-visualização
- [Modo Simulação](simulation-mode) - Pré-visualizando cortes com lacunas de aba
