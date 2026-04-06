# Importando arquivos

O Rayforge suporta a importação de vários formatos de arquivo, vetoriais e
raster. Esta página explica como importar arquivos e otimizá-los para obter os
melhores resultados.

## Formatos de arquivo suportados

### Formatos vetoriais

| Formato   | Extensão | Método de importação           | Ideal para                             |
| --------- | -------- | ------------------------------ | -------------------------------------- |
| **SVG**   | `.svg`   | Vetores diretos ou vetorização | Gráficos vetoriais, logos, designs     |
| **DXF**   | `.dxf`   | Vetores diretos                | Desenhos CAD, projetos técnicos        |
| **PDF**   | `.pdf`   | Vetores diretos ou vetorização | Documentos com conteúdo vetorial       |
| **Ruida** | `.rd`    | Vetores diretos                | Arquivos de trabalho controlador Ruida |

### Formatos raster

| Formato  | Extensão        | Método de importação | Ideal para                         |
| -------- | --------------- | -------------------- | ---------------------------------- |
| **PNG**  | `.png`          | Vetorização          | Fotos, imagens com transparência   |
| **JPEG** | `.jpg`, `.jpeg` | Vetorização          | Fotos, imagens de tom contínuo     |
| **BMP**  | `.bmp`          | Vetorização          | Gráficos simples, capturas de tela |

:::note Importação de imagens raster
:::

Todas as imagens raster são **vetorizadas** para criar caminhos vetoriais que
podem ser usados em operações a laser. A qualidade depende da configuração da
vetorização.

---

## Importando arquivos

### O diálogo de importação

O Rayforge possui um diálogo de importação unificado que oferece visualização em
tempo real e opções de configuração para todos os tipos de arquivo suportados. O
diálogo permite:

- **Visualizar a importação** antes de adicioná-la ao documento
- **Configurar as definições de vetorização** para imagens raster
- **Escolher o método de importação** para arquivos SVG (vetores diretos ou
  vetorização)
- **Ajustar parâmetros** como limiar, inverter e limiar automático

![Diálogo de importação](/screenshots/import-dialog.png)

### Método 1: Menu Arquivo

1. **Importar arquivo** (ou Ctrl+I)
2. **Selecionar o arquivo** no seletor de arquivos
3. **Configurar as definições de importação** no diálogo de importação
4. **Visualizar** o resultado antes de importar
5. **Clicar em Importar** para adicionar à tela e à árvore do documento

### Método 2: Arrastar e soltar

1. **Arrastar o arquivo** do gerenciador de arquivos
2. **Soltar sobre** a tela do Rayforge
3. **Configurar as definições de importação** no diálogo de importação
4. **Visualizar** o resultado antes de importar
5. **Clicar em Importar** para adicionar à tela e à árvore do documento

### Método 3: Linha de comando

```bash
# Abrir o Rayforge com um arquivo
rayforge myfile.svg

# Múltiplos arquivos
rayforge file1.svg file2.dxf
```

### Redimensionamento automático na importação

Ao importar arquivos maiores que a área de trabalho da sua máquina, o Rayforge
automaticamente:

1. **Reduz a escala** do conteúdo importado para caber dentro dos limites da
   máquina
2. **Preserva a proporção** durante o redimensionamento
3. **Centraliza** o conteúdo redimensionado no espaço de trabalho
4. **Exibe uma notificação** com a opção de desfazer o redimensionamento

A notificação de redimensionamento aparece como uma mensagem toast:

- ⚠️ "O item importado era maior que a área de trabalho e foi reduzido para
  caber."
- Inclui um botão **"Redefinir"** para desfazer o redimensionamento automático
- O toast permanece visível até ser descartado ou a ação de redefinição ser
  executada

Isso garante que seus designs sempre se ajustem às capacidades da sua máquina,
dando-lhe a flexibilidade de restaurar o tamanho original, se necessário.

---

## Importação SVG

SVG (Scalable Vector Graphics) é o **formato recomendado** para designs
vetoriais.

### Opções de importação no diálogo

Ao importar SVG, o diálogo de importação oferece um interruptor para escolher
entre dois métodos:

#### 1. Usar vetores originais (Recomendado)

Esta opção está habilitada por padrão no diálogo de importação.

**Como funciona:**

- Analisa o SVG e converte caminhos diretamente em geometria do Rayforge
- Preservação de alta fidelidade de curvas e formas
- Mantém os dados vetoriais exatos

**Vantagens:**

- Melhor qualidade e precisão
- Caminhos editáveis
- Menor tamanho de arquivo

**Desvantagens:**

- Alguns recursos SVG avançados não são suportados
- SVGs complexos podem apresentar problemas

**Usar para:**

- Designs vetoriais limpos do Inkscape, Illustrator
- Complexidade de simples a moderada
- Designs sem recursos SVG avançados

#### 2. Vetorizar bitmap

Desative "Usar vetores originais" para usar este método.

**Como funciona:**

- Renderiza o SVG primeiro como imagem raster
- Vetoriza a imagem renderizada para criar vetores
- Mais compatível, porém menos preciso

**Vantagens:**

- Lida com recursos SVG complexos
- Método de fallback robusto
- Suporta efeitos e filtros

**Desvantagens:**

- Perda de qualidade por rasterização
- Tamanhos de arquivo maiores
- Menos preciso

**Usar para:**

- SVGs cuja importação direta falha
- SVGs com efeitos, filtros, gradientes
- Quando a importação direta produz erros

### Visualização em tempo real

O diálogo de importação mostra uma visualização em tempo real de como seu SVG
será importado:

- Os caminhos vetoriais são exibidos em sobreposição azul
- No modo de vetorização, a imagem original é mostrada com os caminhos
  vetorizados
- A visualização atualiza em tempo real conforme você altera as definições

### Boas práticas SVG

**Prepare seu SVG para obter os melhores resultados:**

1. **Converter texto em caminhos:**
   - Inkscape: `Caminho → Objeto para caminho`
   - Illustrator: `Tipo → Criar contornos`

2. **Simplificar caminhos complexos:**
   - Inkscape: `Caminho → Simplificar` (Ctrl+L)
   - Remover nós desnecessários

3. **Desagrupar grupos aninhados:**
   - Achatar a hierarquia quando possível
   - `Objeto → Desagrupar` (Ctrl+Shift+G)

4. **Remover elementos ocultos:**
   - Excluir guias, grades, linhas de construção
   - Remover objetos invisíveis/transparentes

5. **Salvar como SVG simples:**
   - Inkscape: "SVG simples" ou "SVG otimizado"
   - Não "SVG do Inkscape" (contém metadados extras)

6. **Verificar unidades do documento:**
   - Definir como mm ou polegadas, conforme apropriado
   - O Rayforge usa mm internamente

**Recursos SVG comuns que podem não ser importados:**

- Gradientes (converter em preenchimentos sólidos ou raster)
- Filtros e efeitos (achatar para caminhos)
- Máscaras e caminhos de recorte (expandir/achatar)
- Imagens raster incorporadas (exportar separadamente)
- Texto (converter em caminhos primeiro)

---

## Importação DXF

DXF (Drawing Exchange Format) é comum em softwares CAD.

### Versões DXF

O Rayforge suporta formatos DXF padrão:

- **R12/LT2** (recomendado) - Melhor compatibilidade
- **R13, R14** - Bom suporte
- **R2000+** - Geralmente funciona, mas R12 é mais seguro

**Dica:** Exporte como DXF R12/LT2 para compatibilidade máxima.

### Dicas para importação DXF

**Antes de exportar do CAD:**

1. **Simplificar o desenho:**
   - Remover camadas desnecessárias
   - Excluir dimensões e anotações
   - Remover objetos 3D (usar projeção 2D)

2. **Verificar unidades:**
   - Confirmar as unidades do desenho (mm vs polegadas)
   - O Rayforge assume mm por padrão

3. **Achatar camadas:**
   - Considerar exportar apenas camadas relevantes
   - Ocultar ou excluir camadas de construção

4. **Usar precisão adequada:**
   - A precisão do laser é tipicamente 0,1 mm
   - Não exagere na especificação de precisão

**Após a importação:**

- Verificar a escala (unidades DXF podem precisar de ajuste)
- Confirmar se todos os caminhos foram importados corretamente
- Excluir elementos de construção indesejados

---

## Importação PDF

Os arquivos PDF podem conter gráficos vetoriais, imagens raster, ou ambos.

### Importação vetorial direta

Ao importar um PDF que contém caminhos vetoriais, o Rayforge pode importá-los
diretamente — assim como arquivos SVG ou DXF. Isso fornece geometria limpa e
escalável sem perda de qualidade por rasterização.

Se o PDF contiver camadas, o Rayforge as detecta e permite escolher quais
importar. Cada camada torna-se uma peça de trabalho separada no seu documento.
Isso funciona da mesma forma que a importação de camadas SVG: ative ou desative
camadas individuais no diálogo de importação antes de importar.

Isso é especialmente útil para PDFs exportados de softwares de design como
Illustrator ou Inkscape, onde os caminhos vetoriais estão limpos e bem
organizados.

### Fallback: Renderizar e vetorizar

Para PDFs que não contêm dados vetoriais utilizáveis — documentos digitalizados,
fotos incorporadas, ou PDFs onde o texto não foi convertido em contornos — o
Rayforge pode recorrer à renderização do PDF como imagem e vetorizá-lo. Isso
funciona da mesma forma que a importação de imagens raster.

### Dicas para importação PDF

**Melhores resultados:**

1. **Usar PDFs vetoriais**: PDFs criados a partir de softwares vetoriais
   (Illustrator, Inkscape) produzem os resultados mais limpos com importação
   direta.

2. **Verificar camadas**: Se o seu PDF tiver camadas, elas serão listadas no
   diálogo de importação. Selecione apenas as camadas necessárias.

3. **Para documentos com texto**: Exporte como SVG com fontes convertidas em
   caminhos para melhor qualidade, ou use o fallback de renderizar e vetorizar.

4. **Usar a visualização do diálogo de importação**: Ajuste as definições de
   limiar e inversão ao usar o modo de vetorização. A visualização mostra
   exatamente como o PDF será vetorizado.

---

## Importação Ruida

Os arquivos Ruida (.rd) são arquivos de trabalho binários proprietários usados
pelos controladores Ruida em muitas máquinas de corte a laser. Esses arquivos
contêm tanto geometria vetorial quanto configurações do laser organizadas em
camadas (cores).

**Após a importação:**

- **Verificar escala** - Confirmar se as dimensões correspondem ao tamanho
  esperado
- **Revisar camadas** - Garantir que todas as camadas foram importadas
  corretamente
- **Validar caminhos** - Confirmar se todos os caminhos de corte estão presentes

### Limitações

- **Importação somente leitura** - Arquivos Ruida só podem ser importados, não
  exportados
- **Formato binário** - Edição direta dos arquivos .rd originais não é suportada
- **Recursos proprietários** - Alguns recursos avançados do Ruida podem não ser
  totalmente suportados

---

## Importação de imagens raster (PNG, JPG, BMP)

As imagens raster são **vetorizadas** para criar caminhos vetoriais usando o
diálogo de importação.

### Processo de vetorização no diálogo

**Como funciona:**

1. **Imagem carregada** no diálogo de importação
2. **Visualização em tempo real** mostra o resultado da vetorização
3. **Definições de vetorização** podem ser ajustadas em tempo real
4. **Caminhos vetoriais criados** a partir das bordas vetorizadas
5. **Caminhos adicionados** ao documento como peças de trabalho ao importar

### Configuração da vetorização no diálogo

O diálogo de importação fornece estes parâmetros ajustáveis:

| Parâmetro             | Descrição           | Efeito                                                  |
| --------------------- | ------------------- | ------------------------------------------------------- |
| **Limiar automático** | Detecção automática | Quando ativado, encontra automaticamente o limiar ideal |
| **Limiar**            | Corte preto/branco  | Menor = mais detalhes, maior = mais simples             |
| **Inverter**          | Inverter cores      | Vetoriza objetos claros em fundo escuro                 |

**As definições padrão** funcionam bem para a maioria das imagens. O diálogo
mostra uma visualização em tempo real que atualiza conforme você ajusta esses
parâmetros, permitindo refinar a vetorização antes de importar.

### Preparando imagens para vetorização

**Para melhores resultados:**

1. **Alto contraste:**
   - Ajustar brilho/contraste em um editor de imagens
   - Distinção clara entre primeiro plano e fundo

2. **Fundo limpo:**
   - Remover ruído e artefatos
   - Fundo branco sólido ou transparente

3. **Resolução adequada:**
   - 300-500 DPI para fotos
   - Muito alta = vetorização lenta, muito baixa = qualidade ruim

4. **Recortar ao conteúdo:**
   - Remover bordas desnecessárias
   - Focar na área a ser gravada/cortada

5. **Converter para preto e branco:**
   - Para corte: preto e branco puro
   - Para gravação: tons de cinza são aceitáveis

**Ferramentas de edição de imagens:**

- GIMP (gratuito)
- Photoshop
- Krita (gratuito)
- Paint.NET (gratuito, Windows)

### Qualidade da vetorização

**Bons candidatos para vetorização:**

- Logos com bordas definidas
- Imagens de alto contraste
- Arte linear e desenhos
- Texto (embora vetorial seja melhor)

**Maus candidatos para vetorização:**

- Imagens de baixa resolução
- Fotos com bordas suaves
- Imagens com gradientes
- Fotos muito detalhadas ou complexas

---

## Páginas relacionadas

- [Formatos suportados](formats) - Especificações detalhadas dos formatos
- [Exportando G-code](exporting) - Opções de saída
- [Início rápido](../getting-started/quick-start) - Tutorial de primeira importação
