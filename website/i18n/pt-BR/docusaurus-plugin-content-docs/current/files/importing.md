# Importando Arquivos

O Rayforge suporta importação de vários formatos de arquivo, tanto vetoriais quanto raster. Esta página explica como importar arquivos e otimizá-los para melhores resultados.

## Formatos de Arquivo Suportados

### Formatos Vetoriais

| Formato    | Extensão | Método de Importação           | Melhor Para                        |
| --------- | --------- | ----------------------- | ------------------------------- |
| **SVG**   | `.svg`    | Vetores diretos ou traço | Gráficos vetoriais, logos, designs |
| **DXF**   | `.dxf`    | Vetores diretos          | Desenhos CAD, designs técnicos |
| **PDF**   | `.pdf`    | Renderizar e traçar        | Documentos com conteúdo vetorial   |
| **Ruida** | `.rd`     | Vetores diretos          | Arquivos de trabalho de controlador Ruida      |

### Formatos Raster

| Formato   | Extensão       | Método de Importação    | Melhor Para                         |
| -------- | --------------- | ---------------- | -------------------------------- |
| **PNG**  | `.png`          | Traçar para vetores | Fotos, imagens com transparência |
| **JPEG** | `.jpg`, `.jpeg` | Traçar para vetores | Fotos, imagens de tom contínuo   |
| **BMP**  | `.bmp`          | Traçar para vetores | Gráficos simples, capturas de tela     |

:::note Importação Raster
:::

Todas imagens raster são **traçadas** para criar caminhos vetoriais que podem ser usados para operações de laser. A qualidade depende da configuração de traçamento.

---

## Importando Arquivos

### O Diálogo de Importação

O Rayforge apresenta um diálogo de importação unificado que fornece pré-visualização ao vivo e opções de configuração para todos os tipos de arquivo suportados. O diálogo permite:

- **Pré-visualizar sua importação** antes de adicionar ao documento
- **Configurar configurações de traçamento** para imagens raster
- **Escolher método de importação** para arquivos SVG (vetores diretos ou traço)
- **Ajustar parâmetros** como limiar, inverter e auto-limiar

![Diálogo de Importação](/screenshots/import-dialog.png)

### Método 1: Menu Arquivo

1. **Arquivo Importar** (ou Ctrl+I)
2. **Selecione seu arquivo** no seletor de arquivos
3. **Configure configurações de importação** no diálogo de importação
4. **Pré-visualize** o resultado antes de importar
5. **Clique em Importar** para adicionar à tela e árvore de documento

### Método 2: Arrastar e Soltar

1. **Arraste o arquivo** do seu gerenciador de arquivos
2. **Solte na** tela do Rayforge
3. **Configure configurações de importação** no diálogo de importação
4. **Pré-visualize** o resultado antes de importar
5. **Clique em Importar** para adicionar à tela e árvore de documento

### Método 3: Linha de Comando

```bash
# Abrir Rayforge com um arquivo
rayforge meuarquivo.svg

# Múltiplos arquivos
rayforge arquivo1.svg arquivo2.dxf
```

### Auto-Redimensionamento na Importação

Ao importar arquivos maiores que a área de trabalho da sua máquina, o Rayforge automaticamente:

1. **Redimensiona para baixo** o conteúdo importado para caber dentro dos limites da máquina
2. **Preserva proporção** durante redimensionamento
3. **Centraliza** o conteúdo redimensionado na área de trabalho
4. **Mostra uma notificação** com a opção de desfazer o redimensionamento

A notificação de redimensionamento aparece como mensagem toast:

- ⚠️ "Item importado era maior que a área de trabalho e foi redimensionado para caber."
- Inclui um botão **"Resetar"** para desfazer o auto-redimensionamento
- O toast permanece visível até ser dispensado ou a ação de reset ser tomada

Isso garante que seus designs sempre caibam dentro das capacidades da sua máquina enquanto lhe dá a flexibilidade de restaurar o tamanho original se necessário.

---

## Importação SVG

SVG (Scalable Vector Graphics) é o **formato recomendado** para designs vetoriais.

### Opções de Importação no Diálogo

Ao importar SVG, o diálogo de importação fornece um interruptor para escolher entre dois métodos:

#### 1. Usar Vetores Originais (Recomendado)

Esta opção está habilitada por padrão no diálogo de importação.

**Como funciona:**

- Analisa SVG e converte caminhos diretamente para geometria Rayforge
- Preservação de alta fidelidade de curvas e formas
- Mantém dados vetoriais exatos

**Vantagens:**

- Melhor qualidade e precisão
- Caminhos editáveis
- Tamanho de arquivo pequeno

**Desvantagens:**

- Alguns recursos SVG avançados não suportados
- SVGs complexos podem ter problemas

**Use para:**

- Designs vetoriais limpos de Inkscape, Illustrator
- Complexidade simples a moderada
- Designs sem recursos SVG avançados

#### 2. Traçar Bitmap

Desabilite "Usar Vetores Originais" para usar este método.

**Como funciona:**

- Renderiza SVG para imagem raster primeiro
- Traça a imagem renderizada para criar vetores
- Mais compatível mas menos preciso

**Vantagens:**

- Lida com recursos SVG complexos
- Método de fallback robusto
- Suporta efeitos e filtros

**Desvantagens:**

- Perda de qualidade da rasterização
- Tamanhos de arquivo maiores
- Não tão preciso

**Use para:**

- SVGs que falham na importação direta
- SVGs com efeitos, filtros, gradientes
- Quando importação direta produz erros

### Pré-visualização ao Vivo

O diálogo de importação mostra uma pré-visualização ao vivo de como seu SVG será importado:

- Caminhos vetoriais são exibidos em sobreposição azul
- Para modo de traço, a imagem original é mostrada com os caminhos traçados
- Pré-visualização atualiza em tempo real conforme você muda configurações

### Melhores Práticas SVG

**Prepare seu SVG para melhores resultados:**

1. **Converta texto para caminhos:**

   - Inkscape: `Caminho → Objeto para Caminho`
   - Illustrator: `Tipo → Criar Contornos`

2. **Simplifique caminhos complexos:**

   - Inkscape: `Caminho → Simplificar` (Ctrl+L)
   - Remova nós desnecessários

3. **Desagrupe grupos aninhados:**

   - Achate hierarquia onde possível
   - `Objeto → Desagrupar` (Ctrl+Shift+G)

4. **Remova elementos ocultos:**

   - Exclua guias, grades, linhas de construção
   - Remova objetos invisíveis/transparentes

5. **Salve como SVG Simples:**

   - Inkscape: "SVG Simples" ou "SVG Otimizado"
   - Não "SVG Inkscape" (tem metadados extras)

6. **Verifique unidades do documento:**
   - Defina para mm ou polegadas conforme apropriado
   - Rayforge usa mm internamente

**Recursos SVG comuns que podem não importar:**

- Gradientes (converta para preenchimentos sólidos ou raster)
- Filtros e efeitos (achatar para caminhos)
- Máscaras e caminhos de recorte (expandir/achatar)
- Imagens raster embutidas (exporte separadamente)
- Texto (converta para caminhos primeiro)

---

## Importação DXF

DXF (Drawing Exchange Format) é comum para software CAD.

### Versões DXF

O Rayforge suporta formatos DXF padrão:

- **R12/LT2** (recomendado) - Melhor compatibilidade
- **R13, R14** - Bom suporte
- **R2000+** - Geralmente funciona, mas R12 é mais seguro

**Dica:** Exporte como DXF R12/LT2 para máxima compatibilidade.

### Dicas de Importação DXF

**Antes de exportar do CAD:**

1. **Simplifique o desenho:**

   - Remova camadas desnecessárias
   - Exclua cotas e anotações
   - Remova objetos 3D (use projeção 2D)

2. **Verifique unidades:**

   - Verifique unidades do desenho (mm vs polegadas)
   - Rayforge assume mm por padrão

3. **Achate camadas:**

   - Considere exportar apenas camadas relevantes
   - Oculte ou exclua camadas de construção

4. **Use precisão apropriada:**
   - Precisão de laser é tipicamente 0.1mm
   - Não superespecifique precisão

**Após importação:**

- Verifique escala (unidades DXF podem precisar de ajuste)
- Verifique se todos caminhos importaram corretamente
- Exclua quaisquer elementos de construção indesejados

---

## Importação PDF

Arquivos PDF podem conter gráficos vetoriais, imagens raster ou ambos.

### Como Funciona a Importação PDF

Ao importar arquivos PDF através do diálogo de importação, o Rayforge **renderiza o PDF** para uma imagem, depois **traça** para criar vetores.

**Processo:**

1. PDF renderizado e exibido na pré-visualização do diálogo de importação
2. Você pode ajustar configurações de traçamento em tempo real
3. Imagem renderizada traçada usando vetorização com suas configurações
4. Caminhos resultantes adicionados ao documento quando você clica em Importar

**Limitações:**

- Texto é rasterizado (não editável como caminhos)
- Qualidade vetorial depende do DPI de renderização
- PDFs multi-página: apenas primeira página importada

### Dicas de Importação PDF

**Melhores resultados:**

1. **Use PDFs vetoriais:**

   - PDFs criados de software vetorial (Illustrator, Inkscape)
   - Não documentos escaneados ou imagens embutidas

2. **Exporte SVG em vez se possível:**

   - Maioria do software de design pode exportar SVG diretamente
   - SVG terá melhor qualidade que importação PDF

3. **Para documentos com texto:**

   - Exporte como SVG com fontes convertidas para caminhos
   - Ou renderize PDF em alto DPI (600+) e traçe

4. **Use a pré-visualização do diálogo de importação:**
   - Ajuste configurações de limiar e inverter para melhores resultados
   - Pré-visualização mostra exatamente como o PDF será traçado

---

## Importação Ruida

Arquivos Ruida (.rd) são arquivos de trabalho binários proprietários usados por controladores Ruida em muitas máquinas de corte a laser. Estes arquivos contêm tanto geometria vetorial quanto configurações de laser organizados em camadas (cores).

**Após importação:**

- **Verifique escala** - Verifique se dimensões correspondem ao tamanho esperado
- **Revise camadas** - Certifique-se de que todas camadas importaram corretamente
- **Valide caminhos** - Confirme que todos caminhos de corte estão presentes

### Limitações

- **Importação somente leitura** - Arquivos Ruida só podem ser importados, não exportados
- **Formato binário** - Edição direta de arquivos .rd originais não suportada
- **Recursos proprietários** - Alguns recursos Ruida avançados podem não ser totalmente suportados

---

## Importação de Imagem Raster (PNG, JPG, BMP)

Imagens raster são **traçadas** para criar caminhos vetoriais usando o diálogo de importação.

### Processo de Traçamento no Diálogo

**Como funciona:**

1. **Imagem carregada** no diálogo de importação
2. **Pré-visualização ao vivo** mostra o resultado traçado
3. **Configurações de traçamento** podem ser ajustadas em tempo real
4. **Caminhos vetoriais criados** das bordas traçadas
5. **Caminhos adicionados** ao documento como peças quando importados

### Configuração de Traçamento no Diálogo

O diálogo de importação fornece estes parâmetros ajustáveis:

| Parâmetro          | Descrição         | Efeito                                              |
| ------------------ | ------------------- | --------------------------------------------------- |
| **Auto Limiar** | Detecção automática | Quando habilitado, encontra automaticamente limiar ótimo |
| **Limiar**      | Corte preto/branco  | Menor = mais detalhe, maior = mais simples               |
| **Inverter**         | Inverter cores      | Traçar objetos claros em fundo escuro              |

**Configurações padrão** funcionam bem para a maioria das imagens. O diálogo mostra uma pré-visualização ao vivo que atualiza conforme você ajusta essas configurações, permitindo ajustar o traçamento antes de importar.

### Preparando Imagens para Traçamento

**Para melhores resultados:**

1. **Alto contraste:**

   - Ajuste brilho/contraste em editor de imagem
   - Distinção clara entre primeiro plano e fundo

2. **Fundo limpo:**

   - Remova ruído e artefatos
   - Fundo branco sólido ou transparente

3. **Resolução apropriada:**

   - 300-500 DPI para fotos
   - Muito alto = traçamento lento, muito baixo = qualidade ruim

4. **Recortar para conteúdo:**

   - Remova bordas desnecessárias
   - Foque na área a ser gravada/cortada

5. **Converta para preto e branco:**
   - Para corte: B&W puro
   - Para gravação: escala de cinza está bem

**Ferramentas de edição de imagem:**

- GIMP (gratuito)
- Photoshop
- Krita (gratuito)
- Paint.NET (gratuito, Windows)

### Qualidade do Traçamento

**Bons candidatos a traçado:**

- Logos com bordas claras
- Imagens de alto contraste
- Arte linear e desenhos
- Texto (embora melhor como vetor)

**Maus candidatos a traçado:**

- Imagens de baixa resolução
- Fotos com bordas suaves
- Imagens com gradientes
- Fotos muito detalhadas ou complexas

---

## Páginas Relacionadas

- [Formatos Suportados](formats) - Especificações detalhadas de formato
- [Exportando G-code](exporting) - Opções de saída
- [Início Rápido](../getting-started/quick-start) - Tutorial de primeira importação
