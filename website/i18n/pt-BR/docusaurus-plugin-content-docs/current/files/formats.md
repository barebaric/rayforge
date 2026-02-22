# Formatos de Arquivo Suportados

Esta página fornece informações detalhadas sobre todos os formatos de arquivo suportados pelo Rayforge, incluindo capacidades, limitações e recomendações.

## Visão Geral de Formatos

### Referência Rápida

| Formato               | Tipo    | Importar   | Exportar          | Uso Recomendado           |
| -------------------- | ------- | -------- | --------------- | ------------------------- |
| **SVG**              | Vetor  | ✓ Direto | ✓ Exportação de objeto | Formato de design principal     |
| **DXF**              | Vetor  | ✓ Direto | ✓ Exportação de objeto | Intercâmbio CAD           |
| **PDF**              | Misto   | ✓ Traçar  | –               | Exportação de documento (limitado) |
| **PNG**              | Raster  | ✓ Traçar  | –               | Fotos, imagens            |
| **JPEG**             | Raster  | ✓ Traçar  | –               | Fotos                    |
| **BMP**              | Raster  | ✓ Traçar  | –               | Gráficos simples           |
| **RFS**              | Esboço  | ✓ Direto | ✓ Exportação de objeto | Esboços paramétricos       |
| **G-code**           | Controle | –        | ✓ Primário       | Saída para máquina            |
| **Projeto Rayforge** | Projeto | ✓        | ✓               | Salvar/carregar projetos        |

---

## Formatos Vetoriais

### SVG (Scalable Vector Graphics)

**Extensão:** `.svg`
**Tipo MIME:** `image/svg+xml`
**Importar:** Análise vetorial direta ou traçado de bitmap
**Exportar:** Exportação de objeto (apenas geometria)

**O que é SVG?**

SVG é um formato de imagem vetorial baseado em XML. É o **formato preferido** para importar designs no Rayforge.

**Recursos Suportados:**

- ✓ Caminhos (linhas, curvas, arcos)
- ✓ Formas básicas (retângulos, círculos, elipses, polígonos)
- ✓ Grupos e transformações
- ✓ Cores de contorno e preenchimento
- ✓ Múltiplas camadas
- ✓ Transformações de coordenadas (translação, rotação, escala)

**Recursos Não Suportados/Limitados:**

- ✗ Texto (deve ser convertido em caminhos primeiro)
- ✗ Gradientes (simplificados ou ignorados)
- ✗ Filtros e efeitos (ignorados)
- ✗ Máscaras e caminhos de recorte (podem não funcionar corretamente)
- ✗ Imagens raster embutidas (importadas separadamente se possível)
- ✗ Estilos de contorno complexos (tracejados podem ser simplificados)
- ✗ Símbolos e elementos use (instâncias podem não atualizar)

**Notas de Exportação:**

Ao exportar uma peça de trabalho para SVG, o Rayforge exporta a geometria como caminhos vetoriais com:

- Renderização apenas de contorno (sem preenchimento)
- Unidades em milímetros
- Cor de contorno preta

**Melhores Práticas:**

1. **Use formato SVG simples** (não Inkscape SVG ou outras variantes específicas de ferramenta)
2. **Converta texto em caminhos** antes de exportar
3. **Simplifique caminhos complexos** para reduzir contagem de nós
4. **Achate grupos** quando possível
5. **Remova elementos não utilizados** (guias, grades, camadas ocultas)
6. **Defina unidades do documento** para mm (unidade nativa do Rayforge)

**Recomendações de Software:**

- **Inkscape** (gratuito) - Excelente suporte SVG, formato nativo

---

### DXF (Drawing Exchange Format)

**Extensão:** `.dxf`
**Tipo MIME:** `application/dxf`, `image/vnd.dxf`
**Importar:** Análise vetorial direta
**Exportar:** Exportação de objeto (apenas geometria)

**O que é DXF?**

DXF é um formato de desenho AutoCAD, amplamente usado para intercâmbio CAD.

**Versões Suportadas:**

- ✓ **R12/LT2** (recomendado - melhor compatibilidade)
- ✓ R13, R14
- ✓ R2000 e posterior (geralmente funciona, mas R12 é mais seguro)

**Entidades Suportadas:**

- ✓ Linhas (LINE)
- ✓ Polilinhas (LWPOLYLINE, POLYLINE)
- ✓ Arcos (ARC)
- ✓ Círculos (CIRCLE)
- ✓ Splines (SPLINE) - convertidas em polilinhas
- ✓ Elipses (ELLIPSE)
- ✓ Camadas

**Recursos Não Suportados/Limitados:**

- ✗ Entidades 3D (use projeção 2D)
- ✗ Cotas e anotações (ignoradas)
- ✗ Blocos/inserções (podem não instanciar corretamente)
- ✗ Tipos de linha complexos (simplificados para sólido)
- ✗ Texto (ignorado, converta em contornos primeiro)
- ✗ Hachuras (podem ser simplificadas ou ignoradas)

**Notas de Exportação:**

Ao exportar uma peça de trabalho para DXF, o Rayforge exporta:

- Linhas como entidades LWPOLYLINE
- Arcos como entidades ARC
- Curvas de Bezier como entidades SPLINE
- Unidades em milímetros (INSUNITS = 4)

---

### RFS (Esboço Rayforge)

**Extensão:** `.rfs`
**Tipo MIME:** `application/x-rayforge-sketch`
**Importar:** Direto (peças de trabalho baseadas em esboço)
**Exportar:** Exportação de objeto (peças de trabalho baseadas em esboço)

**O que é RFS?**

RFS é o formato de esboço paramétrico nativo do Rayforge. Preserva todos os elementos
geométricos e restrições paramétricas, permitindo salvar e compartilhar esboços
totalmente editáveis.

**Recursos:**

- ✓ Todos os elementos geométricos (linhas, arcos, círculos, retângulos, etc.)
- ✓ Todas as restrições paramétricas
- ✓ Valores dimensionais e expressões
- ✓ Áreas de preenchimento

**Quando Usar:**

- Salvar designs paramétricos reutilizáveis
- Compartilhar esboços editáveis com outros usuários Rayforge
- Arquivar trabalho em andamento

---

### PDF (Portable Document Format)

**Extensão:** `.pdf`
**Tipo MIME:** `application/pdf`
**Importar:** Renderizado para bitmap, depois traçado
**Exportar:** Não suportado

**O que é Importação de PDF?**

O Rayforge pode importar arquivos PDF rasterizando-os primeiro, depois traçando para vetores.

**Processo:**

1. PDF renderizado para imagem raster (padrão 300 DPI)
2. Raster traçado para criar caminhos vetoriais
3. Caminhos adicionados ao documento

**Limitações:**

- **Não é importação vetorial real** - Mesmo PDFs vetoriais são rasterizados
- **Perda de qualidade** da rasterização
- **Apenas primeira página** - PDFs de múltiplas páginas importam apenas a página 1
- **Lento para PDFs complexos** - Renderização e traçado levam tempo

**Quando Usar:**

- Último recurso quando SVG/DXF não está disponível
- Importação rápida de designs simples
- Documentos com conteúdo misto

**Alternativas Melhores:**

- **Exporte SVG da fonte** em vez de PDF
- **Use formatos vetoriais** (SVG, DXF) quando possível
- **Para texto:** Exporte com texto convertido em contornos

---

## Formatos Raster

Todos os formatos raster são **importados por traçado** - convertidos automaticamente em caminhos vetoriais.

### PNG (Portable Network Graphics)

**Extensão:** `.png`
**Tipo MIME:** `image/png`
**Importar:** Traçar para vetores
**Exportar:** Não suportado

**Características:**

- **Compressão sem perdas** - Sem perda de qualidade
- **Suporte a transparência** - Canal alfa preservado
- **Bom para:** Logos, arte linear, capturas de tela, qualquer coisa precisando de transparência

**Qualidade de Traçado:**  (Excelente para imagens de alto contraste)

**Melhores Práticas:**

- Use PNG para logos e gráficos com bordas nítidas
- Garanta alto contraste entre primeiro plano e fundo
- Fundo transparente funciona melhor que branco

---

### JPEG (Joint Photographic Experts Group)

**Extensão:** `.jpg`, `.jpeg`
**Tipo MIME:** `image/jpeg`
**Importar:** Traçar para vetores
**Exportar:** Não suportado

**Características:**

- **Compressão com perdas** - Alguma perda de qualidade
- **Sem transparência** - Sempre tem fundo
- **Bom para:** Fotos, imagens de tom contínuo

**Qualidade de Traçado:**  (Bom para fotos, mas complexo)

**Melhores Práticas:**

- Use JPEG de alta qualidade (baixa compressão)
- Aumente o contraste antes de importar
- Considere pré-processar em um editor de imagens
- Melhor converter para PNG primeiro se possível

---

### BMP (Bitmap)

**Extensão:** `.bmp`
**Tipo MIME:** `image/bmp`
**Importar:** Traçar para vetores
**Exportar:** Não suportado

**Características:**

- **Sem compressão** - Tamanhos de arquivo grandes
- **Formato simples** - Amplamente compatível
- **Bom para:** Gráficos simples, saída de software antigo

**Qualidade de Traçado:**  (Bom, mas não melhor que PNG)

**Melhores Práticas:**

- Converta para PNG para tamanho de arquivo menor (sem diferença de qualidade)
- Use apenas se o software fonte não puder exportar PNG/SVG

---

## Páginas Relacionadas

- [Importando Arquivos](importing) - Como importar cada formato
- [Exportando](exporting) - Opções de exportação G-code
