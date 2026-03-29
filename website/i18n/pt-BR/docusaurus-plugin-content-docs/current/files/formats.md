# Formatos de arquivo suportados

Esta página fornece informações detalhadas sobre todos os formatos de arquivo
suportados pelo Rayforge, incluindo capacidades, limitações e recomendações.

## Visão geral dos formatos

### Referência rápida

| Formato              | Tipo     | Importação        | Exportação      | Uso recomendado              |
| -------------------- | -------- | ----------------- | --------------- | ---------------------------- |
| **SVG**              | Vetor    | ✓ Direto / Traçar | ✓ Exportar obj. | Formato de design principal  |
| **DXF**              | Vetor    | ✓ Direto          | ✓ Exportar obj. | Intercâmbio CAD              |
| **PDF**              | Misto    | ✓ Direto / Traçar | –               | Documentos com conteúdo vet. |
| **PNG**              | Raster   | ✓ Traçar          | –               | Fotos, imagens               |
| **JPEG**             | Raster   | ✓ Traçar          | –               | Fotos                        |
| **BMP**              | Raster   | ✓ Traçar          | –               | Gráficos simples             |
| **RFS**              | Esboço   | ✓ Direto          | ✓ Exportar obj. | Esboços paramétricos         |
| **G-code**           | Controle | –                 | ✓ Principal     | Saída da máquina             |
| **Projeto Rayforge** | Projeto  | ✓                 | ✓               | Salvar/carregar projetos     |

---

## Formatos vetoriais

### SVG (Gráficos Vetoriais Escaláveis)

**Extensão:** `.svg`
**Tipo MIME:** `image/svg+xml`
**Importar:** Análise vetorial direta ou traçado de bitmap
**Exportar:** Exportação de objeto (apenas geometria)

**O que é SVG?**

SVG é um formato de imagem vetorial baseado em XML. É o **formato preferido**
para importar designs no Rayforge.

**Recursos suportados:**

- ✓ Caminhos (linhas, curvas, arcos)
- ✓ Formas básicas (retângulos, círculos, elipses, polígonos)
- ✓ Grupos e transformações
- ✓ Cores de contorno e preenchimento
- ✓ Múltiplas camadas
- ✓ Transformações de coordenadas (translação, rotação, escala)

**Recursos não suportados/limitados:**

- ✗ Texto (deve ser convertido em caminhos primeiro)
- ✗ Gradientes (simplificados ou ignorados)
- ✗ Filtros e efeitos (ignorados)
- ✗ Máscaras e caminhos de recorte (podem não funcionar corretamente)
- ✗ Imagens raster embutidas (importadas separadamente se possível)
- ✗ Estilos de contorno complexos (tracejados podem ser simplificados)
- ✗ Símbolos e elementos use (instâncias podem não atualizar)

**Notas de exportação:**

Ao exportar uma peça de trabalho para SVG, o Rayforge exporta a geometria como
caminhos vetoriais com:

- Renderização apenas de contorno (sem preenchimento)
- Unidades em milímetros
- Cor de contorno preta

**Melhores práticas:**

1. **Use formato SVG simples** (não Inkscape SVG ou outras variantes
   específicas de ferramenta)
2. **Converta texto em caminhos** antes de exportar
3. **Simplifique caminhos complexos** para reduzir a contagem de nós
4. **Achate grupos** quando possível
5. **Remova elementos não utilizados** (guias, grades, camadas ocultas)
6. **Defina unidades do documento** para mm (unidade nativa do Rayforge)

**Recomendações de software:**

- **Inkscape** (gratuito) - Excelente suporte SVG, formato nativo

---

### DXF (Drawing Exchange Format)

**Extensão:** `.dxf`
**Tipo MIME:** `application/dxf`, `image/vnd.dxf`
**Importar:** Análise vetorial direta
**Exportar:** Exportação de objeto (apenas geometria)

**O que é DXF?**

DXF é um formato de desenho do AutoCAD, amplamente usado para intercâmbio CAD.

**Versões suportadas:**

- ✓ **R12/LT2** (recomendado - melhor compatibilidade)
- ✓ R13, R14
- ✓ R2000 e posterior (geralmente funciona, mas R12 é mais seguro)

**Entidades suportadas:**

- ✓ Linhas (LINE)
- ✓ Polilinhas (LWPOLYLINE, POLYLINE)
- ✓ Arcos (ARC)
- ✓ Círculos (CIRCLE)
- ✓ Splines (SPLINE) - convertidas em polilinhas
- ✓ Elipses (ELLIPSE)
- ✓ Camadas

**Recursos não suportados/limitados:**

- ✗ Entidades 3D (use projeção 2D)
- ✗ Cotas e anotações (ignoradas)
- ✗ Blocos/inserções (podem não instanciar corretamente)
- ✗ Tipos de linha complexos (simplificados para sólido)
- ✗ Texto (ignorado, converta em contornos primeiro)
- ✗ Hachuras (podem ser simplificadas ou ignoradas)

**Notas de exportação:**

Ao exportar uma peça de trabalho para DXF, o Rayforge exporta:

- Linhas como entidades LWPOLYLINE
- Arcos como entidades ARC
- Curvas de Bézier como entidades SPLINE
- Unidades em milímetros (INSUNITS = 4)

---

### RFS (Esboço Rayforge)

**Extensão:** `.rfs`
**Tipo MIME:** `application/x-rayforge-sketch`
**Importar:** Direto (peças de trabalho baseadas em esboço)
**Exportar:** Exportação de objeto (peças baseadas em esboço)

**O que é RFS?**

RFS é o formato nativo de esboço paramétrico do Rayforge. Ele preserva todos
os elementos geométricos e restrições paramétricas, permitindo salvar e
compartilhar esboços totalmente editáveis.

**Recursos:**

- ✓ Todos os elementos geométricos (linhas, arcos, círculos, retângulos, etc.)
- ✓ Todas as restrições paramétricas
- ✓ Valores dimensionais e expressões
- ✓ Áreas de preenchimento

**Quando usar:**

- Salvar designs paramétricos reutilizáveis
- Compartilhar esboços editáveis com outros usuários do Rayforge
- Arquivar trabalho em andamento

---

### PDF (Portable Document Format)

**Extensão:** `.pdf`
**Tipo MIME:** `application/pdf`
**Importar:** Vetores diretos (com suporte a camadas) ou renderizar e traçar
**Exportar:** Não suportado

**O que é a importação de PDF?**

Arquivos PDF podem conter caminhos vetoriais reais, e o Rayforge os importa
diretamente quando disponíveis — fornecendo a mesma geometria limpa que você
obteria de um SVG. Se o PDF tiver camadas, cada camada pode ser importada como
uma peça de trabalho separada.

Para PDFs sem conteúdo vetorial utilizável (documentos digitalizados, fotos),
o Rayforge recorre à renderização e ao traçado.

**Capacidades:**

- ✓ **Importação vetorial direta** para PDFs baseados em vetores
- ✓ **Detecção e seleção de camadas** — escolha quais camadas importar
- ✓ Renderização e traçado alternativos para conteúdo raster

**Limitações:**

- Apenas a primeira página — PDFs de múltiplas páginas importam a página 1
- O texto pode precisar ser convertido em contornos no aplicativo de origem

**Quando usar:**

- PDFs recebidos de designers que contêm ilustrações vetoriais
- Qualquer PDF com camadas bem organizadas
- Quando SVG ou DXF não estão disponíveis na fonte

---

## Formatos raster

Todos os formatos raster são **importados por traçado** — convertidos
automaticamente em caminhos vetoriais.

### PNG (Portable Network Graphics)

**Extensão:** `.png`
**Tipo MIME:** `image/png`
**Importar:** Traçar para vetores
**Exportar:** Não suportado

**Características:**

- **Compressão sem perdas** - Sem perda de qualidade
- **Suporte a transparência** - Canal alfa preservado
- **Bom para:** Logos, arte linear, capturas de tela, qualquer coisa
  precisando de transparência

**Qualidade de traçado:** (Excelente para imagens de alto contraste)

**Melhores práticas:**

- Use PNG para logos e gráficos com bordas nítidas
- Garanta alto contraste entre o primeiro plano e o fundo
- Fundo transparente funciona melhor que fundo branco

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

**Qualidade de traçado:** (Bom para fotos, mas complexo)

**Melhores práticas:**

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

**Qualidade de traçado:** (Bom, mas não melhor que PNG)

**Melhores práticas:**

- Converta para PNG para tamanho de arquivo menor (sem diferença de qualidade)
- Use apenas se o software de origem não puder exportar PNG/SVG

---

## Páginas relacionadas

- [Importando arquivos](importing) - Como importar cada formato
- [Exportando](exporting) - Opções de exportação de G-code
