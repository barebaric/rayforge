# Exportando do Rayforge

O Rayforge suporta várias opções de exportação para diferentes propósitos:

- **G-code** - Saída de controle da máquina para executar trabalhos
- **Exportação de Objeto** - Exporta peças de trabalho individuais para formatos vetoriais
- **Exportação de Documento** - Exporta todas as peças de trabalho como um único arquivo

---

## Exportando Objetos

Você pode exportar qualquer peça de trabalho para formatos vetoriais para uso em software de design, aplicações CAD ou para arquivamento.

### Como Exportar

1. **Selecione uma peça de trabalho** na tela
2. **Escolha Objeto → Exportar Objeto...** (ou clique direito → Exportar Objeto...)
3. **Selecione o formato** e local para salvar

### Formatos Disponíveis

| Formato  | Extensão | Descrição                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **RFS** | `.rfs`    | Formato de esboço paramétrico nativo do Rayforge. Preserva todas as restrições e pode ser reimportado para edição.      |
| **SVG** | `.svg`    | Scalable Vector Graphics. Amplamente compatível com software de design como Inkscape, Illustrator e navegadores web. |
| **DXF** | `.dxf`    | Drawing Exchange Format. Compatível com a maioria das aplicações CAD como AutoCAD, FreeCAD e LibreCAD.            |

### Notas de Exportação

- **SVG e DXF** exportam a geometria resolvida (não restrições paramétricas)
- Exportações usam **unidades em milímetros**
- A geometria é escalada para dimensões reais (espaço do mundo)
- Múltiplos subcaminhos (formas desconectadas) são preservados como elementos separados

### Casos de Uso

**Compartilhando designs:**

- Exporte para SVG para compartilhar com usuários Inkscape
- Exporte para DXF para usuários de software CAD

**Editando posteriormente:**

- Exporte para SVG/DXF, edite em software externo, reimporte

**Arquivando:**

- Use RFS para designs baseados em esboço para preservar editabilidade
- Use SVG/DXF para armazenamento de longo prazo ou usuários não-Rayforge

---

## Exportando Documentos

Você pode exportar todas as peças de trabalho em um documento para um único arquivo vetorial. Isso é
útil para compartilhar projetos completos ou criar backups em formatos padrão.

### Como Exportar

1. **Escolha Arquivo → Exportar Documento...**
2. **Selecione o formato** (SVG ou DXF)
3. **Escolha o local para salvar**

### Formatos Disponíveis

| Formato  | Extensão | Descrição                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **SVG** | `.svg`    | Scalable Vector Graphics. Amplamente compatível com software de design como Inkscape, Illustrator e navegadores web. |
| **DXF** | `.dxf`    | Drawing Exchange Format. Compatível com a maioria das aplicações CAD como AutoCAD, FreeCAD e LibreCAD.            |

### Notas de Exportação

- Todas as peças de trabalho de todas as camadas são combinadas em um único arquivo
- Posições das peças de trabalho são preservadas
- Peças de trabalho vazias são ignoradas
- A caixa delimitadora engloba toda a geometria

### Casos de Uso

- **Compartilhamento de projeto**: Exporte o projeto inteiro para colaboração
- **Backup**: Crie um arquivo visual do seu trabalho
- **Edição posterior**: Leve o design inteiro para o Inkscape ou software CAD

---

## Exportando G-code

O G-code gerado contém tudo exatamente como seria enviado para a máquina.
O formato exato, comandos, precisão numérica, etc. dependem das configurações da
máquina atualmente selecionada e seu dialeto G-code.

---

### Métodos de Exportação

### Método 1: Menu Arquivo

**Arquivo Exportar G-code** (Ctrl+E)

- Abre diálogo de salvar arquivo
- Escolha local e nome do arquivo
- G-code gerado e salvo

### Método 2: Linha de Comando

```bash
# Exportar via linha de comando (se suportado)
rayforge --export saida.gcode entrada.svg
```

---

### Saída G-code

O G-code gerado contém tudo exatamente como seria enviado para a máquina.
O formato exato, comandos, precisão numérica, etc. dependem das configurações da
máquina atualmente selecionada e seu dialeto G-code.

---

## Páginas Relacionadas

- [Importando Arquivos](importing) - Trazendo designs para o Rayforge
- [Formatos Suportados](formats) - Detalhes de formatos de arquivo
- [Dialeto G-code](../reference/gcode-dialects) - Diferenças de dialeto
- [Hooks & Macros](../machine/hooks-macros) - Personalizando a saída
- [Modo de Simulação](../features/simulation-mode) - Visualizar antes de exportar
