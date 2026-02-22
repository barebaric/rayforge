# Fluxo de Trabalho Multi-Camadas

O sistema multi-camadas do Rayforge permite organizar trabalhos complexos em estágios de processamento separados, cada um com suas próprias operações e configurações. Isso é essencial para combinar processos diferentes como gravação e corte, ou trabalhar com múltiplos materiais.

## O Que São Camadas?

Uma **camada** no Rayforge é:

- **Um contêiner** para peças (formas importadas, imagens, texto)
- **Um fluxo de trabalho** definindo como essas peças são processadas
- **Um passo** processado sequencialmente durante trabalhos

**Conceito chave:** Camadas são processadas em ordem, uma após outra, permitindo que você controle a sequência de operações.

:::note Camadas e Peças
Uma camada contém uma ou mais peças. Ao importar arquivos SVG com camadas, cada camada do seu design se torna uma camada separada no Rayforge.
Isso permite manter seu design organizado exatamente como você o criou.
:::


---

## Por Que Usar Múltiplas Camadas?

### Casos de Uso Comuns

**1. Grave Então Corte**

O fluxo de trabalho multi-camadas mais comum:

- **Camada 1:** Gravação raster do design
- **Camada 2:** Corte de contorno do contorno

**Por que camadas separadas?**

- Gravação primeiro garante que a peça não se move durante gravação
- Cortar por último previne peças de caírem antes de gravação completar
- Diferentes configurações de potência/velocidade para cada operação

**2. Corte Multi-Passagem**

Para materiais espessos:

- **Camada 1:** Primeira passagem em potência moderada
- **Camada 2:** Segunda passagem em potência total (mesma geometria)
- **Camada 3:** Terceira passagem opcional se necessário

**Benefícios:**

- Reduz carbonização comparado a uma única passagem de alta potência
- Cada camada pode ter diferentes configurações de velocidade/potência

**3. Projetos Multi-Material**

Diferentes materiais em um trabalho:

- **Camada 1:** Corta peças de acrílico
- **Camada 2:** Grava peças de madeira
- **Camada 3:** Marca peças de metal

**Requisitos:**

- Cada camada mira em áreas diferentes da mesa
- Diferentes velocidade/potência/foco para cada material

**4. Importação de Camadas SVG**

Importe arquivos SVG com estrutura de camadas existente:

- **Camada 1:** Elementos de gravação do SVG
- **Camada 2:** Elementos de corte do SVG
- **Camada 3:** Elementos de marcação do SVG

**Fluxo de trabalho:**

- Importe um arquivo SVG que tem camadas
- Habilite "Usar Vetores Originais" no diálogo de importação
- Selecione quais camadas importar da lista de camadas detectadas
- Cada camada se torna uma camada separada no Rayforge

**Requisitos:**

- Seu arquivo SVG deve usar camadas (criado no Inkscape ou software similar)
- Habilite "Usar Vetores Originais" ao importar
- Nomes de camadas são preservados do seu software de design

---

## Criando e Gerenciando Camadas

### Adicionando uma Nova Camada

1. **Clique no botão "+"** no painel de Camadas
2. **Nomeie a camada** descritivamente (ex: "Camada de Gravação", "Camada de Corte")
3. **A camada aparece** na lista de camadas

**Padrão:** Novos documentos começam com uma camada.

### Propriedades da Camada

Cada camada tem:

| Propriedade       | Descrição                                          |
| -------------- | ---------------------------------------------------- |
| **Nome**       | O nome mostrado na lista de camadas                     |
| **Visível**    | Alterna visibilidade na tela e pré-visualização              |
| **Item de Estoque** | Associação de material opcional                        |
| **Fluxo de Trabalho**   | A(s) operação(ões) aplicada(s) a peças nesta camada |
| **Peças** | As formas/imagens contidas nesta camada            |

:::note Camadas como Contêineres
Camadas são contêineres para suas peças. Ao importar arquivos SVG com camadas, cada camada do seu design se torna uma camada separada no Rayforge.
:::


### Reordenando Camadas

**Ordem de execução = ordem de camadas na lista (de cima para baixo)**

Para reordenar:

1. **Arraste e solte** camadas no painel de Camadas
2. **A ordem importa** - camadas executam de cima para baixo

**Exemplo:**

```
Painel de Camadas:
1. Camada de Gravação     Executa primeiro
2. Camada de Marcação       Executa segundo
3. Camada de Corte         Executa por último (recomendado)
```

### Excluindo Camadas

1. **Selecione a camada** no painel de Camadas
2. **Clique no botão de exclusão** ou pressione Delete
3. **Confirme a exclusão** (todas as peças na camada são removidas)

:::warning Exclusão é Permanente
Excluir uma camada remove todas as suas peças e configurações de fluxo de trabalho. Use Desfazer se você excluir acidentalmente.
:::


---

## Atribuindo Peças a Camadas

### Atribuição Manual

1. **Importe ou crie** uma peça
2. **Arraste a peça** para a camada desejada no painel de Camadas
3. **Ou use o painel de propriedades** para mudar a camada da peça

### Importação de Camadas SVG

Ao importar arquivos SVG com "Usar Vetores Originais" habilitado:

1. **Habilite "Usar Vetores Originais"** no diálogo de importação
2. **O Rayforge detecta camadas** do seu arquivo SVG
3. **Selecione quais camadas** importar usando os interruptores de camada
4. **Cada camada selecionada** se torna uma camada separada com sua própria peça

:::note Detecção de Camadas
O Rayforge detecta automaticamente camadas do seu arquivo SVG. Cada camada
que você criou no seu software de design aparecerá como uma camada separada no
Rayforge.
:::


:::note Somente Importação Vetorial
Seleção de camadas está disponível apenas ao usar importação vetorial direta.
Ao usar modo de traço, todo SVG é processado como uma peça.
:::


### Movendo Peças Entre Camadas

**Arraste e solte:**

- Selecione peça(s) na tela ou painel de Documento
- Arraste para camada alvo no painel de Camadas

**Recortar e colar:**

- Recorte peça da camada atual (Ctrl+X)
- Selecione camada alvo
- Cole (Ctrl+V)

### Diálogo de Importação SVG

Ao importar arquivos SVG, o diálogo de importação fornece opções que afetam o manuseio de camadas:

**Modo de Importação:**

- **Usar Vetores Originais:** Preserva seus caminhos vetoriais e estrutura de camadas.
  Quando habilitado, uma seção "Camadas" aparece mostrando todas as camadas do seu arquivo.
- **Modo de Traço:** Converte o SVG para um bitmap e traça os contornos.
  Seleção de camada é desabilitada neste modo.

**Seção de Camadas (Somente Importação Vetorial):**

- Mostra todas as camadas do seu arquivo SVG
- Cada camada tem um interruptor para habilitar/desabilitar importação
- Nomes de camadas do seu software de design são preservados
- Apenas camadas selecionadas são importadas como camadas separadas

:::tip Preparando Arquivos SVG para Importação de Camadas
Para usar importação de camadas SVG, crie seu design com camadas em software como
Inkscape. Use o painel de Camadas para organizar seu design, e o Rayforge
preservará essa estrutura.
:::


---

## Fluxos de Trabalho de Camadas

Cada camada tem um **Fluxo de Trabalho** que define como suas peças são processadas.

### Configurando Fluxos de Trabalho de Camadas

Para cada camada, você escolhe um tipo de operação e configura suas definições:

**Tipos de Operação:**

- **Contorno** - Segue contornos (para corte ou marcação)
- **Gravação Raster** - Grava imagens e preenche áreas
- **Gravação em Profundidade** - Cria gravações de profundidade variável

**Aprimoramentos Opcionais:**

- **Abas** - Pequenas pontes para segurar peças no lugar durante o corte
- **Overscan** - Estende cortes além da forma para bordas mais limpas
- **Ajuste de Kerf** - Compensa a largura de corte do laser

### Configurações de Camadas Comuns

**Camada de Gravação:**

- Operação: Gravação Raster
- Configurações: 300-500 DPI, velocidade moderada
- Tipicamente não precisa de opções adicionais

**Camada de Corte:**

- Operação: Corte de Contorno
- Opções: Abas (para segurar peças), Overscan (para bordas limpas)
- Configurações: Velocidade mais lenta, potência mais alta

**Camada de Marcação:**

- Operação: Contorno (potência leve, não corta através)
- Configurações: Baixa potência, velocidade rápida
- Propósito: Linhas de dobra, linhas decorativas

---

## Visibilidade de Camadas

Controle quais camadas são mostradas na tela e pré-visualizações:

### Visibilidade na Tela

- **Ícone de olho** no painel de Camadas alterna visibilidade
- **Camadas ocultas:**
  - Não mostradas na tela 2D
  - Não mostradas na pré-visualização 3D
  - **Ainda incluídas no G-code gerado**

**Casos de uso:**

- Ocultar camadas de gravação complexas enquanto posiciona camadas de corte
- Desorganizar a tela quando trabalhando em camadas específicas
- Focar em uma camada de cada vez

### Visibilidade vs Habilitado

| Estado                  | Tela | Pré-visualização | G-code |
| ---------------------- | ------ | ------- | ------ |
| **Visível e Habilitado**  | Sim    | Sim     | Sim    |
| **Oculto e Habilitado**   | Não     | Não      | Sim    |
| **Visível e Desabilitado** | Sim    | Sim     | Não     |
| **Oculto e Desabilitado**  | Não     | Não      | Não     |

:::note Desabilitando Camadas
:::

Para temporariamente excluir uma camada de trabalhos sem excluí-la, desligue a operação da camada ou desabilite-a nas configurações da camada.

---

## Ordem de Execução de Camadas

### Como Camadas São Processadas

Durante execução de trabalho, o Rayforge processa cada camada em ordem de cima para baixo. Dentro de cada camada, todas as peças são processadas antes de mover para a próxima camada.

### A Ordem Importa

**Ordem errada:**

```
1. Camada de Corte
2. Camada de Gravação
```

**Problema:** Peças cortadas podem cair ou mover antes de gravar!

**Ordem correta:**

```
1. Camada de Gravação
2. Camada de Corte
```

**Por que:** Gravação acontece enquanto peça ainda está anexada, então corte a liberta.

### Múltiplas Passagens

Para materiais espessos, crie múltiplas camadas de corte:

```
1. Camada de Gravação
2. Camada de Corte (Passagem 1) - 50% potência
3. Camada de Corte (Passagem 2) - 75% potência
4. Camada de Corte (Passagem 3) - 100% potência
```

**Dica:** Use a mesma geometria para todas passagens de corte (duplique a camada).

---

## Técnicas Avançadas

### Agrupamento de Camadas por Material

Use camadas para organizar por material ao executar trabalhos mistos:

```
Material 1 (Acrílico 3mm):
  - Camada de Gravação Acrílico
  - Camada de Corte Acrílico

Material 2 (Compensado 3mm):
  - Camada de Gravação Madeira
  - Camada de Corte Madeira
```

**Fluxo de trabalho:**

1. Processe todas camadas do Material 1
2. Troque materiais
3. Processe todas camadas do Material 2

**Alternativa:** Use documentos separados para materiais diferentes.

### Pausando Entre Camadas

Você pode configurar o Rayforge para pausar entre camadas. Isso é útil quando você precisa:

- Trocar materiais no meio do trabalho
- Inspecionar progresso antes de continuar
- Ajustar foco para operações diferentes

Para configurar pausas de camada, use o recurso de hooks nas configurações da sua máquina.

### Configurações Específicas por Camada

O fluxo de trabalho de cada camada pode ter configurações únicas:

| Camada   | Operação | Velocidade      | Potência | Passagens |
| ------- | --------- | ---------- | ----- | ------ |
| Grave | Raster    | 300 mm/min | 20%   | 1      |
| Marcação   | Contour   | 500 mm/min | 10%   | 1      |
| Corte     | Contour   | 100 mm/min | 90%   | 2      |

---

## Melhores Práticas

### Convenções de Nomenclatura

**Bons nomes de camadas:**

- "Gravação - Logo"
- "Corte - Contorno Externo"
- "Marcação - Linhas de Dobra"
- "Passagem 1 - Corte Grosseiro"
- "Passagem 2 - Corte Final"

**Nomes de camadas ruins:**

- "Camada 1", "Camada 2" (não descritivo)
- Descrições longas (mantenha conciso)

### Organização de Camadas

1. **De cima para baixo = ordem de execução**
2. **Gravação antes de corte** (regra geral)
3. **Agrupe operações relacionadas** (todo corte, toda gravação)
4. **Use visibilidade** para focar no trabalho atual
5. **Exclua camadas não utilizadas** para manter projetos limpos

### Preparando Arquivos SVG para Importação de Camadas

**Para melhores resultados ao importar camadas SVG:**

1. **Use o painel de Camadas** no seu software de design para organizar seu design
2. **Atribua nomes significativos** a cada camada (ex: "Gravação", "Corte")
3. **Mantenha camadas planas** - evite colocar camadas dentro de outras camadas
4. **Salve seu arquivo** e importe no Rayforge
5. **Verifique detecção de camadas** checando o diálogo de importação

O Rayforge funciona melhor com arquivos SVG criados no Inkscape ou software similar de design vetorial que suporta camadas.

### Desempenho

**Muitas camadas:**

- Sem impacto significativo de desempenho
- 10-20 camadas é comum para trabalhos complexos
- Organize logicamente, não para minimizar contagem de camadas

**Simplifique se necessário:**

- Combine operações similares em uma camada quando possível
- Use menos gravações raster (mais intensivas em recursos)

---

## Solução de Problemas

### Camada Não Está Gerando G-code

**Problema:** Camada aparece no documento mas não no G-code gerado.

**Soluções:**

1. **Verifique se camada tem peças** - Camadas vazias são puladas
2. **Verifique se fluxo de trabalho está configurado** - Camada precisa de uma operação
3. **Verifique configurações de operação** - Potência > 0, velocidade válida, etc.
4. **Verifique visibilidade da peça** - Peças ocultas podem não processar
5. **Regenere G-code** - Faça uma pequena mudança para forçar regeneração

### Ordem Errada de Camadas

**Problema:** Operações executam em ordem inesperada.

**Solução:** Reordene camadas no painel de Camadas. Lembre: cima = primeiro.

### Camadas Sobrepondo na Pré-visualização

**Problema:** Múltiplas camadas mostram conteúdo sobreposto na pré-visualização.

**Esclarecimento:** Isso é normal se camadas compartilham a mesma área XY.

**Soluções:**

- Use visibilidade de camada para ocultar outras camadas temporariamente
- Verifique pré-visualização 3D para ver profundidade/ordem
- Verifique se isso é intencional (ex: gravar então cortar mesma forma)

### Peça na Camada Errada

**Problema:** Peça foi atribuída a camada incorreta.

**Solução:** Arraste peça para camada correta no painel de Camadas ou árvore de Documento.

### Camadas SVG Não Detectadas

**Problema:** Importando um arquivo SVG mas nenhuma camada aparece no diálogo de importação.

**Soluções:**

1. **Verifique estrutura SVG** - Abra seu arquivo no Inkscape ou software similar
   para verificar se tem camadas
2. **Habilite "Usar Vetores Originais"** - Seleção de camada está disponível apenas em
   este modo de importação
3. **Verifique se seu design tem camadas** - Certifique-se de que criou camadas no
   seu software de design, não apenas grupos
4. **Verifique camadas aninhadas** - Camadas dentro de outras camadas podem não ser
   detectadas propriamente
5. **Salve novamente seu arquivo** - Às vezes salvar novamente com uma versão atual do
   seu software de design ajuda

### Importação de Camada SVG Mostra Conteúdo Errado

**Problema:** Camada importada mostra conteúdo de outras camadas ou está vazia.

**Soluções:**

1. **Verifique seleção de camada** - Verifique se as camadas corretas estão habilitadas no
   diálogo de importação
2. **Verifique seu design** - Abra o arquivo original no seu software de design
   para confirmar que cada camada contém o conteúdo certo
3. **Verifique elementos compartilhados** - Elementos que aparecem em múltiplas camadas
   podem causar confusão
4. **Tente modo de traço** - Use modo de traço como alternativa se importação vetorial tem
   problemas

---

## Páginas Relacionadas

- [Operações](./operations/contour) - Tipos de operação para fluxos de trabalho de camadas
- [Modo Simulação](./simulation-mode) - Pré-visualize execução multi-camadas
- [Macros & Hooks](../machine/hooks-macros) - Hooks de nível de camada para automação
- [Visualização 3D](../ui/3d-preview) - Visualize pilha de camadas
