# Guia de Posicionamento de Peça de Trabalho

Este guia cobre todos os métodos disponíveis no Rayforge para posicionar com
precisão sua peça de trabalho e alinhar seus designs antes de cortar ou
gravar.

## Visão Geral

O posicionamento preciso da peça de trabalho é essencial para:

- **Prevenir desperdício**: Evitar cortar no local errado
- **Alinhamento preciso**: Posicionar designs em materiais pré-impressos
- **Resultados repetíveis**: Executar o mesmo trabalho várias vezes de forma
  consistente
- **Trabalhos de múltiplas peças**: Alinhar múltiplas peças em uma única folha

O Rayforge fornece várias ferramentas complementares para posicionamento:

| Método                     | Propósito                       | Melhor Para                                       |
| -------------------------- | ------------------------------- | ------------------------------------------------- |
| **Modo Foco**              | Ver posição do laser            | Alinhamento visual rápido                         |
| **Enquadramento**          | Pré-visualizar limites          | Verificar se o design cabe no material            |
| **Zero SCT**               | Definir origem                  | Posicionamento repetível                          |
| **Sobreposição de Câmera** | Posicionamento visual do design | Alinhamento preciso em características existentes |

---

## Modo Foco (Ponteiro Laser)

O modo foco liga o laser em um nível de potência baixo, atuando como um
"ponteiro laser" para ajudá-lo a ver exatamente onde a cabeça do laser está
posicionada.

### Ativar o Modo Foco

1. **Conectar à sua máquina**
2. **Clicar no botão Foco** na barra de ferramentas (ícone de laser)
3. O laser liga no nível de potência de foco configurado
4. **Mover a cabeça do laser** para ver a posição do feixe no seu material
5. **Clicar no botão Foco novamente** para desligar quando terminar

:::warning Segurança
Mesmo em baixa potência, o laser pode danificar os olhos. Nunca olhe
diretamente para o feixe ou aponte para superfícies reflexivas. Use proteção
ocular adequada.
:::

### Configurar a Potência de Foco

A potência de foco determina o quão brilhante o ponto laser aparece:

1. Vá para **Configurações → Máquina → Laser**
2. Encontre a configuração **Potência de Foco**
3. Defina um valor que torne o ponto visível sem marcar seu material
   - Valores típicos: 1-5% para a maioria dos materiais
   - Defina como 0 para desativar o recurso

:::tip Encontrando a Potência Certa
Comece com 1% e aumente gradualmente. O ponto deve ser visível, mas não
deixar nenhuma marca no seu material. Materiais mais escuros podem precisar
de maior potência para ver o ponto claramente.
:::

### Quando Usar o Modo Foco

- **Verificações rápidas de alinhamento**: Ver se o laser está
  aproximadamente onde você espera
- **Encontrar bordas do material**: Mover para os cantos para verificar o
  posicionamento do material
- **Definir origem SCT**: Posicionar laser no ponto zero desejado antes de
  definir SCT
- **Verificar posição inicial**: Verificar se o referenciamento funcionou
  corretamente

---

## Enquadramento

O enquadramento traça o retângulo delimitador do seu trabalho em potência
baixa (ou zero), mostrando exatamente onde seu design será cortado ou gravado.

### Como Enquadrar

1. **Carregar e posicionar seu design** no Rayforge
2. **Clicar em Máquina → Enquadrar** ou pressionar `Ctrl+F`
3. A cabeça do laser traça a caixa delimitadora do seu trabalho
4. **Verificar o contorno** que cabe dentro do seu material

### Configurações de Enquadramento

Configurar comportamento de enquadramento em **Configurações → Máquina →
Laser**:

- **Velocidade de Enquadramento**: Quão rápido a cabeça se move durante o
  enquadramento (mais lento = mais fácil de ver)
- **Potência de Enquadramento**: Potência do laser durante o enquadramento
  - Defina como 0 para enquadramento a ar (laser desligado, apenas movimento)
  - Defina como 1-5% para um rastro visível no material

:::tip Enquadramento a Ar vs. Baixa Potência

- **Enquadramento a ar (0% potência)**: Seguro para qualquer material, mas
  você só vê o movimento da cabeça
- **Enquadramento de baixa potência**: Deixa uma marca visível fraca, útil
  para alinhamento preciso em materiais escuros
  :::

### Quando Enquadrar

- **Antes de cada trabalho**: Verificação rápida de que o design cabe
- **Após mudanças de posição**: Confirmar que o novo posicionamento está
  correto
- **Materiais caros**: Verificar duas vezes antes de se comprometer
- **Trabalhos de múltiplas peças**: Verificar que todas as peças cabem no
  material

Veja [Enquadrando Seu Trabalho](framing-your-job) para mais detalhes.

---

## Definir Zero SCT (Sistema de Coordenadas de Trabalho)

Os Sistemas de Coordenadas de Trabalho (SCT) permitem que você defina "pontos
zero" personalizados para seus trabalhos. Isso facilita alinhar trabalhos à
posição do seu material.

### Configuração Rápida de SCT

1. **Mover a cabeça do laser** para o canto do seu material (ou ponto de
   origem desejado)
2. **Abrir o Painel de Controle** (`Ctrl+L`)
3. **Selecionar um SCT** (G54 é o sistema de coordenadas de trabalho padrão)
4. **Clicar em Zero X e Zero Y** para definir a posição atual como origem
5. O ponto (0,0) do seu design agora será alinhado com esta posição

### Entendendo os Sistemas de Coordenadas

O Rayforge usa vários sistemas de coordenadas:

| Sistema     | Descrição                                               |
| ----------- | ------------------------------------------------------- |
| **G53**     | Coordenadas de máquina (fixas, não podem ser alteradas) |
| **G54**     | Sistema de coordenadas de trabalho 1 (padrão)           |
| **G55-G59** | Sistemas de coordenadas de trabalho adicionais          |

:::tip Múltiplas Áreas de Trabalho
Use slots SCT diferentes para diferentes posições de fixação. Por exemplo:

- G54 para o lado esquerdo da sua mesa
- G55 para o lado direito
- G56 para um acessório rotativo
  :::

### Quando Definir Zero SCT

- **Novo posicionamento de material**: Alinhar origem ao canto do material
- **Trabalho com fixação**: Definir origem ao ponto de referência da fixação
- **Trabalhos repetíveis**: Mesmo trabalho, diferentes posições
- **Lotes de produção**: Posicionamento consistente através de múltiplas peças

Veja [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems)
para documentação completa.

---

## Posicionamento Baseado em Câmera

A sobreposição de câmera mostra uma visualização ao vivo do seu material com
seu design sobreposto, permitindo alinhamento visual preciso.

### Configurar a Câmera

1. **Conectar uma câmera USB** acima da sua área de trabalho
2. Vá para **Configurações → Câmera** e adicione seu dispositivo de câmera
3. **Ativar a câmera** para ver a sobreposição na sua tela
4. **Alinhar a câmera** usando o procedimento de alinhamento (necessário
   para posicionamento preciso)

### Alinhamento da Câmera

O alinhamento da câmera mapeia os pixels da câmera para coordenadas do mundo
real:

1. Abrir **Câmera → Alinhar Câmera**
2. Colocar marcadores de alinhamento em posições conhecidas (pelo menos 4
   pontos)
3. Inserir as coordenadas X/Y do mundo real para cada ponto
4. Clicar em **Aplicar** para calcular a transformação

:::tip Precisão do Alinhamento

- Use pontos distribuídos por toda sua área de trabalho
- Meça as coordenadas do mundo cuidadosamente com uma régua
- Use posições de máquina (mover para coordenadas conhecidas) para maior
  precisão
  :::

### Posicionamento com Sobreposição de Câmera

1. **Ativar a sobreposição de câmera** para ver seu material
2. **Importar seu design**
3. **Arrastar o design** para alinhar com características visíveis na câmera
4. **Ajuste fino** usando as teclas de seta para posicionamento perfeito ao
   pixel
5. **Enquadrar para verificar** antes de executar o trabalho

### Quando Usar Posicionamento com Câmera

- **Materiais pré-impressos**: Alinhar cortes a impressões existentes
- **Materiais irregulares**: Posicionar em peças não retangulares
- **Posicionamento preciso**: Requisitos de precisão sub-milimétrica
- **Layouts complexos**: Múltiplos elementos com espaçamento específico

Veja [Integração de Câmera](../machine/camera) para documentação completa.

---

## Fluxos de Trabalho Recomendados

### Fluxo de Trabalho de Posicionamento Básico

Para trabalhos simples em materiais retangulares:

1. **Colocar material** na mesa do laser
2. **Ativar modo foco** e mover para verificar posição do material
3. **Definir zero SCT** no canto do material
4. **Posicionar seu design** na tela
5. **Enquadrar o trabalho** para verificar posicionamento
6. **Executar o trabalho**

### Fluxo de Trabalho de Alinhamento de Precisão

Para posicionamento preciso em materiais pré-impressos ou marcados:

1. **Configurar e alinhar câmera** (configuração única)
2. **Colocar material** na mesa do laser
3. **Ativar sobreposição de câmera** para ver o material
4. **Importar e posicionar design** visualmente na imagem da câmera
5. **Desativar câmera** e enquadrar para verificar
6. **Executar o trabalho**

### Fluxo de Trabalho de Produção

Para executar múltiplos trabalhos idênticos:

1. **Configurar fixação** na mesa do laser
2. **Definir zero SCT** alinhado à fixação (ex. G54)
3. **Carregar e configurar** seu design
4. **Enquadrar para verificar** alinhamento com a fixação
5. **Executar o trabalho**
6. **Substituir material** e repetir (SCT permanece o mesmo)

### Fluxo de Trabalho de Múltiplas Posições

Para executar o mesmo trabalho em diferentes locais:

1. **Configurar múltiplas posições SCT**:
   - Mover para posição 1, definir zero G54
   - Mover para posição 2, definir zero G55
   - Mover para posição 3, definir zero G56
2. **Carregar seu design** (mesmo design para todas as posições)
3. **Selecionar G54**, enquadrar e executar
4. **Selecionar G55**, enquadrar e executar
5. **Selecionar G56**, enquadrar e executar

---

## Solução de Problemas

### Ponto laser não visível no modo foco

- **Aumentar potência de foco** nas configurações do laser
- **Materiais escuros** podem precisar de maior potência (5-10%)
- **Verificar conexão do laser** e garantir que a máquina está respondendo
- **Verificar se a potência de foco** não está definida como 0

### Sobreposição de câmera desalinhada

- **Executar alinhamento de câmera novamente** com mais pontos de referência
- **Verificar montagem da câmera** - ela pode ter se movido
- **Verificar se as coordenadas do mundo** foram medidas com precisão
- **Veja solução de problemas da câmera** na documentação de Integração de
  Câmera

---

## Tópicos Relacionados

- [Enquadrando Seu Trabalho](framing-your-job) - Documentação detalhada de
  enquadramento
- [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems) -
  Referência SCT
- [Integração de Câmera](../machine/camera) - Configuração e alinhamento de
  câmera
- [Painel de Controle](../ui/bottom-panel) - Controles de movimento e gestão
  SCT
- [Guia de Início Rápido](../getting-started/quick-start) - Fluxo de trabalho
  básico
