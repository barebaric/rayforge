# Guia de Posicionamento de Peça de Trabalho

Este guia cobre todos os métodos disponíveis no Rayforge para posicionar com precisão sua peça de
trabalho e alinhar seus designs antes de cortar ou gravar.

## Visão Geral

O posicionamento preciso da peça de trabalho é essencial para:

- **Prevenir desperdício**: Evitar cortar no local errado
- **Alinhamento preciso**: Posicionar designs em materiais pré-impressos
- **Resultados repetíveis**: Executar o mesmo trabalho várias vezes de forma consistente
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

O modo foco liga o laser em um nível de potência baixo, atuando como um "ponteiro laser" para
ajudá-lo a ver exatamente onde a cabeça do laser está posicionada.

### Ativar o Modo Foco

1. **Conectar à sua máquina**
2. **Clicar no botão Foco** na barra de ferramentas (ícone de laser)
3. O laser liga no nível de potência de foco configurado
4. **Mover a cabeça do laser** para ver a posição do feixe no seu material
5. **Clicar no botão Foco novamente** para desligar quando terminar

<!-- prettier-ignore-start -->
:::warning[Segurança]
Mesmo em baixa potência, o laser pode danificar os olhos. Nunca olhe
diretamente para o feixe ou aponte para superfícies reflexivas. Use proteção ocular adequada.
:::
<!-- prettier-ignore-end -->

### Configurar a Potência de Foco

A potência de foco determina o quão brilhante o ponto laser aparece:

1. Vá para **Configurações → Máquina → Laser**
2. Encontre a configuração **Potência de Foco**
3. Defina um valor que torne o ponto visível sem marcar seu material
   - Valores típicos: 1-5% para a maioria dos materiais
   - Defina como 0 para desativar o recurso

<!-- prettier-ignore-start -->
:::tip[Encontrando a Potência Certa]
Comece com 1% e aumente gradualmente. O ponto deve ser visível,
mas não deixar nenhuma marca no seu material. Materiais mais escuros podem precisar de maior
potência para ver o ponto claramente.
:::
<!-- prettier-ignore-end -->

### Quando Usar o Modo Foco

- **Verificações rápidas de alinhamento**: Ver se o laser está aproximadamente onde você espera
- **Encontrar bordas do material**: Mover para os cantos para verificar o posicionamento do material
- **Definir origem SCT**: Posicionar laser no ponto zero desejado antes de definir SCT
- **Verificar posição inicial**: Verificar se o referenciamento funcionou corretamente

---

## Ponteiro Laser com Deslocamento

Algumas máquinas têm um ponteiro laser dedicado (um pequeno laser de ponto vermelho) montado a uma
distância fixa do feixe de corte. Quando você alinha o material usando o ponto do ponteiro, o feixe
de corte na verdade ficaria deslocado desse ponto — a menos que o Rayforge compense.

A configuração de laser [Deslocamento do Ponteiro](../machine/laser.md) permite informar essa
distância e ativar ou desativar a compensação. Com ela ativada:

- **Definir zero na posição atual** (e Zerar X / Zerar Y) coloca a origem de trabalho na posição do
  ponto do ponteiro.
- A tela mostra um ponto amarelo do ponteiro ao lado do ponto vermelho do feixe, marcando onde o
  ponto do ponteiro está no seu material.
- Uma chave de **alinhamento do ponteiro** aparece no popover de movimentação (o ícone de bússola ao
  lado da leitura de posição).

### Alinhamento do Ponteiro

Enquanto o **alinhamento do ponteiro** está ativado, cada operação absoluta de mira — Mover para, os
atalhos de canto, ir à origem do SCT, Clicar para mover, Mover cabeça para cá e enquadrar — faz o
_ponto do ponteiro_ ficar na posição mirada, de modo que você pode posicionar o material
inteiramente pelo ponto visível. A entrada de coordenadas do popover é pré-preenchida com a posição
do ponto do ponteiro. Na tela, sempre há exatamente um ponto preenchido: o ponto do ponteiro é
desenhado preenchido enquanto o alinhamento está ativado (o ponto do feixe é então um anel oco), e
oco enquanto desativado (o ponto do feixe é preenchido).

O fluxo de trabalho combina perfeitamente com o zeramento pelo ponto do ponteiro: a origem fica onde
o ponteiro marcou, a mira desloca cada alvo pelo deslocamento, e a gravação não é deslocada —
alinhar com o ponto e cortar com o feixe acabam sendo consistentes.

Ao pressionar **Enviar** com o alinhamento ativado, um aviso aparece a cada envio (não há opção de
"não perguntar novamente"). Você pode escolher:

- **Simulação com ponteiro**: o trabalho é executado com o deslocamento do ponteiro aplicado, de
  modo que o ponto do ponteiro traga o trajeto enquanto o feixe corre deslocado. O laser continua
  disparando na potência do trabalho — certifique-se de que o feixe deslocado não possa atingir nada
  que não deva.
- **Desativar e gravar**: o alinhamento é desativado e o trabalho grava normalmente com o feixe nas
  posições do SCT.
- **Cancelar**.

Movimentos de jog e trabalhos nunca são deslocados pela chave: o jog é relativo, e a saída G-code é
idêntica com o alinhamento ativado ou não. A chave vale apenas para a sessão — não é salva no perfil
da máquina e é redefinida ao trocar de máquina.

O deslocamento do ponteiro é especialmente útil para material maior que a mesa laser (trabalho de
passagem contínua), onde você alinha repetidamente o desenho a uma marca de referência no material
em movimento.

---

## Enquadramento

O enquadramento traça o retângulo delimitador do seu trabalho em potência baixa (ou zero), mostrando
exatamente onde seu design será cortado ou gravado.

### Como Enquadrar

1. **Carregar e posicionar seu design** no Rayforge
2. **Clicar em Máquina → Enquadrar** ou pressionar `Ctrl+F`
3. A cabeça do laser traça a caixa delimitadora do seu trabalho
4. **Verificar o contorno** que cabe dentro do seu material

### Configurações de Enquadramento

Configurar comportamento de enquadramento em **Configurações → Máquina → Laser**:

- **Velocidade de Enquadramento**: Quão rápido a cabeça se move durante o enquadramento (mais lento
  = mais fácil de ver)
- **Potência de Enquadramento**: Potência do laser durante o enquadramento
  - Defina como 0 para enquadramento a ar (laser desligado, apenas movimento)
  - Defina como 1-5% para um rastro visível no material

<!-- prettier-ignore-start -->
:::tip[Enquadramento a Ar vs. Baixa Potência]
- **Enquadramento a ar (0% potência)**: Seguro para qualquer material, mas você só vê o movimento da
  cabeça
- **Enquadramento de baixa potência**: Deixa uma marca visível fraca, útil para alinhamento preciso
  em materiais escuros
:::
<!-- prettier-ignore-end -->

### Quando Enquadrar

- **Antes de cada trabalho**: Verificação rápida de que o design cabe
- **Após mudanças de posição**: Confirmar que o novo posicionamento está correto
- **Materiais caros**: Verificar duas vezes antes de se comprometer
- **Trabalhos de múltiplas peças**: Verificar que todas as peças cabem no material

Veja [Enquadrando Seu Trabalho](framing-your-job) para mais detalhes.

---

## Definir Zero SCT (Sistema de Coordenadas de Trabalho)

Os Sistemas de Coordenadas de Trabalho (SCT) permitem que você defina "pontos zero" personalizados
para seus trabalhos. Isso facilita alinhar trabalhos à posição do seu material.

### Configuração Rápida de SCT

1. **Mover a cabeça do laser** para o canto do seu material (ou ponto de origem desejado)
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

<!-- prettier-ignore-start -->
:::tip[Múltiplas Áreas de Trabalho]
Use slots SCT diferentes para diferentes posições de fixação. Por
exemplo:

- G54 para o lado esquerdo da sua mesa
- G55 para o lado direito
- G56 para um acessório rotativo
:::
<!-- prettier-ignore-end -->

### Quando Definir Zero SCT

- **Novo posicionamento de material**: Alinhar origem ao canto do material
- **Trabalho com fixação**: Definir origem ao ponto de referência da fixação
- **Trabalhos repetíveis**: Mesmo trabalho, diferentes posições
- **Lotes de produção**: Posicionamento consistente através de múltiplas peças

Veja [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems.md) para documentação
completa.

---

## Posicionamento Baseado em Câmera

A sobreposição de câmera mostra uma visualização ao vivo do seu material com seu design sobreposto,
permitindo alinhamento visual preciso.

### Configurar a Câmera

1. **Conectar uma câmera USB** acima da sua área de trabalho
2. Vá para **Configurações → Câmera** e adicione seu dispositivo de câmera
3. **Ativar a câmera** para ver a sobreposição na sua tela
4. **Alinhar a câmera** usando o procedimento de alinhamento (necessário para posicionamento
   preciso)

### Alinhamento da Câmera

O alinhamento da câmera mapeia os pixels da câmera para coordenadas do mundo real:

1. Abrir **Câmera → Alinhar Câmera**
2. Colocar marcadores de alinhamento em posições conhecidas (pelo menos 4 pontos)
3. Inserir as coordenadas X/Y do mundo real para cada ponto
4. Clicar em **Aplicar** para calcular a transformação

<!-- prettier-ignore-start -->
:::tip[Precisão do Alinhamento]
- Use pontos distribuídos por toda sua área de trabalho
- Meça as coordenadas do mundo cuidadosamente com uma régua
- Use posições de máquina (mover para coordenadas conhecidas) para maior precisão
:::
<!-- prettier-ignore-end -->

### Posicionamento com Sobreposição de Câmera

1. **Ativar a sobreposição de câmera** para ver seu material
2. **Importar seu design**
3. **Arrastar o design** para alinhar com características visíveis na câmera
4. **Ajuste fino** usando as teclas de seta para posicionamento perfeito ao pixel
5. **Enquadrar para verificar** antes de executar o trabalho

### Quando Usar Posicionamento com Câmera

- **Materiais pré-impressos**: Alinhar cortes a impressões existentes
- **Materiais irregulares**: Posicionar em peças não retangulares
- **Posicionamento preciso**: Requisitos de precisão sub-milimétrica
- **Layouts complexos**: Múltiplos elementos com espaçamento específico

Veja [Integração de Câmera](../machine/camera.md) para documentação completa.

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

### Fluxo de Trabalho de Passagem Contínua

Para material mais longo que a mesa laser (alimentado pela máquina em segmentos), o **deslocamento
do ponteiro** elimina o cálculo manual:

1. **Corte o segmento 1** do seu desenho normalmente.
2. **Alimente o material para frente** pela passagem contínua para que o próximo segmento fique
   sobre a mesa.
3. **Movimente a máquina** até o ponto do ponteiro marcar um ponto de referência no material (por
   exemplo, um canto de um elemento já cortado).
4. **Defina o zero do SCT** (ou Zerar X / Zerar Y) — com o deslocamento do ponteiro ativado, a
   origem fica exatamente onde o ponteiro apontou.
5. **Posicione o próximo segmento** do desenho em relação a essa origem na tela.
6. **Enquadre para verificar**, então **execute o trabalho**.
7. Repita a partir do passo 2 para cada segmento restante.

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
- **Veja solução de problemas da câmera** na documentação de Integração de Câmera

---

## Tópicos Relacionados

- [Enquadrando Seu Trabalho](framing-your-job) - Documentação detalhada de enquadramento
- [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems.md) - Referência SCT
- [Integração de Câmera](../machine/camera.md) - Configuração e alinhamento de câmera
- [Painel de Controle](../ui/bottom-panel.md) - Controles de movimento e gestão SCT
- [Guia de Início Rápido](../getting-started/quick-start.md) - Fluxo de trabalho básico
