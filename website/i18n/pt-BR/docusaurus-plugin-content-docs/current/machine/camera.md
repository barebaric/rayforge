---
description: "Configure a calibração da câmera no Rayforge para alinhamento preciso da peça de trabalho. Use sua câmera para visualizar e posicionar designs em materiais."
---

# Integração com Câmera

O Rayforge suporta integração com câmera USB para alinhamento e posicionamento
preciso de materiais. O recurso de sobreposição de câmera permite ver
exatamente onde seu laser vai cortar ou gravar no material, eliminando
suposições e reduzindo o desperdício de material.

![Configurações da Câmera](/screenshots/machine-camera.png)

## Fluxo de trabalho de configuração

A configuração de uma câmera segue quatro etapas:

1. **Adicionar uma câmera** — Conecte sua câmera e adicione-a à configuração
   da máquina
2. **Ajustar configurações de imagem** — Ajuste brilho, contraste, balanço de
   branco e redução de ruído
3. **Calibrar a lente** — Corrija a distorção com o assistente de calibração
   ou coeficientes manuais
4. **Alinhar a câmera** — Mapeie pixels da câmera para coordenadas da máquina
   para posicionamento preciso

As etapas 2–4 são acessadas no painel de propriedades da câmera, onde ícones
de status mostram o progresso rapidamente:

- ✓ **Calibração de Lente** — A calibração foi realizada
- ⚠ **Alinhamento de Imagem** — Aviso quando o alinhamento precisa ser refeito
  (p. ex., após calibração de lente)
- ✓ **Alinhamento de Imagem** — O alinhamento está atual e válido

---

## Passo 1: Adicionar uma câmera

### Requisitos de Hardware

**Câmeras compatíveis:**

- Webcams USB (mais comum)
- Câmeras integradas de laptop (se executar Rayforge em laptop perto da
  máquina)
- Qualquer câmera suportada por Video4Linux2 (V4L2) no Linux ou
  DirectShow no Windows

**Configuração recomendada:**

- Câmera montada acima da área de trabalho com visão clara do material
- Condições de iluminação consistentes
- Câmera posicionada para capturar a área de trabalho do laser
- Montagem segura para prevenir movimento da câmera

### Adicionando uma Câmera

1. **Conecte sua câmera** ao computador via USB

2. **Abra Configurações da Câmera:**
   - Navegue até **Configurações → Preferências → Câmera**
   - Ou use o botão da câmera na barra de ferramentas

3. **Adicione uma nova câmera:**
   - Clique no botão "+" para adicionar uma câmera
   - Digite um nome descritivo (ex: "Câmera Superior",
     "Câmera da Área de Trabalho")
   - Selecione o dispositivo no menu suspenso
     - No Linux: `/dev/video0`, `/dev/video1`, etc.
     - No Windows: Camera 0, Camera 1, etc.

4. **Habilite a câmera:**
   - Ative o interruptor de habilitação da câmera
   - O feed ao vivo deve aparecer na sua tela

---

## Passo 2: Ajustar configurações de imagem

![Diálogo de Configurações de Imagem](/screenshots/camera-image-settings.png)

Clique em **Configurar** ao lado de **Configurações de Imagem** nas
propriedades da câmera para abrir o diálogo de configurações de imagem. Ajuste
estes parâmetros para obter uma visão clara da câmera:

| Configuração          | Descrição                                                                              |
| --------------------- | -------------------------------------------------------------------------------------- |
| **Brilho**            | Brilho geral da imagem (-100 a +100)                                                   |
| **Contraste**         | Definição de bordas e contraste (0 a 100)                                              |
| **Preferir YUYV**     | Usar YUYV não comprimido em vez de MJPEG. Mais lento mas pode corrigir alguns glitches |
| **Transparência**     | Opacidade da sobreposição na tela (0% opaco a 100% transparente)                       |
| **Balanço de Branco** | Correção de temperatura de cor (Auto ou 2500-10000K)                                   |
| **Redução de Ruído**  | Redução de ruído temporal (0.0 a 0.95)                                                 |

A opção YUYV é útil se sua câmera produz imagens esverdeadas com o formato
MJPEG padrão. Note que YUYV é não comprimido e pode reduzir a resolução
disponível ou a taxa de quadros em conexões USB 2.0.

---

## Passo 3: Calibração de lente

Se sua câmera tem uma lente grande-angular ou está montada em um ângulo,
a imagem pode mostrar curvatura visível — linhas retas aparecem curvadas,
especialmente perto das bordas do quadro. Isso é chamado de distorção de
lente, e pode comprometer o alinhamento mesmo que seus pontos de
alinhamento sejam cuidadosamente medidos.

O Rayforge inclui um assistente de calibração guiado que corrige essa
distorção automaticamente. Você também pode ajustar os coeficientes de
distorção manualmente.

### Diálogo de Calibração de Lente

![Diálogo de Calibração de Lente](/screenshots/camera-lens-calibration.png)

Abra o diálogo de calibração de lente clicando em **Configurar** ao lado
de **Calibração de Lente** nas propriedades da câmera. A partir daqui
você pode:

- **Ajustar coeficientes de distorção manualmente** — Ajuste fino dos
  parâmetros de distorção radial (k1–k3) e tangencial (p1–p2)
- **Iniciar o assistente de calibração** — Clique no botão **Assistente**
  para calibração automática guiada

Ajustes manuais são úteis para ajuste fino após o assistente ter
calculado uma solução inicial, ou quando você conhece os valores
aproximados de distorção para sua lente.

### Assistente de Calibração

O assistente de calibração orienta você a capturar várias imagens de um
cartão de calibração impresso de diferentes posições na mesa. Ele então
calcula um modelo de distorção automaticamente.

**Passo 1: Configurar o cartão de calibração**

![Assistente — Configurações do
Cartão](/screenshots/camera-lens-calibration-wizard-card.png)

1. Clique em **Assistente** no diálogo de calibração de lente para iniciar
2. Defina a **Largura** e **Altura** do seu cartão impresso
3. A visualização é atualizada em tempo real — o cartão deve cobrir
   cerca de 70% da vista da câmera
4. Clique em **Salvar como PDF** para exportar o cartão para impressão
5. Imprima o cartão e coloque-o na mesa do laser

**Passo 2: Capturar quadros**

![Assistente —
Captura](/screenshots/camera-lens-calibration-wizard-capture.png)

1. Clique em **Avançar** para entrar no modo de captura
2. Posicione o cartão de calibração em diferentes locais e ângulos
   dentro da vista da câmera
3. Clique em **Capturar Quadro** para cada posição
4. Visel pelo menos 8 capturas cobrindo todo o quadro, incluindo cantos
   e bordas
5. A barra de progresso e indicadores de status mostram a qualidade da captura

**Passo 3: Aplicar calibração**

1. Quando quadros suficientes forem capturados, clique em **Calibrar**
2. Os coeficientes de distorção calculados são automaticamente aplicados
   à câmera
3. A sobreposição da câmera agora mostra uma imagem corrigida e reta

---

## Passo 4: Alinhamento de imagem

![Diálogo de Alinhamento de Imagem](/screenshots/camera-image-alignment.png)

O alinhamento da câmera calibra a relação entre pixels da câmera e coordenadas
do mundo real, permitindo posicionamento preciso.

### Por Que o Alinhamento é Necessário

A câmera vê a área de trabalho de cima, mas a imagem pode estar:

- Rotacionada em relação aos eixos da máquina
- Escalada de forma diferente nas direções X e Y
- Distorcida pela perspectiva da lente

O alinhamento cria uma matriz de transformação que mapeia pixels da câmera
para coordenadas da máquina.

### Procedimento de Alinhamento

1. **Abra o Diálogo de Alinhamento:**
   - Clique no botão **Configurar** ao lado de **Alinhamento de Imagem** nas
     propriedades da câmera
   - O diálogo mostra o feed da câmera com a sobreposição de alinhamento atual

2. **Coloque marcadores de alinhamento:**
   - Você precisa de pelo menos 3 pontos de referência (4 recomendados para
     melhor precisão)
   - Os pontos de alinhamento devem ser espalhados pela área de trabalho
   - Use posições conhecidas como:
     - Posição de origem da máquina
     - Marcações de régua
     - Furos de alinhamento pré-cortados
     - Grade de calibração

3. **Marque pontos na imagem:**
   - Clique na imagem da câmera para colocar um ponto em um local conhecido
   - O widget de balão aparece mostrando as coordenadas do ponto
   - Repita para cada ponto de referência

4. **Insira coordenadas do mundo real:**
   - Para cada ponto na imagem, insira as coordenadas X/Y reais em mm
   - Estas são as coordenadas reais da máquina onde cada ponto está localizado
   - Meça com precisão com uma régua ou use posições conhecidas da máquina

5. **Aplique o alinhamento:**
   - Clique em **Aplicar** para calcular a transformação
   - A sobreposição da câmera estará agora devidamente alinhada

6. **Verifique o alinhamento:**
   - Mova a cabeça do laser para uma posição conhecida
   - Verifique se o ponto do laser se alinha com a posição esperada na visão
     da câmera
   - Ajuste fino realinhando se necessário

### Status de Alinhamento

O painel de propriedades da câmera mostra o status de alinhamento com um
ícone:

- **Marca de verificação** — O alinhamento está atual e válido
- **Aviso** — O alinhamento precisa ser refeito. Isso acontece quando a
  calibração de lente é atualizada, porque a correção de distorção altera a
  imagem da câmera e invalida o alinhamento existente. Seus pontos de
  alinhamento são preservados — basta abrir o diálogo e clicar em
  **Aplicar** novamente.

### Exemplo de fluxo de trabalho

1. Mova o laser para a posição de origem (0, 0) e marque na câmera
2. Mova o laser para (100, 0) e marque na câmera
3. Mova o laser para (100, 100) e marque na câmera
4. Mova o laser para (0, 100) e marque na câmera
5. Insira coordenadas exatas para cada ponto
6. Aplique e verifique

:::tip Melhores Práticas

- Use pontos nos cantos da sua área de trabalho para cobertura máxima
- Evite agrupar pontos em uma área
- Meça as coordenadas do mundo real com cuidado - a precisão aqui
  determina a qualidade geral do
  alinhamento
- Realinhe se você mover a câmera ou alterar a distância do foco
- Realinhe após atualizar a calibração de lente
- Salve seu alinhamento - ele persiste entre sessões
  :::

---

## Usando a Sobreposição de Câmera

Uma vez alinhada, a sobreposição de câmera ajuda a posicionar trabalhos
com precisão. Alterne clicando no ícone da câmera na barra de ferramentas
da janela principal.

---

### Múltiplas Câmeras

O Rayforge suporta múltiplas câmeras para diferentes visões ou máquinas:

- Adicione múltiplas câmeras nas preferências
- Cada câmera pode ter alinhamento independente
- Alterne entre câmeras usando o seletor de câmera
- Casos de uso:
  - Visão superior + visão lateral para objetos 3D
  - Câmeras diferentes para máquinas diferentes
  - Grande angular + câmera de detalhe

---

## Solução de Problemas

### Câmera Não Detectada

**Problema:** A câmera não aparece na lista de dispositivos.

**Soluções:**

**Linux:**
Verifique se a câmera é reconhecida pelo sistema:

```bash
# Liste dispositivos de vídeo
ls -l /dev/video*

# Verifique a câmera com v4l2
v4l2-ctl --list-devices

# Teste com outra aplicação
cheese  # ou VLC, etc.
```

**Para usuários Snap:**

```bash
# Conceda acesso à câmera
sudo snap connect rayforge:camera
```

**Windows:**

- Verifique o Gerenciador de Dispositivos para câmera sob "Cameras" ou
  "Dispositivos de imagem"
- Certifique-se de que nenhuma outra aplicação está usando a câmera
  (feche Zoom, Skype, etc.)
- Tente uma porta USB diferente
- Atualize os drivers da câmera

### Câmera Mostra Tela Preta

**Problema:** Câmera detectada mas não mostra imagem.

**Possíveis causas:**

1. **Câmera em uso por outra aplicação** - Feche outros apps de vídeo
2. **Dispositivo incorreto selecionado** - Tente IDs de dispositivo diferentes
3. **Permissões da câmera** - No Linux Snap, certifique-se de que a
   interface da câmera está conectada
4. **Problema de hardware** - Teste a câmera com outra aplicação

**Soluções:**

```bash
# Linux: Libere dispositivo de câmera
sudo killall cheese  # ou outros apps de câmera

# Verifique qual processo está usando a câmera
sudo lsof /dev/video0
```

### Alinhamento Não Preciso

**Problema:** A sobreposição da câmera não corresponde à posição real do laser.

**Diagnóstico:**

1. **Pontos de alinhamento insuficientes** - Use pelo menos 4 pontos
2. **Erros de medição** - Verifique as coordenadas do mundo real
3. **Câmera moveu** - Realinhe se a posição da câmera mudou
4. **Distorção não linear** - Pode precisar de calibração da lente

**Melhore a precisão:**

- Use mais pontos de alinhamento (6-8 para áreas muito grandes)
- Espalhe pontos por toda a área de trabalho
- Meça as coordenadas do mundo real com muito cuidado
- Use comandos de movimento da máquina para posicionar precisamente o
  laser em coordenadas conhecidas
- Realinhe após qualquer ajuste da câmera

### Qualidade de Imagem Ruim

**Problema:** A imagem da câmera está borrada, escura ou lavada.

**Soluções:**

1. **Ajuste brilho/contraste** nas configurações da câmera
2. **Melhore a iluminação** - Adicione iluminação consistente na área de
   trabalho
3. **Limpe a lente da câmera** - Poeira e detritos reduzem a clareza
4. **Verifique o foco** - Auto-foco pode não funcionar bem; use manual
   se possível
5. **Reduza a transparência** temporariamente para ver a imagem da
   câmera mais claramente
6. **Tente diferentes configurações de balanço de branco**
7. **Ajuste a redução de ruído** se a imagem aparecer granulada

### Atraso ou Travamento da Câmera

**Problema:** O feed ao vivo da câmera está instável ou atrasado.

**Soluções:**

- Reduza a resolução da câmera nas configurações do dispositivo (se acessível)
- Feche outras aplicações que usam CPU/GPU
- Atualize drivers gráficos

---

## Páginas Relacionadas

- [Visualização 3D](../ui/3d-preview) — Visualizar execução com sobreposição
  de câmera
- [Enquadramento de Trabalhos](../features/framing-your-job) — Verificar
  posição do trabalho
- [Configurações Gerais](general) — Configuração da máquina
