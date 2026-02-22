# Integração com Câmera

O Rayforge suporta integração com câmera USB para alinhamento e posicionamento preciso de materiais. O recurso de sobreposição de câmera permite ver exatamente onde seu laser vai cortar ou gravar no material, eliminando suposições e reduzindo o desperdício de material.

![Configurações da Câmera](/screenshots/machine-camera.png)

## Visão Geral

A integração com câmera fornece:

- **Sobreposição de vídeo ao vivo** na tela mostrando seu material em tempo real
- **Alinhamento de imagem** para calibrar a posição da câmera relativa ao laser
- **Posicionamento visual** para colocar trabalhos com precisão em materiais irregulares ou pré-marcados
- **Pré-visualização do material** antes de executar trabalhos
- **Suporte a múltiplas câmeras** para diferentes configurações de máquina

:::tip Casos de Uso

- Alinhar cortes em materiais pré-impressos
- Trabalhar com materiais de formas irregulares
- Posicionamento preciso de gravações em objetos existentes
- Reduzir cortes de teste e desperdício de material
  :::

---

## Configuração da Câmera

### Requisitos de Hardware

**Câmeras compatíveis:**

- Webcams USB (mais comum)
- Câmeras integradas de laptop (se executar Rayforge em laptop perto da máquina)
- Qualquer câmera suportada por Video4Linux2 (V4L2) no Linux ou DirectShow no Windows

**Configuração recomendada:**

- Câmera montada acima da área de trabalho com visão clara do material
- Condições de iluminação consistentes
- Câmera posicionada para capturar a área de trabalho do laser
- Montagem segura para prevenir movimento da câmera

### Adicionando uma Câmera

1. **Conecte sua câmera** ao computador via USB

2. **Abra Configurações da Câmera:**
   - Navegue até **Configurações Preferências Câmera**
   - Ou use o botão da câmera na barra de ferramentas

3. **Adicione uma nova câmera:**
   - Clique no botão "+" para adicionar uma câmera
   - Digite um nome descritivo (ex: "Câmera Superior", "Câmera da Área de Trabalho")
   - Selecione o dispositivo no menu suspenso
     - No Linux: `/dev/video0`, `/dev/video1`, etc.
     - No Windows: Camera 0, Camera 1, etc.

4. **Habilite a câmera:**
   - Ative o interruptor de habilitação da câmera
   - O feed ao vivo deve aparecer na sua tela

5. **Ajuste as configurações da câmera:**
   - **Brilho:** Ajuste se o material estiver muito escuro/claro
   - **Contraste:** Melhore a visibilidade das bordas
   - **Transparência:** Controle a opacidade da sobreposição (20-50% recomendado)
   - **Balanço de Branco:** Auto ou temperatura Kelvin manual

---

## Alinhamento da Câmera

O alinhamento da câmera calibra a relação entre pixels da câmera e coordenadas do mundo real, permitindo posicionamento preciso.

### Por Que o Alinhamento é Necessário

A câmera vê a área de trabalho de cima, mas a imagem pode estar:

- Rotacionada em relação aos eixos da máquina
- Escalada de forma diferente nas direções X e Y
- Distorcida pela perspectiva da lente

O alinhamento cria uma matriz de transformação que mapeia pixels da câmera para coordenadas da máquina.

### Procedimento de Alinhamento

1. **Abra o Diálogo de Alinhamento:**
   - Clique no botão de alinhamento de câmera na barra de ferramentas
   - Ou vá para **Câmera Alinhar Câmera**

2. **Coloque marcadores de alinhamento:**
   - Você precisa de pelo menos 3 pontos de referência (4 recomendados para melhor precisão)
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
   - Clique em "Aplicar" para calcular a transformação
   - A sobreposição da câmera estará agora devidamente alinhada

6. **Verifique o alinhamento:**
   - Mova a cabeça do laser para uma posição conhecida
   - Verifique se o ponto do laser se alinha com a posição esperada na visão da câmera
   - Ajuste fino realinhando se necessário

### Dicas de Alinhamento

:::tip Melhores Práticas
- Use pontos nos cantos da sua área de trabalho para cobertura máxima
- Evite agrupar pontos em uma área
- Meça as coordenadas do mundo real com cuidado - a precisão aqui determina a qualidade geral do alinhamento
- Realinhe se você mover a câmera ou alterar a distância do foco
- Salve seu alinhamento - ele persiste entre sessões
  :::

**Fluxo de trabalho de alinhamento de exemplo:**

1. Mova o laser para a posição de origem (0, 0) e marque na câmera
2. Mova o laser para (100, 0) e marque na câmera
3. Mova o laser para (100, 100) e marque na câmera
4. Mova o laser para (0, 100) e marque na câmera
5. Insira coordenadas exatas para cada ponto
6. Aplique e verifique

---

## Usando a Sobreposição de Câmera

Uma vez alinhada, a sobreposição de câmera ajuda a posicionar trabalhos com precisão.

### Habilitando/Desabilitando a Sobreposição

- **Alternar câmera:** Clique no ícone da câmera na barra de ferramentas
- **Ajustar transparência:** Use o controle deslizante nas configurações da câmera (20-50% funciona bem)
- **Atualizar imagem:** A câmera atualiza continuamente enquanto habilitada

### Posicionando Trabalhos com a Câmera

**Fluxo de trabalho para posicionamento preciso:**

1. **Habilite a sobreposição da câmera** para ver seu material

2. **Importe seu design** (SVG, DXF, etc.)

3. **Posicione o design** na tela:
   - Arraste o design para alinhar com recursos visíveis na câmera
   - Use zoom para ver detalhes finos
   - Rotacione/escalone conforme necessário

4. **Pré-visualize o alinhamento:**
   - Use o [Modo Simulação](../features/simulation-mode) para visualizar
   - Verifique se cortes/gravações estarão onde você espera

5. **Enquadre o trabalho** para verificar o posicionamento antes de executar

6. **Execute o trabalho** com confiança

### Exemplo: Gravando em um Cartão Pré-Impresso

1. Coloque o cartão impresso na mesa do laser
2. Habilite a sobreposição da câmera
3. Importe seu design de gravação
4. Arraste e posicione o design para alinhar com recursos impressos
5. Ajuste fino da posição usando as teclas de seta
6. Enquadre para verificar
7. Execute o trabalho

---

## Referência de Configurações da Câmera

### Configurações do Dispositivo

| Configuração       | Descrição                     | Valores                               |
| ------------- | ------------------------------- | ------------------------------------ |
| **Nome**      | Nome descritivo para a câmera | Qualquer texto                             |
| **ID do Dispositivo** | Identificador do dispositivo no sistema        | `/dev/video0` (Linux), `0` (Windows) |
| **Habilitado**   | Estado ativo da câmera             | Ligado/Desligado                               |

### Ajuste de Imagem

| Configuração           | Descrição                  | Faixa                             |
| ----------------- | ---------------------------- | --------------------------------- |
| **Brilho**    | Brilho geral da imagem     | -100 a +100                      |
| **Contraste**      | Definição de bordas e contraste | 0 a 100                          |
| **Transparência**  | Opacidade da sobreposição na tela    | 0% (opaco) a 100% (transparente) |
| **Balanço de Branco** | Correção de temperatura de cor | Auto ou 2000-10000K               |

### Dados de Alinhamento

| Propriedade                  | Descrição                         |
| ------------------------- | ----------------------------------- |
| **Pontos na Imagem**          | Coordenadas de pixel na imagem da câmera   |
| **Pontos do Mundo Real**          | Coordenadas da máquina no mundo real (mm) |
| **Matriz de Transformação** | Mapeamento calculado (interno)       |

---

## Recursos Avançados

### Calibração da Câmera (Correção de Distorção de Lente)

Para trabalho preciso, você pode calibrar a câmera para corrigir distorção de barril/almofada:

1. **Imprima um padrão xadrez** (ex: grade 8x6 com quadrados de 25mm)
2. **Capture 10+ imagens** do padrão de diferentes ângulos/posições
3. **Use ferramentas de calibração OpenCV** para calcular matriz da câmera e coeficientes de distorção
4. **Aplique a calibração** no Rayforge (configurações avançadas)

:::note Quando Calibrar
A correção de distorção de lente só é necessária para:

- Lentes grande-angular com distorção de barril perceptível
- Trabalho de precisão exigindo &lt;1mm de precisão
- Grandes áreas de trabalho onde a distorção se acumula

A maioria das webcams padrão funciona bem sem calibração para trabalho típico de laser.
:::

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

- Verifique o Gerenciador de Dispositivos para câmera sob "Cameras" ou "Dispositivos de imagem"
- Certifique-se de que nenhuma outra aplicação está usando a câmera (feche Zoom, Skype, etc.)
- Tente uma porta USB diferente
- Atualize os drivers da câmera

### Câmera Mostra Tela Preta

**Problema:** Câmera detectada mas não mostra imagem.

**Possíveis causas:**

1. **Câmera em uso por outra aplicação** - Feche outros apps de vídeo
2. **Dispositivo incorreto selecionado** - Tente IDs de dispositivo diferentes
3. **Permissões da câmera** - No Linux Snap, certifique-se de que a interface da câmera está conectada
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
- Use comandos de movimento da máquina para posicionar precisamente o laser em coordenadas conhecidas
- Realinhe após qualquer ajuste da câmera

### Qualidade de Imagem Ruim

**Problema:** A imagem da câmera está borrada, escura ou lavada.

**Soluções:**

1. **Ajuste brilho/contraste** nas configurações da câmera
2. **Melhore a iluminação** - Adicione iluminação consistente na área de trabalho
3. **Limpe a lente da câmera** - Poeira e detritos reduzem a clareza
4. **Verifique o foco** - Auto-foco pode não funcionar bem; use manual se possível
5. **Reduza a transparência** temporariamente para ver a imagem da câmera mais claramente
6. **Tente diferentes configurações de balanço de branco**

### Atraso ou Travamento da Câmera

**Problema:** O feed ao vivo da câmera está instável ou atrasado.

**Soluções:**

- Reduza a resolução da câmera nas configurações do dispositivo (se acessível)
- Feche outras aplicações usando CPU/GPU
- Atualize drivers gráficos
- No Linux, certifique-se de usar o backend V4L2 (automático no Rayforge)
- Desabilite a câmera quando não precisar para economizar recursos

---

## Páginas Relacionadas

- [Modo Simulação](../features/simulation-mode) - Pré-visualizar execução com sobreposição de câmera
- [Visualização 3D](../ui/3d-preview) - Visualizar trabalhos em 3D
- [Enquadramento de Trabalhos](../features/framing-your-job) - Verificar posição do trabalho
- [Configurações Gerais](general) - Configuração da máquina
