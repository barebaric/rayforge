---
description: "Configure sua cortadora ou gravadora a laser pela primeira vez. Use o assistente de configuração para criar sua máquina, depois conecte-se e prepare-se para cortar com o Rayforge."
---

# Configuração Inicial

Após instalar o Rayforge, você precisará configurar sua cortadora ou gravadora a laser. Este guia irá orientá-lo na criação de sua primeira máquina com o assistente de configuração e no estabelecimento de uma conexão.

## Etapa 1: Iniciar o Rayforge

Inicie o Rayforge a partir do menu de aplicativos ou executando `rayforge` em um terminal. Você verá a interface principal com uma tela vazia.

## Etapa 2: Criar uma Máquina com o Assistente

Navegue até **Configurações → Máquinas** ou pressione <kbd>ctrl+comma</kbd> para abrir o diálogo de configurações, depois selecione a página **Máquinas**.

![Configurações da Máquina](/screenshots/app-settings-machines.png)

Clique em **Add Machine** para abrir o seletor de máquinas.

![Adicionar Máquina](/screenshots/app-settings-machines-add.png)

O assistente de configuração abre e adapta quais etapas ele mostra de acordo com suas escolhas:

- Escolher um **perfil integrado** pré-preenche o controlador, a área de
  trabalho e a cabeça — o assistente pula direto para as etapas de rotativo,
  câmera e revisão
- **Importar um perfil** mantém as etapas de hardware e cabeça para que você
  possa corrigir o que a importação errou
- **Device Not Listed** orienta você em todas as etapas, incluindo o
  controlador e a consulta de especificações por IA

### Escolher um Ponto de Partida

Escolha um perfil de dispositivo integrado para pré-preencher o controlador,
a área de trabalho e as configurações de cabeça, ou clique em
**Device Not Listed** para configurar tudo manualmente. Você também pode
**Import from File…** um perfil exportado anteriormente ou um perfil de
dispositivo LightBurn (.lbdev) com calibração de câmera e configurações de
laser.

![Assistente — Escolher um Ponto de Partida](/screenshots/config-wizard-profile.png)

### Escolher um Controlador

Escolha a família de firmware ou protocolo que corresponde à placa
controladora da sua máquina (GRBL, Marlin, Smoothie, Ruida, OctoPrint, …).
Escolha **None — G-code export only** se você quiser apenas exportar G-code
para arquivos e nunca operar uma máquina física. Esta etapa é pulada quando
você começa a partir de um perfil integrado ou de uma importação.

![Assistente — Escolher um Controlador](/screenshots/config-wizard-controller.png)

### Conexão

Insira os parâmetros de conexão que sua máquina exige. Os campos exatos
dependem do controlador que você escolheu:

- **Drivers seriais** — caminho do dispositivo USB (ex.: `/dev/ttyUSB0` no
  Linux, `COM3` no Windows) e taxa de transmissão
- **Drivers de rede** — endereço do host e porta (ex.: `192.168.1.100`)
- **OctoPrint** — URL do servidor e chave de API

![Assistente — Conexão](/screenshots/config-wizard-connect.png)

### Descobrir o Dispositivo

Quando seu controlador suporta, o assistente oferece conectar-se ao
dispositivo e ler sua configuração automaticamente — área de trabalho,
velocidades, aceleração e recursos do firmware. Clique em **Probe Now** para
detectar automaticamente esses valores, ou use **Next** para inseri-los
manualmente nas etapas seguintes.

![Assistente — Descobrir o Dispositivo](/screenshots/config-wizard-probe.png)

### Provedor de IA

Mostrado somente quando nenhum provedor de IA está configurado ainda. Insira
um endpoint compatível com OpenAI (URL base e chave de API) para que a próxima
etapa possa consultar as especificações de máquinas comerciais conhecidas.
Pule esta etapa para inserir os valores manualmente.

![Assistente — Provedor de IA](/screenshots/config-wizard-ai-provider.png)

### Consulta de Especificações por IA

Se sua máquina é um modelo comercial conhecido, a IA pode pré-preencher suas
especificações a partir da documentação do fabricante. Insira o fabricante e
o modelo e clique em **Look Up Specs**. Os valores sugeridos aparecem como
linhas de alternância e começam aceitos — desative qualquer coisa que você
não queira aplicar.

![Assistente — Consulta de Especificações por IA](/screenshots/config-wizard-ai-lookup.png)

### Hardware

Configure a configuração física da máquina:

- **Eixos** — extensões X/Y da área de trabalho e o canto da origem das
  coordenadas (0,0)
- **Direção do eixo** — inverta um eixo se as coordenadas saírem negativas
- **Eixo Z** — se a máquina tem um eixo Z (motor de foco, mesa móvel);
  quando ausente, nenhum movimento Z é gerado e o canvas 3D distribui o
  conteúdo no plano de gravação
- **Orientação do painel** — gira a área de trabalho plana como apresentada
  na tela (Nativo, Girar para a esquerda, Girar para a direita); camadas
  rotativas exigem Nativo
- **Área de Trabalho** — margens ao redor do espaço inutilizável da
  superfície de trabalho
- **Limites de Software** — limites de segurança opcionais para jog
- **Velocidades** — velocidade máxima de deslocamento, velocidade máxima de
  corte e aceleração
- **Comportamento** — origem (home) na inicialização e homing de eixo único

![Assistente — Hardware](/screenshots/config-wizard-hardware.png)

### Cabeça

Declare o que está acoplado ao pórtico — uma cabeça de laser ou de spindle —
e defina seus parâmetros. Para um laser: potência máxima (valor S), tamanho
do ponto, frequência PWM e distância focal. Para um spindle: RPM máximo e
mínimo.

![Assistente — Cabeça](/screenshots/config-wizard-head.png)

### Módulo Rotativo

Opcionalmente, configure um acessório rotativo: tipo (mandril ou rolos),
eixo (A/B/C), modo (4º eixo verdadeiro vs. substituição de eixo), geometria
e sinalizador de direção reversa. Pule esta etapa para adicionar um módulo
rotativo mais tarde nas configurações da máquina.

![Assistente — Módulo Rotativo](/screenshots/config-wizard-rotary.png)

### Câmeras

Opcionalmente, habilite qualquer câmera que você queira usar para
visualização e alinhamento. Quando você habilita uma câmera e continua, o
[Assistente de Câmera](../machine/camera.md#etapa-2-assistente-de-câmera)
abre para orientá-lo nas configurações de imagem, calibração de lente e
alinhamento de imagem. Você pode pular isso e configurar câmeras mais tarde
nas configurações de câmera da máquina.

![Assistente — Câmeras](/screenshots/config-wizard-camera.png)

### Revisar e Nomear

Dê um nome à máquina e revise um resumo de tudo o que você configurou —
driver, conexão, área de trabalho, velocidades, cabeças, módulos rotativos e
câmeras. O assistente também exibe quaisquer avisos, como um driver ausente
ou uma área de trabalho não definida.

![Assistente — Revisar e Nomear](/screenshots/config-wizard-review.png)

Clique em **Create Machine** para finalizar. O diálogo de Configurações da
Máquina abre para sua nova máquina, onde você pode ajustar qualquer uma das
definições pré-preenchidas pelo assistente. Veja as páginas de
[Configuração da Máquina](../machine/general.md) para detalhes.

## Etapa 3: Conexão Automática

O Rayforge conecta-se automaticamente à sua máquina quando o aplicativo
inicia (se a máquina estiver ligada e conectada). Você não precisa clicar
manualmente em um botão de conectar.

O status da conexão é exibido no canto inferior esquerdo da janela principal
com um ícone de status e rótulo mostrando o estado atual (Conectado,
Conectando, Desconectado, Erro, etc.).

:::success Conectado!
Se sua máquina mostrar status "Conectado", você está pronto para começar a usar o Rayforge!
:::

---

## Solução de Problemas de Conexão

### Dispositivo Não Encontrado

- **Linux (Serial)**: Adicione seu usuário ao grupo `dialout`. Isto é
  necessário para **ambas instalações Snap e não-Snap** em distribuições
  baseadas em Debian para evitar mensagens AppArmor DENIED:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  Saia e entre novamente para que as alterações tenham efeito.

- **Pacote Snap**: Além do grupo `dialout` acima, certifique-se de ter
  concedido permissões de porta serial:
  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Verifique o Gerenciador de Dispositivos para confirmar se o
  dispositivo é reconhecido e anote o número da porta COM.

### Conexão Recusada

- Verifique se o endereço IP e número da porta estão corretos
- Certifique-se de que sua máquina está ligada e conectada à rede
- Verifique as configurações de firewall se estiver usando conexão de rede

### Máquina Não Responde

- Tente uma taxa de transmissão diferente (alguns dispositivos usam `9600` ou `57600`)
- Verifique se há cabos soltos ou conexões ruins
- Desligue e ligue novamente sua cortadora a laser e tente novamente

Para mais ajuda, veja [Problemas de Conexão](../troubleshooting/connection.md).

---

**Próximo:** [Guia de Início Rápido →](quick-start)
