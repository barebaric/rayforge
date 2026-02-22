# Configuração Inicial

Após instalar o Rayforge, você precisará configurar sua cortadora ou gravadora a laser. Este guia irá orientá-lo na criação de sua primeira máquina e no estabelecimento de uma conexão.

## Passo 1: Iniciar o Rayforge

Inicie o Rayforge a partir do menu de aplicativos ou executando `rayforge` em um terminal. Você verá a interface principal com uma tela vazia.

## Passo 2: Criar uma Máquina

Navegue até **Configurações → Máquinas** ou pressione <kbd>ctrl+comma</kbd> para abrir o diálogo de configurações, depois selecione a página **Máquinas**.

Clique em **Adicionar Máquina** para criar uma nova máquina. Você pode:

1. **Escolher um perfil integrado** - Selecione entre modelos de máquina predefinidos
2. **Selecionar "Personalizado"** - Comece com uma configuração em branco

Após selecionar, o diálogo de Configurações da Máquina abre para sua nova máquina.

![Configurações da Máquina](/screenshots/application-machines.png)

## Passo 3: Configurar Definições Gerais

A página **Geral** contém informações básicas da máquina, seleção de driver e configurações de conexão.

![Configurações Gerais](/screenshots/machine-general.png)

### Informações da Máquina

1. **Nome da Máquina**: Dê à sua máquina um nome descritivo (ex: "K40 Laser", "Ortur LM2")

### Seleção de Driver

Selecione o driver apropriado para seu dispositivo no menu suspenso:

- **GRBL Serial** - Para dispositivos GRBL conectados via porta USB/serial
- **GRBL Network** - Para dispositivos GRBL com conectividade WiFi/Ethernet
- **Smoothie** - Para dispositivos baseados em Smoothieware

### Configurações do Driver

Dependendo do driver selecionado, configure os parâmetros de conexão:

#### GRBL Serial (USB)

1. **Porta**: Escolha seu dispositivo no menu suspenso (ex: `/dev/ttyUSB0` no Linux, `COM3` no Windows)
2. **Taxa de Transmissão**: Selecione `115200` (padrão para a maioria dos dispositivos GRBL)

:::info
Se seu dispositivo não aparecer na lista, verifique se está conectado e se você tem as permissões necessárias. No Linux, você pode precisar adicionar seu usuário ao grupo `dialout`.
:::

#### GRBL Network / Smoothie (WiFi/Ethernet)

1. **Host**: Digite o endereço IP do seu dispositivo (ex: `192.168.1.100`)
2. **Porta**: Digite o número da porta (tipicamente `23` ou `8080`)

### Velocidades e Aceleração

Estas configurações são usadas para estimativa de tempo de trabalho e otimização de caminho:

- **Velocidade Máxima de Deslocamento**: Velocidade máxima de movimento rápido
- **Velocidade Máxima de Corte**: Velocidade máxima de corte
- **Aceleração**: Usada para estimativas de tempo e cálculos de overscan

## Passo 4: Configurar Definições de Hardware

Mude para a aba **Hardware** para configurar as dimensões físicas da sua máquina.

![Configurações de Hardware](/screenshots/machine-hardware.png)

### Dimensões

- **Largura**: Digite a largura máxima da sua área de trabalho em milímetros
- **Altura**: Digite a altura máxima da sua área de trabalho em milímetros

### Eixos

- **Origem das Coordenadas (0,0)**: Selecione onde a origem da sua máquina está localizada:
  - Inferior Esquerdo (mais comum para GRBL)
  - Superior Esquerdo
  - Superior Direito
  - Inferior Direito

### Deslocamentos dos Eixos (Opcional)

Configure deslocamentos X e Y se sua máquina precisar deles para posicionamento preciso.

## Passo 5: Conexão Automática

O Rayforge conecta-se automaticamente à sua máquina quando o aplicativo inicia (se a máquina estiver ligada e conectada). Você não precisa clicar manualmente em um botão de conectar.

O status da conexão é exibido no canto inferior esquerdo da janela principal com um ícone de status e rótulo mostrando o estado atual (Conectado, Conectando, Desconectado, Erro, etc.).

:::success Conectado!
Se sua máquina mostrar status "Conectado", você está pronto para começar a usar o Rayforge!
:::

## Opcional: Configuração Avançada

### Múltiplos Lasers

Se sua máquina tem múltiplos módulos de laser (ex: diodo e CO2), você pode configurá-los na página **Laser**.

![Configurações do Laser](/screenshots/machine-laser.png)

Veja [Configuração do Laser](../machine/laser) para detalhes.

### Configuração de Câmera

Se você tem uma câmera USB para alinhamento e posicionamento, configure-a na página **Câmera**.

![Configurações da Câmera](/screenshots/machine-camera.png)

Veja [Integração com Câmera](../machine/camera) para detalhes.

### Configurações do Dispositivo

A página **Dispositivo** permite ler e modificar configurações de firmware diretamente no seu dispositivo conectado (como parâmetros GRBL). Este é um recurso avançado e deve ser usado com cautela.

:::warning
Editar configurações do dispositivo pode ser perigoso e pode tornar sua máquina inoperante se valores incorretos forem aplicados!
:::

---

## Solução de Problemas de Conexão

### Dispositivo Não Encontrado

- **Linux (Serial)**: Adicione seu usuário ao grupo `dialout`:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  Saia e entre novamente para que as alterações tenham efeito.

- **Pacote Snap**: Certifique-se de ter concedido permissões de porta serial:
  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Verifique o Gerenciador de Dispositivos para confirmar se o dispositivo é reconhecido e anote o número da porta COM.

### Conexão Recusada

- Verifique se o endereço IP e número da porta estão corretos
- Certifique-se de que sua máquina está ligada e conectada à rede
- Verifique as configurações de firewall se estiver usando conexão de rede

### Máquina Não Responde

- Tente uma taxa de transmissão diferente (alguns dispositivos usam `9600` ou `57600`)
- Verifique se há cabos soltos ou conexões ruins
- Desligue e ligue novamente sua cortadora a laser e tente novamente

Para mais ajuda, veja [Problemas de Conexão](../troubleshooting/connection).

---

**Próximo:** [Guia de Início Rápido →](quick-start)
