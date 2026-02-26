# Janela Principal

A janela principal do Rayforge é sua área de trabalho principal para criar e gerenciar trabalhos de laser.

## Layout da Janela

![Janela Principal](/screenshots/main-standard.png)

### 1. Barra de Menu

Acesse todas as funções do Rayforge através de menus organizados:

- **Arquivo**: Abrir, salvar, importar, exportar e arquivos recentes
- **Editar**: Desfazer, refazer, copiar, colar, preferências
- **Visualizar**: Zoom, grade, réguas, painéis e modos de visualização
- **Operações**: Adicionar, editar e gerenciar operações
- **Máquina**: Conectar, jog, origem, iniciar/parar trabalhos
- **Ajuda**: Documentação, sobre e suporte

### 2. Barra de Ferramentas

Acesso rápido a ferramentas frequentemente usadas:

- **Ferramenta de seleção**: Selecionar e mover objetos
- **Ferramenta de pan**: Navegar pela tela
- **Ferramenta de zoom**: Zoom para dentro/fora em áreas específicas
- **Ferramenta de medição**: Medir distâncias e ângulos
- **Ferramentas de alinhamento**: Alinhar e distribuir objetos
- **Menu suspenso WCS**: Selecionar o Sistema de Coordenadas de Trabalho ativo (G53-G59)

O menu suspenso WCS permite alternar rapidamente entre sistemas de coordenadas.
Veja [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems) para
mais informação.

### 3. Tela

A área de trabalho principal onde você:

- Importa e organiza designs
- Pré-visualiza caminhos de ferramenta
- Posiciona objetos relativos à origem da máquina
- Testa limites de enquadramento

**Controles da Tela:**

- **Pan**: Arrastar clique do meio ou <kbd>espaço</kbd> + arrastar
- **Zoom**: Roda do mouse ou <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Resetar Visualização**: <kbd>ctrl+0</kbd> ou Visualizar → Resetar Zoom

### 4. Painel de Camadas

Gerencia operações e atribuições de camadas:

- Ver todas as operações no seu projeto
- Atribuir operações a elementos de design
- Reordenar execução de operações
- Habilitar/desabilitar operações individuais
- Configurar parâmetros de operação

### 5. Painel de Propriedades

Configura definições para objetos selecionados ou operações:

- Tipo de operação (Contorno, Raster, etc.)
- Configurações de potência e velocidade
- Número de passagens
- Opções avançadas (overscan, kerf, abas)

### 6. Painel de Controle

O Painel de Controle na parte inferior da janela fornece:

- **Controles de Jog**: Movimento e posicionamento manual da máquina
- **Status da Máquina**: Posição em tempo real e estado de conexão
- **Visão de Log**: Comunicação G-code e histórico de operações
- **Gerenciamento WCS**: Seleção e zeramento do sistema de coordenadas de trabalho

Veja [Painel de Controle](control-panel) para informação detalhada.

## Gerenciamento de Janela

### Painéis

Mostrar/ocultar painéis conforme necessário:

- **Painel de Camadas**: Visualizar → Painel de Camadas (<kbd>ctrl+l</kbd>)
- **Painel de Propriedades**: Visualizar → Painel de Propriedades (<kbd>ctrl+i</kbd>)

### Modo Tela Cheia

Foque no seu trabalho com tela cheia:

- Entrar: <kbd>f11</kbd> ou Visualizar → Tela Cheia
- Sair: <kbd>f11</kbd> ou <kbd>esc</kbd>

## Personalização

Personalize a interface em **Editar → Preferências**:

- **Tema**: Claro, escuro ou sistema
- **Unidades**: Milímetros ou polegadas
- **Grade**: Mostrar/ocultar e configurar espaçamento da grade
- **Réguas**: Mostrar/ocultar réguas na tela
- **Barra de ferramentas**: Personalizar botões visíveis

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems) - WCS
- [Ferramentas da Tela](canvas-tools) - Ferramentas para manipular designs
- [Painel de Controle](control-panel) - Controle manual da máquina, status e logs
- [Visualização 3D](3d-preview) - Visualizar caminhos de ferramenta em 3D
