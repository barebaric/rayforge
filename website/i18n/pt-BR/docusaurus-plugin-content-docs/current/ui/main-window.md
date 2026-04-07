# Janela Principal

A janela principal do Rayforge é sua área de trabalho principal para criar e
gerenciar trabalhos de laser.

## Layout da Janela

![Janela Principal](/screenshots/main-standard.png)

### 1. Barra de Menu

Acesse todas as funções do Rayforge através de menus organizados:

- **Arquivo**: Abrir, salvar, importar, exportar e arquivos recentes
- **Editar**: Desfazer, refazer, copiar, colar, preferências
- **Visualizar**: Zoom, grade, réguas, painéis e modos de visualização
- **Objeto**: Adicionar, editar e gerenciar operações
- **Máquina**: Conectar, jog, origem, iniciar/parar trabalhos
- **Ajuda**: Sobre, Doar, Salvar Log de Debug

### 2. Barra de Ferramentas

Acesso rápido a controles frequentemente usados:

- **Menu suspenso de máquina**: Selecione sua máquina, veja o status de conexão e
  veja o tempo estimado durante os trabalhos
- **Menu suspenso WCS**: Selecione o Sistema de Coordenadas de Trabalho ativo
  (G53-G59)
- **Alternar simulação**: Habilitar/desabilitar modo de simulação de trabalho
- **Focar laser**: Alternar modo de focagem do laser
- **Controles de trabalho**: Botões de Origem, Enquadrar, Enviar, Pausar e
  Cancelar

O menu suspenso de máquina mostra o status de conexão e o estado atual da sua
máquina (ex: Ocioso, Executando) diretamente na barra de ferramentas. Durante a
execução do trabalho, também exibe o tempo restante estimado.

O menu suspenso WCS permite alternar rapidamente entre sistemas de coordenadas.
Veja [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems)
para mais informação.

Botões de alternância de visibilidade para peças de trabalho, abas, feed de
câmera, movimentos de deslocamento e outros elementos foram movidos para botões
de sobreposição na própria tela, então estão sempre à mão enquanto você
trabalha.

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

### 6. Painel Inferior

O Painel Inferior fornece abas para o Console, Visualizador G-code e seus
ativos do documento (estoque e esboços). Os controles de jog e gerenciamento
WCS estão sempre visíveis no lado direito. O tempo estimado do trabalho é
mostrado no cabeçalho da lista de camadas acima do painel de camadas.

Veja [Painel Inferior](bottom-panel) para informação detalhada.

## Gerenciamento de Janela

### Painéis

Mostrar/ocultar painéis conforme necessário:

- **Painel Inferior**: Visualizar → Painel Inferior (<kbd>ctrl+l</kbd>)

### Modo Tela Cheia

Foque no seu trabalho com tela cheia:

- Entrar: <kbd>f11</kbd> ou Visualizar → Tela Cheia
- Sair: <kbd>f11</kbd> ou <kbd>esc</kbd>

## Personalização

Personalize a interface em **Editar → Configurações**:

- **Tema**: Claro, escuro ou sistema
- **Unidades**: Milímetros ou polegadas
- **Grade**: Mostrar/ocultar e configurar espaçamento da grade
- **Réguas**: Mostrar/ocultar réguas na tela

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabalho](../general-info/coordinate-systems) - WCS
- [Ferramentas da Tela](canvas-tools) - Ferramentas para manipular designs
- [Painel Inferior](bottom-panel) - Controle manual da máquina, status e logs
- [Visualização 3D](3d-preview) - Visualizar caminhos de ferramenta em 3D
