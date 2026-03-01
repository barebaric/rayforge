# Guia do Desenvolvedor de Addons Rayforge

O Rayforge usa um sistema de addons baseado em [pluggy](https://pluggy.readthedocs.io/)
para permitir que desenvolvedores estendam funcionalidade, adicionem novos drivers de máquina, ou
integrem lógica personalizada sem modificar o código base principal.

## 1. Início Rápido

A forma mais rápida de começar é usando o template oficial.

1. **Faça Fork ou Clone** do
   [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Renomeie** o diretório e atualize os metadados.

## 2. Estrutura do Addon

O `AddonManager` escaneia o diretório `addons`. Um addon válido deve ser um
diretório contendo um arquivo de manifesto:

**Layout do Diretório:**

```text
meu-addon-rayforge/
├── rayforge-addon.yaml  <-- Manifesto Obrigatório
├── meu_addon/           <-- Pacote Python
│   ├── __init__.py
│   ├── backend.py       <-- Ponto de entrada backend
│   └── frontend.py      <-- Ponto de entrada frontend (opcional)
├── assets/              <-- Recursos opcionais
├── locales/             <-- Traduções opcionais (arquivos .po)
└── README.md
```

## 3. O Manifesto (`rayforge-addon.yaml`)

Este arquivo diz ao Rayforge como carregar seu addon.

```yaml
# rayforge-addon.yaml

# Identificador único para seu addon (nome do diretório)
name: meu_addon_personalizado

# Nome de exibição legível para humanos
display_name: "Meu Addon Personalizado"

# Descrição exibida na UI
description: "Adiciona suporte para o cortador a laser XYZ."

# Versão da API (deve corresponder ao PLUGIN_API_VERSION do Rayforge)
api_version: 1

# Dependências de versão do Rayforge
depends:
  - rayforge>=0.27.0,<2.0.0

# Opcional: Dependências de outros addons
requires:
  - algum-outro-addon>=1.0.0

# O que o addon fornece
provides:
  # Módulo backend (carregado em processos principal e worker)
  backend: meu_addon.backend
  # Módulo frontend (carregado apenas no processo principal, para UI)
  frontend: meu_addon.frontend
  # Arquivos de asset opcionais
  assets:
    - path: assets/profiles.json
      type: profiles

# Metadados do autor
author:
  name: Maria Silva
  email: maria@example.com

url: https://github.com/usuario/meu-addon-personalizado
```

### Campos Obrigatórios

- `name`: Identificador único (deve corresponder ao nome do diretório)
- `display_name`: Nome legível exibido na UI
- `description`: Descrição breve da funcionalidade do addon
- `api_version`: Deve ser `1` (corresponde ao `PLUGIN_API_VERSION` do Rayforge)
- `depends`: Lista de restrições de versão para o Rayforge
- `author`: Objeto com `name` (obrigatório) e `email` (opcional)

### Campos Opcionais

- `requires`: Lista de dependências de outros addons
- `provides`: Pontos de entrada e assets
- `url`: Página do projeto ou repositório

## 4. Pontos de Entrada

Addons podem fornecer dois tipos de pontos de entrada:

### Backend (`provides.backend`)

Carregado tanto no processo principal quanto nos processos worker. Use para:
- Drivers de máquina
- Tipos de passos
- Produtores de ops
- Funcionalidade principal sem dependências de UI

### Frontend (`provides.frontend`)

Carregado apenas no processo principal. Use para:
- Componentes UI
- Widgets GTK
- Itens de menu
- Ações que requerem a janela principal

Os pontos de entrada são especificados como caminhos de módulo com pontos (ex., `meu_addon.backend`).

## 5. Escrevendo o Código do Addon

O Rayforge usa hooks `pluggy`. Para conectar ao Rayforge, defina funções decoradas
com `@pluggy.HookimplMarker("rayforge")`.

### Boilerplate Básico (`backend.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Chamado quando o Rayforge está totalmente inicializado.
    Este é seu ponto de entrada principal para acessar gerenciadores.
    """
    logger.info("Meu Addon Personalizado foi iniciado!")

    machine = context.machine
    if machine:
        logger.info(f"Addon rodando na máquina: {machine.id}")

@hookimpl
def on_unload():
    """
    Chamado quando o addon está sendo desabilitado ou descarregado.
    Limpar recursos, fechar conexões, desregistrar handlers.
    """
    logger.info("Meu Addon Personalizado está encerrando")

@hookimpl
def register_machines(machine_manager):
    """
    Chamado durante a inicialização para registrar novos drivers de máquina.
    """
    from .meu_driver import MinhaNovaMaquina
    machine_manager.register("minha_nova_maquina", MinhaNovaMaquina)

@hookimpl
def register_steps(step_registry):
    """
    Chamado para registrar tipos de passos personalizados.
    """
    from .meu_passo import MeuPassoPersonalizado
    step_registry.register("meu_passo_personalizado", MeuPassoPersonalizado)

@hookimpl
def register_producers(producer_registry):
    """
    Chamado para registrar produtores de ops personalizados.
    """
    from .meu_produtor import MeuProdutor
    producer_registry.register("meu_produtor", MeuProdutor)

@hookimpl
def register_step_widgets(widget_registry):
    """
    Chamado para registrar widgets de configuração de passos personalizados.
    """
    from .meu_widget import MeuWidgetPasso
    widget_registry.register("meu_passo_personalizado", MeuWidgetPasso)

@hookimpl
def register_menu_items(menu_registry):
    """
    Chamado para registrar itens de menu.
    """
    from .menu_items import register_menus
    register_menus(menu_registry)

@hookimpl
def register_commands(command_registry):
    """
    Chamado para registrar comandos do editor.
    """
    from .commands import register_commands
    register_commands(command_registry)

@hookimpl
def register_actions(window):
    """
    Chamado para registrar ações da janela.
    """
    from .actions import setup_actions
    setup_actions(window)
```

### Hooks Disponíveis

Definidos em `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Ponto de Entrada Principal.** Chamado após configuração, câmera e hardware serem carregados.
  Use isso para lógica, injeções de UI ou listeners.

**`on_unload`** ()
: Chamado quando um addon está sendo desabilitado ou descarregado. Use para limpar
  recursos, fechar conexões, desregistrar handlers, etc.

**`register_machines`** (`machine_manager`)
: Chamado durante a inicialização para registrar novos drivers de máquina.

**`register_steps`** (`step_registry`)
: Chamado para permitir que plugins registrem tipos de passos personalizados.

**`register_producers`** (`producer_registry`)
: Chamado para permitir que plugins registrem produtores de ops personalizados.

**`register_step_widgets`** (`widget_registry`)
: Chamado para permitir que plugins registrem widgets de configuração de passos personalizados.

**`register_menu_items`** (`menu_registry`)
: Chamado para permitir que plugins registrem itens de menu.

**`register_commands`** (`command_registry`)
: Chamado para permitir que plugins registrem comandos do editor.

**`register_actions`** (`window`)
: Chamado para permitir que plugins registrem ações da janela.

## 6. Acessando Dados do Rayforge

O hook `rayforge_init` fornece o **`RayforgeContext`**. Através deste objeto,
você pode acessar:

- **`context.machine`**: A instância da máquina atualmente ativa.
- **`context.config`**: Configurações globais.
- **`context.config_mgr`**: Gerenciador de configuração.
- **`context.machine_mgr`**: Gerenciador de máquinas (todas as máquinas).
- **`context.camera_mgr`**: Acesse feeds de câmera e ferramentas de visão computacional.
- **`context.material_mgr`**: Acesse a biblioteca de materiais.
- **`context.recipe_mgr`**: Acesse receitas de processamento.
- **`context.dialect_mgr`**: Gerenciador de dialetos G-code.
- **`context.language`**: Código de idioma atual para conteúdo localizado.
- **`context.addon_mgr`**: Instância do gerenciador de addons.
- **`context.plugin_mgr`**: Instância do gerenciador de plugins.
- **`context.debug_dump_manager`**: Gerenciador de dumps de debug.
- **`context.artifact_store`**: Armazenamento de artefatos do pipeline.

## 7. Localização

Addons podem fornecer traduções usando arquivos `.po`:

```text
meu-addon-rayforge/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── meu_addon.po
│   └── pt_BR/
│       └── LC_MESSAGES/
│           └── meu_addon.po
```

Os arquivos `.po` são automaticamente compilados para arquivos `.mo` quando o addon
é instalado ou carregado.

## 8. Desenvolvimento e Testes

Para testar seu addon localmente sem publicá-lo:

1.  **Localize seu Diretório de Configuração:**
    O Rayforge usa `platformdirs`.

    - **Windows:** `C:\Users\<Usuario>\AppData\Local\rayforge\rayforge\addons`
    - **macOS:** `~/Library/Application Support/rayforge/addons`
    - **Linux:** `~/.config/rayforge/addons`
      _(Verifique os logs na inicialização por `Config dir is ...`)_

2.  **Crie um link simbólico do seu addon:**
    Em vez de copiar arquivos repetidamente, crie um link simbólico da sua pasta
    de desenvolvimento para a pasta de addons do Rayforge.

    _Linux/macOS:_

    ```bash
    ln -s /caminho/para/meu-addon-rayforge ~/.config/rayforge/addons/meu-addon-rayforge
    ```

3.  **Reinicie o Rayforge:**
    A aplicação escaneia o diretório na inicialização. Verifique os logs do console para:
    > `Loaded addon: meu_addon_personalizado`

## 9. Publicação

Para compartilhar seu addon com a comunidade:

1.  **Hospede no Git:** Envie seu código para um repositório Git público (GitHub, GitLab,
    etc.).
2.  **Envie ao Registro:**
    - Vá para [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Faça fork do repositório.
    - Adicione a URL Git e metadados do seu addon à lista do registro.
    - Envie um Pull Request.

Uma vez aceito, usuários podem instalar seu addon diretamente via UI do Rayforge ou
usando a URL Git.
