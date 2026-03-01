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
diretório contendo pelo menos dois arquivos:

1. `rayforge_addon.yaml` (Metadados)
2. Um ponto de entrada Python (ex.: `addon.py`)

**Layout do Diretório:**

```text
meu-addon-rayforge/
├── rayforge_addon.yaml  &lt;-- Manifesto Obrigatório
├── addon.py             &lt;-- Ponto de entrada (lógica)
├── assets/              &lt;-- Recursos opcionais
└── README.md
```

## 3. O Manifesto (`rayforge_addon.yaml`)

Este arquivo diz ao Rayforge como carregar seu addon.

```yaml
# rayforge_addon.yaml

# Identificador único para seu addon
name: meu_addon_personalizado

# Nome de exibição legível para humanos
display_name: "Meu Addon Personalizado"

# Descrição exibida na UI
description: "Adiciona suporte para o cortador a laser XYZ."

# Dependências (addon e restrições de versão)
depends:
  - rayforge&gt;=0.27.0,~0.27

# O arquivo python a carregar (relativo à pasta do addon)
entry_point: addon.py

# Metadados do autor
author: Maria Silva
url: https://github.com/usuario/meu-addon-personalizado
```

## 4. Escrevendo o Código do Addon

O Rayforge usa hooks `pluggy`. Para conectar ao Rayforge, defina funções decoradas
com `@pluggy.HookimplMarker("rayforge")`.

### Boilerplate Básico (`addon.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

# Define o marcador de implementação de hook
hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Chamado quando o Rayforge está totalmente inicializado.
    Este é seu ponto de entrada principal para acessar gerenciadores.
    """
    logger.info("Meu Addon Personalizado foi iniciado!")

    # Acesse sistemas principais via contexto
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Addon rodando na máquina: {machine.id}")

@hookimpl
def register_machines(machine_manager):
    """
    Chamado durante a inicialização para registrar novos drivers de máquina.
    """
    # from .meu_driver import MinhaNovaMaquina
    # machine_manager.register("minha_nova_maquina", MinhaNovaMaquina)
    pass
```

### Hooks Disponíveis

Definidos em `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Ponto de Entrada Principal.** Chamado após configuração, câmera e hardware serem carregados.
  Use isso para lógica, injeções de UI ou listeners.

**`register_machines`** (`machine_manager`)
: Chamado cedo no processo de inicialização. Use isso para registrar novas classes/drivers
  de hardware.

## 5. Acessando Dados do Rayforge

O hook `rayforge_init` fornece o **`RayforgeContext`**. Através deste objeto,
você pode acessar:

- **`context.machine`**: A instância da máquina atualmente ativa.
- **`context.config`**: Configurações globais.
- **`context.camera_mgr`**: Acesse feeds de câmera e ferramentas de visão computacional.
- **`context.material_mgr`**: Acesse a biblioteca de materiais.
- **`context.recipe_mgr`**: Acesse receitas de processamento.

## 6. Desenvolvimento e Testes

Para testar seu addon localmente sem publicá-lo:

1.  **Localize seu Diretório de Configuração:**
    O Rayforge usa `platformdirs`.

    - **Windows:** `C:\Users\&lt;Usuario&gt;\AppData\Local\rayforge\rayforge\addons`
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

## 7. Publicação

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
