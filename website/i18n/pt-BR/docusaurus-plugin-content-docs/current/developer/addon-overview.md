# Visão Geral do Desenvolvimento de Addons

O Rayforge utiliza um sistema de addons baseado no [pluggy](https://pluggy.readthedocs.io/) que permite estender a funcionalidade, adicionar novos drivers de máquinas ou integrar lógica personalizada sem modificar o código-base principal.

## Início Rápido

A maneira mais rápida de começar é com o [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template) oficial. Faça um fork ou clone, renomeie o diretório e atualize os metadados para corresponder ao seu addon.

## Como os Addons Funcionam

O `AddonManager` escaneia o diretório `addons` em busca de addons válidos. Um addon é simplesmente um diretório contendo um arquivo de manifesto `rayforge-addon.yaml` junto com seu código Python.

Veja como é um addon típico:

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Manifesto obrigatório
├── my_addon/            <-- Seu pacote Python
│   ├── __init__.py
│   ├── backend.py       <-- Ponto de entrada do backend
│   └── frontend.py      <-- Ponto de entrada do frontend (opcional)
├── assets/              <-- Recursos opcionais
├── locales/             <-- Traduções opcionais (arquivos .po)
└── README.md
```

## Seu Primeiro Addon

Vamos criar um addon simples que registra um driver de máquina personalizado. Primeiro, crie o manifesto:

```yaml title="rayforge-addon.yaml"
name: my_laser_driver
display_name: "My Laser Driver"
description: "Adds support for the XYZ laser cutter."
api_version: 9

author:
  name: Jane Doe
  email: jane@example.com

provides:
  backend: my_addon.backend
```

Agora crie o módulo backend que registra seu driver:

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

É isso! Seu addon será carregado quando o Rayforge iniciar, e seu driver de máquina estará disponível para os usuários.

A documentação do [Manifesto](./addon-manifest.md) cobre todas as opções de configuração disponíveis.

## Compreendendo os Pontos de Entrada

Os addons podem fornecer dois pontos de entrada, cada um carregado em momentos diferentes:

O ponto de entrada **backend** é carregado tanto no processo principal quanto nos processos de trabalho. Use-o para drivers de máquinas, tipos de passos, produtores e transformadores de ops, ou qualquer funcionalidade principal que não precise de dependências de interface.

O ponto de entrada **frontend** é carregado apenas no processo principal. É aqui que você colocaria componentes de interface, widgets GTK, itens de menu e qualquer coisa que precise acessar a janela principal.

Ambos são especificados como caminhos de módulo pontilhados como `my_addon.backend`.

## Conectando-se ao Rayforge com Hooks

O Rayforge usa hooks do `pluggy` para permitir que os addons se integrem à aplicação. Simplesmente decore suas funções com `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context: RayforgeContext):
    """Called when Rayforge is fully initialized."""
    # Seu código de configuração aqui
    pass

@hookimpl
def on_unload():
    """Called when the addon is being disabled or unloaded."""
    # Limpe recursos aqui
    pass
```

A documentação de [Hooks](./addon-hooks.md) descreve cada hook disponível e quando ele é chamado.

## Registrando Seus Componentes

A maioria dos hooks recebe um objeto de registro que você usa para registrar seus componentes personalizados:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)

@hookimpl
def register_actions(action_registry):
    from .actions import setup_actions
    setup_actions(action_registry)
```

A documentação de [Registros](./addon-registries.md) explica cada registro e como usá-los.

## Acessando os Dados do Rayforge

O hook `rayforge_init` dá acesso a um objeto `RayforgeContext`. Através deste contexto, você pode acessar tudo no Rayforge:

Você pode obter a máquina atualmente ativa via `context.machine`, ou acessar todas as máquinas através de `context.machine_mgr`. O objeto `context.config` contém as configurações globais, enquanto `context.camera_mgr` fornece acesso aos feeds de câmera. Para materiais, use `context.material_mgr`, e para receitas de processamento, use `context.recipe_mgr`. O gerenciador de dialetos G-code está disponível como `context.dialect_mgr`, e recursos de IA passam por `context.ai_provider_mgr`. Para localização, verifique `context.language` para o código do idioma atual. O gerenciador de addons em si está disponível como `context.addon_mgr`, e se você estiver criando addons pagos, `context.license_validator` cuida da validação de licenças.

## Adicionando Traduções

Os addons podem fornecer traduções usando arquivos `.po` padrão. Organize-os assim:

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── es/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

O Rayforge compila automaticamente arquivos `.po` para arquivos `.mo` quando seu addon é carregado.

## Testando Durante o Desenvolvimento

Para testar seu addon localmente, crie um link simbólico da sua pasta de desenvolvimento para o diretório de addons do Rayforge.

Primeiro, encontre seu diretório de configuração. No Windows, é `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`. No macOS, procure em `~/Library/Application Support/rayforge/addons`. No Linux, é `~/.config/rayforge/addons`.

Então crie o link simbólico:

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

Reinicie o Rayforge e verifique o console por uma mensagem como `Loaded addon: my_laser_driver`.

## Compartilhando Seu Addon

Quando estiver pronto para compartilhar seu addon, envie-o para um repositório Git público no GitHub ou GitLab. Então envie-o para o [rayforge-registry](https://github.com/barebaric/rayforge-registry) fazendo fork do repositório, adicionando os metadados do seu addon e abrindo um pull request.

Uma vez aceito, os usuários poderão instalar seu addon diretamente através do gerenciador de addons do Rayforge.
