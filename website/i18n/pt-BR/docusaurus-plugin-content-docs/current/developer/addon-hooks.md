# Hooks de Addons

Hooks são os pontos de conexão entre seu addon e o Rayforge. Quando algo acontece na aplicação—um passo é criado, um diálogo abre, ou a janela inicializa—o Rayforge chama quaisquer hooks registrados para que seu addon possa responder.

## Como os Hooks Funcionam

O Rayforge usa [pluggy](https://pluggy.readthedocs.io/) para seu sistema de hooks. Para implementar um hook, decore uma função com `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Seu código executa quando o Rayforge termina de inicializar
    pass
```

Você não precisa implementar todos os hooks—apenas aqueles que precisa. Todos os hooks são opcionais.

## Hooks de Ciclo de Vida

Estes hooks gerenciam o ciclo de vida geral do seu addon.

### `rayforge_init(context)`

Este é seu ponto de entrada principal. O Rayforge chama este hook depois que o contexto da aplicação está totalmente inicializado, significando que todos os gerenciadores, configurações e hardware estão prontos. Use-o para configuração geral, logging ou injeção de elementos de interface.

O parâmetro `context` é uma instância de `RayforgeContext` que dá acesso a tudo no Rayforge. Veja [Acessando Dados do Rayforge](./addon-overview.md#accessing-rayforges-data) para detalhes.

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

O Rayforge chama isto quando seu addon está sendo desabilitado ou descarregado. Use-o para limpar recursos, fechar conexões ou remover handlers registrados.

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Limpe quaisquer recursos aqui
```

### `main_window_ready(main_window)`

Este hook é disparado quando a janela principal está totalmente inicializada. É útil para registrar páginas de interface, comandos ou outros componentes que precisam que a janela principal exista primeiro.

O parâmetro `main_window` é a instância de `MainWindow`.

```python
@hookimpl
def main_window_ready(main_window):
    # Adiciona uma página personalizada à janela principal
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## Hooks de Registro

Estes hooks permitem que você registre componentes personalizados nos vários registros do Rayforge.

### `register_machines(machine_manager)`

Use isto para registrar novos drivers de máquinas. O `machine_manager` é uma instância de `MachineManager` que gerencia todas as configurações de máquinas.

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

Registre tipos de passos personalizados que aparecem no painel de operações. O `step_registry` é uma instância de `StepRegistry`.

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

Registre produtores de ops personalizados que geram trajetórias. O `producer_registry` é uma instância de `ProducerRegistry`.

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

Registre transformadores de ops personalizados para operações de pós-processamento. Transformadores modificam operações depois que os produtores as geram. O `transformer_registry` é uma instância de `TransformerRegistry`.

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

Registre comandos de editor que estendem a funcionalidade do editor de documentos. O `command_registry` é uma instância de `CommandRegistry`.

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

Registre ações de janela com posicionamento opcional em menu e barra de ferramentas. Ações são como você adiciona botões, itens de menu e atalhos de teclado. O `action_registry` é uma instância de `ActionRegistry`.

```python
from gi.repository import Gio
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement

@hookimpl
def register_actions(action_registry):
    action = Gio.SimpleAction.new("my-action", None)
    action.connect("activate", on_my_action_activated)
    
    action_registry.register(
        action_name="my-action",
        action=action,
        addon_name="my_addon",
        label="My Action",
        icon_name="document-new-symbolic",
        shortcut="<Ctrl><Alt>m",
        menu=MenuPlacement(menu_id="tools", priority=50),
        toolbar=ToolbarPlacement(group="main", priority=50),
    )
```

### `register_layout_strategies(layout_registry)`

Registre estratégias de layout personalizadas para organizar conteúdo no documento. O `layout_registry` é uma instância de `LayoutStrategyRegistry`. Note que metadados de interface como rótulos e atalhos devem ser registrados via `register_actions`, não aqui.

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

Registre tipos de assets personalizados que podem ser armazenados em documentos. Isso habilita desserialização dinâmica de assets fornecidos por addons. O `asset_type_registry` é uma instância de `AssetTypeRegistry`.

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

Registre renderizadores personalizados para exibir seus tipos de assets na interface. O `renderer_registry` é uma instância de `RendererRegistry`.

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

Registre exportadores de arquivos para formatos de exportação personalizados. O `exporter_registry` é uma instância de `ExporterRegistry`.

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

Registre importadores de arquivos para formatos de importação personalizados. O `importer_registry` é uma instância de `ImporterRegistry`.

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

Registre bibliotecas de materiais adicionais. Chame `library_manager.add_library_from_path(path)` para registrar diretórios contendo arquivos YAML de materiais. Por padrão, bibliotecas registradas são somente leitura.

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## Hooks de Extensão de Interface

Estes hooks permitem que você estenda componentes de interface existentes.

### `step_settings_loaded(dialog, step, producer)`

O Rayforge chama isto quando um diálogo de configurações de passo está sendo populado. Você pode adicionar widgets personalizados ao diálogo baseado no tipo de produtor do passo.

O `dialog` é uma instância de `GeneralStepSettingsView`. O `step` é o `Step` sendo configurado. O `producer` é a instância de `OpsProducer`, ou `None` se não estiver disponível.

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Só adiciona widgets para tipos específicos de produtor
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

Chamado quando as configurações de pós-processamento estão sendo populadas. Adicione widgets personalizados para seus transformadores aqui.

O `dialog` é uma instância de `PostProcessingSettingsView`. O `step` é o `Step` sendo configurado. O `transformer` é a instância de `OpsTransformer`.

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## Histórico de Versões da API

Os hooks são versionados para manter compatibilidade retroativa. Quando novos hooks são adicionados ou existentes mudam, a versão da API é incrementada. O campo `api_version` do seu addon deve ser pelo menos a versão mínima suportada.

A versão atual da API é 9. Aqui está o que mudou nas versões recentes:

**Versão 9** adicionou `main_window_ready`, `register_exporters`, `register_importers` e `register_renderers`.

**Versão 8** adicionou `register_asset_types` para tipos de assets personalizados.

**Versão 7** adicionou `register_material_libraries`.

**Versão 6** adicionou `register_transformers`.

**Versão 5** substituiu `register_step_widgets` por `step_settings_loaded` e `transformer_settings_loaded`.

**Versão 4** removeu `register_menu_items` e consolidou o registro de ações em `register_actions`.

**Versão 2** adicionou `register_layout_strategies`.

**Versão 1** foi o lançamento inicial com hooks principais para ciclo de vida de addons, registro de recursos e integração de interface.
