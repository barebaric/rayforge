# Хуки аддонів

Хуки — це точки з'єднання між вашим аддоном та Rayforge. Коли в додатку щось відбувається — створюється крок, відкривається діалог або ініціалізується вікно — Rayforge викликає всі зареєстровані хуки, щоб ваш аддон міг відреагувати.

## Як працюють хуки

Rayforge використовує [pluggy](https://pluggy.readthedocs.io/) для своєї системи хуків. Щоб реалізувати хук, декоруйте функцію за допомогою `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Your code runs when Rayforge finishes initializing
    pass
```

Вам не потрібно реалізовувати кожен хук — лише ті, які вам потрібні. Всі хуки опціональні.

## Хуки життєвого циклу

Ці хуки обробляють загальний життєвий цикл вашого аддону.

### `rayforge_init(context)`

Це ваша головна точка входу. Rayforge викликає цей хук після повної ініціалізації контексту додатку, що означає, що всі менеджери, конфігурації та обладнання готові. Використовуйте це для загального налаштування, логування або впровадження UI-елементів.

Параметр `context` — це екземпляр `RayforgeContext`, який дає вам доступ до всього в Rayforge. Дивіться [Доступ до даних Rayforge](./addon-overview.md#accessing-rayforges-data) для деталей.

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

Rayforge викликає це, коли ваш аддон відключається або вивантажується. Використовуйте це для очищення ресурсів, закриття з'єднань або скасування реєстрації обробників.

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Clean up any resources here
```

### `main_window_ready(main_window)`

Цей хук спрацьовує, коли головне вікно повністю ініціалізоване. Він корисний для реєстрації UI-сторінок, команд або інших компонентів, які потребують, щоб головне вікно вже існувало.

Параметр `main_window` — це екземпляр `MainWindow`.

```python
@hookimpl
def main_window_ready(main_window):
    # Add a custom page to the main window
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## Хуки реєстрації

Ці хуки дозволяють реєструвати власні компоненти в різних реєстрах Rayforge.

### `register_machines(machine_manager)`

Використовуйте це для реєстрації нових драйверів пристроїв. `machine_manager` — це екземпляр `MachineManager`, який керує всіма конфігураціями пристроїв.

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

Реєструє власні типи кроків, які з'являються в панелі операцій. `step_registry` — це екземпляр `StepRegistry`.

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

Реєструє власних продюсерів операцій, які генерують траєкторії інструменту. `producer_registry` — це екземпляр `ProducerRegistry`.

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

Реєструє власні трансформери операцій для постобробки. Трансформери модифікують операції після того, як продюсери їх згенерували. `transformer_registry` — це екземпляр `TransformerRegistry`.

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

Реєструє команди редактора, які розширюють функціональність редактора документів. `command_registry` — це екземпляр `CommandRegistry`.

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

Реєструє дії вікна з опціональним розміщенням у меню та панелі інструментів. Дії — це спосіб додавання кнопок, пунктів меню та клавіатурних скорочень. `action_registry` — це екземпляр `ActionRegistry`.

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

Реєструє власні стратегії компонування для розміщення контенту в документі. `layout_registry` — це екземпляр `LayoutStrategyRegistry`. Зауважте, що UI-метадані, такі як мітки та скорочення, мають реєструватися через `register_actions`, а не тут.

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

Реєструє власні типи ресурсів, які можуть зберігатися в документах. Це забезпечує динамічну десеріалізацію ресурсів, наданих аддонами. `asset_type_registry` — це екземпляр `AssetTypeRegistry`.

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

Реєструє власні рендерери для відображення ваших типів ресурсів в UI. `renderer_registry` — це екземпляр `RendererRegistry`.

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

Реєструє експортери файлів для власних форматів експорту. `exporter_registry` — це екземпляр `ExporterRegistry`.

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

Реєструє імпортери файлів для власних форматів імпорту. `importer_registry` — це екземпляр `ImporterRegistry`.

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

Реєструє додаткові бібліотеки матеріалів. Викличте `library_manager.add_library_from_path(path)`, щоб зареєструвати директорії, що містять YAML-файли матеріалів. За замовчуванням зареєстровані бібліотеки доступні лише для читання.

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## Хуки розширення UI

Ці хуки дозволяють розширювати існуючі UI-компоненти.

### `step_settings_loaded(dialog, step, producer)`

Rayforge викликає це, коли діалог налаштувань кроку заповнюється. Ви можете додати власні віджети до діалогу на основі типу продюсера кроку.

`dialog` — це екземпляр `GeneralStepSettingsView`. `step` — це `Step`, що налаштовується. `producer` — це екземпляр `OpsProducer` або `None`, якщо недоступний.

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Only add widgets for specific producer types
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

Викликається при заповненні налаштувань постобробки. Додавайте власні віджети для ваших трансформерів тут.

`dialog` — це екземпляр `PostProcessingSettingsView`. `step` — це `Step`, що налаштовується. `transformer` — це екземпляр `OpsTransformer`.

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## Історія версій API

Хуки версіонуються для підтримки зворотної сумісності. Коли додаються нові хуки або змінюються існуючі, версія API збільшується. Поле `api_version` вашого аддону має бути не менше мінімальної підтримуваної версії.

Поточна версія API — 9. Ось що змінилося в останніх версіях:

**Версія 9** додала `main_window_ready`, `register_exporters`, `register_importers` та `register_renderers`.

**Версія 8** додала `register_asset_types` для власних типів ресурсів.

**Версія 7** додала `register_material_libraries`.

**Версія 6** додала `register_transformers`.

**Версія 5** замінила `register_step_widgets` на `step_settings_loaded` та `transformer_settings_loaded`.

**Версія 4** видалила `register_menu_items` та консолідувала реєстрацію дій у `register_actions`.

**Версія 2** додала `register_layout_strategies`.

**Версія 1** була початковим релізом з базовими хуками для життєвого циклу аддону, реєстрації ресурсів та UI-інтеграції.
