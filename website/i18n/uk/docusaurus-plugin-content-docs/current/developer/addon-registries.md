# Реєстри аддонів

Реєстри — це спосіб, яким Rayforge керує розширюваністю. Кожен реєстр містить колекцію споріднених компонентів — кроків, продюсерів, дій тощо. Коли ваш аддон реєструє щось, це стає доступним у всьому додатку.

## Як працюють реєстри

Всі реєстри слідують схожому шаблону. Вони надають метод `register()` для додавання елементів та різні методи пошуку для їх отримання. Більшість реєстрів також відстежують, який аддон зареєстрував кожен елемент, щоб вони могли очистити все при вивантаженні аддону.

Ось загальний шаблон:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

Параметр `addon_name` опціональний, але рекомендований. Він гарантує, що ваші компоненти будуть належним чином видалені, якщо користувач відключить ваш аддон.

## Реєстр кроків

Реєстр кроків (`StepRegistry`) керує типами кроків, які з'являються в панелі операцій. Кожен крок представляє тип операції, яку користувачі можуть додати до свого завдання.

### Реєстрація кроку

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

Ім'я класу кроку використовується як ключ реєстру. Ваш клас кроку має успадковуватися від `Step` та визначати атрибути на кшталт `TYPELABEL`, `HIDDEN`, а також реалізовувати метод класу `create()`.

### Отримання кроків

Реєстр надає кілька методів для пошуку кроків:

```python
# Get a step by its class name
step_class = step_registry.get("MyCustomStep")

# Get a step by its TYPELABEL (for backward compatibility)
step_class = step_registry.get_by_typelabel("My Custom Step")

# Get all registered steps
all_steps = step_registry.all_steps()

# Get factory methods for UI menus (excludes hidden steps)
factories = step_registry.get_factories()
```

## Реєстр продюсерів

Реєстр продюсерів (`ProducerRegistry`) керує продюсерами операцій. Продюсери генерують траєкторії операцій для кроку — по суті, вони конвертують вашу робочу деталь у інструкції для машини.

### Реєстрація продюсера

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

За замовчуванням ім'я класу стає ключем реєстру. Ви можете вказати власне ім'я:

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### Отримання продюсерів

```python
# Get a producer by name
producer_class = producer_registry.get("MyCustomProducer")

# Get all producers
all_producers = producer_registry.all_producers()
```

## Реєстр трансформерів

Реєстр трансформерів (`TransformerRegistry`) керує трансформерами операцій. Трансформери виконують постобробку операцій після того, як продюсери їх згенерували — подумайте про такі завдання, як оптимізація шляху, згладжування або додавання утримуючих табів.

### Реєстрація трансформера

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### Отримання трансформерів

```python
# Get a transformer by name
transformer_class = transformer_registry.get("MyCustomTransformer")

# Get all transformers
all_transformers = transformer_registry.all_transformers()
```

## Реєстр дій

Реєстр дій (`ActionRegistry`) керує діями вікна. Дії — це спосіб додавання пунктів меню, кнопок панелі інструментів та клавіатурних скорочень. Це один із найбільш функціонально насичених реєстрів.

### Реєстрація дії

```python
from gi.repository import Gio
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement

@hookimpl
def register_actions(action_registry):
    # Create the action
    action = Gio.SimpleAction.new("my-action", None)
    action.connect("activate", lambda a, p: do_something())
    
    # Register with optional menu and toolbar placement
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

### Параметри дії

При реєстрації дії ви можете надати:

- `action_name`: Ідентифікатор дії (без префікса "win.")
- `action`: Екземпляр `Gio.SimpleAction`
- `addon_name`: Ім'я вашого аддону для очищення
- `label`: Текст, зрозумілий для людини, для меню та підказок
- `icon_name`: Ідентифікатор іконки для панелей інструментів
- `shortcut`: Клавіатурне скорочення з використанням синтаксису GTK accelerator
- `menu`: Об'єкт `MenuPlacement`, що вказує меню та пріоритет
- `toolbar`: Об'єкт `ToolbarPlacement`, що вказує групу панелі інструментів та пріоритет

### Розміщення в меню

Клас `MenuPlacement` приймає:

- `menu_id`: До якого меню додати (наприклад, "tools", "arrange")
- `priority`: Нижчі числа з'являються раніше

### Розміщення в панелі інструментів

Клас `ToolbarPlacement` приймає:

- `group`: Ідентифікатор групи панелі інструментів (наприклад, "main", "arrange")
- `priority`: Нижчі числа з'являються раніше

### Отримання дій

```python
# Get action info
info = action_registry.get("my-action")

# Get all actions for a specific menu
menu_items = action_registry.get_menu_items("tools")

# Get all actions for a toolbar group
toolbar_items = action_registry.get_toolbar_items("main")

# Get all actions with keyboard shortcuts
shortcuts = action_registry.get_all_with_shortcuts()
```

## Реєстр команд

Реєстр команд (`CommandRegistry`) керує командами редактора. Команди розширюють функціональність редактора документів.

### Реєстрація команди

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

Класи команд мають приймати екземпляр `DocEditor` у своєму конструкторі.

### Отримання команд

```python
# Get a command by name
command_class = command_registry.get("my_command")

# Get all commands
all_commands = command_registry.all_commands()
```

## Реєстр типів ресурсів

Реєстр типів ресурсів (`AssetTypeRegistry`) керує типами ресурсів, які можуть зберігатися в документах. Це забезпечує динамічну десеріалізацію — коли Rayforge завантажує документ, що містить ваш власний ресурс, він знає, як його реконструювати.

### Реєстрація типу ресурсу

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(
        MyCustomAsset,
        type_name="my_asset",
        addon_name="my_addon"
    )
```

`type_name` — це рядок, що використовується в серіалізованих документах для ідентифікації вашого типу ресурсу.

### Отримання типів ресурсів

```python
# Get an asset class by type name
asset_class = asset_type_registry.get("my_asset")

# Get all registered asset types
all_types = asset_type_registry.all_types()
```

## Реєстр стратегій компонування

Реєстр стратегій компонування (`LayoutStrategyRegistry`) керує стратегіями компонування для розміщення контенту в редакторі документів.

### Реєстрація стратегії компонування

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(
        MyLayoutStrategy,
        name="my_layout",
        addon_name="my_addon"
    )
```

Зауважте, що UI-метадані, такі як мітки та скорочення, мають реєструватися через реєстр дій, а не тут.

### Отримання стратегій компонування

```python
# Get a strategy by name
strategy_class = layout_registry.get("my_layout")

# Get all strategy classes
all_strategies = layout_registry.list_all()

# Get all strategy names
strategy_names = layout_registry.list_names()
```

## Реєстр імпортерів

Реєстр імпортерів (`ImporterRegistry`) керує імпортерами файлів. Імпортери обробляють завантаження зовнішніх файлів у Rayforge.

### Реєстрація імпортера

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

Ваш клас імпортера має визначати атрибути класу `extensions` та `mime_types`, щоб реєстр знав, які файли він обробляє.

### Отримання імпортерів

```python
# Get importer by file extension
importer_class = importer_registry.get_by_extension(".xyz")

# Get importer by MIME type
importer_class = importer_registry.get_by_mime_type("application/x-xyz")

# Get importer by class name
importer_class = importer_registry.get_by_name("MyCustomImporter")

# Get appropriate importer for a file path
importer_class = importer_registry.get_for_file(Path("file.xyz"))

# Get all supported file extensions
extensions = importer_registry.get_supported_extensions()

# Get all file filters for file dialogs
filters = importer_registry.get_all_filters()

# Get importers that support a specific feature
importers = importer_registry.by_feature(ImporterFeature.SOME_FEATURE)
```

## Реєстр експортерів

Реєстр експортерів (`ExporterRegistry`) керує експортерами файлів. Експортери обробляють збереження документів або операцій Rayforge у зовнішні формати.

### Реєстрація експортера

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

Ваш клас експортера має визначати атрибути класу `extensions` та `mime_types`.

### Отримання експортерів

```python
# Get exporter by file extension
exporter_class = exporter_registry.get_by_extension(".xyz")

# Get exporter by MIME type
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Get all file filters for file dialogs
filters = exporter_registry.get_all_filters()
```

## Реєстр рендерерів

Реєстр рендерерів (`RendererRegistry`) керує рендерерами ресурсів. Рендерери відображають ресурси в UI.

### Реєстрація рендерера

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

Зауважте, що ви реєструєте екземпляр рендерера, а не клас. Ім'я класу рендерера використовується як ключ реєстру.

### Отримання рендерерів

```python
# Get renderer by class name
renderer = renderer_registry.get("MyAssetRenderer")

# Get renderer by name (same as get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Get all renderers
all_renderers = renderer_registry.all()
```

## Менеджер бібліотек

Менеджер бібліотек (`LibraryManager`) керує бібліотеками матеріалів. Хоча технічно це не реєстр, він слідує схожим шаблонам для реєстрації бібліотек, наданих аддонами.

### Реєстрація бібліотеки матеріалів

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

Зареєстровані бібліотеки за замовчуванням доступні лише для читання. Користувачі можуть переглядати та використовувати матеріали, але не можуть змінювати їх через UI.
