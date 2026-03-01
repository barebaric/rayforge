# Посібник розробника аддонів Rayforge

Rayforge використовує систему аддонів на основі [pluggy](https://pluggy.readthedocs.io/)
щоб дозволити розробникам розширювати функціональність, додавати нові драйвери машин або
інтегрувати власну логіку без модифікації основної кодової бази.

## 1. Швидкий старт

Найшвидший спосіб почати - використати офіційний шаблон.

1. **Форкніть або клонуйте** [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Перейменуйте** директорію та оновіть метадані.

## 2. Структура аддону

`AddonManager` сканує директорію `addons`. Валідний аддон має бути
директорією, що містить файл маніфесту:

**Макет директорії:**

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Обов'язковий маніфест
├── my_addon/            <-- Python пакет
│   ├── __init__.py
│   ├── backend.py       <-- Точка входу backend
│   └── frontend.py      <-- Точка входу frontend (опціонально)
├── assets/              <-- Опціональні ресурси
├── locales/             <-- Опціональні переклади (файли .po)
└── README.md
```

## 3. Маніфест (`rayforge-addon.yaml`)

Цей файл каже Rayforge, як завантажити ваш аддон.

```yaml
# rayforge-addon.yaml

# Унікальний ідентифікатор для вашого аддону (ім'я директорії)
name: my_custom_addon

# Зрозуміла для людини відображувана назва
display_name: "Мій власний аддон"

# Опис, що відображається в UI
description: "Додає підтримку лазерного різака XYZ."

# Версія API (має відповідати PLUGIN_API_VERSION Rayforge)
api_version: 1

# Залежності версії Rayforge
depends:
  - rayforge>=0.27.0,<2.0.0

# Опціонально: Залежності від інших аддонів
requires:
  - some-other-addon>=1.0.0

# Що надає аддон
provides:
  # Модуль backend (завантажується в основному та worker процесах)
  backend: my_addon.backend
  # Модуль frontend (завантажується лише в основному процесі, для UI)
  frontend: my_addon.frontend
  # Опціональні файли asset
  assets:
    - path: assets/profiles.json
      type: profiles

# Метадані автора
author:
  name: Іван Петренко
  email: ivan@example.com

url: https://github.com/username/my-custom-addon
```

### Обов'язкові поля

- `name`: Унікальний ідентифікатор (має відповідати імені директорії)
- `display_name`: Зрозуміла назва, що відображається в UI
- `description`: Короткий опис функціональності аддону
- `api_version`: Має бути `1` (відповідає `PLUGIN_API_VERSION` Rayforge)
- `depends`: Список обмежень версії для Rayforge
- `author`: Об'єкт з `name` (обов'язково) та `email` (опціонально)

### Опціональні поля

- `requires`: Список залежностей від інших аддонів
- `provides`: Точки входу та assets
- `url`: Сторінка проекту або репозиторій

## 4. Точки входу

Аддони можуть надавати два типи точок входу:

### Backend (`provides.backend`)

Завантажується як в основному процесі, так і в worker процесах. Використовуйте для:
- Драйверів машин
- Типів кроків
- Продюсерів ops
- Основної функціональності без UI залежностей

### Frontend (`provides.frontend`)

Завантажується лише в основному процесі. Використовуйте для:
- UI компонентів
- GTK віджетів
- Пунктів меню
- Дій, що потребують головного вікна

Точки входу вказуються як шляхи модулів з крапками (напр., `my_addon.backend`).

## 5. Написання коду аддону

Rayforge використовує хуки `pluggy`. Щоб підключитися до Rayforge, визначте функції, декоровані
з `@pluggy.HookimplMarker("rayforge")`.

### Базовий шаблон (`backend.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Викликається коли Rayforge повністю ініціалізовано.
    Це ваша головна точка входу для доступу до менеджерів.
    """
    logger.info("Мій власний аддон запущено!")

    machine = context.machine
    if machine:
        logger.info(f"Аддон працює на машині: {machine.id}")

@hookimpl
def on_unload():
    """
    Викликається коли аддон вимикається або вивантажується.
    Очистити ресурси, закрити з'єднання, скасувати реєстрацію handlers.
    """
    logger.info("Мій власний аддон завершує роботу")

@hookimpl
def register_machines(machine_manager):
    """
    Викликається під час запуску для реєстрації нових драйверів машин.
    """
    from .my_driver import MyNewMachine
    machine_manager.register("my_new_machine", MyNewMachine)

@hookimpl
def register_steps(step_registry):
    """
    Викликається для реєстрації власних типів кроків.
    """
    from .my_step import MyCustomStep
    step_registry.register("my_custom_step", MyCustomStep)

@hookimpl
def register_producers(producer_registry):
    """
    Викликається для реєстрації власних продюсерів ops.
    """
    from .my_producer import MyProducer
    producer_registry.register("my_producer", MyProducer)

@hookimpl
def register_step_widgets(widget_registry):
    """
    Викликається для реєстрації власних віджетів налаштувань кроків.
    """
    from .my_widget import MyStepWidget
    widget_registry.register("my_custom_step", MyStepWidget)

@hookimpl
def register_menu_items(menu_registry):
    """
    Викликається для реєстрації пунктів меню.
    """
    from .menu_items import register_menus
    register_menus(menu_registry)

@hookimpl
def register_commands(command_registry):
    """
    Викликається для реєстрації команд редактора.
    """
    from .commands import register_commands
    register_commands(command_registry)

@hookimpl
def register_actions(window):
    """
    Викликається для реєстрації дій вікна.
    """
    from .actions import setup_actions
    setup_actions(window)
```

### Доступні хуки

Визначені в `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Головна точка входу.** Викликається після завантаження конфігурації, камери та обладнання.
  Використовуйте це для логіки, ін'єкцій UI або слухачів.

**`on_unload`** ()
: Викликається коли аддон вимикається або вивантажується. Використовуйте для очищення
  ресурсів, закриття з'єднань, скасування реєстрації handlers тощо.

**`register_machines`** (`machine_manager`)
: Викликається під час запуску для реєстрації нових драйверів машин.

**`register_steps`** (`step_registry`)
: Викликається для дозволу плагінам реєструвати власні типи кроків.

**`register_producers`** (`producer_registry`)
: Викликається для дозволу плагінам реєструвати власних продюсерів ops.

**`register_step_widgets`** (`widget_registry`)
: Викликається для дозволу плагінам реєструвати власні віджети налаштувань кроків.

**`register_menu_items`** (`menu_registry`)
: Викликається для дозволу плагінам реєструвати пункти меню.

**`register_commands`** (`command_registry`)
: Викликається для дозволу плагінам реєструвати команди редактора.

**`register_actions`** (`window`)
: Викликається для дозволу плагінам реєструвати дії вікна.

## 6. Доступ до даних Rayforge

Хук `rayforge_init` надає **`RayforgeContext`**. Через цей об'єкт
ви можете отримати доступ:

- **`context.machine`**: Поточний активний екземпляр машини.
- **`context.config`**: Глобальні налаштування конфігурації.
- **`context.config_mgr`**: Менеджер конфігурації.
- **`context.machine_mgr`**: Менеджер машин (всі машини).
- **`context.camera_mgr`**: Доступ до камер та інструментів комп'ютерного зору.
- **`context.material_mgr`**: Доступ до бібліотеки матеріалів.
- **`context.recipe_mgr`**: Доступ до рецептів обробки.
- **`context.dialect_mgr`**: Менеджер діалектів G-code.
- **`context.language`**: Поточний код мови для локалізованого контенту.
- **`context.addon_mgr`**: Екземпляр менеджера аддонів.
- **`context.plugin_mgr`**: Екземпляр менеджера плагінів.
- **`context.debug_dump_manager`**: Менеджер debug дампів.
- **`context.artifact_store`**: Сховище артефактів пайплайну.

## 7. Локалізація

Аддони можуть надавати переклади використовуючи файли `.po`:

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── uk/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

Файли `.po` автоматично компілюються в файли `.mo` коли аддон
встановлюється або завантажується.

## 8. Розробка та тестування

Щоб протестувати ваш аддон локально без публікації:

1.  **Знайдіть вашу директорію конфігурації:**
    Rayforge використовує `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`
    - **macOS:** `~/Library/Application Support/rayforge/addons`
    - **Linux:** `~/.config/rayforge/addons`
      _(Перевірте логи при запуску на `Config dir is ...`)_

2.  **Створіть символьне посилання на ваш аддон:**
    Замість копіювання файлів туди-сюди, створіть символьне посилання з вашої папки
    розробки до папки аддонів Rayforge.

    _Linux/macOS:_

    ```bash
    ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
    ```

3.  **Перезапустіть Rayforge:**
    Додаток сканує директорію при запуску. Перевірте консольні логи на:
    > `Loaded addon: my_custom_addon`

## 9. Публікація

Щоб поділитися вашим аддоном зі спільнотою:

1.  **Опублікуйте на Git:** Завантажте код у публічний Git репозиторій (GitHub, GitLab,
    тощо).
2.  **Надішліть до реєстру:**
    - Перейдіть до [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Зробіть форк репозиторію.
    - Додайте Git URL вашого аддону та метадані до списку реєстру.
    - Надішліть Pull Request.

Після прийняття користувачі можуть встановити ваш аддон напряму через UI Rayforge або
використовуючи Git URL.
