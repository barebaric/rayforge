# Посібник розробника пакетів Rayforge

Rayforge використовує систему пакетів на основі [pluggy](https://pluggy.readthedocs.io/)
щоб дозволити розробникам розширювати функціональність, додавати нові драйвери машин або
інтегрувати власну логіку без модифікації основної кодової бази.

## 1. Швидкий старт

Найшвидший спосіб почати - використати офіційний шаблон.

1. **Форкніть або клонуйте** [rayforge-package-template](https://github.com/barebaric/rayforge-package-template).
2. **Перейменуйте** директорію та оновіть метадані.

## 2. Структура пакету

`PackageManager` сканує директорію `packages`. Валідний пакет має бути
директорією, що містить принаймні два файли:

1. `rayforge_package.yaml` (Метадані)
2. Точка входу Python (наприклад, `package.py`)

**Макет директорії:**

```text
my-rayforge-package/
├── rayforge_package.yaml  <-- Обов'язковий маніфест
├── package.py             <-- Точка входу (логіка)
├── assets/                <-- Опціональні ресурси
└── README.md
```

## 3. Маніфест (`rayforge_package.yaml`)

Цей файл каже Rayforge, як завантажити ваш пакет.

```yaml
# rayforge_package.yaml

# Унікальний ідентифікатор для вашого пакету
name: my_custom_package

# Зрозуміла для людини відображувана назва
display_name: "Мій власний пакет"

# Рядок версії
version: 0.1.0

# Опис, що відображається в UI
description: "Додає підтримку лазерного різака XYZ."

# Залежності (пакет та обмеження версій)
depends:
  - rayforge>=0.27.0,~0.27

# Python файл для завантаження (відносно папки пакету)
entry_point: package.py

# Метадані автора
author: Jane Doe
url: https://github.com/username/my-custom-package
```

## 4. Написання коду пакету

Rayforge використовує хуки `pluggy`. Щоб підключитися до Rayforge, визначте функції, декоровані
з `@pluggy.HookimplMarker("rayforge")`.

### Базовий шаблон (`package.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

# Визначте маркер реалізації хуку
hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Викликається коли Rayforge повністю ініціалізовано.
    Це ваша головна точка входу для доступу до менеджерів.
    """
    logger.info("Мій власний пакет запущено!")

    # Доступ до основних систем через контекст
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Пакет працює на машині: {machine.id}")

@hookimpl
def register_machines(machine_manager):
    """
    Викликається під час запуску для реєстрації нових драйверів машин.
    """
    # from .my_driver import MyNewMachine
    # machine_manager.register("my_new_machine", MyNewMachine)
    pass
```

### Доступні хуки

Визначені в `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Головна точка входу.** Викликається після завантаження конфігурації, камери та обладнання.
  Використовуйте це для логіки, ін'єкцій UI або слухачів.

**`register_machines`** (`machine_manager`)
: Викликається на ранній стадії завантаження. Використовуйте для реєстрації нових класів/драйверів
  обладнання.

## 5. Доступ до даних Rayforge

Хук `rayforge_init` надає **`RayforgeContext`**. Через цей об'єкт
ви можете отримати доступ:

- **`context.machine`**: Поточний активний екземпляр машини.
- **`context.config`**: Глобальні налаштування конфігурації.
- **`context.camera_mgr`**: Доступ до камер та інструментів комп'ютерного зору.
- **`context.material_mgr`**: Доступ до бібліотеки матеріалів.
- **`context.recipe_mgr`**: Доступ до рецептів обробки.

## 6. Розробка та тестування

Щоб протестувати ваш пакет локально без публікації:

1.  **Знайдіть вашу директорію конфігурації:**
    Rayforge використовує `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\packages`
    - **macOS:** `~/Library/Application Support/rayforge/packages`
    - **Linux:** `~/.config/rayforge/packages`
      _(Перевірте логи при запуску на `Config dir is ...`)_

2.  **Створіть символьне посилання на ваш пакет:**
    Замість копіювання файлів туди-сюди, створіть символьне посилання з вашої папки
    розробки до папки пакетів Rayforge.

    _Linux/macOS:_

    ```bash
    ln -s /path/to/my-rayforge-package ~/.config/rayforge/packages/my-rayforge-package
    ```

3.  **Перезапустіть Rayforge:**
    Додаток сканує директорію при запуску. Перевірте консольні логи на:
    > `Loaded package: my_custom_package`

## 7. Публікація

Щоб поділитися вашим пакетом зі спільнотою:

1.  **Опублікуйте на Git:** Завантажте код у публічний Git репозиторій (GitHub, GitLab,
    тощо.).
2.  **Надішліть до реєстру:**
    - Перейдіть до [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Зробіть форк репозиторію.
    - Додайте Git URL вашого пакету та метадані до списку реєстру.
    - Надішліть Pull Request.

Після прийняття користувачі можуть встановити ваш пакет напряму через UI Rayforge або
використовуючи Git URL.
