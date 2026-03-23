# Огляд розробки аддонів

Rayforge використовує систему аддонів на основі [pluggy](https://pluggy.readthedocs.io/), яка дозволяє розширювати функціональність, додавати нові драйвери пристроїв або інтегрувати власну логіку без зміни основного коду.

## Швидкий старт

Найшвидший спосіб почати — використати офіційний [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template). Зробіть форк або клонуйте його, перейменуйте директорію та оновіть метадані відповідно до вашого аддону.

## Як працюють аддони

`AddonManager` сканує директорію `addons` на наявність валідних аддонів. Аддон — це просто директорія, що містить файл маніфесту `rayforge-addon.yaml` разом із вашим Python-кодом.

Ось як виглядає типовий аддон:

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Обов'язковий маніфест
├── my_addon/            <-- Ваш Python-пакет
│   ├── __init__.py
│   ├── backend.py       <-- Точка входу бекенду
│   └── frontend.py      <-- Точка входу фронтенду (опціонально)
├── assets/              <-- Опціональні ресурси
├── locales/             <-- Опціональні переклади (.po файли)
└── README.md
```

## Ваш перший аддон

Створимо простий аддон, який реєструє власний драйвер пристрою. Спочатку створіть маніфест:

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

Тепер створіть модуль бекенду, який реєструє ваш драйвер:

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

Це все! Ваш аддон буде завантажено при запуску Rayforge, і драйвер вашого пристрою стане доступним для користувачів.

Документація [Маніфест](./addon-manifest.md) описує всі доступні опції конфігурації.

## Розуміння точок входу

Аддони можуть надавати дві точки входу, кожна з яких завантажується в різний час:

Точка входу **backend** завантажується як у головному процесі, так і в робочих процесах. Використовуйте її для драйверів пристроїв, типів кроків, продюсерів та трансформерів операцій, або будь-якої основної функціональності, яка не потребує UI-залежностей.

Точка входу **frontend** завантажується лише в головному процесі. Тут ви розміщуєте UI-компоненти, GTK-віджети, пункти меню та все, що потребує доступу до головного вікна.

Обидві вказуються як шляхи до модулів у нотації з крапкою, наприклад `my_addon.backend`.

## Підключення до Rayforge через хуки

Rayforge використовує хуки `pluggy`, щоб дозволити аддонам інтегруватися з додатком. Просто декоруйте ваші функції за допомогою `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context: RayforgeContext):
    """Called when Rayforge is fully initialized."""
    # Your setup code here
    pass

@hookimpl
def on_unload():
    """Called when the addon is being disabled or unloaded."""
    # Clean up resources here
    pass
```

Документація [Хуки](./addon-hooks.md) описує кожен доступний хук і коли він викликається.

## Реєстрація ваших компонентів

Більшість хуків отримують об'єкт реєстру, який ви використовуєте для реєстрації власних компонентів:

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

Документація [Реєстри](./addon-registries.md) пояснює кожен реєстр та як ним користуватися.

## Доступ до даних Rayforge

Хук `rayforge_init` надає вам доступ до об'єкта `RayforgeContext`. Через цей контекст ви можете отримати доступ до всього в Rayforge:

Ви можете отримати поточний активний пристрій через `context.machine` або доступ до всіх пристроїв через `context.machine_mgr`. Об'єкт `context.config` містить глобальні налаштування, а `context.camera_mgr` надає доступ до відеопотоків камер. Для матеріалів використовуйте `context.material_mgr`, а для рецептів обробки — `context.recipe_mgr`. Менеджер діалектів G-код доступний як `context.dialect_mgr`, а AI-функції — через `context.ai_provider_mgr`. Для локалізації перевірте `context.language` на поточний код мови. Сам менеджер аддонів доступний як `context.addon_mgr`, а якщо ви створюєте платні аддони, `context.license_validator` обробляє перевірку ліцензій.

## Додавання перекладів

Аддони можуть надавати переклади за допомогою стандартних файлів `.po`. Організуйте їх так:

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

Rayforge автоматично компілює файли `.po` у файли `.mo` при завантаженні вашого аддону.

## Тестування під час розробки

Щоб протестувати ваш аддон локально, створіть символічне посилання з вашої директорії розробки до директорії аддонів Rayforge.

Спочатку знайдіть вашу директорію конфігурації. На Windows це `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`. На macOS шукайте в `~/Library/Application Support/rayforge/addons`. На Linux це `~/.config/rayforge/addons`.

Потім створіть символічне посилання:

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

Перезапустіть Rayforge і перевірте консоль на наявність повідомлення на кшталт `Loaded addon: my_laser_driver`.

## Поширення вашого аддону

Коли ви готові поширити свій аддон, завантажте його до публічного Git-репозиторію на GitHub або GitLab. Потім додайте його до [rayforge-registry](https://github.com/barebaric/rayforge-registry), зробивши форк репозиторію, додавши метадані вашого аддону та відкривши pull request.

Після прийняття користувачі зможуть встановити ваш аддон безпосередньо через менеджер аддонів Rayforge.
