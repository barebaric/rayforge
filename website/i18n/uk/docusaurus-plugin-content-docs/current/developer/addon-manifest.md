# Маніфест аддону

Кожен аддон потребує файл `rayforge-addon.yaml` у своїй кореневій директорії. Цей маніфест повідомляє Rayforge про ваш аддон — його назву, що він надає та як його завантажити.

## Базова структура

Ось повний маніфест з усіма загальними полями:

```yaml
name: my_custom_addon
display_name: "My Custom Addon"
description: "Adds support for the XYZ laser cutter."
api_version: 9
url: https://github.com/username/my-custom-addon

author:
  name: Jane Doe
  email: jane@example.com

depends:
  - rayforge>=0.27.0

requires:
  - some-other-addon>=1.0.0

provides:
  backend: my_addon.backend
  frontend: my_addon.frontend
  assets:
    - path: assets/profiles.json
      type: profiles

license:
  name: MIT
```

## Обов'язкові поля

### `name`

Унікальний ідентифікатор вашого аддону. Це має бути валідне ім'я Python-модуля — лише літери, цифри та підкреслення, і воно не може починатися з цифри.

```yaml
name: my_custom_addon
```

### `display_name`

Зрозуміла для людини назва, що відображається в UI. Може містити пробіли та спеціальні символи.

```yaml
display_name: "My Custom Addon"
```

### `description`

Короткий опис того, що робить ваш аддон. Відображається в менеджері аддонів.

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

Версія API, яку target-ує ваш аддон. Вона має бути не менше 1 (мінімальна підтримувана версія) і не більше поточної версії (9). Використання вищої версії, ніж підтримується, призведе до помилки валідації вашого аддону.

```yaml
api_version: 9
```

Дивіться документацію [Хуки](./addon-hooks.md#api-version-history) для інформації про те, що змінилося в кожній версії.

### `author`

Інформація про автора аддону. Поле `name` є обов'язковим; `email` — опціональне, але рекомендоване для зв'язку з користувачами.

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## Опціональні поля

### `url`

URL до домашньої сторінки або репозиторію вашого аддону.

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Обмеження версій для самого Rayforge. Вкажіть мінімальну версію, яку потребує ваш аддон.

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

Залежності від інших аддонів. Перелічіть імена аддонів з обмеженнями версій.

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

Номер версії вашого аддону. Зазвичай визначається автоматично з git-тегів, але ви можете вказати його явно. Використовуйте семантичне версіонування (наприклад, `1.0.0`).

```yaml
version: 1.0.0
```

## Точки входу

Секція `provides` визначає, що ваш аддон надає Rayforge.

### Backend

Модуль бекенду завантажується як у головному процесі, так і в робочих процесах. Використовуйте це для драйверів пристроїв, типів кроків, продюсерів операцій та будь-якої основної функціональності.

```yaml
provides:
  backend: my_addon.backend
```

Значення — це шлях до Python-модуля в нотації з крапкою, відносно директорії вашого аддону.

### Frontend

Модуль фронтенду завантажується лише в головному процесі. Використовуйте це для UI-компонентів, GTK-віджетів та всього, що потребує головного вікна.

```yaml
provides:
  frontend: my_addon.frontend
```

### Assets

Ви можете включити файли ресурсів, які Rayforge розпізнає. Кожен ресурс має шлях і тип:

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

`path` відносний до кореня вашого аддону і має існувати. Типи ресурсів визначаються Rayforge і можуть включати такі речі, як профілі пристроїв, бібліотеки матеріалів або шаблони.

## Інформація про ліцензію

Поле `license` описує, як ліцензований ваш аддон. Для безкоштовних аддонів просто вкажіть назву ліцензії, використовуючи SPDX-ідентифікатор:

```yaml
license:
  name: MIT
```

Поширені SPDX-ідентифікатори включають `MIT`, `Apache-2.0`, `GPL-3.0` та `BSD-3-Clause`.

## Платні аддони

Rayforge підтримує платні аддони через перевірку ліцензій Gumroad. Якщо ви хочете продавати свій аддон, ви можете налаштувати його на вимогу валідної ліцензії перед початком роботи.

### Базова конфігурація платного аддону

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

Коли `required` дорівнює true, Rayforge перевірить наявність валідної ліцензії перед завантаженням вашого аддону. `purchase_url` показується користувачам, які не мають ліцензії.

### Gumroad Product ID

Додайте ваш Gumroad Product ID для активації перевірки ліцензії:

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

Для кількох Product ID (наприклад, різні цінові рівні):

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### Повний приклад платного аддону

Ось повний маніфест для платного аддону:

```yaml
name: premium_laser_pack
display_name: "Premium Laser Pack"
description: "Advanced features for professional laser cutting."
api_version: 9
url: https://example.com/premium-laser-pack

author:
  name: Your Name
  email: you@example.com

depends:
  - rayforge>=0.27.0

provides:
  backend: premium_pack.backend
  frontend: premium_pack.frontend

license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/premium-laser-pack
  product_ids:
    - "standard_tier_id"
    - "pro_tier_id"
```

### Перевірка статусу ліцензії в коді

У коді вашого аддону ви можете перевірити, чи ліцензія валідна:

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Check if user has a valid license for your product
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Optionally show a message or limit functionality
            logger.warning("License not found - some features disabled")
```

## Правила валідації

Rayforge валідує ваш маніфест при завантаженні аддону. Ось правила:

Поле `name` має бути валідним Python-ідентифікатором (літери, цифри, підкреслення, без цифр на початку). Поле `api_version` має бути цілим числом від 1 до поточної версії. Поле `author.name` не може бути порожнім або містити текст-заглушку на кшталт "your-github-username". Точки входу мають бути валідними шляхами до модулів, і модулі мають існувати. Шляхи до ресурсів мають бути відносними (без `..` або початкового `/`), і файли мають існувати.

Якщо валідація не пройде, Rayforge залогує помилку та пропустить ваш аддон. Перевірте вивід консолі під час розробки, щоб виявити ці проблеми.
