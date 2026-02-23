# Tasker: Керування фоновими завданнями

`tasker` - це модуль для запуску довготривалих завдань у фоні GTK-додатку без заморожування UI. Він надає простий, уніфікований API як для I/O-зв'язаних (`asyncio`), так і для CPU-зв'язаних (`multiprocessing`) завдань.

## Основні концепції

1. **`task_mgr`**: Глобальний синглтон-проксі, який ви використовуєте для запуску та скасування всіх завдань
2. **`Task`**: Об'єкт, що представляє одне фонове завдання. Ви використовуєте його для відстеження статусу
3. **`ExecutionContext` (`context`)**: Об'єкт, що передається як перший аргумент у вашу фонову функцію. Ваш код використовує його для звітування прогресу, надсилання повідомлень та перевірки скасування
4. **`TaskManagerProxy`**: Потокобезпечний проксі, що пересилає виклики до фактичного TaskManager, що працює в головному потоці

## Швидкий старт

Всі фонові завдання керуються глобальним `task_mgr`.

### Запуск I/O-зв'язаного завдання (наприклад, мережа, доступ до файлів)

Використовуйте `add_coroutine` для `async` функцій. Вони легковагі та ідеальні для завдань, що очікують на I/O.

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# Ваша фоновая функція ПОВИННА приймати `context` як перший аргумент.
async def my_io_task(context, url):
    context.set_message("Завантаження...")
    # ... виконати асинхронне завантаження ...
    await asyncio.sleep(2) # Симуляція роботи
    context.set_progress(1.0)
    context.set_message("Завантаження завершено!")

# Запустіть завдання з вашого UI коду (наприклад, клік кнопки)
task_mgr.add_coroutine(my_io_task, "http://example.com", key="downloader")
```

### Запуск CPU-зв'язаного завдання (наприклад, важкі обчислення)

Використовуйте `run_process` для звичайних функцій. Вони виконуються в окремому процесі щоб уникнути GIL та зберегти UI відгуковим.

```python
import time
from rayforge.shared.tasker import task_mgr

# Звичайна функція, не async.
def my_cpu_task(context, iterations):
    context.set_total(iterations)
    context.set_message("Обчислення...")
    for i in range(iterations):
        # ... виконати важкі обчислення ...
        time.sleep(0.1) # Симуляція роботи
        context.set_progress(i + 1)
    return "Фінальний результат"

# Запустіть завдання
task_mgr.run_process(my_cpu_task, 50, key="calculator")
```

### Запуск потокового завдання

Використовуйте `run_thread` для завдань, які мають виконуватися в потоці, але не потребують повної ізоляції процесу. Це корисно для завдань, що спільно використовують пам'ять, але все ще не повинні блокувати UI.

```python
import time
from rayforge.shared.tasker import task_mgr

# Звичайна функція, яка виконуватиметься в потоці
def my_thread_task(context, duration):
    context.set_message("Робота в потоці...")
    time.sleep(duration) # Симуляція роботи
    context.set_progress(1.0)
    return "Потокове завдання завершено"

# Запустіть завдання в потоці
task_mgr.run_thread(my_thread_task, 2, key="thread_worker")
```

## Основні патерни

### Оновлення UI

Підключіться до сигналу `tasks_updated` щоб реагувати на зміни. Обробник буде безпечно викликаний в головному GTK потоці.

```python
def setup_ui(progress_bar, status_label):
    # Цей обробник оновлює UI на основі загального прогресу
    def on_tasks_updated(sender, tasks, progress):
        progress_bar.set_fraction(progress)
        if tasks:
            status_label.set_text(tasks[-1].get_message() or "Працює...")
        else:
            status_label.set_text("Очікування")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# Пізніше у вашому UI...
# setup_ui(my_progress_bar, my_label)
```

### Скасування

Давайте вашим завданням `key` щоб скасувати їх пізніше. Ваша фоновая функція повинна періодично перевіряти `context.is_cancelled()`.

```python
# У вашій фоновій функції:
if context.is_cancelled():
    print("Завдання скасовано, зупиняю роботу.")
    return

# У вашому UI коді:
task_mgr.cancel_task("calculator")
```

### Обробка завершення

Використовуйте зворотний виклик `when_done` щоб отримати результат або побачити, чи сталася помилка.

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        print(f"Завдання завершено з результатом: {task.result()}")
    elif task.get_status() == 'failed':
        print(f"Завдання не вдалося: {task._task_exception}")

task_mgr.run_process(my_cpu_task, 10, when_done=on_task_finished)
```

## Довідник API

### `task_mgr` (Проксі менеджера)

- `add_coroutine(coro, *args, key=None, when_done=None)`: Додати завдання на основі asyncio
- `run_process(func, *args, key=None, when_done=None, when_event=None)`: Запустити CPU-зв'язане завдання в окремому процесі
- `run_thread(func, *args, key=None, when_done=None)`: Запустити завдання в потоці (спільно використовує пам'ять з основним процесом)
- `cancel_task(key)`: Скасувати завдання, що виконується, за його ключем
- `tasks_updated` (сигнал для оновлень UI): Емітується коли статус завдання змінюється

### `context` (Всередині вашої фонової функції)

- `set_progress(value)`: Повідомити поточний прогрес (наприклад, `i + 1`)
- `set_total(total)`: Встановити максимальне значення для `set_progress`
- `set_message("...")`: Оновити текст статусу
- `is_cancelled()`: Перевірити, чи варто зупинитися
- `sub_context(...)`: Створити підзавдання для багатостадійних операцій
- `send_event("name", data)`: (Тільки для процесів) Надіслати власні дані назад до UI
- `flush()`: Негайно надіслати будь-які очікувальні оновлення до UI

## Використання в Rayforge

Tasker використовується по всьому Rayforge для:

- **Обробки конвеєра**: Запуск конвеєра документа у фоні
- **Файлових операцій**: Імпорт та експорт файлів без блокування UI
- **Комунікації з пристроєм**: Керування довготривалими операціями з лазерними різаками
- **Обробки зображень**: Виконання CPU-інтенсивного трасування та обробки зображень

При роботі з tasker в Rayforge завжди переконайтеся, що ваші фонові функції правильно обробляють скасування та надають значущі оновлення прогресу для підтримки відгукового користувацького досвіду.
