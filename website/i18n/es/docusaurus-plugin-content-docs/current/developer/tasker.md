# Tasker: Gestión de Tareas en Segundo Plano

`tasker` es un módulo para ejecutar tareas de larga duración en segundo plano de una aplicación GTK sin congelar la UI. Proporciona una API simple y unificada para trabajo limitado por I/O (`asyncio`) y limitado por CPU (`multiprocessing`).

## Conceptos Core

1. **`task_mgr`**: El proxy singleton global que usas para iniciar y cancelar todas las tareas
2. **`Task`**: Un objeto que representa un solo trabajo en segundo plano. Lo usas para rastrear el estado
3. **`ExecutionContext` (`context`)**: Un objeto pasado como primer argumento a tu función en segundo plano. Tu código lo usa para reportar progreso, enviar mensajes y verificar cancelación
4. **`TaskManagerProxy`**: Un proxy thread-safe que reenvía llamadas al TaskManager real ejecutándose en el hilo principal

## Inicio Rápido

Todas las tareas en segundo plano son gestionadas por el `task_mgr` global.

### Ejecutar una Tarea Limitada por I/O (ej., red, acceso a archivos)

Usa `add_coroutine` para funciones `async`. Son ligeras e ideales para tareas que esperan I/O.

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# Tu función en segundo plano DEBE aceptar `context` como primer argumento.
async def my_io_task(context, url):
    context.set_message("Descargando...")
    # ... realizar descarga asíncrona ...
    await asyncio.sleep(2) # Simular trabajo
    context.set_progress(1.0)
    context.set_message("¡Descarga completa!")

# Iniciar la tarea desde tu código UI (ej., un clic de botón)
task_mgr.add_coroutine(my_io_task, "http://example.com", key="downloader")
```

### Ejecutar una Tarea Limitada por CPU (ej., cálculo pesado)

Usa `run_process` para funciones regulares. Estas se ejecutan en un proceso separado para evitar el GIL y mantener la UI responsiva.

```python
import time
from rayforge.shared.tasker import task_mgr

# Una función regular, no async.
def my_cpu_task(context, iterations):
    context.set_total(iterations)
    context.set_message("Calculando...")
    for i in range(iterations):
        # ... realizar cálculo pesado ...
        time.sleep(0.1) # Simular trabajo
        context.set_progress(i + 1)
    return "Resultado Final"

# Iniciar la tarea
task_mgr.run_process(my_cpu_task, 50, key="calculator")
```

### Ejecutar una Tarea en un Hilo

Usa `run_thread` para tareas que deberían ejecutarse en un hilo pero no requieren el aislamiento completo de proceso. Esto es útil para tareas que comparten memoria pero aún no deberían bloquear la UI.

```python
import time
from rayforge.shared.tasker import task_mgr

# Una función regular que se ejecutará en un hilo
def my_thread_task(context, duration):
    context.set_message("Trabajando en hilo...")
    time.sleep(duration) # Simular trabajo
    context.set_progress(1.0)
    return "Tarea de hilo completa"

# Iniciar la tarea en un hilo
task_mgr.run_thread(my_thread_task, 2, key="thread_worker")
```

## Patrones Esenciales

### Actualizar la UI

Conéctate a la señal `tasks_updated` para reaccionar a cambios. El handler será llamado de forma segura en el hilo principal de GTK.

```python
def setup_ui(progress_bar, status_label):
    # Este handler actualiza la UI basándose en el progreso general
    def on_tasks_updated(sender, tasks, progress):
        progress_bar.set_fraction(progress)
        if tasks:
            status_label.set_text(tasks[-1].get_message() or "Trabajando...")
        else:
            status_label.set_text("Inactivo")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# Más tarde en tu UI...
# setup_ui(my_progress_bar, my_label)
```

### Cancelación

Dale a tus tareas una `key` para cancelarlas más tarde. Tu función en segundo plano debería verificar periódicamente `context.is_cancelled()`.

```python
# En tu función en segundo plano:
if context.is_cancelled():
    print("La tarea fue cancelada, deteniendo trabajo.")
    return

# En tu código UI:
task_mgr.cancel_task("calculator")
```

### Manejar Completado

Usa el callback `when_done` para obtener el resultado o ver si ocurrió un error.

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        print(f"Tarea terminada con resultado: {task.result()}")
    elif task.get_status() == 'failed':
        print(f"Tarea fallida: {task._task_exception}")

task_mgr.run_process(my_cpu_task, 10, when_done=on_task_finished)
```

## Referencia de API

### `task_mgr` (El Proxy Manager)

- `add_coroutine(coro, *args, key=None, when_done=None)`: Añadir una tarea basada en asyncio
- `run_process(func, *args, key=None, when_done=None, when_event=None)`: Ejecutar una tarea limitada por CPU en un proceso separado
- `run_thread(func, *args, key=None, when_done=None)`: Ejecutar una tarea en un hilo (comparte memoria con el proceso principal)
- `cancel_task(key)`: Cancelar una tarea en ejecución por su key
- `tasks_updated` (señal para actualizaciones UI): Emitida cuando el estado de la tarea cambia

### `context` (Dentro de tu función en segundo plano)

- `set_progress(value)`: Reportar progreso actual (ej., `i + 1`)
- `set_total(total)`: Establecer el valor máximo para `set_progress`
- `set_message("...")`: Actualizar el texto de estado
- `is_cancelled()`: Verificar si deberías detener
- `sub_context(...)`: Crear una sub-tarea para operaciones multi-etapa
- `send_event("name", data)`: (Solo proceso) Enviar datos personalizados de vuelta a la UI
- `flush()`: Enviar inmediatamente cualquier actualización pendiente a la UI

## Uso en Rayforge

El tasker se usa en todo Rayforge para:

- **Procesamiento de pipeline**: Ejecutar el pipeline del documento en segundo plano
- **Operaciones de archivo**: Importar y exportar archivos sin bloquear la UI
- **Comunicación con dispositivo**: Gestionar operaciones de larga duración con cortadoras láser
- **Procesamiento de imagen**: Realizar trazado y procesamiento de imagen intensivo en CPU

Al trabajar con el tasker en Rayforge, siempre asegúrate que tus funciones en segundo plano manejen correctamente la cancelación y proporcionen actualizaciones de progreso significativas para mantener una experiencia de usuario responsiva.
