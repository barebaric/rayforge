# Tasker: Background Task Management

`tasker` is a module for running long-running tasks in the background of a GTK application without freezing the UI. It provides a simple, unified API for both I/O-bound (`asyncio`) and CPU-bound (`multiprocessing`) work.

## Core Concepts

1. **`task_mgr`**: The global singleton proxy you use to start and cancel all tasks
2. **`Task`**: An object representing a single background job. You use it to track status
3. **`ExecutionContext` (`context`)**: An object passed as the first argument to your background function. Your code uses it to report progress, send messages, and check for cancellation
4. **`TaskManagerProxy`**: A thread-safe proxy that forwards calls to the actual TaskManager running in the main thread

## Quick Start

All background tasks are managed by the global `task_mgr`.

### Running an I/O-Bound Task (e.g., network, file access)

Use `add_coroutine` for `async` functions. These are lightweight and ideal for tasks that wait for I/O.

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# Your background function MUST accept `context` as the first argument.
async def my_io_task(context, url):
    context.set_message("Downloading...")
    # ... perform async download ...
    await asyncio.sleep(2) # Simulate work
    context.set_progress(1.0)
    context.set_message("Download complete!")

# Start the task from your UI code (e.g., a button click)
task_mgr.add_coroutine(my_io_task, "http://example.com", key="downloader")
```

### Running a CPU-Bound Task (e.g., heavy computation)

Use `run_process` for regular functions. These run in a separate process to avoid the GIL and keep the UI responsive.

```python
import time
from rayforge.shared.tasker import task_mgr

# A regular function, not async.
def my_cpu_task(context, iterations):
    context.set_total(iterations)
    context.set_message("Calculating...")
    for i in range(iterations):
        # ... perform heavy calculation ...
        time.sleep(0.1) # Simulate work
        context.set_progress(i + 1)
    return "Final Result"

# Start the task
task_mgr.run_process(my_cpu_task, 50, key="calculator")
```

### Running a Thread-Bound Task

Use `run_thread` for tasks that should run in a thread but don't require the full process isolation. This is useful for tasks that share memory but still shouldn't block the UI.

```python
import time
from rayforge.shared.tasker import task_mgr

# A regular function that will run in a thread
def my_thread_task(context, duration):
    context.set_message("Working in thread...")
    time.sleep(duration) # Simulate work
    context.set_progress(1.0)
    return "Thread task complete"

# Start the task in a thread
task_mgr.run_thread(my_thread_task, 2, key="thread_worker")
```

## Essential Patterns

### Updating the UI

Connect to the `tasks_updated` signal to react to changes. The handler will be safely called on the main GTK thread.

```python
def setup_ui(progress_bar, status_label):
    # This handler updates the UI based on the overall progress
    def on_tasks_updated(sender, tasks, progress):
        progress_bar.set_fraction(progress)
        if tasks:
            status_label.set_text(tasks[-1].get_message() or "Working...")
        else:
            status_label.set_text("Idle")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# Later in your UI...
# setup_ui(my_progress_bar, my_label)
```

### Cancellation

Give your tasks a `key` to cancel them later. Your background function should periodically check `context.is_cancelled()`.

```python
# In your background function:
if context.is_cancelled():
    print("Task was cancelled, stopping work.")
    return

# In your UI code:
task_mgr.cancel_task("calculator")
```

### Handling Completion

Use the `when_done` callback to get the result or see if an error occurred.

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        print(f"Task finished with result: {task.result()}")
    elif task.get_status() == 'failed':
        print(f"Task failed: {task._task_exception}")

task_mgr.run_process(my_cpu_task, 10, when_done=on_task_finished)
```

## API Reference

### `task_mgr` (The Manager Proxy)

- `add_coroutine(coro, *args, key=None, when_done=None)`: Add an asyncio-based task
- `run_process(func, *args, key=None, when_done=None, when_event=None)`: Run a CPU-bound task in a separate process
- `run_thread(func, *args, key=None, when_done=None)`: Run a task in a thread (shares memory with main process)
- `cancel_task(key)`: Cancel a running task by its key
- `tasks_updated` (signal for UI updates): Emitted when task status changes

### `context` (Inside your background function)

- `set_progress(value)`: Report current progress (e.g., `i + 1`)
- `set_total(total)`: Set the max value for `set_progress`
- `set_message("...")`: Update the status text
- `is_cancelled()`: Check if you should stop
- `sub_context(...)`: Create a sub-task for multi-stage operations
- `send_event("name", data)`: (Process-only) Send custom data back to the UI
- `flush()`: Immediately send any pending updates to the UI

## Usage in Rayforge

The tasker is used throughout Rayforge for:

- **Pipeline processing**: Running the document pipeline in the background
- **File operations**: Importing and exporting files without blocking the UI
- **Device communication**: Managing long-running operations with laser cutters
- **Image processing**: Performing CPU-intensive image tracing and processing

When working with the tasker in Rayforge, always ensure your background functions properly handle cancellation and provide meaningful progress updates to maintain a responsive user experience.