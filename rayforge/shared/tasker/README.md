You're right. A good README should be concise and get to the point quickly. Here is a much shorter version that focuses on the essentials.

---

# Tasker: Background Task Management for Python/GTK

`tasker` is a module for running long-running tasks in the background of a GTK application without freezing the UI. It provides a simple, unified API for both I/O-bound (`asyncio`) and CPU-bound (`multiprocessing`) work.

## Core Concepts

1.  **`task_mgr`**: The global singleton you use to start and cancel all tasks.
2.  **`Task`**: An object representing a single background job. You use it to track status.
3.  **`ExecutionContext` (`context`)**: An object passed as the first argument to your background function. Your code uses it to report progress, send messages, and check for cancellation.

## Quick Start

All background tasks are managed by the global `task_mgr`.

### 1. Running an I/O-Bound Task (e.g., network, file access)

Use `add_coroutine` for `async` functions. These are lightweight and ideal for tasks that wait for I/O.

```python
import asyncio
from your_project.tasker import task_mgr

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

### 2. Running a CPU-Bound Task (e.g., heavy computation)

Use `run_process` for regular functions. These run in a separate process to avoid the GIL and keep the UI responsive.

```python
import time
from your_project.tasker import task_mgr

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

## API Cheat Sheet

### `task_mgr` (The Manager)
- `add_coroutine(coro, *args, key=None, when_done=None)`
- `run_process(func, *args, key=None, when_done=None, when_event=None)`
- `cancel_task(key)`
- `tasks_updated` (signal for UI updates)

### `context` (Inside your background function)
- `set_progress(value)`: Report current progress (e.g., `i + 1`).
- `set_total(total)`: Set the max value for `set_progress`.
- `set_message("...")`: Update the status text.
- `is_cancelled()`: Check if you should stop.
- `sub_context(...)`: Create a sub-task for multi-stage operations.
- `send_event("name", data)`: (Process-only) Send custom data back to the UI.
