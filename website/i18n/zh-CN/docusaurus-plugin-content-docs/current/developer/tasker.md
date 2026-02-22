# Tasker：后台任务管理

`tasker` 是一个用于在 GTK 应用程序后台运行长时间任务而不冻结 UI 的模块。它为 I/O 密集型（`asyncio`）和 CPU 密集型（`multiprocessing`）工作提供了简单统一的 API。

## 核心概念

1. **`task_mgr`**：用于启动和取消所有任务的全局单例代理
2. **`Task`**：表示单个后台作业的对象。您用它来跟踪状态
3. **`ExecutionContext`（`context`）**：作为第一个参数传递给后台函数的对象。您的代码使用它来报告进度、发送消息和检查取消
4. **`TaskManagerProxy`**：一个线程安全代理，将调用转发到主线程中运行的真正 TaskManager

## 快速入门

所有后台任务由全局 `task_mgr` 管理。

### 运行 I/O 密集型任务（例如网络、文件访问）

对 `async` 函数使用 `add_coroutine`。这些是轻量级的，适合等待 I/O 的任务。

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# 您的后台函数必须接受 `context` 作为第一个参数。
async def my_io_task(context, url):
    context.set_message("正在下载...")
    # ... 执行异步下载 ...
    await asyncio.sleep(2) # 模拟工作
    context.set_progress(1.0)
    context.set_message("下载完成！")

# 从您的 UI 代码启动任务（例如按钮点击）
task_mgr.add_coroutine(my_io_task, "http://example.com", key="downloader")
```

### 运行 CPU 密集型任务（例如繁重计算）

对常规函数使用 `run_process`。这些在单独的进程中运行以避免 GIL 并保持 UI 响应。

```python
import time
from rayforge.shared.tasker import task_mgr

# 常规函数，不是 async。
def my_cpu_task(context, iterations):
    context.set_total(iterations)
    context.set_message("正在计算...")
    for i in range(iterations):
        # ... 执行繁重计算 ...
        time.sleep(0.1) # 模拟工作
        context.set_progress(i + 1)
    return "最终结果"

# 启动任务
task_mgr.run_process(my_cpu_task, 50, key="calculator")
```

### 运行线程绑定任务

对应该在线程中运行但不需要完整进程隔离的任务使用 `run_thread`。这对于共享内存但仍不应阻塞 UI 的任务很有用。

```python
import time
from rayforge.shared.tasker import task_mgr

# 将在线程中运行的常规函数
def my_thread_task(context, duration):
    context.set_message("在线程中工作...")
    time.sleep(duration) # 模拟工作
    context.set_progress(1.0)
    return "线程任务完成"

# 在线程中启动任务
task_mgr.run_thread(my_thread_task, 2, key="thread_worker")
```

## 基本模式

### 更新 UI

连接到 `tasks_updated` 信号以响应变化。处理程序将在主 GTK 线程上安全调用。

```python
def setup_ui(progress_bar, status_label):
    # 此处理程序根据总体进度更新 UI
    def on_tasks_updated(sender, tasks, progress):
        progress_bar.set_fraction(progress)
        if tasks:
            status_label.set_text(tasks[-1].get_message() or "工作中...")
        else:
            status_label.set_text("空闲")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# 稍后在您的 UI 中...
# setup_ui(my_progress_bar, my_label)
```

### 取消

为您的任务提供一个 `key` 以便稍后取消。您的后台函数应定期检查 `context.is_cancelled()`。

```python
# 在您的后台函数中：
if context.is_cancelled():
    print("任务已取消，停止工作。")
    return

# 在您的 UI 代码中：
task_mgr.cancel_task("calculator")
```

### 处理完成

使用 `when_done` 回调获取结果或查看是否发生错误。

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        print(f"任务完成，结果：{task.result()}")
    elif task.get_status() == 'failed':
        print(f"任务失败：{task._task_exception}")

task_mgr.run_process(my_cpu_task, 10, when_done=on_task_finished)
```

## API 参考

### `task_mgr`（管理器代理）

- `add_coroutine(coro, *args, key=None, when_done=None)`：添加基于 asyncio 的任务
- `run_process(func, *args, key=None, when_done=None, when_event=None)`：在单独的进程中运行 CPU 密集型任务
- `run_thread(func, *args, key=None, when_done=None)`：在线程中运行任务（与主进程共享内存）
- `cancel_task(key)`：通过 key 取消正在运行的任务
- `tasks_updated`（UI 更新信号）：任务状态变化时发出

### `context`（在您的后台函数内部）

- `set_progress(value)`：报告当前进度（例如 `i + 1`）
- `set_total(total)`：设置 `set_progress` 的最大值
- `set_message("...")`：更新状态文本
- `is_cancelled()`：检查是否应该停止
- `sub_context(...)`：为多阶段操作创建子任务
- `send_event("name", data)`：（仅进程）将自定义数据发送回 UI
- `flush()`：立即将任何待处理的更新发送到 UI

## 在 Rayforge 中的使用

Tasker 在整个 Rayforge 中用于：

- **流水线处理**：在后台运行文档流水线
- **文件操作**：导入和导出文件而不阻塞 UI
- **设备通信**：管理与激光切割机的长时间操作
- **图像处理**：执行 CPU 密集型的图像描摹和处理

在 Rayforge 中使用 tasker 时，请始终确保您的后台函数正确处理取消并提供有意义的进度更新，以保持响应式用户体验。
