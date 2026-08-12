"""
Tasker package for managing tasks, contexts, and execution.
"""

from .manager import TaskManager, TaskManagerProxy
from .task import Task

# This is the global, thread-safe, and process-safe singleton.
# It's a lightweight proxy that will create the real TaskManager on
# first use.
# We hint it as TaskManager so type checkers and IDEs provide
# correct autocompletion.
task_mgr: TaskManager = TaskManagerProxy()  # type: ignore


__all__ = [
    "Task",
    "TaskManager",
    "task_mgr",
]
