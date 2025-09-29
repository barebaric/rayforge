"""
Tasker package for managing tasks, contexts, and execution.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .manager import TaskManagerProxy
from .task import Task

if TYPE_CHECKING:
    from .manager import TaskManager

# This is the global, thread-safe, and process-safe singleton.
# It's a lightweight proxy that will create the real TaskManager on
# first use.
# We hint it as TaskManager so type checkers and IDEs provide
# correct autocompletion.
task_mgr: TaskManager = TaskManagerProxy()  # type: ignore


__all__ = [
    "TaskManager",
    "Task",
    "task_mgr",
]
