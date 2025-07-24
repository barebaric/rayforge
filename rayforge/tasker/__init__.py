"""
Tasker package for managing tasks, contexts, and execution.
"""
from .manager import TaskManager
from .context import ExecutionContext, ExecutionContextProxy
from .task import Task, CancelledError

task_mgr = TaskManager()

__all__ = [
    "TaskManager",
    "ExecutionContext",
    "ExecutionContextProxy",
    "Task",
    "CancelledError",
]
