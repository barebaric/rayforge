"""
Tasker package for managing tasks, contexts, and execution.
"""
from .manager import TaskManager

# Initialize the task manager. This is a singleton that manages all tasks.
# It is initialized here to ensure it is available globally.
task_mgr = TaskManager()
