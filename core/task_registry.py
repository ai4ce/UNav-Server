# core/task_registry.py

from core.tasks.general import GENERAL_TASKS
from core.tasks.unav import UNAV_TASKS

TASKS = {}
TASKS.update(GENERAL_TASKS)
TASKS.update(UNAV_TASKS)

def get_task(name):
    return TASKS.get(name)
