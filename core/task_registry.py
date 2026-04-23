# core/task_registry.py

from core.tasks.general import GENERAL_TASKS
from core.tasks.unav import UNAV_TASKS
from core.tasks.agent import AGENT_TASKS
from core.tasks.debug import DEBUG_TASKS

TASKS = {}
TASKS.update(GENERAL_TASKS)
TASKS.update(UNAV_TASKS)
TASKS.update(AGENT_TASKS)
TASKS.update(DEBUG_TASKS)


def get_task(name):
    return TASKS.get(name)
