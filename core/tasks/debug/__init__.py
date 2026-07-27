from core.tasks.debug.localize_only import DEBUG_TASKS as _L1
from core.tasks.debug.localize_verbose import DEBUG_TASKS as _L2
from core.tasks.debug.localize_deep import DEBUG_TASKS as _L3
DEBUG_TASKS = {}
DEBUG_TASKS.update(_L1)
DEBUG_TASKS.update(_L2)
DEBUG_TASKS.update(_L3)
