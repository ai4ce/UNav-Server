# core/tasks/general.py
# General session/context/unit management tasks

from core.unav_state import user_sessions

def select_unit(inputs):
    """
    Store user's preferred unit (feet/meters) for navigation instructions.
    Inputs: unit, user_id
    """
    user_id = inputs["user_id"]
    unit = inputs["unit"]
    session = user_sessions.setdefault(user_id, {})
    session["unit"] = unit
    return {"success": True}

# ...未来可扩展其它session/用户管理相关通用方法

GENERAL_TASKS = {
    "select_unit": select_unit,
    # e.g. "reset_session": reset_session, ...
}
