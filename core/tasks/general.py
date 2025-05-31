# core/tasks/general.py
# General-purpose session, context, and user management task definitions for the UNav system.

from core.unav_state import get_session

def select_unit(inputs):
    """
    Store the user's preferred unit for navigation instructions (e.g., 'feet' or 'meters').

    Args:
        inputs (dict): {
            "user_id": str,     # Unique user identifier
            "unit": str         # Preferred unit ("feet" or "meters")
        }

    Returns:
        dict: {"success": True}
    """
    user_id = inputs["user_id"]
    unit = inputs["unit"]
    session = get_session(user_id)
    session["unit"] = unit
    return {"success": True}

# --- Placeholder for future extensible session or user management tasks ---
# Example: reset_session, set_language, etc.

GENERAL_TASKS = {
    "select_unit": select_unit,
    # "reset_session": reset_session,      # Example for future extensibility
}
