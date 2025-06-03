# core/tasks/general.py
# General-purpose session, context, and user management task definitions for the UNav system.

from core.unav_state import get_session

def select_unit(inputs):
    """
    Store the user's preferred unit for navigation instructions (e.g., 'feet' or 'meters').
    """
    user_id = inputs["user_id"]
    unit = inputs["unit"]
    session = get_session(user_id)
    session["unit"] = unit
    return {"success": True}

def select_language(inputs):
    """
    Store the user's preferred language for interface and TTS prompts.

    Args:
        inputs (dict): {
            "user_id": str,       # Unique user identifier
            "language": str       # Preferred language code ("en", "zh", "th", etc.)
        }

    Returns:
        dict: {"success": True}
    """
    user_id = inputs["user_id"]
    language = inputs["language"]
    session = get_session(user_id)
    session["language"] = language
    return {"success": True}

GENERAL_TASKS = {
    "select_unit": select_unit,
    "select_language": select_language,
    # "reset_session": reset_session,      # Example for future extensibility
}
