from core.unav_state import get_session


def build_session_context(user_id: str) -> dict:
    session = get_session(user_id)
    return {
        "current_place": session.get("current_place"),
        "current_building": session.get("current_building"),
        "current_floor": session.get("current_floor"),
        "target_place": session.get("target_place"),
        "target_building": session.get("target_building"),
        "target_floor": session.get("target_floor"),
        "selected_dest_id": session.get("selected_dest_id"),
        "language": session.get("language", "en"),
        "unit": session.get("unit", "meters"),
        "turn_mode": session.get("turn_mode", "default"),
    }
