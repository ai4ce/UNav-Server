from core.unav_state import get_session
from models.agent_schemas import NavigationStateSnapshot


def build_navigation_state_snapshot(user_id: str) -> NavigationStateSnapshot:
    session = get_session(user_id)
    return NavigationStateSnapshot(
        has_active_navigation=bool(session.get("selected_dest_id") is not None),
        destination_id=str(session.get("selected_dest_id")) if session.get("selected_dest_id") is not None else None,
        destination_name=session.get("selected_destination_name"),
        next_waypoint_name=None,
        distance_to_waypoint_m=None,
        heading_error_deg=None,
        off_route=False,
    )
