from core.agent.tools.navigation_tools import build_navigation_state_snapshot


def build_explanation_context(user_id: str) -> dict:
    snapshot = build_navigation_state_snapshot(user_id)
    return snapshot.model_dump()
