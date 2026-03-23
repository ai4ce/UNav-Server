from core.agent.runtime import AgentRuntime

runtime = AgentRuntime()


def agent_interpret_destination(inputs):
    utterance = str(inputs.get("utterance", "")).strip()
    if not utterance:
        return {"error": "Missing utterance."}
    return runtime.interpret_destination(
        user_id=inputs["user_id"],
        utterance=utterance,
        language=inputs.get("language"),
    )


def agent_resolve_destination(inputs):
    destination_query = inputs.get("destination_query")
    if not isinstance(destination_query, dict):
        return {"error": "Missing or invalid destination_query."}
    return runtime.resolve_destination(
        user_id=inputs["user_id"],
        destination_query=destination_query,
        response_language=inputs.get("response_language"),
    )


def agent_followup_destination(inputs):
    utterance = str(inputs.get("utterance", "")).strip()
    candidates = inputs.get("candidates")
    if not utterance:
        return {"error": "Missing utterance."}
    if not isinstance(candidates, list):
        return {"error": "Missing or invalid candidates."}
    return runtime.followup_destination(
        user_id=inputs["user_id"],
        utterance=utterance,
        candidates=candidates,
        response_language=inputs.get("response_language"),
    )


def agent_adjust_preferences(inputs):
    utterance = str(inputs.get("utterance", "")).strip()
    if not utterance:
        return {"error": "Missing utterance."}
    return runtime.adjust_preferences(
        user_id=inputs["user_id"],
        utterance=utterance,
    )


def agent_explain_navigation_state(inputs):
    question = str(inputs.get("question", "")).strip()
    if not question:
        return {"error": "Missing question."}
    return runtime.explain_navigation_state(
        user_id=inputs["user_id"],
        question=question,
    )


def agent_reset_session_context(inputs):
    user_id = inputs["user_id"]
    from core.unav_state import get_session

    session = get_session(user_id)
    for key in [
        "refinement_queue",
        "floorplan_pose",
        "current_place",
        "current_building",
        "current_floor",
        "selected_dest_id",
        "target_place",
        "target_building",
        "target_floor",
    ]:
        session.pop(key, None)
    return {"success": True, "message": "Smart destination context cleared."}


AGENT_TASKS = {
    "agent_interpret_destination": agent_interpret_destination,
    "agent_resolve_destination": agent_resolve_destination,
    "agent_followup_destination": agent_followup_destination,
    "agent_adjust_preferences": agent_adjust_preferences,
    "agent_explain_navigation_state": agent_explain_navigation_state,
    "agent_reset_session_context": agent_reset_session_context,
}
