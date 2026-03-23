from core.agent.policy import sanitize_preference_patch
from core.agent.profile_store import ProfileStore

store = ProfileStore()


def get_profile(user_id: str):
    return store.get_user_profile(user_id)


def update_profile(user_id: str, patch: dict):
    return store.upsert_user_profile(user_id, sanitize_preference_patch(patch))
