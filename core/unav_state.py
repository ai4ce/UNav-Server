import time
import asyncio
from config import (
    DATA_ROOT,
    FEATURE_MODEL,
    LOCAL_FEATURE_MODEL,
    PLACES
)
from unav.config import UNavConfig
from unav.localizer.localizer import UNavLocalizer
from unav.navigator.multifloor import FacilityNavigator
from unav.navigator.commander import commands_from_result
from unav.navigator.commander import I18NLabels
from core.i18n_labels import init_labels, get_all_labels

SESSION_TIMEOUT_SECONDS = 30 * 60  # 30 minutes

# Flatten PLACES dict to extract place, building, and floor lists for configuration if needed
place_names = list(PLACES.keys())
building_names = [b for p in PLACES.values() for b in p.keys()]
floor_names = [f for p in PLACES.values() for b in p.values() for f in b]

# Initialize UNav configuration with all necessary parameters
config = UNavConfig(
    data_final_root=DATA_ROOT,
    places=PLACES,  # Keep hierarchical structure for flexible access
    global_descriptor_model=FEATURE_MODEL,
    local_feature_model=LOCAL_FEATURE_MODEL
)

# Extract specific sub-configs for localization and navigation modules
localizor_config = config.localizer_config
navigator_config = config.navigator_config

# Initialize labels.json path for i18n
init_labels(DATA_ROOT) # labels at <DATA_ROOT>/_i18n/labels.json
LABELS = I18NLabels(payload=get_all_labels())

# Initialize global singletons for UNav algorithm modules
places = PLACES  # Global place/building/floor info

localizer = UNavLocalizer(localizor_config)
localizer.load_maps_and_features()  # Preload all maps and features for fast localization

nav = FacilityNavigator(navigator_config)  # Initialize multi-floor navigator instance

commander = commands_from_result  # Navigation command generator function

# In-memory per-user session store
user_sessions = {}
"""
user_sessions: dict
    Maintains the stateful session information per user in memory.
    Key: user_id (str) — Unique identifier for each user.
    Value: dict — User-specific data including:
        - 'refinement_queue': dict, historical data for pose refinement
        - 'floorplan_pose': dict, latest estimated user pose on floorplan
        - 'current_place', 'current_building', 'current_floor': str, current floor context
        - 'selected_dest_id': str or int, currently chosen destination identifier
        - 'target_place', 'target_building', 'target_floor': str, destination floor context
        - 'unit': str, preferred unit for navigation commands (e.g., 'feet' or 'meters')
    Note: This store is volatile and will reset on server restart. Consider persistent storage if needed.
"""

def get_session(user_id):
    """Get or create user session, and update last_active timestamp."""
    now = time.time()
    if user_id not in user_sessions:
        user_sessions[user_id] = {"data": {}, "last_active": now}
    else:
        user_sessions[user_id]["last_active"] = now
    return user_sessions[user_id]["data"]

async def cleanup_sessions():
    """Background task to remove expired sessions."""
    while True:
        now = time.time()
        expired = []
        for user_id, info in user_sessions.items():
            if now - info.get("last_active", 0) > SESSION_TIMEOUT_SECONDS:
                expired.append(user_id)
        for user_id in expired:
            print(f"[Session Cleanup] Removing expired session for user_id: {user_id}")
            user_sessions.pop(user_id, None)
        await asyncio.sleep(300)
