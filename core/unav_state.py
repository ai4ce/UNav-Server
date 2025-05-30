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

# Flatten PLACES dict for UNavConfig (as you already did)
place_names = list(PLACES.keys())
building_names = [b for p in PLACES.values() for b in p.keys()]
floor_names = [f for p in PLACES.values() for b in p.values() for f in b]

config = UNavConfig(
    data_final_root=DATA_ROOT,
    places=PLACES,
    global_descriptor_model=FEATURE_MODEL,
    local_feature_model=LOCAL_FEATURE_MODEL
)
localizor_config = config.localizer_config
navigator_config = config.navigator_config

# --- Global singletons for UNav algorithm modules ---
places = PLACES
localizer = UNavLocalizer(localizor_config)
localizer.load_maps_and_features()
nav = FacilityNavigator(navigator_config)
commander = commands_from_result

# --- In-memory user session store ---
user_sessions = {}
"""
user_sessions: dict
    Stores per-user state for refinement_queue and last localization result.
    Key: user_id (str)
    Value: dict (e.g. output from localization, incl. 'refinement_queue', 'floorplan_pose', etc.)
"""
