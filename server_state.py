from unav.config import UNavConfig
from unav.localizer.localizer import UNavLocalizer
from unav.navigator.multifloor import FacilityNavigator

# --------------------------
# Data and Model Parameters
# --------------------------

DATA_FINAL_ROOT = "/mnt/data/UNav-IO/data"
DATA_TEMP_ROOT = "/mnt/data/UNav-IO/temp"
FEATURE_MODEL = "DinoV2Salad"
LOCAL_FEATURE_MODEL = "superpoint+lightglue"
PLACES = ["New_York_City"]
BUILDINGS = ["LightHouse"]
FLOORS = ["3_floor", "4_floor", "6_floor"]

# -----------------------------------
# UNavConfig: Centralized Configuration
# -----------------------------------

config = UNavConfig(
    data_final_root=DATA_FINAL_ROOT,
    places=PLACES,
    buildings=BUILDINGS,
    floors=FLOORS,
    global_descriptor_model=FEATURE_MODEL,
    local_feature_model=LOCAL_FEATURE_MODEL
)
localizor_config = config.localizer_config

# -----------------------------------------------------
# Global Singletons: Algorithm Modules (Initialized Once)
# -----------------------------------------------------

localizer = UNavLocalizer(localizor_config)
nav = FacilityNavigator(localizor_config)

# ---------------------------------------------------
# User Sessions: Stores Each User's Refinement Queue
# and Most Recent Localization Results in Memory
# ---------------------------------------------------

# Maps user_id (str) -> dict containing refinement_queue and last localization output
user_sessions = {}

