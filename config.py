# config.py
# Centralized server configuration file.
# All key parameters for server startup and runtime should be set here.
# Team members only need to modify this file to adjust deployment or environment settings.

# --- Path Configuration ---
DATA_ROOT = "/data"

# --- Model Settings ---
FEATURE_MODEL = "DinoV2Salad"
LOCAL_FEATURE_MODEL = "superpoint+lightglue"

# --- Place/Building/Floor Structure ---
# Add or modify locations, buildings, and available floors as needed.
PLACES = {
    "New_York_University": {
        "Langone": ["15_floor", "16_floor", "17_floor"],
        "Tandon": ["4_floor"]
    },
    "New_York_City": {
        "LightHouse": ["3_floor", "4_floor", "6_floor"]
    }
}
