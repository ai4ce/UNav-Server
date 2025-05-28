#!/bin/bash

# ===== UNav Server Configuration =====
export DATA_FINAL_ROOT="/mnt/data/UNav-IO/data"
export DATA_TEMP_ROOT="/mnt/data/UNav-IO/temp"
export FEATURE_MODEL="DinoV2Salad"
export LOCAL_FEATURE_MODEL="superpoint+lightglue"
export PLACES="New_York_City"
export BUILDINGS="LightHouse"
export FLOORS="3_floor,4_floor,6_floor"

echo "=== UNav Server Configuration ==="
echo "DATA_FINAL_ROOT = $DATA_FINAL_ROOT"
echo "DATA_TEMP_ROOT = $DATA_TEMP_ROOT"
echo "FEATURE_MODEL = $FEATURE_MODEL"
echo "LOCAL_FEATURE_MODEL = $LOCAL_FEATURE_MODEL"
echo "PLACES = $PLACES"
echo "BUILDINGS = $BUILDINGS"
echo "FLOORS = $FLOORS"
echo

# ===== Launch FastAPI Server =====
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
