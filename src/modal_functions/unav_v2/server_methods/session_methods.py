from modal import method



def get_session(self, user_id: str) -> dict:
    """Get or create user session"""
    if user_id not in self.user_sessions:
        self.user_sessions[user_id] = {}
    return self.user_sessions[user_id]



def update_session(self, user_id: str, updates: dict):
    """Update user session with new data"""
    session = self.get_session(user_id)
    session.update(updates)



def _construct_mock_localization_output(
    self,
    x: float,
    y: float,
    angle: float,
    place: str,
    building: str,
    floor: str,
) -> dict:
    """
    Construct a mock localization output from user-provided coordinates.
    This allows skipping the actual localization phase when coordinates are known.

    Args:
        x: X coordinate on the floor plan
        y: Y coordinate on the floor plan
        angle: Heading angle (direction user is facing)
        place: Place name
        building: Building name
        floor: Floor name

    Returns:
        dict: Mock localization output matching the structure from localizer.localize()
    """
    return {
        "floorplan_pose": {"xy": [x, y], "ang": angle},
        "best_map_key": (place, building, floor),
        "refinement_queue": {},  # Empty since we're not doing actual localization
    }


@method()
def set_navigation_context(
    self,
    user_id: str,
    dest_id: str,
    target_place: str,
    target_building: str,
    target_floor: str,
    unit: str = "meter",
    language: str = "en",
):
    """Set navigation context for a user session"""
    try:
        self.update_session(
            user_id,
            {
                "selected_dest_id": dest_id,
                "target_place": target_place,
                "target_building": target_building,
                "target_floor": target_floor,
                "unit": unit,
                "language": language,
            },
        )

        return {
            "status": "success",
            "message": "Navigation context set successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "type": type(e).__name__}
