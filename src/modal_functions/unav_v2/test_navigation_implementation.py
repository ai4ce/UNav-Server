#!/usr/bin/env python3
"""
Test script to verify the navigation implementation logic.
This simulates the navigation flow without requiring the actual UNav dependencies.
"""

import numpy as np
import json


class MockLocalizer:
    """Mock localizer for testing"""

    def localize(self, image, refinement_queue, top_k=None):
        return {
            "floorplan_pose": {"xy": [10.5, 20.3], "ang": 45.0},
            "best_map_key": ("New_York_City", "LightHouse", "6_floor"),
            "refinement_queue": {"mock": "queue"},
        }


class MockNavigator:
    """Mock navigator for testing"""

    def find_path(
        self,
        start_place,
        start_building,
        start_floor,
        start_xy,
        target_place,
        target_building,
        target_floor,
        dest_id,
    ):
        return {
            "path": [[10.5, 20.3], [15.0, 25.0], [20.0, 30.0]],
            "distance": 25.5,
            "estimated_time": 120,
        }


def mock_commander(nav, result, initial_heading=0, unit="feet", language="en"):
    """Mock commander for testing"""
    return [
        "Head straight for 10 feet",
        "Turn right and continue for 15 feet",
        "You have arrived at your destination",
    ]


class MockUnavServer:
    """Simplified version of UnavServer for testing"""

    def __init__(self):
        self.user_sessions = {}
        self.localizer = MockLocalizer()
        self.nav = MockNavigator()
        self.commander = mock_commander

    def get_session(self, user_id: str) -> dict:
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        return self.user_sessions[user_id]

    def update_session(self, user_id: str, updates: dict):
        """Update user session with new data"""
        session = self.get_session(user_id)
        session.update(updates)

    def _safe_serialize(self, obj):
        """Helper method to safely serialize objects for JSON response"""

        def convert_obj(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, dict):
                return {k: convert_obj(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [convert_obj(item) for item in o]
            else:
                return o

        return convert_obj(obj)

    def unav_navigation(
        self,
        user_id: str,
        image,
        dest_id: str,
        target_place: str,
        target_building: str,
        target_floor: str,
        top_k: int = None,
        unit: str = "feet",
        language: str = "en",
        refinement_queue: dict = None,
    ):
        """Navigation implementation (simplified for testing)"""
        try:
            print(f"🧭 Starting navigation for user {user_id}")

            # Get user session
            session = self.get_session(user_id)

            # Check for required navigation context
            if not all([dest_id, target_place, target_building, target_floor]):
                return {
                    "status": "error",
                    "error": "Incomplete navigation context. Please select a destination.",
                }

            # Automatically set/update navigation context in session
            print("📝 Setting navigation context in user session")
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

            # Use provided refinement queue or get from session
            if refinement_queue is None:
                refinement_queue = session.get("refinement_queue") or {}

            print(
                f"📍 Localizing user at {target_place}/{target_building}/{target_floor}"
            )

            # Perform localization
            output = self.localizer.localize(image, refinement_queue, top_k=top_k)
            if output is None or "floorplan_pose" not in output:
                return {
                    "status": "error",
                    "error": "Localization failed, no pose found.",
                }

            floorplan_pose = output["floorplan_pose"]
            start_xy, start_heading = floorplan_pose["xy"], -floorplan_pose["ang"]
            source_key = output["best_map_key"]
            start_place, start_building, start_floor = source_key

            print(f"✅ Localized at {start_place}/{start_building}/{start_floor}")
            print(f"📍 Position: {start_xy}, Heading: {start_heading}")

            # Update user's current floor context and pose for real-time tracking
            self.update_session(
                user_id,
                {
                    "current_place": start_place,
                    "current_building": start_building,
                    "current_floor": start_floor,
                    "floorplan_pose": floorplan_pose,
                    "refinement_queue": output["refinement_queue"],
                },
            )

            print(f"🎯 Planning path to destination {dest_id}")

            # Plan navigation path to destination
            result = self.nav.find_path(
                start_place,
                start_building,
                start_floor,
                start_xy,
                target_place,
                target_building,
                target_floor,
                dest_id,
            )

            if result is None:
                return {
                    "status": "error",
                    "error": "Path planning failed. Could not find route to destination.",
                }

            print("🗣️ Generating navigation commands")

            # Generate spoken/navigation commands
            cmds = self.commander(
                self.nav,
                result,
                initial_heading=start_heading,
                unit=unit,
                language=language,
            )

            print(f"✅ Navigation completed successfully with {len(cmds)} commands")

            # Return all relevant info safely serialized for JSON
            return {
                "status": "success",
                "result": self._safe_serialize(result),
                "cmds": self._safe_serialize(cmds),
                "best_map_key": self._safe_serialize(source_key),
                "floorplan_pose": self._safe_serialize(floorplan_pose),
                "navigation_info": {
                    "start_location": f"{start_place}/{start_building}/{start_floor}",
                    "destination": f"{target_place}/{target_building}/{target_floor}",
                    "dest_id": dest_id,
                    "unit": unit,
                    "language": language,
                },
            }

        except Exception as e:
            print(f"❌ Navigation error: {e}")
            return {"status": "error", "error": str(e), "type": type(e).__name__}


def test_navigation():
    """Test the navigation implementation"""
    print("🧪 Testing UNav Navigation Implementation")
    print("=" * 50)

    # Create server instance
    server = MockUnavServer()

    # Test data
    user_id = "test_user_123"
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Mock BGR image
    dest_id = "dest_001"
    target_place = "New_York_City"
    target_building = "LightHouse"
    target_floor = "6_floor"

    # Test navigation
    result = server.unav_navigation(
        user_id=user_id,
        image=image,
        dest_id=dest_id,
        target_place=target_place,
        target_building=target_building,
        target_floor=target_floor,
        unit="feet",
        language="en",
    )

    print("\n📊 Navigation Result:")
    print(json.dumps(result, indent=2))

    print("\n🗂️ User Session After Navigation:")
    session = server.get_session(user_id)
    print(json.dumps(server._safe_serialize(session), indent=2))

    # Test error case - missing destination
    print("\n🚫 Testing Error Case (Missing Destination):")
    error_result = server.unav_navigation(
        user_id="error_user",
        image=image,
        dest_id="",  # Missing destination
        target_place=target_place,
        target_building=target_building,
        target_floor=target_floor,
    )
    print(json.dumps(error_result, indent=2))


if __name__ == "__main__":
    test_navigation()
