#!/usr/bin/env python3
"""
Usage examples for the UNav Navigation implementation.
Shows how to use the navigation system with proper session management.
"""

import numpy as np
from unav_modal import UnavServer


def example_usage():
    """Example of how to use the UNav navigation system"""

    # Initialize the server (in Modal, this happens automatically)
    server = UnavServer()

    # Example user and navigation setup
    user_id = "user_12345"
    dest_id = "bathroom_001"
    target_place = "New_York_City"
    target_building = "LightHouse"
    target_floor = "6_floor"

    # Create a mock image (in real usage, this would be from camera)
    # Image should be BGR format as expected by the system
    mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("🧭 Performing navigation (context will be set automatically)...")

    # Perform navigation - context is set automatically when calling this method
    navigation_result = server.unav_navigation(
        user_id=user_id,
        image=mock_image,
        dest_id=dest_id,
        target_place=target_place,
        target_building=target_building,
        target_floor=target_floor,
        top_k=5,
        unit="feet",
        language="en",
    )

    print(f"Navigation result: {navigation_result['status']}")

    if navigation_result["status"] == "success":
        print(f"📍 Current location: {navigation_result['best_map_key']}")
        print(f"🎯 Navigation commands ({len(navigation_result['cmds'])}):")
        for i, cmd in enumerate(navigation_result["cmds"], 1):
            print(f"  {i}. {cmd}")
    else:
        print(f"❌ Navigation failed: {navigation_result.get('error')}")

    print("\n📋 Alternative: Using simplified interface...")

    # Alternative - using the simplified interface (context already set above)
    inputs = {"user_id": user_id, "image": mock_image, "top_k": 5}

    simple_result = server.unav_navigation_simple(inputs)

    if "error" not in simple_result:
        print(f"✅ Simple navigation successful")
        print(f"📍 Location: {simple_result['best_map_key']}")
        print(f"🗣️ Commands: {len(simple_result['cmds'])} steps")
    else:
        print(f"❌ Simple navigation failed: {simple_result['error']}")

    print("\n👤 Checking user session...")

    # Check user session state
    session_result = server.get_user_session(user_id)
    if session_result["status"] == "success":
        session = session_result["session"]
        print(f"Current floor: {session.get('current_floor')}")
        print(f"Target destination: {session.get('selected_dest_id')}")
        print(f"Last known position: {session.get('floorplan_pose', {}).get('xy')}")


def example_error_handling():
    """Example of error handling scenarios"""

    server = UnavServer()
    user_id = "test_user"
    mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("🚫 Testing error scenarios...")

    # Error 1: Missing navigation context
    result1 = server.unav_navigation(
        user_id=user_id,
        image=mock_image,
        dest_id="",  # Missing
        target_place="",  # Missing
        target_building="",  # Missing
        target_floor="",  # Missing
    )
    print(f"Missing context error: {result1.get('error')}")

    # Error 2: Using simplified interface without setting context first
    inputs = {"user_id": "new_user_no_context", "image": mock_image}

    result2 = server.unav_navigation_simple(inputs)
    print(f"No context error: {result2.get('error')}")


def example_session_management():
    """Example of session management operations"""

    server = UnavServer()
    user_id = "session_test_user"

    print("📝 Testing session management...")

    # Set some context
    server.set_navigation_context(
        user_id=user_id,
        dest_id="office_001",
        target_place="New_York_City",
        target_building="LightHouse",
        target_floor="4_floor",
    )

    # Check session
    session = server.get_user_session(user_id)
    print(f"Session created: {len(session['session'])} items")

    # Clear session
    clear_result = server.clear_user_session(user_id)
    print(f"Session cleared: {clear_result['message']}")

    # Check empty session
    empty_session = server.get_user_session(user_id)
    print(f"Empty session: {len(empty_session['session'])} items")


if __name__ == "__main__":
    print("🏗️ UNav Navigation System Usage Examples")
    print("=" * 50)

    try:
        example_usage()
        print("\n" + "=" * 50)
        example_error_handling()
        print("\n" + "=" * 50)
        example_session_management()
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("Note: This example requires the actual UNav dependencies to run fully.")
