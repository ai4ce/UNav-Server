from typing import Dict, Any


def _get_queue_key_for_image_shape(image_shape):
    """Get a queue key based on image shape for bucket-based refinement queue handling."""
    if image_shape is None:
        return "default"
    h, w = image_shape[:2]
    return f"{h}x{w}"


def _get_refinement_queue_for_map(queue_dict, map_key, queue_key):
    """Get the refinement queue for a specific map_key and queue_key (image shape bucket)."""
    if map_key not in queue_dict:
        return {"pairs": [], "initial_poses": [], "pps": []}
    map_queues = queue_dict[map_key]
    if queue_key not in map_queues:
        return {"pairs": [], "initial_poses": [], "pps": []}
    return map_queues[queue_key]


def _update_refinement_queue(queue_dict, map_key, queue_key, new_queue_state):
    """Update the refinement queue for a specific map_key and queue_key."""
    if map_key not in queue_dict:
        queue_dict[map_key] = {}
    queue_dict[map_key][queue_key] = new_queue_state
    return queue_dict
