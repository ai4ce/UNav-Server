import base64

import cv2
import numpy as np


def decode_image_input(base_64_image, allow_none: bool = False):
    """Decode base64 string or passthrough numpy array to BGR image."""
    if base_64_image is None:
        if allow_none:
            return None, None
        return None, "No image provided. base_64_image parameter is required."

    if isinstance(base_64_image, str):
        try:
            base64_string = base_64_image
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]

            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += "=" * (4 - missing_padding)

            image_bytes = base64.b64decode(base64_string)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return None, "Failed to decode base64 image. Invalid image format."
            return image, None
        except Exception as img_error:
            return None, f"Error processing base64 image: {str(img_error)}"

    if isinstance(base_64_image, np.ndarray):
        return base_64_image, None

    return (
        None,
        f"Unsupported image format. Expected base64 string or numpy array, got {type(base_64_image)}",
    )
