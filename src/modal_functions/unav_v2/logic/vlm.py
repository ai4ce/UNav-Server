from typing import Any

import numpy as np


def run_vlm_on_image(server: Any, image: np.ndarray) -> str:
    """
    Run VLM on the provided image to extract text using Gemini 2.5 Flash.
    """
    if hasattr(server, "tracer") and server.tracer:
        with server.tracer.start_as_current_span(
            "vlm_text_extraction_span"
        ) as vlm_span:
            return _vlm_extract_text(image)

    return _vlm_extract_text(image)


def _vlm_extract_text(image: np.ndarray) -> str:
    try:
        from google import genai
        from google.genai import types
        import cv2
        import os

        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            error_msg = "GEMINI_API_KEY environment variable not set. Please set it with your API key in Modal Secrets."
            print(f"❌ {error_msg}")
            return error_msg

        client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = "gemini-2.5-flash"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, image_bytes = cv2.imencode(".jpg", image_rgb)
        image_bytes = image_bytes.tobytes()

        prompt = """Analyze this image and extract all visible text content. 
            Please provide:
            1. All readable text, signs, labels, and written content
            2. Any numbers, codes, or identifiers visible
            3. Location descriptions or directional information if present
            
            Format the response as clear, readable text without extra formatting."""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes, mime_type="image/jpeg"
                ),
            ],
        )

        extracted_text = (
            response.text if response.text else "No text extracted"
        )

        print(
            f"✅ VLM extraction successful: {len(extracted_text)} characters extracted"
        )

        return extracted_text

    except ImportError as e:
        error_msg = f"Missing required library for VLM: {str(e)}. Please install: pip install google-genai"
        print(f"❌ {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"VLM extraction failed: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
