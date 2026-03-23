import json
import os
import urllib.request

from core.agent.prompt_builder import (
    build_destination_interpretation_prompt,
    build_navigation_explanation_prompt,
    build_preference_adjustment_prompt,
)


class GeminiProvider:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"
        self.url = (
            os.getenv("GEMINI_API_URL", "").strip()
            or f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        )

    def _post_json(self, prompt: str) -> dict:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(content)

    def interpret_destination(self, *, utterance: str, language: str, context: dict) -> dict:
        return self._post_json(
            build_destination_interpretation_prompt(
                utterance=utterance,
                language=language,
                context=context,
            )
        )

    def adjust_preferences(self, *, utterance: str, profile: dict, context: dict) -> dict:
        return self._post_json(
            build_preference_adjustment_prompt(
                utterance=utterance,
                profile=profile,
                context=context,
            )
        )

    def explain_navigation_state(self, *, question: str, context: dict) -> dict:
        return self._post_json(
            build_navigation_explanation_prompt(
                question=question,
                context=context,
            )
        )
