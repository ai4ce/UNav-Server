import json
import os
import urllib.request

from core.agent.prompt_builder import (
    build_destination_followup_prompt,
    build_destination_interpretation_prompt,
    build_navigation_explanation_prompt,
    build_preference_adjustment_prompt,
)


class OpenAIProvider:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        self.url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions").strip()

    def _post_json(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    def interpret_destination(self, *, utterance: str, language: str, context: dict) -> dict:
        return self._post_json(
            build_destination_interpretation_prompt(
                utterance=utterance,
                language=language,
                context=context,
            )
        )

    def followup_destination(self, *, utterance: str, candidates: list[dict], language: str) -> dict:
        return self._post_json(
            build_destination_followup_prompt(
                utterance=utterance,
                candidates=candidates,
                language=language,
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
