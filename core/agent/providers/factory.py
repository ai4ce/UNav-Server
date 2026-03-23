import os

from core.agent.providers.gemini_provider import GeminiProvider
from core.agent.providers.openai_provider import OpenAIProvider
from core.agent.providers.rule_provider import RuleBasedProvider


def get_agent_provider():
    provider = os.getenv("LLM_PROVIDER", "rule").strip().lower()

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return GeminiProvider()

    fallback = os.getenv("LLM_FALLBACK_PROVIDER", "rule").strip().lower()
    if fallback == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    if fallback == "gemini" and os.getenv("GEMINI_API_KEY"):
        return GeminiProvider()
    return RuleBasedProvider()
