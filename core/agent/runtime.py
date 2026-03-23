from collections import defaultdict

from models.agent_schemas import (
    AgentAdjustPreferencesResponse,
    AgentExplainNavigationStateResponse,
    AgentFollowUpDestinationResponse,
    AgentInterpretDestinationResponse,
    AgentResolveDestinationResponse,
    DestinationQuery,
)
from core.agent.providers.factory import get_agent_provider
from core.agent.session_context import build_session_context
from core.agent.tools.destination_tools import resolve_query_against_destinations
from core.agent.tools.explanation_tools import build_explanation_context
from core.agent.tools.preference_tools import get_profile, update_profile


class AgentRuntime:
    def __init__(self):
        self.provider = get_agent_provider()

    def _humanize_floor(self, floor: str | None, language: str = "en") -> str | None:
        if not floor:
            return None
        normalized = floor.replace("_floor", "").replace("_", " ").strip()
        if normalized.isdigit():
            if language == "zh":
                return f"{normalized}楼"
            if language == "th":
                return f"ชั้น {normalized}"
            return f"floor {normalized}"
        return normalized

    def _article_count(self, count: int, language: str = "en") -> str:
        if language == "zh":
            return str(count)
        mapping = {2: "two", 3: "three", 4: "four", 5: "five"}
        return mapping.get(count, str(count))

    def _group_candidates(self, candidates):
        grouped = defaultdict(list)
        for candidate in candidates:
            key = (candidate.name, candidate.place or "", candidate.building or "")
            grouped[key].append(candidate)
        return grouped

    def _naturalize_ambiguous_message(self, candidates, language: str) -> str:
        grouped = self._group_candidates(candidates)

        if language == "zh":
            if len(grouped) == 1:
                (name, place, building), items = next(iter(grouped.items()))
                floors = [self._humanize_floor(item.floor, language) for item in items[:4] if item.floor]
                location_base = "，".join([part for part in [building, place] if part])
                if floors:
                    floor_phrase = "、".join(floors)
                    if location_base:
                        return f"我找到了{len(items)}个{name}，都在{location_base}，分别是{floor_phrase}。你想去哪个？"
                    return f"我找到了{len(items)}个{name}，分别是{floor_phrase}。你想去哪个？"
                return f"我找到了{len(items)}个{name}。你想去哪个？"

            summary = []
            for (name, place, building), items in list(grouped.items())[:3]:
                floor = self._humanize_floor(items[0].floor, language)
                loc = "，".join([part for part in [building, place, floor] if part])
                summary.append(f"{name}{'（' + loc + '）' if loc else ''}")
            return f"我找到了几个可能的地点：{'；'.join(summary)}。你想去哪个？"

        if len(grouped) == 1:
            (name, place, building), items = next(iter(grouped.items()))
            floors = [self._humanize_floor(item.floor, language) for item in items[:4] if item.floor]
            if floors:
                floor_phrase = ", ".join(floors[:-1]) + (f", and {floors[-1]}" if len(floors) > 1 else floors[0])
                location_bits = [bit for bit in [building, place] if bit]
                location_base = " in ".join(location_bits) if location_bits else None
                if location_base:
                    return (
                        f"I found {self._article_count(len(items), language)} {name}s in {location_base}, "
                        f"on {floor_phrase}. Which one do you want?"
                    )
                return f"I found {self._article_count(len(items), language)} {name}s on {floor_phrase}. Which one do you want?"
            return f"I found {self._article_count(len(items), language)} {name}s. Which one do you want?"

        summary = []
        for (name, place, building), items in list(grouped.items())[:3]:
            floor = self._humanize_floor(items[0].floor, language)
            loc = ", ".join([part for part in [building, place, floor] if part])
            summary.append(f"{name}{' (' + loc + ')' if loc else ''}")
        return f"I found a few likely places: {'; '.join(summary)}. Which one do you want?"

    def interpret_destination(self, *, user_id: str, utterance: str, language=None) -> dict:
        context = build_session_context(user_id)
        resolved_language = language or context.get("language") or "en"
        raw = self.provider.interpret_destination(
            utterance=utterance,
            language=resolved_language,
            context=context,
        )
        response_language = raw.get("response_language") or resolved_language or "en"
        response = AgentInterpretDestinationResponse(
            intent=raw.get("intent", "navigate"),
            destination_query=DestinationQuery.model_validate(raw.get("destination_query", {})),
            needs_clarification=bool(raw.get("needs_clarification", False)),
            message=raw.get("message"),
            response_language=response_language,
        )
        return response.model_dump()

    def resolve_destination(self, *, user_id: str, destination_query: dict, response_language: str | None = None) -> dict:
        query = DestinationQuery.model_validate(destination_query)
        context = build_session_context(user_id)
        language = response_language or context.get("language") or "en"
        candidates = resolve_query_against_destinations(query, user_id)
        if not candidates:
            response = AgentResolveDestinationResponse(
                status="not_found",
                needs_confirmation=False,
                candidates=[],
                message=("我没有找到匹配的目的地。" if language == "zh" else "I could not find a matching destination."),
                response_language=language,
            )
        elif len(candidates) == 1:
            candidate = candidates[0]
            floor = self._humanize_floor(candidate.floor, language)
            location_bits = [part for part in [candidate.building, candidate.place, floor] if part]
            if language == "zh":
                location = "，".join(location_bits)
                message = f"我找到了{candidate.name}{'，位置在' + location if location else ''}。你想去那里吗？"
            else:
                location = ", ".join(location_bits)
                message = f"I found {candidate.name}{' in ' + location if location else ''}. Would you like to go there?"
            response = AgentResolveDestinationResponse(
                status="resolved",
                needs_confirmation=True,
                candidates=candidates,
                message=message,
                response_language=language,
            )
        else:
            response = AgentResolveDestinationResponse(
                status="ambiguous",
                needs_confirmation=True,
                candidates=candidates,
                message=self._naturalize_ambiguous_message(candidates, language),
                response_language=language,
            )
        return response.model_dump()

    def followup_destination(self, *, user_id: str, utterance: str, candidates: list[dict], response_language: str | None = None) -> dict:
        context = build_session_context(user_id)
        language = response_language or context.get("language") or "en"
        raw = self.provider.followup_destination(utterance=utterance, candidates=candidates, language=language)
        response = AgentFollowUpDestinationResponse(
            status=raw.get("status", "unclear"),
            selected_destination_ids=[str(item) for item in raw.get("selected_destination_ids", [])],
            message=raw.get("message"),
            response_language=raw.get("response_language") or language,
        )
        return response.model_dump()

    def adjust_preferences(self, *, user_id: str, utterance: str) -> dict:
        context = build_session_context(user_id)
        profile = get_profile(user_id).model_dump()
        raw = self.provider.adjust_preferences(utterance=utterance, profile=profile, context=context)
        patch = raw.get("patch", {})
        updated = update_profile(user_id, patch)
        message = raw.get("message")
        if not message:
            message = "Preferences updated." if patch else "No preference changes detected."
        response = AgentAdjustPreferencesResponse(
            applied_changes=[{"key": key, "value": value} for key, value in patch.items()],
            message=message,
        )
        payload = response.model_dump()
        payload["profile"] = updated.model_dump()
        return payload

    def explain_navigation_state(self, *, user_id: str, question: str) -> dict:
        context = build_explanation_context(user_id)
        raw = self.provider.explain_navigation_state(question=question, context=context)
        response = AgentExplainNavigationStateResponse(
            message=raw.get("message", "I could not explain the current navigation state."),
            state_summary=context,
        )
        return response.model_dump()
