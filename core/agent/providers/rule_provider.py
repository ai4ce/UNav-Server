from models.agent_schemas import DestinationQuery


CATEGORY_ALIASES = {
    "restroom": [
        "restroom", "bathroom", "washroom", "toilet", "wc", "ladies room", "mens room",
        "pee", "urgent", "尿急", "上厕所", "厕所", "洗手间", "卫生间", "ห้องน้ำ",
        "男厕", "女厕", "男洗手间", "女洗手间", "men's restroom", "women's restroom",
    ],
    "elevator": ["elevator", "lift", "电梯", "ลิฟต์"],
    "stairs": ["stairs", "stair", "staircase", "楼梯", "บันได"],
    "exit": ["exit", "entrance", "door", "way out", "出口", "ทางออก"],
    "service_desk": ["service desk", "front desk", "reception", "help desk", "服务台", "接待处"],
}

LANGUAGE_ALIASES = {
    "english": "en",
    "chinese": "zh",
    "mandarin": "zh",
}


def _infer_restroom_name_hint(text: str):
    if any(token in text for token in ["women", "woman", "female", "ladies", "女厕", "女洗手间", "女厕所", "女生厕所"]):
        return "women"
    if any(token in text for token in ["men", "man's", "male", "mens", "男厕", "男洗手间", "男厕所", "男生厕所"]):
        return "men"
    if any(token in text for token in ["accessible", "wheelchair", "无障碍"]):
        return "accessible"
    if any(token in text for token in ["family", "parent", "亲子"]):
        return "family"
    return None


class RuleBasedProvider:
    def interpret_destination(self, *, utterance: str, language: str, context: dict) -> dict:
        text = utterance.lower().strip()
        category = None
        for key, aliases in CATEGORY_ALIASES.items():
            if any(alias in text for alias in aliases):
                category = key
                break

        urgency_words = ["nearest", "closest", "nearby", "near me", "urgent", "pee", "尿急", "急"]
        preference = "nearest" if any(word in text for word in urgency_words) else "specific"
        name_hint = None if category else utterance.strip()
        if category == "restroom":
            name_hint = _infer_restroom_name_hint(text)
        query = DestinationQuery(
            category=category,
            name_hint=name_hint,
            preference=preference if (category or name_hint) else "any",
        )
        needs = not bool(category or name_hint)
        response_language = language or "en"
        return {
            "intent": "navigate",
            "destination_query": query.model_dump(),
            "needs_clarification": needs,
            "message": "Please tell me where you want to go." if needs else None,
            "response_language": response_language,
        }

    def adjust_preferences(self, *, utterance: str, profile: dict, context: dict) -> dict:
        text = utterance.lower().strip()
        patch = {}
        if "slow" in text:
            patch["guidance_tempo_multiplier"] = 0.8
        elif "faster" in text or "quicker" in text:
            patch["guidance_tempo_multiplier"] = 1.2
        for alias, code in LANGUAGE_ALIASES.items():
            if alias in text:
                patch["language"] = code
        if "spatial" in text:
            patch["preferred_audio_mode"] = "spatial"
        elif "stereo" in text:
            patch["preferred_audio_mode"] = "stereo"
        if "less" in text and "verbose" in text:
            patch["verbosity"] = "low"
        elif "more" in text and "verbose" in text:
            patch["verbosity"] = "high"
        return {"patch": patch}

    def explain_navigation_state(self, *, question: str, context: dict) -> dict:
        if not context.get("has_active_navigation"):
            return {"message": "There is no active navigation session right now."}
        return {"message": "Navigation is active. You can ask for the next step or remaining distance."}


    def followup_destination(self, *, utterance: str, candidates: list[dict], language: str) -> dict:
        text = utterance.lower().strip()
        response_language = language or "en"
        if not candidates:
            return {
                "status": "restart",
                "selected_destination_ids": [],
                "message": None,
                "response_language": response_language,
            }

        affirmative_tokens = ["yes", "yeah", "correct", "that one", "对", "是的", "想去", "就这个", "这个", "好的"]
        if len(candidates) == 1 and any(token in text for token in affirmative_tokens):
            return {
                "status": "confirm",
                "selected_destination_ids": [str(candidates[0].get("destination_id"))],
                "message": None,
                "response_language": response_language,
            }

        ordinal = None
        if any(token in text for token in ["first", "option one", "number one", "第一个", "1"]):
            ordinal = 0
        elif any(token in text for token in ["second", "option two", "number two", "第二个", "2"]):
            ordinal = 1
        elif any(token in text for token in ["third", "option three", "number three", "第三个", "3"]):
            ordinal = 2
        if ordinal is not None and ordinal < len(candidates):
            chosen = candidates[ordinal]
            return {
                "status": "matched",
                "selected_destination_ids": [str(chosen.get("destination_id"))],
                "message": None,
                "response_language": response_language,
            }

        floor_aliases = {"一":"1","二":"2","两":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10"}
        floor_values = set()
        import re
        for m in re.finditer(r'(\d+)', text):
            floor_values.add(m.group(1))
        for key, value in floor_aliases.items():
            if key + '楼' in text or key + ' floor' in text or key in text and '楼' in text:
                floor_values.add(value)
        if floor_values:
            narrowed = []
            for candidate in candidates:
                floor = str(candidate.get('floor', '')).lower()
                for value in floor_values:
                    if floor == f'{value}_floor' or floor == f'{value} floor':
                        narrowed.append(candidate)
                        break
            if narrowed:
                return {
                    "status": "matched" if len(narrowed) == 1 else "narrowed",
                    "selected_destination_ids": [str(item.get("destination_id")) for item in narrowed],
                    "message": None,
                    "response_language": response_language,
                }

        filtered = []
        for candidate in candidates:
            haystack = ' '.join([str(candidate.get('name', '')), str(candidate.get('building', '')), str(candidate.get('place', '')), str(candidate.get('floor', ''))]).lower()
            if any(token in haystack for token in text.split() if len(token) >= 2):
                filtered.append(candidate)
        if filtered:
            return {
                "status": "matched" if len(filtered) == 1 else "narrowed",
                "selected_destination_ids": [str(item.get("destination_id")) for item in filtered],
                "message": None,
                "response_language": response_language,
            }

        return {
            "status": "unclear",
            "selected_destination_ids": [],
            "message": "我还没听出你指的是哪个选项。你可以说楼层，或者说第一个、第二个。" if response_language.startswith('zh') else "I still could not tell which option you meant. You can say the floor, or say first, second, or third.",
            "response_language": response_language,
        }
