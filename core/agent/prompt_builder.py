import json


DESTINATION_CATEGORY_HINTS = {
    "restroom": [
        "restroom",
        "bathroom",
        "washroom",
        "toilet",
        "wc",
        "ladies room",
        "mens room",
        "I need the toilet",
        "I need to pee",
        "I am desperate for the bathroom",
        "尿急",
        "想上厕所",
        "洗手间",
        "卫生间",
        "ห้องน้ำ",
        "男厕",
        "女厕",
    ],
    "elevator": [
        "elevator",
        "lift",
        "I need the elevator",
        "take me to the lift",
        "电梯",
        "ลิฟต์",
    ],
    "stairs": [
        "stairs",
        "staircase",
        "楼梯",
        "บันได",
    ],
    "exit": [
        "exit",
        "entrance",
        "way out",
        "出口",
        "ทางออก",
    ],
    "service_desk": [
        "service desk",
        "front desk",
        "reception",
        "help desk",
        "服务台",
        "接待处",
    ],
}


def build_destination_interpretation_prompt(*, utterance: str, language: str, context: dict) -> str:
    schema = {
        "intent": "navigate",
        "destination_query": {
            "category": "string or null",
            "name_hint": "string or null",
            "building_hint": "string or null",
            "floor_hint": "string or null",
            "preference": "nearest | specific | any",
        },
        "needs_clarification": "boolean",
        "message": "short clarification or null",
        "response_language": "BCP-47-like language tag such as en, zh, th, es, fr",
    }
    return (
        "You are a destination understanding model for an indoor accessibility navigation app. "
        "Return only valid JSON matching this schema: "
        f"{json.dumps(schema, ensure_ascii=False)}. "
        f"Preferred UI language: {language}. "
        f"Session context: {json.dumps(context, ensure_ascii=False)}. "
        "Infer the user's actual navigation goal, even when they speak indirectly, emotionally, colloquially, or in another language. "
        "For example, needing to pee, needing a toilet, or saying 尿急 should map to the restroom category. "
        "Map likely intents to one of these categories when appropriate: "
        f"{json.dumps(DESTINATION_CATEGORY_HINTS, ensure_ascii=False)}. "
        "If the user describes a need rather than naming a place, infer the most likely destination category instead of copying the utterance into name_hint. "
        "If the user specifies a subtype inside a category, preserve it in name_hint. For example: men's restroom, women's restroom, family restroom, accessible restroom, 男厕, 女厕 should set category='restroom' and a matching name_hint such as 'men' or 'women'. "
        "Only use name_hint when the user appears to mention a specific room, office, named place, destination label, or a subtype like men's/women's restroom. "
        "If the user says nearest, close, nearby, near me, or implies urgency, prefer 'nearest'. "
        "Always set response_language to the language the user is speaking in, unless it is unclear, in which case use the preferred UI language. "
        "If you include a message, write it naturally in response_language. "
        "Keep message null unless a short clarification is really needed. "
        f"User utterance: {utterance}"
    )


def build_destination_followup_prompt(*, utterance: str, candidates: list[dict], language: str) -> str:
    schema = {
        "status": "confirm | matched | narrowed | unclear | restart",
        "selected_destination_ids": ["destination id strings"],
        "message": "short natural-language response or null",
        "response_language": "BCP-47-like language tag such as en, zh, th, es, fr",
    }
    return (
        "You are a destination follow-up selection model for an indoor accessibility navigation app. "
        "The user is responding to previously proposed destination candidates. "
        "Return only valid JSON matching this schema: "
        f"{json.dumps(schema, ensure_ascii=False)}. "
        f"Preferred UI language: {language}. "
        f"Current candidates: {json.dumps(candidates, ensure_ascii=False)}. "
        "Use the current candidate list as conversation memory. Do not restart the search unless the user clearly asks for a different destination. "
        "If the user says yes / correct / 想去 / 对 / 就这个 and there is one current candidate, return status='confirm' with that candidate id. "
        "If the user picks by ordinal like first/second/third or 第一个/第二个, return the matching id. "
        "If the user narrows by floor like floor 3 / 3楼 / 三楼, return only matching candidate ids. "
        "If the user refers to name, building, place, or subtype, return only matching candidate ids. "
        "If the user is unclear, keep status='unclear'. "
        "If the user clearly changes topic to a different destination search, use status='restart'. "
        "Always set response_language to the language the user is speaking in if you can tell. "
        "If you include a message, write it naturally and concisely in response_language. "
        f"User utterance: {utterance}"
    )


def build_preference_adjustment_prompt(*, utterance: str, profile: dict, context: dict) -> str:
    schema = {
        "patch": {
            "language": "optional language code",
            "unit": "optional meters|feet",
            "preferred_audio_mode": "optional auto|stereo|spatial",
            "guidance_tempo_multiplier": "optional float",
            "countdown_enabled": "optional boolean",
            "haptic_level": "optional low|medium|high",
            "verbosity": "optional low|medium|high",
        },
        "message": "short natural-language confirmation",
    }
    return (
        "You are a preference parser for an indoor accessibility navigation app. "
        "Return only valid JSON matching this schema: "
        f"{json.dumps(schema)}. "
        f"Current profile: {json.dumps(profile, ensure_ascii=False)}. "
        f"Session context: {json.dumps(context, ensure_ascii=False)}. "
        f"User request: {utterance}"
    )


def build_navigation_explanation_prompt(*, question: str, context: dict) -> str:
    schema = {"message": "short navigation explanation"}
    return (
        "You are a navigation explanation assistant for a blind-friendly indoor navigation app. "
        "Return only valid JSON matching this schema: "
        f"{json.dumps(schema)}. "
        f"Navigation context: {json.dumps(context, ensure_ascii=False)}. "
        f"User question: {question}"
    )
