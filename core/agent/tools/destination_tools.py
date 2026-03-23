from difflib import SequenceMatcher
from typing import List
import re

from config import PLACES
from core.unav_state import nav, get_session
from core.i18n_labels import get_label
from core.destination_display import humanize_destination_name
from models.agent_schemas import DestinationCandidate, DestinationQuery


CATEGORY_KEYWORDS = {
    "restroom": ["restroom", "bathroom", "washroom", "toilet", "wc", "厕", "厕所", "洗手间", "卫生间", "盥洗室"],
    "elevator": ["elevator", "lift", "电梯", "升降机"],
    "stairs": ["stairs", "stair", "staircase", "楼梯", "楼道"],
    "exit": ["exit", "door", "entrance", "出口", "门口", "大门"],
    "service_desk": ["service desk", "front desk", "reception", "help desk", "服务台", "前台", "接待处"],
}

_RESTROOM_SUBTYPE_ALIASES = {
    "men": ["men", "mens", "mens restroom", "male", "boys", "gentlemen", "男厕", "男洗手间", "男卫生间", "男厕所"],
    "women": ["women", "womens", "womens restroom", "female", "girls", "ladies", "女厕", "女洗手间", "女卫生间", "女厕所"],
    "family": ["family", "all gender", "unisex", "家庭", "家庭厕所"],
    "accessible": ["accessible", "wheelchair", "ada", "无障碍", "无障碍厕所"],
}

_CATEGORY_DEFAULT_SCOPE = {
    "restroom": "building",
    "elevator": "building",
    "exit": "building",
    "stairs": "building",
    "service_desk": "building",
}

_NAME_STOPWORDS = {
    "dr", "doctor", "prof", "professor", "mr", "mrs", "ms",
    "office", "room", "suite", "lab", "laboratory",
    "of", "the", "go", "to", "take", "me",
    "办公室", "老师", "教授", "房间", "去", "带我", "我想去",
}


def infer_category_from_name(name: str):
    lower = name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return category
    return None


def _alias_matches(normalized: str, alias: str) -> bool:
    if re.search(r'[\u4e00-\u9fff]', alias):
        return alias in normalized
    pattern = r'(?<![a-z])' + re.escape(alias) + r'(?![a-z])'
    return re.search(pattern, normalized) is not None


def _infer_restroom_subtype(text: str | None):
    if not text:
        return None
    lowered = text.lower()
    normalized = lowered.replace("'", "").replace("’", "")
    for subtype, aliases in _RESTROOM_SUBTYPE_ALIASES.items():
        if any(_alias_matches(normalized, alias) for alias in aliases):
            return subtype
    return None


def _restroom_subtype_for_item(name: str | None):
    return _infer_restroom_subtype(name)


def build_destination_catalog(user_id: str, scope: str = "session") -> List[dict]:
    session = get_session(user_id)
    lang = session.get("language", "en")
    contexts = []

    target_place = session.get("target_place")
    target_building = session.get("target_building")
    target_floor = session.get("target_floor")
    current_place = session.get("current_place")
    current_building = session.get("current_building")
    current_floor = session.get("current_floor")

    def append_global_contexts():
        for place, buildings in PLACES.items():
            for building, floors in buildings.items():
                for floor in floors:
                    contexts.append((place, building, floor))

    if scope == "building":
        place = target_place or current_place
        building = target_building or current_building
        if place and building and place in PLACES and building in PLACES[place]:
            for floor in PLACES[place][building]:
                contexts.append((place, building, floor))
        elif place and place in PLACES:
            for bldg, floors in PLACES[place].items():
                for floor in floors:
                    contexts.append((place, bldg, floor))
        else:
            append_global_contexts()
    elif scope == "place":
        place = target_place or current_place
        if place and place in PLACES:
            for building, floors in PLACES[place].items():
                for floor in floors:
                    contexts.append((place, building, floor))
        else:
            append_global_contexts()
    elif scope == "global":
        append_global_contexts()
    elif target_place and target_building and target_floor:
        contexts.append((target_place, target_building, target_floor))
    elif current_place and current_building and current_floor:
        contexts.append((current_place, current_building, current_floor))
    else:
        append_global_contexts()

    catalog = []
    seen = set()
    for place, building, floor in contexts:
        key = (place, building, floor)
        if key in seen or key not in nav.pf_map:
            continue
        seen.add(key)
        pf_target = nav.pf_map[key]
        for did in pf_target.dest_ids:
            did_str = str(did)
            label_key = f"{place}/{building}/{floor}/{did_str}"
            fallback = pf_target.labels[did]
            raw_name = get_label("destinations", label_key, lang, fallback=fallback)
            category = infer_category_from_name(raw_name) or infer_category_from_name(str(fallback))
            name = humanize_destination_name(raw_name, fallback=fallback, category=category, floor=floor, lang=lang)
            catalog.append(
                {
                    "destination_id": did_str,
                    "name": name,
                    "category": category,
                    "building": building,
                    "floor": floor,
                    "place": place,
                }
            )
    return catalog


def _context_bonus(item: dict, session: dict) -> float:
    bonus = 0.0
    if session.get("target_place") and item.get("place") == session.get("target_place"):
        bonus += 0.2
    if session.get("target_building") and item.get("building") == session.get("target_building"):
        bonus += 0.5
    if session.get("target_floor") and item.get("floor") == session.get("target_floor"):
        bonus += 1.0
    if session.get("current_place") and item.get("place") == session.get("current_place"):
        bonus += 0.1
    if session.get("current_building") and item.get("building") == session.get("current_building"):
        bonus += 0.2
    if session.get("current_floor") and item.get("floor") == session.get("current_floor"):
        bonus += 0.6
    return bonus


def _normalize_name_for_match(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[’'`]+", "", lowered)
    lowered = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", lowered)
    normalized_tokens = []
    for token in lowered.split():
        if not token or token in _NAME_STOPWORDS:
            continue
        if re.fullmatch(r"[a-z]+", token) and len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def _fuzzy_name_score(name_hint: str, item_name: str) -> float | None:
    if not name_hint:
        return None

    normalized_hint = _normalize_name_for_match(name_hint)
    normalized_item = _normalize_name_for_match(item_name)
    if not normalized_hint or not normalized_item:
        return None

    if normalized_hint in normalized_item:
        return 1.5
    if normalized_item in normalized_hint:
        return 1.1

    hint_tokens = normalized_hint.split()
    item_tokens = normalized_item.split()
    overlap = set(hint_tokens) & set(item_tokens)
    best_token_ratio = 0.0
    for hint_token in hint_tokens:
        for item_token in item_tokens:
            best_token_ratio = max(best_token_ratio, SequenceMatcher(None, hint_token, item_token).ratio())

    phrase_ratio = SequenceMatcher(None, normalized_hint, normalized_item).ratio()
    if phrase_ratio >= 0.9:
        return 1.35
    if phrase_ratio >= 0.8:
        return 1.05
    if phrase_ratio >= 0.72 and (best_token_ratio >= 0.6 or overlap):
        return 0.85
    if best_token_ratio >= 0.82:
        return 0.8
    if best_token_ratio >= 0.68 and overlap:
        return 0.65
    if len(hint_tokens) == 1 and len(item_tokens) == 1 and best_token_ratio >= 0.58:
        return 0.72
    if len(hint_tokens) == 1 and best_token_ratio >= 0.72:
        return 0.7
    if len(item_tokens) == 1 and best_token_ratio >= 0.72:
        return 0.7
    return None


def _score_candidate(item: dict, *, category: str | None, name_hint: str, building_hint: str, floor_hint: str,
                     requested_restroom_subtype: str | None, session: dict) -> float | None:
    score = 0.0
    if category:
        if item.get("category") == category:
            score += 1.5
        else:
            return None

    if category == "restroom" and requested_restroom_subtype:
        item_subtype = _restroom_subtype_for_item(item.get("name"))
        if item_subtype and item_subtype != requested_restroom_subtype:
            return None
        if item_subtype == requested_restroom_subtype:
            score += 1.0

    if name_hint:
        name_score = _fuzzy_name_score(name_hint, item["name"])
        if name_score is None:
            if category == "restroom" and requested_restroom_subtype and _restroom_subtype_for_item(item.get("name")) == requested_restroom_subtype:
                name_score = 0.55
            else:
                return None
        score += name_score

    if building_hint:
        if building_hint in item["building"].lower():
            score += 0.6
        else:
            return None
    if floor_hint:
        if floor_hint in item["floor"].lower():
            score += 0.6
        else:
            return None
    if not category and not name_hint and not building_hint and not floor_hint:
        score += 0.1
    score += _context_bonus(item, session)
    return score


def resolve_query_against_destinations(destination_query: DestinationQuery, user_id: str) -> List[DestinationCandidate]:
    requested_scope = destination_query.search_scope
    effective_scope = requested_scope or "session"
    if effective_scope == "session" and destination_query.category in _CATEGORY_DEFAULT_SCOPE and not destination_query.floor_hint:
        effective_scope = _CATEGORY_DEFAULT_SCOPE[destination_query.category]

    session = get_session(user_id)
    name_hint = (destination_query.name_hint or "").strip().lower()
    category = destination_query.category
    building_hint = (destination_query.building_hint or "").strip().lower()
    floor_hint = (destination_query.floor_hint or "").strip().lower()
    requested_restroom_subtype = _infer_restroom_subtype(name_hint) if category == "restroom" else None

    def collect(scope: str) -> List[DestinationCandidate]:
        results: List[DestinationCandidate] = []
        catalog = build_destination_catalog(user_id, scope=scope)
        for item in catalog:
            score = _score_candidate(
                item,
                category=category,
                name_hint=name_hint,
                building_hint=building_hint,
                floor_hint=floor_hint,
                requested_restroom_subtype=requested_restroom_subtype,
                session=session,
            )
            if score is None:
                continue
            confidence_base = 0.45 if scope != "global" else 0.4
            confidence_divisor = 3.2 if scope != "global" else 3.4
            results.append(
                DestinationCandidate(
                    destination_id=item["destination_id"],
                    name=item["name"],
                    category=item.get("category"),
                    place=item.get("place"),
                    building=item.get("building"),
                    floor=item.get("floor"),
                    confidence=min(0.99, confidence_base + score / confidence_divisor),
                )
            )
        results.sort(key=lambda x: (x.confidence or 0.0, x.name), reverse=True)
        return results[:5]

    results = collect(effective_scope)
    if not results and effective_scope in {"session", "building", "place"}:
        results = collect("global")
    return results
