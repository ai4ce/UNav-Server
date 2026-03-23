from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class DestinationQuery(BaseModel):
    category: Optional[str] = None
    name_hint: Optional[str] = None
    building_hint: Optional[str] = None
    floor_hint: Optional[str] = None
    preference: Literal["nearest", "specific", "any"] = "any"
    search_scope: Optional[Literal["session", "building", "place", "global"]] = None


class DestinationCandidate(BaseModel):
    destination_id: str
    name: str
    category: Optional[str] = None
    place: Optional[str] = None
    building: Optional[str] = None
    floor: Optional[str] = None
    distance_hint_m: Optional[float] = None
    confidence: Optional[float] = None


class UserProfile(BaseModel):
    user_id: str
    language: str = "en"
    unit: str = "meters"
    preferred_audio_mode: str = "auto"
    guidance_tempo_multiplier: float = 1.0
    countdown_enabled: bool = True
    haptic_level: str = "medium"
    verbosity: str = "low"
    favorite_destination_ids: List[str] = Field(default_factory=list)


class PreferencePatch(BaseModel):
    language: Optional[str] = None
    unit: Optional[str] = None
    preferred_audio_mode: Optional[str] = None
    guidance_tempo_multiplier: Optional[float] = None
    countdown_enabled: Optional[bool] = None
    haptic_level: Optional[str] = None
    verbosity: Optional[str] = None


class NavigationStateSnapshot(BaseModel):
    has_active_navigation: bool
    destination_id: Optional[str] = None
    destination_name: Optional[str] = None
    next_waypoint_name: Optional[str] = None
    distance_to_waypoint_m: Optional[float] = None
    heading_error_deg: Optional[float] = None
    off_route: bool = False


class AgentInterpretDestinationResponse(BaseModel):
    intent: str = "navigate"
    destination_query: DestinationQuery
    needs_clarification: bool = False
    message: Optional[str] = None
    response_language: str = "en"


class AgentResolveDestinationResponse(BaseModel):
    status: Literal["resolved", "ambiguous", "not_found"]
    needs_confirmation: bool = True
    candidates: List[DestinationCandidate] = Field(default_factory=list)
    message: Optional[str] = None
    response_language: str = "en"


class AgentAdjustPreferencesResponse(BaseModel):
    applied_changes: List[dict] = Field(default_factory=list)
    message: Optional[str] = None


class AgentExplainNavigationStateResponse(BaseModel):
    message: str
    state_summary: dict = Field(default_factory=dict)


class AgentFollowUpDestinationResponse(BaseModel):
    status: Literal["confirm", "matched", "narrowed", "unclear", "restart"]
    selected_destination_ids: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    response_language: str = "en"
