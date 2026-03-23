class AgentProvider:
    def interpret_destination(self, *, utterance: str, language: str, context: dict) -> dict:
        raise NotImplementedError

    def adjust_preferences(self, *, utterance: str, profile: dict, context: dict) -> dict:
        raise NotImplementedError

    def explain_navigation_state(self, *, question: str, context: dict) -> dict:
        raise NotImplementedError
