# ============================================================
#  src/memory.py  -- Bo nho hoi thoai (Chuong 2 - Module 2)
#  Ghi nho toi da 100 turn / cuoc hoi thoai
#  Tich luy entities qua cac turn de ho tro context resolution
# ============================================================

from datetime import datetime

MAX_TURNS = 100


class ConversationMemory:
    """Bo nho cho 1 cuoc hoi thoai.

    - Luu lich su toi da MAX_TURNS turn (FIFO khi vuot).
    - Tich luy entities qua cac turn de phuc vu context resolution.
    - Serializable de app.py luu/load JSON.
    """

    def __init__(self):
        self.session_start: str = datetime.now().isoformat()
        self.history: list[dict] = []
        self.accumulated_entities: dict[str, list[str]] = {}

    # ----------------------------------------------------------
    #  Them turn moi
    # ----------------------------------------------------------
    def add_turn(self, user_input: str, nlu_result: dict, bot_response: str):
        turn = {
            "turn_id": len(self.history) + 1,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "intent": nlu_result.get("intent", ""),
            "confidence": round(nlu_result.get("confidence", 0.0), 4),
            "entities": nlu_result.get("entities", {}),
            "bot_response": bot_response,
        }
        self.history.append(turn)

        # Enforce gioi han 100 turn (FIFO)
        if len(self.history) > MAX_TURNS:
            self.history = self.history[-MAX_TURNS:]

        # Tich luy entities
        for etype, values in nlu_result.get("entities", {}).items():
            if not values:
                continue
            if etype not in self.accumulated_entities:
                self.accumulated_entities[etype] = []
            for v in values:
                if v and v not in self.accumulated_entities[etype]:
                    self.accumulated_entities[etype].append(v)

    # ----------------------------------------------------------
    #  Context cho dialog manager
    # ----------------------------------------------------------
    def get_context(self, last_n: int = 5) -> dict:
        """Tra ve context gan nhat: accumulated entities + last N turns."""
        recent = self.history[-last_n:] if self.history else []
        return {
            "accumulated_entities": self.accumulated_entities,
            "recent_turns": recent,
            "total_turns": len(self.history),
        }

    def get_accumulated_entities(self) -> dict[str, list[str]]:
        return self.accumulated_entities

    def get_last_intent(self) -> str | None:
        if self.history:
            return self.history[-1].get("intent")
        return None

    def get_last_entities(self) -> dict:
        if self.history:
            return self.history[-1].get("entities", {})
        return {}

    # ----------------------------------------------------------
    #  Serialization (app.py goi truc tiep)
    # ----------------------------------------------------------
    def to_serializable(self) -> dict:
        return {
            "session_start": self.session_start,
            "history": self.history,
            "accumulated_entities": self.accumulated_entities,
        }

    def load_from_dict(self, data: dict):
        self.session_start = data.get("session_start", datetime.now().isoformat())
        self.history = data.get("history", [])
        self.accumulated_entities = data.get("accumulated_entities", {})
