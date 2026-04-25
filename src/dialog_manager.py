# ============================================================
#  src/dialog_manager.py  -- Quan ly hoi thoai (Chuong 2 - Module 3)
#  Dieu phoi: NLU Pipeline -> Recommendation Engine -> Response
#  Context resolution tu ConversationMemory
# ============================================================

import logging

logger = logging.getLogger(__name__)


class DialogManager:
    """Quan ly luong hoi thoai chatbot phim.

    Nhan: nlu_pipeline, rec_engine, memory (tu app.py)
    Method chinh: respond(user_message) -> {"text": str, "cards": str}
    """

    def __init__(self, nlu_pipeline, rec_engine, memory):
        self.nlu = nlu_pipeline
        self.rec = rec_engine
        self.memory = memory

    def respond(self, user_message: str) -> dict:
        """Xu ly 1 tin nhan nguoi dung, tra ve {"text": str, "cards": str}."""

        # 1. NLU processing
        nlu_result = self.nlu.process(user_message)
        intent = nlu_result.get("intent", "out_of_scope")
        entities = nlu_result.get("entities", {})

        logger.info(
            f"[DM] intent={intent} conf={nlu_result.get('confidence', 0):.3f} "
            f"entities={entities}"
        )

        # 2. Context resolution: bo sung entity thieu tu memory
        context = self.memory.get_context()
        nlu_result = self._enrich_from_context(nlu_result, context)

        # 3. Dispatch xu ly qua recommendation engine
        result = self.rec.handle(nlu_result, context)

        # 4. Luu turn vao memory
        self.memory.add_turn(
            user_input=user_message,
            nlu_result=nlu_result,
            bot_response=result.get("text", ""),
        )

        return result

    def _enrich_from_context(self, nlu_result: dict, context: dict) -> dict:
        """Bo sung entities thieu tu accumulated context cua cac turn truoc.

        Chi bo sung khi intent yeu cau entity nhung khong co trong ket qua NLU hien tai.
        """
        intent = nlu_result.get("intent", "")
        entities = nlu_result.get("entities", {})
        acc = context.get("accumulated_entities", {})

        if not acc:
            return nlu_result

        # Intent nao can entity gi
        intent_needs = {
            "find_movie":      ["GENRE", "YEAR", "PERSON"],
            "recommendation":  ["GENRE", "YEAR", "PERSON", "MOVIE_TITLE"],
            "genre_filter":    ["GENRE", "YEAR"],
            "movie_info":      ["MOVIE_TITLE"],
            "person_info":     ["PERSON"],
        }

        needed = intent_needs.get(intent, [])
        if not needed:
            return nlu_result

        enriched = False
        for etype in needed:
            current = entities.get(etype, [])
            if not current and acc.get(etype):
                # Chi lay gia tri moi nhat tu accumulated
                entities[etype] = [acc[etype][-1]]
                enriched = True

        if enriched:
            nlu_result["entities"] = entities
            logger.info(f"[DM] Enriched entities from context: {entities}")

        return nlu_result
