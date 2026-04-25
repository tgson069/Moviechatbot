# ============================================================
#  main.py  -- Pipeline NLU hoan chinh (Chuong 1 -- SemanticNLU)
# ============================================================

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from src.semantic_nlu import SemanticNLUInference
from src.config import SEMANTIC_MODEL_DIR


class NLUPipeline:
    """
    Pipeline NLU dung PhoBERT-base-v2 Shared Backbone.
      - Intent  : joint-trained classifier (8 lop)
      - Slots   : per-slot start/end extraction (QA-style)
      - Embedding: mean-pool -> MLP projection (256d, L2-norm)
    Output JSON tuong thich Chuong 2.
    """

    def __init__(self, model_path: str = SEMANTIC_MODEL_DIR,
                 entities: dict = None):
        print("Dang load NLU Pipeline (SemanticNLU)...")
        self.nlu = SemanticNLUInference(model_path, entities=entities)
        print("OK NLU Pipeline san sang!")

    def load_entities(self, entities: dict):
        """Nap lai entity vocab (dung khi entities.json cap nhat)."""
        self.nlu.load_entities(entities)

    def process(self, user_input: str) -> dict:
        """
        Input : cau hoi tieng Viet
        Output: JSON chuong 2
        """
        return self.nlu.process(user_input)

    def process_batch(self, queries: list) -> list:
        return [self.process(q) for q in queries]


if __name__ == "__main__":
    pipeline = NLUPipeline()
    test_query = "Tim phim hanh dong cua Thanh Long nam 2010"
    result = pipeline.process(test_query)
    display = {k: v for k, v in result.items() if k != "query_vector"}
    print(json.dumps(display, ensure_ascii=False, indent=2))
