# ============================================================
#  src/llm_reranker.py  -- LLM Re-ranking qua Ollama local
#  Nhan danh sach candidates tu FAISS, re-rank theo ngu canh
# ============================================================

import json
import logging
import requests

from src.config import RECOMMENDATION_CONFIG

logger = logging.getLogger(__name__)


class OllamaReranker:
    """Re-rank movie candidates bang LLM chay local qua Ollama.

    LLM doc query + candidate movies (plot + metadata)
    roi danh gia: phim nao phu hop ngu canh nhat.
    Graceful degradation: Ollama khong co -> tra candidates nguyen ban.
    """

    def __init__(self, model=None, base_url=None):
        self.model = model or RECOMMENDATION_CONFIG.get("ollama_model", "gemma3:4b")
        self.base_url = base_url or RECOMMENDATION_CONFIG.get(
            "ollama_base_url", "http://localhost:11434"
        )
        self._available = None

    def is_available(self) -> bool:
        """Kiem tra Ollama server co dang chay khong."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def rerank(self, query_text: str, candidates: list,
               top_k: int = 5) -> list:
        """Re-rank candidates bang LLM.

        Args:
            query_text: cau hoi goc cua user
            candidates: list[(movie_dict, score)] tu FAISS + hard filter
            top_k: so phim tra ve

        Returns:
            list[(movie_dict, score)] da re-rank, co them key 'rerank_reason'
        """
        if not candidates:
            return []

        if not self.is_available():
            logger.warning("[Reranker] Ollama not available, skipping re-rank")
            return candidates[:top_k]

        prompt = self._build_prompt(query_text, candidates, top_k)

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024},
                },
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            response_text = result.get("response", "")
            return self._parse_response(response_text, candidates, top_k)
        except Exception as e:
            logger.error(f"[Reranker] Ollama error: {e}")
            return candidates[:top_k]

    def _build_prompt(self, query_text: str, candidates: list,
                       top_k: int = 5) -> str:
        """Tao prompt tieng Viet cho LLM."""
        movie_lines = []
        for i, (movie, score) in enumerate(candidates):
            title = movie.get("title", "N/A")
            year = movie.get("year", "")
            genres = ", ".join(movie.get("genre_names", []))
            cast = ", ".join(movie.get("cast_names", [])[:3])
            overview = movie.get("overview", "")
            if len(overview) > 150:
                overview = overview[:150] + "..."
            movie_lines.append(
                f"{i+1}. [{movie['id']}] {title} ({year}) - {genres}"
                f" - Cast: {cast}\n   {overview}"
            )

        movies_text = "\n".join(movie_lines)
        n = min(top_k, len(candidates))

        return (
            f'Bạn là hệ thống gợi ý phim thông minh. '
            f'Người dùng hỏi: "{query_text}"\n\n'
            f'Danh sách phim ứng viên (đã sắp xếp theo độ tương đồng embedding):\n'
            f'{movies_text}\n\n'
            f'Nhiệm vụ: Chọn và xếp hạng {n} phim PHÙ HỢP NHẤT với yêu cầu.\n'
            f'Đánh giá không chỉ "giống" mà còn "đúng yêu cầu" '
            f'(thể loại, diễn viên, nội dung, ngữ cảnh).\n\n'
            f'Trả về ĐÚNG JSON array, không thêm text:\n'
            f'[{{"rank": 1, "id": <tmdb_id>, "reason": "<lý do ngắn>"}}, ...]'
        )

    def _parse_response(self, response_text: str, candidates: list,
                        top_k: int) -> list:
        """Parse JSON response tu LLM, fallback neu loi."""
        try:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                rankings = json.loads(json_str)

                # Map id -> (movie, score)
                id_map = {movie["id"]: (movie, score)
                          for movie, score in candidates}

                reranked = []
                seen = set()
                for item in rankings:
                    mid = item.get("id")
                    if mid in id_map and mid not in seen:
                        movie, score = id_map[mid]
                        movie_copy = dict(movie)
                        movie_copy["rerank_reason"] = item.get("reason", "")
                        reranked.append((movie_copy, score))
                        seen.add(mid)
                    if len(reranked) >= top_k:
                        break

                if reranked:
                    logger.info(f"[Reranker] Re-ranked {len(reranked)} movies")
                    return reranked

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[Reranker] Parse failed: {e}")

        # Fallback: tra ve thu tu goc
        return candidates[:top_k]
