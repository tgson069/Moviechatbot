# ============================================================
#  src/recommendation.py  -- He thong khuyen nghi phim (Chuong 2)
#  Pipeline: FAISS retrieval -> Hard filter -> Ollama LLM Re-rank
#  Fallback: TMDB API (khi offline DB chua build)
# ============================================================

import logging
from src.api_client import TMDBClient
from src.config import RECOMMENDATION_CONFIG

logger = logging.getLogger(__name__)

# TMDB genre ID mapping (Vietnamese -> TMDB ID)
# https://api.themoviedb.org/3/genre/movie/list?language=vi-VN
GENRE_MAP = {
    "phim hành động": 28,
    "hành động": 28,
    "phim phiêu lưu": 12,
    "phiêu lưu": 12,
    "phim hoạt hình": 16,
    "hoạt hình": 16,
    "phim hài": 35,
    "hài": 35,
    "phim hình sự": 80,
    "hình sự": 80,
    "phim tài liệu": 99,
    "tài liệu": 99,
    "phim chính kịch": 18,
    "chính kịch": 18,
    "phim gia đình": 10751,
    "gia đình": 10751,
    "phim giả tượng": 14,
    "giả tượng": 14,
    "phim lịch sử": 36,
    "lịch sử": 36,
    "phim kinh dị": 27,
    "kinh dị": 27,
    "phim nhạc": 10402,
    "nhạc": 10402,
    "phim bí ẩn": 9648,
    "bí ẩn": 9648,
    "phim lãng mạn": 10749,
    "lãng mạn": 10749,
    "phim khoa học viễn tưởng": 878,
    "khoa học viễn tưởng": 878,
    "chương trình truyền hình": 10770,
    "truyền hình": 10770,
    "phim gây cấn": 53,
    "gây cấn": 53,
    "phim chiến tranh": 10752,
    "chiến tranh": 10752,
    "phim miền tây": 37,
    "miền tây": 37,
}

# Reverse map: TMDB genre_id -> Vietnamese name
GENRE_ID_TO_NAME = {
    28: "Phim Hành Động", 12: "Phim Phiêu Lưu", 16: "Phim Hoạt Hình",
    35: "Phim Hài", 80: "Phim Hình Sự", 99: "Phim Tài Liệu",
    18: "Phim Chính Kịch", 10751: "Phim Gia Đình", 14: "Phim Giả Tượng",
    36: "Phim Lịch Sử", 27: "Phim Kinh Dị", 10402: "Phim Nhạc",
    9648: "Phim Bí Ẩn", 10749: "Phim Lãng Mạn", 878: "Phim Khoa Học Viễn Tưởng",
    10770: "Chương Trình Truyền Hình", 53: "Phim Gây Cấn",
    10752: "Phim Chiến Tranh", 37: "Phim Miền Tây",
}

IMG_BASE = "https://image.tmdb.org/t/p"


class TMDBRecommendationEngine:
    """He thong khuyen nghi phim.

    Pipeline chinh (khi offline DB da build):
      Query -> encode_query -> FAISS top-K -> Hard filter -> Ollama LLM Re-rank
    Fallback (khi DB chua co):
      Goi TMDB API truc tiep (discover / search / similar)
    """

    def __init__(self):
        self.client = TMDBClient()

        # -- Offline components (FAISS + Embedder + LLM) --
        self.movie_db = None
        self.embedder = None
        self.reranker = None

        try:
            from src.movie_database import MovieDatabase
            self.movie_db = MovieDatabase()
            if self.movie_db.load():
                from src.movie_embedder import MovieEmbedder
                self.embedder = MovieEmbedder()
                logger.info("[RecEngine] Offline DB loaded OK")
            else:
                self.movie_db = None
                logger.info("[RecEngine] Offline DB not found, using API fallback")
        except Exception as e:
            logger.warning(f"[RecEngine] Cannot load offline DB: {e}")
            self.movie_db = None

        try:
            from src.llm_reranker import OllamaReranker
            self.reranker = OllamaReranker()
            if self.reranker.is_available():
                logger.info(f"[RecEngine] Ollama reranker ready ({self.reranker.model})")
            else:
                logger.info("[RecEngine] Ollama not available, re-rank disabled")
        except Exception as e:
            logger.warning(f"[RecEngine] Cannot init reranker: {e}")
            self.reranker = None

    @property
    def _use_faiss(self) -> bool:
        return (self.movie_db is not None and self.movie_db.is_loaded
                and self.embedder is not None)

    # ==========================================================
    #  PUBLIC: dispatch theo intent
    # ==========================================================

    def handle(self, nlu_result: dict, context: dict = None) -> dict:
        """Dispatch xu ly theo intent.

        Returns: {"text": str, "cards": str}
        """
        intent = nlu_result.get("intent", "out_of_scope")
        context = context or {}

        try:
            if intent == "find_movie":
                return self._handle_find_movie(nlu_result, context)
            elif intent == "recommendation":
                return self._handle_recommendation(nlu_result, context)
            elif intent == "person_info":
                return self._handle_person_info(nlu_result, context)
            elif intent == "genre_filter":
                return self._handle_genre_filter(nlu_result, context)
            elif intent == "movie_info":
                return self._handle_movie_info(nlu_result, context)
            elif intent == "greeting":
                return self._handle_greeting()
            elif intent == "goodbye":
                return self._handle_goodbye()
            else:
                return self._handle_out_of_scope()
        except Exception as e:
            logger.error(f"Loi xu ly intent '{intent}': {e}", exc_info=True)
            return {
                "text": "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại!",
                "cards": "",
            }

    # ==========================================================
    #  INTENT HANDLERS
    # ==========================================================

    def _handle_find_movie(self, nlu_result: dict, context: dict) -> dict:
        """find_movie: Tim 3 phim giong nhat voi mieu ta."""
        if self._use_faiss:
            result = self._faiss_find_movie(nlu_result, context)
            if result:
                return result
        return self._handle_find_movie_api(nlu_result, context)

    def _handle_recommendation(self, nlu_result: dict, context: dict) -> dict:
        """recommendation: Goi y 5 phim."""
        if self._use_faiss:
            result = self._faiss_recommendation(nlu_result, context)
            if result:
                return result
        return self._handle_recommendation_api(nlu_result, context)

    def _handle_genre_filter(self, nlu_result: dict, context: dict) -> dict:
        """genre_filter: Loc phim theo genre (chinh) + year."""
        if self._use_faiss:
            result = self._faiss_genre_filter(nlu_result, context)
            if result:
                return result
        return self._handle_genre_filter_api(nlu_result, context)

    # ==========================================================
    #  FAISS PIPELINE HANDLERS
    # ==========================================================

    def _faiss_find_movie(self, nlu_result: dict, context: dict) -> dict | None:
        """find_movie qua FAISS: title search -> embedding search."""
        entities = nlu_result.get("entities", {})
        hard_filters = nlu_result.get("hard_filters", {})
        title = self._resolve_title(entities, hard_filters)

        # Neu co title -> tim trong offline DB truoc
        if title:
            results = self.movie_db.search_by_title(title, limit=3)
            if results:
                movies = [self._offline_to_card_format(m) for m, s in results]
                return {
                    "text": f"Tìm thấy {len(movies)} phim liên quan đến \"{title}\":",
                    "cards": self._build_movie_cards(movies),
                }

        # FAISS embedding search
        return self._faiss_pipeline(nlu_result, context, limit=3)

    def _faiss_recommendation(self, nlu_result: dict, context: dict) -> dict | None:
        """recommendation qua FAISS: similar movie -> embedding search."""
        entities = nlu_result.get("entities", {})
        hard_filters = nlu_result.get("hard_filters", {})
        title = self._resolve_title(entities, hard_filters)

        # Neu co like_movie -> tim phim tuong tu trong offline DB
        if title:
            results = self.movie_db.search_by_title(title, limit=1)
            if results:
                liked_movie = results[0][0]
                idx = self.movie_db._id_map.get(liked_movie["id"])
                if idx is not None:
                    query_emb = self.movie_db.embeddings[idx]
                    cfg = RECOMMENDATION_CONFIG
                    candidates = self.movie_db.search(query_emb, k=cfg["faiss_top_k"] + 1)
                    # Loai bo phim goc
                    candidates = [(m, s) for m, s in candidates if m["id"] != liked_movie["id"]]
                    candidates = self.movie_db.hard_filter(candidates, hard_filters)
                    candidates = candidates[:cfg["hard_filter_top_k"]]

                    # LLM re-rank
                    query_text = nlu_result.get("input", "")
                    top_k = cfg["llm_rerank_top_k"]
                    if self.reranker and len(candidates) > top_k:
                        candidates = self.reranker.rerank(query_text, candidates, top_k=top_k)
                    else:
                        candidates = candidates[:top_k]

                    if candidates:
                        movies = [self._offline_to_card_format(m) for m, s in candidates]
                        text = f"Nếu bạn thích \"{title}\", có thể bạn sẽ thích:"
                        reasons = self._collect_reasons(candidates)
                        if reasons:
                            text += "\n\n" + "\n".join(reasons)
                        return {"text": text, "cards": self._build_movie_cards(movies)}

        # Khong co title -> FAISS pipeline voi entities
        return self._faiss_pipeline(nlu_result, context, limit=5,
                                    prefix="Gợi ý {n} phim{desc} cho bạn:")

    def _faiss_genre_filter(self, nlu_result: dict, context: dict) -> dict | None:
        """genre_filter qua FAISS: hard filter la chinh."""
        entities = nlu_result.get("entities", {})
        hard_filters = nlu_result.get("hard_filters", {})

        genre_ids = self._resolve_genre_ids(entities, hard_filters)
        if not genre_ids:
            # Context fallback
            acc = context.get("accumulated_entities", {})
            if acc.get("GENRE"):
                genre_ids = self._names_to_genre_ids(acc["GENRE"])
        if not genre_ids:
            return None  # fallback to API handler

        return self._faiss_pipeline(nlu_result, context, limit=5)

    def _faiss_pipeline(self, nlu_result: dict, context: dict,
                        limit: int = 5, prefix: str = None) -> dict | None:
        """Core FAISS pipeline: encode query -> search -> filter -> re-rank.

        Returns dict or None (de fallback sang API).
        """
        entities = nlu_result.get("entities", {})
        hard_filters = nlu_result.get("hard_filters", {})
        query_vector = nlu_result.get("query_vector")

        # Resolve entities + context fallback
        genre_ids = self._resolve_genre_ids(entities, hard_filters)
        year = self._resolve_year(entities, hard_filters)
        person_name = self._resolve_person(entities, hard_filters)

        acc = context.get("accumulated_entities", {})
        if not genre_ids and acc.get("GENRE"):
            genre_ids = self._names_to_genre_ids(acc["GENRE"])
        if not year and acc.get("YEAR"):
            year = acc["YEAR"][-1]
        if not person_name and acc.get("PERSON"):
            person_name = acc["PERSON"][-1]

        # Encode query
        qv = self.embedder.encode_query(
            query_vector=query_vector,
            genre_ids=genre_ids,
            person_name=person_name,
        )

        # FAISS search
        cfg = RECOMMENDATION_CONFIG
        candidates = self.movie_db.search(qv, k=cfg["faiss_top_k"])
        if not candidates:
            return None

        # --- Rating boost ---
        candidates = self.movie_db.rating_boost(candidates)

        # Hard filter
        filter_dict = dict(hard_filters)
        if year and "YEAR" not in filter_dict:
            filter_dict["YEAR"] = year
        candidates = self.movie_db.hard_filter(candidates, filter_dict)
        candidates = candidates[:cfg["hard_filter_top_k"]]

        if not candidates:
            return None

        # LLM re-rank
        query_text = nlu_result.get("input", "")
        top_k = min(limit, cfg["llm_rerank_top_k"])
        if self.reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(query_text, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        if not candidates:
            return None

        # Build response
        movies = [self._offline_to_card_format(m) for m, s in candidates]

        # Description
        parts = []
        if genre_ids:
            parts.append(self._genre_ids_to_text(genre_ids))
        if person_name:
            parts.append(person_name)
        if year:
            parts.append(f"năm {year}")
        desc = f" ({', '.join(parts)})" if parts else ""

        if prefix:
            text = prefix.format(n=len(candidates), desc=desc)
        else:
            text = f"Tìm thấy {len(candidates)} phim{desc}:"

        reasons = self._collect_reasons(candidates)
        if reasons:
            text += "\n\n" + "\n".join(reasons)

        return {"text": text, "cards": self._build_movie_cards(movies)}

    def _offline_to_card_format(self, movie: dict) -> dict:
        """Convert offline movie metadata sang format tuong thich voi card builders."""
        return {
            "title": movie.get("title", ""),
            "original_title": movie.get("original_title", ""),
            "vote_average": movie.get("vote_average", 0),
            "release_date": movie.get("release_date", ""),
            "overview": movie.get("overview", ""),
            "poster_path": movie.get("poster_path", ""),
            "genre_ids": movie.get("genres", []),
        }

    def _collect_reasons(self, candidates: list) -> list:
        """Thu thap rerank_reason tu LLM (neu co)."""
        reasons = []
        for m, s in candidates:
            reason = m.get("rerank_reason", "")
            if reason:
                reasons.append(f"• {m.get('title', '')}: {reason}")
        return reasons

    # ==========================================================
    #  API FALLBACK HANDLERS (khi offline DB chua build)
    # ==========================================================

    def _handle_find_movie_api(self, nlu_result: dict, context: dict) -> dict:
        """find_movie: Tim 3 phim giong nhat voi mieu ta."""
        entities = nlu_result.get("entities", {})
        hard_filters = nlu_result.get("hard_filters", {})

        # Extract filters
        genre_ids = self._resolve_genre_ids(entities, hard_filters)
        year = self._resolve_year(entities, hard_filters)
        person_name = self._resolve_person(entities, hard_filters)
        title = self._resolve_title(entities, hard_filters)

        # Neu co title cu the -> tim theo title truoc
        if title:
            movies = self._search_movies_by_title(title, limit=3)
            if movies:
                text = f"Tìm thấy {len(movies)} phim liên quan đến \"{title}\":"
                cards = self._build_movie_cards(movies)
                return {"text": text, "cards": cards}

        # Tim theo person (dien vien/dao dien)
        if person_name:
            person_id = self._search_person_id(person_name)
            if person_id:
                movies = self._discover_movies(
                    with_cast=person_id, with_genres=genre_ids,
                    year=year, limit=3,
                )
                if movies:
                    parts = []
                    if person_name:
                        parts.append(person_name)
                    if genre_ids:
                        parts.append(self._genre_ids_to_text(genre_ids))
                    if year:
                        parts.append(f"năm {year}")
                    desc = ", ".join(parts) if parts else ""
                    text = f"Tìm thấy {len(movies)} phim{' (' + desc + ')' if desc else ''}:"
                    cards = self._build_movie_cards(movies)
                    return {"text": text, "cards": cards}

        # Tim theo genre + year (discover)
        if genre_ids or year:
            movies = self._discover_movies(
                with_genres=genre_ids, year=year, limit=3,
            )
            if movies:
                parts = []
                if genre_ids:
                    parts.append(self._genre_ids_to_text(genre_ids))
                if year:
                    parts.append(f"năm {year}")
                desc = ", ".join(parts)
                text = f"Tìm thấy {len(movies)} phim ({desc}):"
                cards = self._build_movie_cards(movies)
                return {"text": text, "cards": cards}

        # Fallback: popular movies
        movies = self._get_popular(limit=3)
        return {
            "text": "Không tìm thấy phim phù hợp chính xác. Đây là một số phim phổ biến:",
            "cards": self._build_movie_cards(movies),
        }

    def _handle_recommendation_api(self, nlu_result: dict, context: dict) -> dict:
        entities = nlu_result.get("entities", {})
        hard_filters = nlu_result.get("hard_filters", {})

        # Neu co like_movie / title -> tim phim tuong tu
        title = self._resolve_title(entities, hard_filters)
        if title:
            movie_id = self._search_movie_id(title)
            if movie_id:
                movies = self._get_similar_movies(movie_id, limit=5)
                if movies:
                    return {
                        "text": f"Nếu bạn thích \"{title}\", có thể bạn sẽ thích:",
                        "cards": self._build_movie_cards(movies),
                    }

        # Goi y theo genre / person / year
        genre_ids = self._resolve_genre_ids(entities, hard_filters)
        year = self._resolve_year(entities, hard_filters)
        person_name = self._resolve_person(entities, hard_filters)

        # Context resolution: lay tu accumulated_entities neu thieu
        acc = context.get("accumulated_entities", {})
        if not genre_ids and acc.get("GENRE"):
            genre_ids = self._names_to_genre_ids(acc["GENRE"])
        if not year and acc.get("YEAR"):
            year = acc["YEAR"][-1]
        if not person_name and acc.get("PERSON"):
            person_name = acc["PERSON"][-1]

        person_id = self._search_person_id(person_name) if person_name else None
        movies = self._discover_movies(
            with_genres=genre_ids, with_cast=person_id,
            year=year, limit=5, sort_by="vote_average.desc",
        )
        if movies:
            parts = []
            if genre_ids:
                parts.append(self._genre_ids_to_text(genre_ids))
            if person_name:
                parts.append(person_name)
            if year:
                parts.append(f"năm {year}")
            desc = ", ".join(parts) if parts else "phổ biến"
            return {
                "text": f"Gợi ý {len(movies)} phim ({desc}) cho bạn:",
                "cards": self._build_movie_cards(movies),
            }

        # Fallback
        movies = self._get_popular(limit=5)
        return {
            "text": "Đây là 5 phim phổ biến được gợi ý cho bạn:",
            "cards": self._build_movie_cards(movies),
        }

    def _handle_person_info(self, nlu_result: dict, context: dict) -> dict:
        """person_info: Tim thong tin theo ten nguoi."""
        entities = nlu_result.get("entities", {})
        name = None
        for key in ("PERSON", "MOVIE_TITLE"):
            vals = entities.get(key, [])
            if vals:
                name = vals[0]
                break
        # Fallback: arguments
        if not name:
            args = nlu_result.get("arguments", {})
            for slot in ("name", "person"):
                if slot in args:
                    name = args[slot].get("value", "") if isinstance(args[slot], dict) else args[slot]
                    if name:
                        break

        if not name:
            return {"text": "Bạn muốn tìm thông tin về ai? Hãy cho tôi biết tên.", "cards": ""}

        data = self._search_person_detail(name)
        if not data:
            return {"text": f"Không tìm thấy thông tin về \"{name}\".", "cards": ""}

        # Build text summary
        pname = data.get("name", name)
        birthday = data.get("birthday", "N/A")
        birthplace = data.get("place_of_birth", "N/A")
        credits_ = data.get("movie_credits", data.get("combined_credits", {}))
        cast_list = credits_.get("cast", [])
        # Sap xep theo vote_average
        cast_list = sorted(cast_list, key=lambda x: x.get("vote_average", 0), reverse=True)
        top_movies = cast_list[:5]
        movie_names = [m.get("title", m.get("name", "")) for m in top_movies]

        text = (
            f"🎭 {pname}\n"
            f"🎂 Sinh: {birthday} | 📍 {birthplace}\n"
            f"🎬 Phim nổi bật: {', '.join(movie_names[:3])}"
        )

        card = self._build_person_card(data, top_movies)
        return {"text": text, "cards": card}

    def _handle_genre_filter_api(self, nlu_result: dict, context: dict) -> dict:
        entities = nlu_result.get("entities", {})
        hard_filters = nlu_result.get("hard_filters", {})

        genre_ids = self._resolve_genre_ids(entities, hard_filters)
        year = self._resolve_year(entities, hard_filters)

        # Context fallback
        acc = context.get("accumulated_entities", {})
        if not genre_ids and acc.get("GENRE"):
            genre_ids = self._names_to_genre_ids(acc["GENRE"])

        if not genre_ids:
            return {"text": "Bạn muốn xem phim thể loại gì? Hãy cho tôi biết.", "cards": ""}

        movies = self._discover_movies(
            with_genres=genre_ids, year=year, limit=5,
            sort_by="popularity.desc",
        )

        if not movies:
            return {
                "text": f"Không tìm thấy phim thể loại {self._genre_ids_to_text(genre_ids)}"
                        + (f" năm {year}" if year else "") + ".",
                "cards": "",
            }

        desc = self._genre_ids_to_text(genre_ids) + (f", năm {year}" if year else "")
        return {
            "text": f"Tìm thấy {len(movies)} phim ({desc}):",
            "cards": self._build_movie_cards(movies),
        }

    def _handle_movie_info(self, nlu_result: dict, context: dict) -> dict:
        """movie_info: Tra ve thong tin chi tiet phim."""
        entities = nlu_result.get("entities", {})
        title = None
        for key in ("MOVIE_TITLE",):
            vals = entities.get(key, [])
            if vals:
                title = vals[0]
                break
        if not title:
            args = nlu_result.get("arguments", {})
            if "title" in args:
                title = args["title"].get("value", "") if isinstance(args["title"], dict) else args["title"]

        if not title:
            return {"text": "Bạn muốn xem thông tin phim nào? Hãy cho tôi biết tên phim.", "cards": ""}

        movie_id = self._search_movie_id(title)
        if not movie_id:
            return {"text": f"Không tìm thấy phim \"{title}\".", "cards": ""}

        details = self.client.get_movie_details(movie_id)
        if not details:
            return {"text": f"Không lấy được thông tin phim \"{title}\".", "cards": ""}

        # Build text
        vi_title = details.get("title", title)
        orig_title = details.get("original_title", "")
        overview = details.get("overview", "Chưa có mô tả.")
        rating = details.get("vote_average", 0)
        release = details.get("release_date", "N/A")
        runtime = details.get("runtime", 0)
        genres = ", ".join(g["name"] for g in details.get("genres", []))

        credits_ = details.get("credits", {})
        cast = [c["name"] for c in credits_.get("cast", [])[:5]]
        directors = [c["name"] for c in credits_.get("crew", []) if c.get("job") == "Director"]

        text = (
            f"🎬 {vi_title}\n"
            f"⭐ {rating}/10 | 📅 {release} | ⏱ {runtime} phút\n"
            f"🎭 {genres}\n"
            f"🎥 Đạo diễn: {', '.join(directors) if directors else 'N/A'}\n"
            f"👥 Diễn viên: {', '.join(cast) if cast else 'N/A'}"
        )
        card = self._build_movie_detail_card(details)
        return {"text": text, "cards": card}

    def _handle_greeting(self) -> dict:
        return {
            "text": "Xin chào! 👋 Tôi là chatbot tư vấn phim. Bạn có thể hỏi tôi về:\n"
                    "- Tìm phim theo thể loại, năm, diễn viên\n"
                    "- Gợi ý phim hay\n"
                    "- Thông tin diễn viên, đạo diễn\n"
                    "- Thông tin chi tiết về một bộ phim\n\n"
                    "Hãy thử hỏi tôi nhé!",
            "cards": "",
        }

    def _handle_goodbye(self) -> dict:
        return {
            "text": "Tạm biệt! 👋 Hẹn gặp lại bạn. Chúc bạn xem phim vui vẻ!",
            "cards": "",
        }

    def _handle_out_of_scope(self) -> dict:
        return {
            "text": "Xin lỗi, tôi chỉ có thể hỗ trợ về phim ảnh. Bạn có thể hỏi tôi về:\n"
                    "- Tìm phim, gợi ý phim\n"
                    "- Thông tin diễn viên, đạo diễn\n"
                    "- Thông tin chi tiết phim",
            "cards": "",
        }

    # ==========================================================
    #  TMDB SEARCH / DISCOVER helpers
    # ==========================================================

    def _search_movies_by_title(self, title: str, limit: int = 3) -> list:
        data = self.client._get("search/movie", query=title)
        results = data.get("results", [])
        return results[:limit]

    def _search_movie_id(self, title: str) -> int | None:
        data = self.client._get("search/movie", query=title)
        results = data.get("results", [])
        return results[0]["id"] if results else None

    def _search_person_id(self, name: str) -> int | None:
        if not name:
            return None
        data = self.client._get("search/person", query=name)
        results = data.get("results", [])
        return results[0]["id"] if results else None

    def _search_person_detail(self, name: str) -> dict | None:
        pid = self._search_person_id(name)
        if not pid:
            return None
        return self.client._get(
            f"person/{pid}", append_to_response="movie_credits"
        )

    def _get_similar_movies(self, movie_id: int, limit: int = 5) -> list:
        data = self.client._get(f"movie/{movie_id}/recommendations")
        results = data.get("results", [])
        if len(results) < limit:
            data2 = self.client._get(f"movie/{movie_id}/similar")
            seen = {m["id"] for m in results}
            for m in data2.get("results", []):
                if m["id"] not in seen:
                    results.append(m)
                    if len(results) >= limit:
                        break
        return results[:limit]

    def _discover_movies(self, with_genres: list = None,
                         with_cast: int = None, year: str = None,
                         limit: int = 5, sort_by: str = "popularity.desc") -> list:
        params = {"sort_by": sort_by, "vote_count.gte": 50}
        if with_genres:
            params["with_genres"] = ",".join(str(g) for g in with_genres)
        if with_cast:
            params["with_cast"] = with_cast
        if year:
            params["primary_release_year"] = year
        data = self.client._get("discover/movie", **params)
        return data.get("results", [])[:limit]

    def _get_popular(self, limit: int = 5) -> list:
        data = self.client._get("movie/popular")
        return data.get("results", [])[:limit]

    # ==========================================================
    #  ENTITY RESOLUTION helpers
    # ==========================================================

    def _resolve_genre_ids(self, entities: dict, hard_filters: dict) -> list[int]:
        names = []
        for g in entities.get("GENRE", []):
            if g:
                names.append(g)
        if not names and hard_filters.get("GENRE"):
            names.append(hard_filters["GENRE"])
        return self._names_to_genre_ids(names)

    def _names_to_genre_ids(self, names: list) -> list[int]:
        ids = []
        for name in names:
            gid = GENRE_MAP.get(name.lower().strip())
            if gid and gid not in ids:
                ids.append(gid)
        return ids

    def _genre_ids_to_text(self, genre_ids: list) -> str:
        names = [GENRE_ID_TO_NAME.get(gid, str(gid)) for gid in genre_ids]
        return ", ".join(names)

    def _resolve_year(self, entities: dict, hard_filters: dict) -> str | None:
        years = entities.get("YEAR", [])
        if years:
            return years[0]
        return hard_filters.get("YEAR")

    def _resolve_person(self, entities: dict, hard_filters: dict) -> str | None:
        persons = entities.get("PERSON", [])
        if persons:
            return persons[0]
        return hard_filters.get("PERSON")

    def _resolve_title(self, entities: dict, hard_filters: dict) -> str | None:
        titles = entities.get("MOVIE_TITLE", [])
        if titles:
            return titles[0]
        return hard_filters.get("MOVIE_TITLE")

    # ==========================================================
    #  HTML CARD BUILDERS
    # ==========================================================

    def _build_movie_cards(self, movies: list) -> str:
        cards = []
        for m in movies:
            cards.append(self._build_single_movie_card(m))
        return "\n".join(cards)

    def _build_single_movie_card(self, m: dict) -> str:
        title = m.get("title", "N/A")
        orig = m.get("original_title", "")
        rating = m.get("vote_average", 0)
        release = m.get("release_date", "N/A")
        overview = m.get("overview", "")
        poster = m.get("poster_path", "")
        genre_ids = m.get("genre_ids", [])
        genres_text = ", ".join(
            GENRE_ID_TO_NAME.get(gid, "") for gid in genre_ids
            if gid in GENRE_ID_TO_NAME
        )
        img_url = f"{IMG_BASE}/w300{poster}" if poster else ""
        img_tag = f'<img src="{img_url}" style="width:130px;border-radius:8px;margin-right:12px;">' if img_url else ""
        overview_short = (overview[:200] + "...") if len(overview) > 200 else overview

        return f"""
        <div style="display:flex;border:1px solid #444;border-radius:10px;padding:12px;margin:6px 0;background:#1a1a2e;color:#eee;font-family:Arial;">
            {img_tag}
            <div style="flex:1;">
                <div style="font-size:15px;font-weight:bold;color:#e94560;margin-bottom:4px;">🎬 {title}</div>
                <div style="color:#aaa;font-size:11px;">{orig}</div>
                <div style="margin:4px 0;">⭐ {rating}/10 &nbsp; 📅 {release}</div>
                <div>🎭 {genres_text}</div>
                <div style="font-size:12px;color:#ccc;margin-top:6px;">{overview_short}</div>
            </div>
        </div>"""

    def _build_movie_detail_card(self, details: dict) -> str:
        title = details.get("title", "N/A")
        orig = details.get("original_title", "")
        rating = details.get("vote_average", 0)
        release = details.get("release_date", "N/A")
        runtime = details.get("runtime", 0)
        overview = details.get("overview", "Chưa có mô tả.")
        poster = details.get("poster_path", "")
        genres = ", ".join(g["name"] for g in details.get("genres", []))

        credits_ = details.get("credits", {})
        cast = credits_.get("cast", [])[:5]
        directors = [c["name"] for c in credits_.get("crew", []) if c.get("job") == "Director"]

        img_url = f"{IMG_BASE}/w300{poster}" if poster else ""
        img_tag = f'<img src="{img_url}" style="width:150px;border-radius:8px;margin-right:14px;">' if img_url else ""

        cast_html = ""
        if cast:
            cast_html = "<div style='margin-top:4px;'><b>Diễn viên:</b> " + ", ".join(
                c["name"] for c in cast
            ) + "</div>"

        dir_html = ""
        if directors:
            dir_html = f"<div><b>Đạo diễn:</b> {', '.join(directors)}</div>"

        return f"""
        <div style="display:flex;border:1px solid #444;border-radius:10px;padding:12px;margin:6px 0;background:#1a1a2e;color:#eee;font-family:Arial;">
            {img_tag}
            <div style="flex:1;">
                <div style="font-size:16px;font-weight:bold;color:#e94560;margin-bottom:4px;">🎬 {title}</div>
                <div style="color:#aaa;font-size:11px;">{orig}</div>
                <div style="margin:4px 0;">⭐ {rating}/10 &nbsp; 📅 {release} &nbsp; ⏱ {runtime} phút</div>
                <div>🎭 {genres}</div>
                {dir_html}
                {cast_html}
                <div style="font-size:12px;color:#ccc;margin-top:6px;">{overview}</div>
            </div>
        </div>"""

    def _build_person_card(self, person: dict, top_movies: list) -> str:
        name = person.get("name", "N/A")
        birthday = person.get("birthday", "N/A")
        birthplace = person.get("place_of_birth", "N/A")
        bio = person.get("biography", "")
        profile = person.get("profile_path", "")

        img_url = f"{IMG_BASE}/w185{profile}" if profile else ""
        img_tag = f'<img src="{img_url}" style="width:100px;border-radius:8px;margin-right:12px;">' if img_url else ""

        movies_html = ""
        if top_movies:
            items = []
            for m in top_movies[:5]:
                mtitle = m.get("title", m.get("name", ""))
                myear = (m.get("release_date", "") or "")[:4]
                mrating = m.get("vote_average", 0)
                items.append(f"<li>{mtitle} ({myear}) - ⭐{mrating}</li>")
            movies_html = "<div><b>Phim nổi bật:</b><ul style='margin:2px 0;'>" + "".join(items) + "</ul></div>"

        bio_short = (bio[:300] + "...") if len(bio) > 300 else bio

        return f"""
        <div style="display:flex;border:1px solid #444;border-radius:10px;padding:12px;margin:6px 0;background:#16213e;color:#eee;font-family:Arial;">
            {img_tag}
            <div style="flex:1;">
                <div style="font-size:15px;font-weight:bold;color:#0f3460;margin-bottom:4px;">🎭 {name}</div>
                <div>🎂 {birthday} &nbsp; 📍 {birthplace}</div>
                {movies_html}
                <div style="font-size:12px;color:#ccc;margin-top:6px;">{bio_short}</div>
            </div>
        </div>"""
