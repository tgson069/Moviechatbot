# ============================================================
#  src/movie_database.py  -- Offline movie database + FAISS index
#  Luu tru metadata + embeddings, search + hard filter
#  CLI: python -m movie_chatbot_nlu.src.movie_database --build
# ============================================================

import json
import os
import logging
import numpy as np

from src.config import MOVIE_DB_DIR, RECOMMENDATION_CONFIG, GENRE_ALIASES

logger = logging.getLogger(__name__)


class MovieDatabase:
        def rating_boost(self, candidates):
            """Blend FAISS score with normalized rating. Returns re-scored, sorted list."""
            from src.config import RECOMMENDATION_CONFIG
            w = RECOMMENDATION_CONFIG.get("rating_boost_weight", 0.15)
            min_votes = RECOMMENDATION_CONFIG.get("rating_min_votes", 50)
            fallback = RECOMMENDATION_CONFIG.get("rating_fallback", 6.0)
            rescored = []
            for movie, faiss_score in candidates:
                vote_count = movie.get("vote_count", 0) or 0
                vote_avg = movie.get("vote_average", None)
                if vote_avg is None:
                    vote_avg = fallback
                elif vote_count < min_votes:
                    vote_avg = fallback
                norm_rating = float(vote_avg) / 10.0
                final_score = faiss_score * (1 - w) + norm_rating * w
                rescored.append((movie, final_score))
            rescored.sort(key=lambda x: x[1], reverse=True)
            return rescored
    """Offline movie database voi FAISS index cho similarity search."""

    def __init__(self, data_dir=MOVIE_DB_DIR):
        self.data_dir = data_dir
        self.movies = []            # list[dict] metadata
        self.embeddings = None      # np.ndarray (N, dim)
        self.index = None           # faiss.IndexFlatIP
        self._id_map = {}           # tmdb_id -> list index

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.movies) > 0

    # ==============================================================
    #  Build: crawl -> embed -> index
    # ==============================================================

    def build(self, embedder, tmdb_client, target=None):
        """Full pipeline: crawl TMDB -> extract metadata -> encode -> FAISS index."""
        import faiss

        target = target or RECOMMENDATION_CONFIG.get("tmdb_crawl_target", 5000)
        print(f"[MovieDatabase] Building with target={target} movies...")

        # Step 1: Crawl
        raw_movies = tmdb_client.crawl_full_database(target=target)
        print(f"  Crawled {len(raw_movies)} movies with details")

        # Step 2: Extract metadata
        self.movies = []
        for m in raw_movies:
            credits = m.get("credits", {})
            cast = credits.get("cast", [])[:5]
            crew = credits.get("crew", [])
            director = next(
                (c["name"] for c in crew if c.get("job") == "Director"), ""
            )
            movie = {
                "id": m["id"],
                "title": m.get("title", ""),
                "original_title": m.get("original_title", ""),
                "overview": m.get("overview", ""),
                "genres": [g["id"] for g in m.get("genres", [])],
                "genre_names": [g["name"] for g in m.get("genres", [])],
                "cast_names": [c["name"] for c in cast],
                "director": director,
                "release_date": m.get("release_date", ""),
                "year": (m.get("release_date", "") or "")[:4],
                "vote_average": m.get("vote_average", 0),
                "vote_count": m.get("vote_count", 0),
                "popularity": m.get("popularity", 0),
                "poster_path": m.get("poster_path", ""),
            }
            self.movies.append(movie)

        print(f"  Extracted metadata for {len(self.movies)} movies")

        # Step 3: Encode
        self.embeddings = embedder.batch_encode_movies(self.movies)
        print(f"  Embeddings shape: {self.embeddings.shape}")

        # Step 4: Build FAISS index (Inner Product = cosine similarity on L2-normed vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        # Step 5: ID map
        self._id_map = {m["id"]: i for i, m in enumerate(self.movies)}

        # Save
        self.save()
        print(f"[MovieDatabase] Build complete: {len(self.movies)} movies")

    # ==============================================================
    #  Save / Load
    # ==============================================================

    def save(self):
        """Luu metadata + embeddings + FAISS index ra disk."""
        import faiss

        os.makedirs(self.data_dir, exist_ok=True)

        meta_path = os.path.join(self.data_dir, "movies_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.movies, f, ensure_ascii=False, indent=2)

        emb_path = os.path.join(self.data_dir, "movie_embeddings.npy")
        np.save(emb_path, self.embeddings)

        idx_path = os.path.join(self.data_dir, "faiss_index.bin")
        faiss.write_index(self.index, idx_path)

        print(f"[MovieDatabase] Saved {len(self.movies)} movies -> {self.data_dir}")

    def load(self) -> bool:
        """Load tu disk. Tra ve True neu thanh cong."""
        import faiss

        meta_path = os.path.join(self.data_dir, "movies_metadata.json")
        emb_path = os.path.join(self.data_dir, "movie_embeddings.npy")
        idx_path = os.path.join(self.data_dir, "faiss_index.bin")

        if not all(os.path.exists(p) for p in [meta_path, emb_path, idx_path]):
            logger.warning(f"MovieDatabase files not found in {self.data_dir}")
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            self.movies = json.load(f)
        self.embeddings = np.load(emb_path)
        self.index = faiss.read_index(idx_path)
        self._id_map = {m["id"]: i for i, m in enumerate(self.movies)}

        print(f"[MovieDatabase] Loaded {len(self.movies)} movies from {self.data_dir}")
        return True

    # ==============================================================
    #  Search
    # ==============================================================

    def search(self, query_vector: np.ndarray, k: int = 100) -> list:
        """FAISS nearest neighbor search.

        Returns: list[(movie_dict, similarity_score)]
        """
        if not self.is_loaded:
            return []
        qv = np.array([query_vector], dtype=np.float32)
        k = min(k, len(self.movies))
        scores, indices = self.index.search(qv, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.movies[idx], float(score)))
        return results

    def search_by_title(self, title: str, limit: int = 5) -> list:
        """Tim phim theo ten (substring match).

        Returns: list[(movie_dict, match_score)]
        """
        if not self.movies:
            return []
        title_lower = title.lower().strip()
        results = []
        for movie in self.movies:
            m_title = movie.get("title", "").lower()
            m_orig = movie.get("original_title", "").lower()
            if title_lower == m_title or title_lower == m_orig:
                results.append((movie, 1.0))
            elif title_lower in m_title or title_lower in m_orig:
                results.append((movie, 0.9))
            elif m_title in title_lower or m_orig in title_lower:
                results.append((movie, 0.7))
        results.sort(key=lambda x: (-x[1], -x[0].get("popularity", 0)))
        return results[:limit]

    # ==============================================================
    #  Hard Filter
    # ==============================================================

    def hard_filter(self, candidates: list, filters: dict) -> list:
        """Loc candidates theo dieu kien cung: year, genre, person.

        Args:
            candidates: list[(movie_dict, score)]
            filters: dict voi keys YEAR, GENRE, PERSON (tu NLU hard_filters)
        Returns:
            list[(movie_dict, score)] da loc
        """
        if not filters:
            return candidates

        filtered = []
        for movie, score in candidates:
            # Year filter (±2 nam)
            if "YEAR" in filters:
                try:
                    target_year = int(filters["YEAR"])
                    movie_year = int(movie.get("year", "0") or "0")
                    if movie_year and abs(movie_year - target_year) > 2:
                        continue
                except (ValueError, TypeError):
                    pass

            # Genre filter
            if "GENRE" in filters:
                genre_filter = filters["GENRE"].lower().strip()
                canonical = GENRE_ALIASES.get(genre_filter, genre_filter).lower()
                movie_genres_lower = [g.lower() for g in movie.get("genre_names", [])]
                match = any(
                    canonical in mg or mg in canonical
                    for mg in movie_genres_lower
                )
                if not match:
                    continue

            # Person filter (in cast or director)
            if "PERSON" in filters:
                person = filters["PERSON"].lower()
                cast_lower = [n.lower() for n in movie.get("cast_names", [])]
                director_lower = movie.get("director", "").lower()
                match = any(person in c for c in cast_lower) or person in director_lower
                if not match:
                    continue

            filtered.append((movie, score))

        return filtered

    # ==============================================================
    #  Lookup
    # ==============================================================

    def get_movie_by_id(self, tmdb_id: int) -> dict | None:
        idx = self._id_map.get(tmdb_id)
        if idx is not None:
            return self.movies[idx]
        return None


# ==============================================================
#  CLI: python -m movie_chatbot_nlu.src.movie_database --build
# ==============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Movie Database Builder")
    parser.add_argument("--build", action="store_true", help="Build offline database")
    parser.add_argument("--target", type=int, default=5000, help="Number of movies to crawl")
    args = parser.parse_args()

    if args.build:
        from src.api_client import TMDBClient
        from src.movie_embedder import MovieEmbedder

        print("Initializing...")
        client = TMDBClient()
        embedder = MovieEmbedder()
        db = MovieDatabase()
        db.build(embedder, client, target=args.target)
    else:
        parser.print_help()
