# ============================================================
#  src/api_client.py  -- Ket noi va lay du lieu tu TMDB
#  Auth: Bearer token (Read Access Token) hoac api_key fallback
# ============================================================

import requests
import json
import time
import os
import re
import logging
from src.config import TMDB_API_KEY, TMDB_BASE_URL, DATA_RAW

logger = logging.getLogger(__name__)

# Regex cho phep Latin co ban + dau Tieng Viet + khoang trang / dau cau thong thuong
_VALID_NAME_RE = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿĀ-žƠơƯưẠ-ỹĐđ\s\-'.]+$"
)

def is_valid_entity(name: str) -> bool:
    """Loc ten: chi giu Latin + Viet (loai Trung/Han/Nhat/Anh-A-rap)."""
    if not name or len(name.strip()) < 2:
        return False
    return bool(_VALID_NAME_RE.match(name.strip()))


class TMDBClient:
    """Client lay du lieu phim tu TMDB.
    
    Doc token trong __init__ (khong o cap module) de tranh cache issue.
    Uu tien Bearer token > api_key query param.
    """

    def __init__(self, api_key: str = None, read_token: str = None):
        # Doc tu env var moi lan khoi tao de tranh module-level cache
        self.api_key    = api_key    or os.environ.get("TMDB_API_KEY", "")
        self.read_token = read_token or os.environ.get("TMDB_READ_TOKEN", "")
        self.base_url   = TMDB_BASE_URL
        self.session    = requests.Session()

        if self.read_token:
            # Dung Bearer token -- xac thuc chinh xac nhat
            self.session.headers.update({
                "Authorization": f"Bearer {self.read_token}",
                "Content-Type": "application/json",
            })
            self.session.params = {"language": "vi-VN"}
            print("[TMDBClient] Dang dung Bearer token (Read Access Token) OK")
        elif self.api_key:
            # Fallback: api_key query param
            self.session.params = {"api_key": self.api_key, "language": "vi-VN"}
            print("[TMDBClient] Dang dung api_key query param")
        else:
            raise ValueError(
                "Chua co credentials TMDB!\n"
                "Set it nhat 1 trong 2:\n"
                "  os.environ['TMDB_READ_TOKEN'] = 'eyJ...'  (uu tien)\n"
                "  os.environ['TMDB_API_KEY']    = 'abc...'"
            )

    def _get(self, endpoint: str, **params) -> dict:
        url = f"{self.base_url}/{endpoint}"
        last_error = None
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if resp.status_code == 401:
                    raise PermissionError(
                        "TMDB 401 Unauthorized!\n"
                        "Token/key khong hop le hoac het han.\n"
                        "Kiem tra lai tai: https://www.themoviedb.org/settings/api"
                    )
                last_error = e
                print(f"   Lan {attempt+1}/3 that bai: {e}")
                time.sleep(2 ** attempt)
            except requests.RequestException as e:
                last_error = e
                print(f"   Lan {attempt+1}/3 that bai: {e}")
                time.sleep(2 ** attempt)
        raise ConnectionError(
            f"TMDB API that bai sau 3 lan thu: {endpoint} - {last_error}"
        )

    def get_popular_movies(self, pages: int = 10) -> list:
        movies = []
        for page in range(1, pages + 1):
            try:
                data = self._get("movie/popular", page=page)
                movies.extend(data.get("results", []))
                print(f"  Trang {page}/{pages}: +{len(data.get('results', []))} phim")
            except ConnectionError as e:
                print(f"  Trang {page}/{pages}: LOI - {e}")
            time.sleep(0.25)
        return movies

    def get_movie_details(self, movie_id: int) -> dict:
        try:
            return self._get(f"movie/{movie_id}", append_to_response="credits")
        except (ConnectionError, PermissionError):
            return {}

    def build_entities(self, pages: int = 15) -> dict:
        movies   = self.get_popular_movies(pages=pages)
        entities = {"movies": [], "actors": set(), "directors": set(), "genres": set()}

        print(f"\nDang lay chi tiet {len(movies)} phim...")
        for i, m in enumerate(movies):
            if i % 50 == 0:
                print(f"  {i}/{len(movies)}")
            details = self.get_movie_details(m["id"])
            if not details:
                continue

            title = details.get("title", "")
            if title and is_valid_entity(title):
                entities["movies"].append(title)

            cast = details.get("credits", {}).get("cast", [])[:5]
            for c in cast:
                name = c["name"]
                if is_valid_entity(name):
                    entities["actors"].add(name)

            crew = details.get("credits", {}).get("crew", [])
            for c in crew:
                if c.get("job") == "Director":
                    name = c["name"]
                    if is_valid_entity(name):
                        entities["directors"].add(name)

            entities["genres"].update(g["name"] for g in details.get("genres", []))
            time.sleep(0.1)

        entities["actors"]    = sorted(entities["actors"])
        entities["directors"] = sorted(entities["directors"])
        entities["genres"]    = sorted(entities["genres"])
        return entities

    def save_entities(self, entities: dict, path: str = None):
        path = path or os.path.join(DATA_RAW, "entities.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        print(f"Da luu {sum(len(v) for v in entities.values())} thuc the -> {path}")

    # ==============================================================
    #  Batch crawl cho Recommendation Engine v2 (offline database)
    # ==============================================================

    def discover_all_movies(self, pages: int = 250, delay: float = 0.25) -> list:
        """Crawl phim qua discover/movie, sap xep theo popularity."""
        movies = []
        seen_ids = set()
        for page in range(1, pages + 1):
            try:
                data = self._get("discover/movie", **{
                    "sort_by": "popularity.desc",
                    "page": page,
                    "vote_count.gte": 10,
                })
                results = data.get("results", [])
                for m in results:
                    if m["id"] not in seen_ids:
                        movies.append(m)
                        seen_ids.add(m["id"])
                if not results:
                    break
                if page % 50 == 0:
                    print(f"  Discover: {page}/{pages} pages, {len(movies)} unique movies")
            except ConnectionError as e:
                print(f"  Page {page}: ERROR - {e}")
            time.sleep(delay)
        print(f"Discovered {len(movies)} unique movies from {page} pages")
        return movies

    def get_movie_details_batch(self, movie_ids: list, delay: float = 0.25) -> list:
        """Lay details+credits cho danh sach movie IDs."""
        details = []
        for i, mid in enumerate(movie_ids):
            try:
                d = self._get(f"movie/{mid}", append_to_response="credits")
                if d:
                    details.append(d)
            except (ConnectionError, PermissionError) as e:
                logger.warning(f"  Movie {mid}: ERROR - {e}")
            if (i + 1) % 100 == 0:
                print(f"  Details: {i+1}/{len(movie_ids)}")
            time.sleep(delay)
        return details

    def crawl_full_database(self, target: int = 5000) -> list:
        """Full pipeline: discover -> details -> list of movie dicts with credits."""
        pages = (target // 20) + 5
        print(f"Starting crawl: target={target} movies, {pages} pages")

        # Step 1: Discover
        basic = self.discover_all_movies(pages=pages, delay=0.25)
        basic = basic[:target]
        print(f"Step 1 done: {len(basic)} movies discovered")

        # Step 2: Get details with credits
        movie_ids = [m["id"] for m in basic]
        print(f"Step 2: Getting details for {len(movie_ids)} movies...")
        detailed = self.get_movie_details_batch(movie_ids, delay=0.25)
        print(f"Crawl complete: {len(detailed)} movies with full details")

        return detailed
