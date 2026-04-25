# ============================================================
#  src/movie_embedder.py  -- Encoding phim & query thanh vector
#  Reuse PhoBERT semantic head tu semantic_model.pt
#  Output: plot(256d) + genre(19d) + cast(256d) = 531d
# ============================================================

import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer

from src.config import (
    PHOBERT_MODEL, NLU_MAX_LEN, INTENT_LABELS, NUM_SLOTS,
    SIMCSE_CONFIG, SEMANTIC_MODEL_DIR, GENRE_ID_LIST,
    RECOMMENDATION_CONFIG, TRAIN_CONFIG,
)

logger = logging.getLogger(__name__)


class MovieEmbedder:
    """Encode phim va query vao cung khong gian embedding 531d.

    Su dung PhoBERT shared backbone (da train) cho plot & cast,
    multi-hot vector cho genre.
    """

    def __init__(self, model_dir=SEMANTIC_MODEL_DIR):
        from src.semantic_nlu import SemanticNLUModel

        self.proj_dim = SIMCSE_CONFIG.get("projection_dim", 256)
        self.genre_dim = len(GENRE_ID_LIST)
        self.weights = RECOMMENDATION_CONFIG["embedding_weights"]
        self.genre_id_list = GENRE_ID_LIST
        self.max_len = NLU_MAX_LEN

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load model
        self.model = SemanticNLUModel(
            PHOBERT_MODEL, len(INTENT_LABELS), NUM_SLOTS,
            dropout=TRAIN_CONFIG["dropout"],
            projection_dim=self.proj_dim,
        )
        ckpt = os.path.join(model_dir, "semantic_model.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=False)
            logger.info(f"Loaded semantic model from {ckpt}")
        else:
            logger.warning(f"No checkpoint found at {ckpt}, using random weights")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[MovieEmbedder] Ready on {self.device} | "
              f"plot={self.proj_dim}d + genre={self.genre_dim}d + cast={self.proj_dim}d "
              f"= {self.proj_dim + self.genre_dim + self.proj_dim}d")

    # ==============================================================
    #  Core encoding methods
    # ==============================================================

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text -> 256d vector qua PhoBERT + projection head."""
        enc = self.tokenizer(
            text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            emb = self.model.get_embedding(ids, attn)
        return emb[0].cpu().float().numpy()

    def encode_plot(self, text: str) -> np.ndarray:
        """Encode plot/overview -> 256d."""
        if not text or not text.strip():
            return np.zeros(self.proj_dim, dtype=np.float32)
        return self._encode_text(text)

    def encode_genre(self, genre_ids: list) -> np.ndarray:
        """Multi-hot encode genre IDs -> 19d, L2-normalized."""
        vec = np.zeros(self.genre_dim, dtype=np.float32)
        for gid in genre_ids:
            try:
                idx = self.genre_id_list.index(gid)
                vec[idx] = 1.0
            except ValueError:
                pass
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_cast(self, cast_names: list) -> np.ndarray:
        """Encode top-5 cast names -> 256d qua PhoBERT."""
        if not cast_names:
            return np.zeros(self.proj_dim, dtype=np.float32)
        text = ", ".join(cast_names[:5])
        return self._encode_text(text)

    def encode_movie(self, movie_data: dict) -> np.ndarray:
        """Encode 1 movie -> 531d weighted concatenation.

        movie_data can co: overview, genres/genre_ids, cast_names
        """
        plot = self.encode_plot(movie_data.get("overview", ""))

        # Genre IDs: ho tro ca format {id, name} va plain int
        raw_genres = movie_data.get("genres", movie_data.get("genre_ids", []))
        genre_ids = []
        for g in raw_genres:
            if isinstance(g, dict):
                genre_ids.append(g["id"])
            else:
                genre_ids.append(g)
        genre = self.encode_genre(genre_ids)

        cast_names = movie_data.get("cast_names", [])
        cast = self.encode_cast(cast_names)

        w = self.weights
        combined = np.concatenate([
            w["plot"] * plot,
            w["genre"] * genre,
            w["cast"] * cast,
        ])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined.astype(np.float32)

    def encode_query(self, query_vector=None, genre_ids=None,
                     person_name=None) -> np.ndarray:
        """Encode query vao cung khong gian voi movie embeddings.

        Args:
            query_vector: 256d vector tu NLU (da co san tu SemanticNLUInference)
            genre_ids: list TMDB genre IDs (da resolve boi recommendation engine)
            person_name: ten dien vien/dao dien
        """
        # Plot
        if query_vector is not None:
            plot = np.array(query_vector, dtype=np.float32)
        else:
            plot = np.zeros(self.proj_dim, dtype=np.float32)

        # Genre
        genre = self.encode_genre(genre_ids or [])

        # Cast
        if person_name:
            cast = self._encode_text(person_name)
        else:
            cast = np.zeros(self.proj_dim, dtype=np.float32)

        w = self.weights
        combined = np.concatenate([
            w["plot"] * plot,
            w["genre"] * genre,
            w["cast"] * cast,
        ])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined.astype(np.float32)

    def batch_encode_movies(self, movies: list) -> np.ndarray:
        """Batch encode danh sach movies -> (N, 531) array."""
        try:
            from tqdm import tqdm
            iterator = tqdm(movies, desc="Encoding movies")
        except ImportError:
            iterator = movies

        embeddings = []
        for movie in iterator:
            emb = self.encode_movie(movie)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)
