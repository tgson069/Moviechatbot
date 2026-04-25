# ============================================================
#  src/config.py  -- Cau hinh tap trung cho toan bo du an
#  Phien ban: Semantic Parsing + Frame Semantics + SimCSE
#  Backbone : vinai/phobert-base-v2 (shared)
# ============================================================

import os

# -- API -------------------------------------------------------
TMDB_API_KEY  = os.environ.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise ValueError("Thieu TMDB_API_KEY! Hay set: os.environ['TMDB_API_KEY'] = 'your_key'")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# -- Duong dan -------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW         = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
INTENT_MODEL_DIR = os.path.join(MODELS_DIR, "intent_model")
SEMANTIC_MODEL_DIR = os.path.join(MODELS_DIR, "semantic_model")

# -- Mo hinh PhoBERT -------------------------------------------
PHOBERT_MODEL = "vinai/phobert-base-v2"
NLU_MAX_LEN   = 128

# -- Nhan y dinh (Intents) -- giu nguyen 8 classes -------------
INTENT_LABELS = [
    "find_movie",
    "recommendation",
    "movie_info",
    "person_info",
    "genre_filter",
    "greeting",
    "goodbye",
    "out_of_scope",
]
LABEL2ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# -- Frame Schema -----------------------------------------------
# Moi intent map sang 1 frame voi cac slots co ngu nghia
# Slot name -> entity type dung trong output cuoi
FRAME_SCHEMA = {
    "find_movie": {
        "frame": "FIND_MOVIE",
        "slots": {
            "genre":  "GENRE",
            "person": "PERSON",
            "year":   "YEAR",
            "title":  "MOVIE_TITLE",
        },
    },
    "recommendation": {
        "frame": "RECOMMENDATION",
        "slots": {
            "genre":     "GENRE",
            "person":    "PERSON",
            "year":      "YEAR",
            "like_movie":"MOVIE_TITLE",
        },
    },
    "movie_info": {
        "frame": "MOVIE_INFO",
        "slots": {
            "title": "MOVIE_TITLE",
        },
    },
    "person_info": {
        "frame": "PERSON_INFO",
        "slots": {
            "name": "PERSON",
        },
    },
    "genre_filter": {
        "frame": "GENRE_FILTER",
        "slots": {
            "genre": "GENRE",
            "year":  "YEAR",
        },
    },
    "greeting":     {"frame": "GREETING",     "slots": {}},
    "goodbye":      {"frame": "GOODBYE",      "slots": {}},
    "out_of_scope": {"frame": "OUT_OF_SCOPE", "slots": {}},
}

# Union cua tat ca slot names (dung cho model heads)
ALL_SLOT_NAMES = sorted({
    slot
    for schema in FRAME_SCHEMA.values()
    for slot in schema["slots"]
})
SLOT2IDX = {name: idx for idx, name in enumerate(ALL_SLOT_NAMES)}
IDX2SLOT = {idx: name for name, idx in SLOT2IDX.items()}
NUM_SLOTS = len(ALL_SLOT_NAMES)

# Mapping: slot_name -> entity_type (dung de convert arguments -> entities)
SLOT_TO_ENTITY = {}
for schema in FRAME_SCHEMA.values():
    for slot_name, entity_type in schema["slots"].items():
        SLOT_TO_ENTITY[slot_name] = entity_type

# Output entity labels cuoi cung (backward compat voi dialog_manager)
FINAL_ENTITY_LABELS = ["PERSON", "GENRE", "YEAR", "MOVIE_TITLE", "RATING"]

# -- Confidence threshold cho inference -------------------------
CONFIDENCE_THRESHOLD = 0.35
GENRE_ALIASES = {
    # Phim Hành Động
    "hành động": "Phim Hành Động", "action": "Phim Hành Động",
    "võ thuật": "Phim Hành Động", "đánh nhau": "Phim Hành Động",
    "bom tấn": "Phim Hành Động", "đấm đá": "Phim Hành Động",
    "đánh đấm": "Phim Hành Động", "kung fu": "Phim Hành Động",
    # Phim Kinh Dị
    "kinh dị": "Phim Kinh Dị", "horror": "Phim Kinh Dị",
    "phim ma": "Phim Kinh Dị", "ma": "Phim Kinh Dị",
    "rùng rợn": "Phim Kinh Dị", "ám ảnh": "Phim Kinh Dị",
    "ghê rợn": "Phim Kinh Dị", "kinh hoàng": "Phim Kinh Dị",
    "ma quỷ": "Phim Kinh Dị", "tâm linh": "Phim Kinh Dị",
    # Phim Lãng Mạn
    "lãng mạn": "Phim Lãng Mạn", "romance": "Phim Lãng Mạn",
    "tình cảm": "Phim Lãng Mạn", "tình yêu": "Phim Lãng Mạn",
    "ngôn tình": "Phim Lãng Mạn", "yêu đương": "Phim Lãng Mạn",
    "romantic": "Phim Lãng Mạn",
    # Phim Hài
    "hài": "Phim Hài", "hài hước": "Phim Hài", "comedy": "Phim Hài",
    "vui": "Phim Hài", "cười": "Phim Hài", "giải trí": "Phim Hài",
    "hài kịch": "Phim Hài", "funny": "Phim Hài", "tấu hài": "Phim Hài",
    # Phim Khoa Học Viễn Tưởng
    "viễn tưởng": "Phim Khoa Học Viễn Tưởng", "sci-fi": "Phim Khoa Học Viễn Tưởng",
    "khoa học viễn tưởng": "Phim Khoa Học Viễn Tưởng",
    "tương lai": "Phim Khoa Học Viễn Tưởng", "vũ trụ": "Phim Khoa Học Viễn Tưởng",
    "ngoài hành tinh": "Phim Khoa Học Viễn Tưởng",
    # Phim Hoạt Hình
    "hoạt hình": "Phim Hoạt Hình", "anime": "Phim Hoạt Hình",
    "cartoon": "Phim Hoạt Hình", "animation": "Phim Hoạt Hình",
    # Phim Phiêu Lưu
    "phiêu lưu": "Phim Phiêu Lưu", "adventure": "Phim Phiêu Lưu",
    "mạo hiểm": "Phim Phiêu Lưu", "thám hiểm": "Phim Phiêu Lưu",
    # Phim Chính Kịch
    "chính kịch": "Phim Chính Kịch", "drama": "Phim Chính Kịch",
    "tâm lý": "Phim Chính Kịch", "xã hội": "Phim Chính Kịch",
    # Phim Gây Cấn
    "gây cấn": "Phim Gây Cấn", "thriller": "Phim Gây Cấn",
    "hồi hộp": "Phim Gây Cấn", "kịch tính": "Phim Gây Cấn",
    "suspense": "Phim Gây Cấn",
    # Phim Bí Ẩn
    "bí ẩn": "Phim Bí Ẩn", "mystery": "Phim Bí Ẩn",
    "trinh thám": "Phim Bí Ẩn", "điều tra": "Phim Bí Ẩn",
    "detective": "Phim Bí Ẩn",
    # Phim Chiến Tranh
    "chiến tranh": "Phim Chiến Tranh", "war": "Phim Chiến Tranh",
    "quân đội": "Phim Chiến Tranh",
    # Phim Tài Liệu
    "tài liệu": "Phim Tài Liệu", "documentary": "Phim Tài Liệu",
    # Phim Gia Đình
    "gia đình": "Phim Gia Đình", "family": "Phim Gia Đình",
    "cho cả nhà": "Phim Gia Đình",
    # Phim Hình Sự
    "hình sự": "Phim Hình Sự", "crime": "Phim Hình Sự",
    "tội phạm": "Phim Hình Sự", "mafia": "Phim Hình Sự",
    "xã hội đen": "Phim Hình Sự", "băng đảng": "Phim Hình Sự",
    # Phim Lịch Sử
    "lịch sử": "Phim Lịch Sử", "history": "Phim Lịch Sử",
    "cổ trang": "Phim Lịch Sử",
    # Phim Giả Tượng
    "giả tưởng": "Phim Giả Tượng", "fantasy": "Phim Giả Tượng",
    "phép thuật": "Phim Giả Tượng", "thần thoại": "Phim Giả Tượng",
    "siêu nhiên": "Phim Giả Tượng",
    # Phim Nhạc
    "nhạc": "Phim Nhạc", "âm nhạc": "Phim Nhạc", "musical": "Phim Nhạc",
    # Phim Miền Tây
    "miền tây": "Phim Miền Tây", "western": "Phim Miền Tây",
    "cao bồi": "Phim Miền Tây",
}
# -- Sieu tham so Stage 1: Joint Intent + Frame Arg Extraction ---
TRAIN_CONFIG = {
    "num_train_epochs"      : 15,
    "batch_size"            : 16,
    "learning_rate"         : 2e-5,
    "warmup_ratio"          : 0.1,
    "weight_decay"          : 0.01,
    "dropout"               : 0.4,
    "early_stopping_patience": 5,
    "label_smoothing"       : 0.1,
    "intent_loss_weight"    : 1.0,   # alpha
    "slot_loss_weight"      : 0.5,   # beta
    # --- New: Focal Loss + R-Drop + Mixup ---
    "use_focal_loss"        : True,
    "focal_gamma"           : 2.0,
    "use_rdrop"             : True,
    "rdrop_alpha"           : 0.7,
    "use_mixup"             : True,
    "mixup_alpha"           : 0.2,
}

# -- Sieu tham so Stage 2: SimCSE Contrastive Embedding ----------
SIMCSE_CONFIG = {
    "num_train_epochs" : 5,
    "batch_size"       : 32,
    "learning_rate"    : 1e-5,
    "warmup_ratio"     : 0.1,
    "weight_decay"     : 0.01,
    "dropout"          : 0.1,
    "projection_dim"   : 256,
    "temperature"      : 0.05,
}

# -- Legacy compat (de cac module cu khong crash khi import) ------
NER_LABELS = ["PERSON", "GENRE", "YEAR", "MOVIE_TITLE", "RATING"]
NER_MODEL_DIR = os.path.join(MODELS_DIR, "ner_model")
NER_TRAIN_CONFIG = TRAIN_CONFIG.copy()
NER_BIO_LABELS = ["O"] + [f"B-{l}" for l in NER_LABELS] + [f"I-{l}" for l in NER_LABELS]
NER_BIO_LABEL2ID = {label: idx for idx, label in enumerate(NER_BIO_LABELS)}
NER_BIO_ID2LABEL = {idx: label for label, idx in NER_BIO_LABEL2ID.items()}

# -- Recommendation Engine v2 (FAISS + Ollama LLM Re-rank) ------
MOVIE_DB_DIR = os.path.join(BASE_DIR, "data", "movies")

# 19 TMDB genre IDs (thu tu co dinh cho multi-hot encoding)
GENRE_ID_LIST = [
    28, 12, 16, 35, 80, 99, 18, 10751, 14,
    36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37,
]

RECOMMENDATION_CONFIG = {
    "faiss_top_k"       : 100,
    "hard_filter_top_k" : 20,
    "llm_rerank_top_k"  : 5,
    "embedding_weights" : {"plot": 0.5, "genre": 0.3, "cast": 0.2},
    "embedding_dim"     : 256 + len(GENRE_ID_LIST) + 256,   # 531
    "ollama_model"      : "gemma3:4b",
    "ollama_base_url"   : "http://localhost:11434",
    "tmdb_crawl_target" : 5000,
    "crawl_delay"       : 0.25,
    # --- Rating boost config ---
    "rating_boost_weight": 0.15,   # Weight for rating in final score
    "rating_min_votes": 50,        # Min votes to trust rating
    "rating_fallback": 6.0,        # Fallback rating if too few votes
}

print("Config loaded OK (Semantic Parsing + SimCSE)")
print(f"PHOBERT_MODEL   : {PHOBERT_MODEL}")
print(f"INTENT_LABELS   : {INTENT_LABELS}")
print(f"ALL_SLOT_NAMES  : {ALL_SLOT_NAMES}")
print(f"NUM_SLOTS       : {NUM_SLOTS}")
print(f"FRAME_SCHEMA    : {len(FRAME_SCHEMA)} frames")
print(f"MOVIE_DB_DIR    : {MOVIE_DB_DIR}")
print(f"EMBEDDING_DIM   : {RECOMMENDATION_CONFIG['embedding_dim']}")
