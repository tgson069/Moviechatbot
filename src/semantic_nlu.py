# ============================================================
#  src/semantic_nlu.py -- Semantic NLU Pipeline
#  Frame Semantics + Joint Intent/Slot Extraction + SimCSE
#  Shared PhoBERT-base-v2 backbone (1 model, 3 heads)
# ============================================================
import json, re, os, logging, time, random
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, RobertaModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from src.config import (
    PHOBERT_MODEL, NLU_MAX_LEN,
    INTENT_LABELS, FINAL_ENTITY_LABELS,
    LABEL2ID, ID2LABEL, TRAIN_CONFIG, CONFIDENCE_THRESHOLD,
    FRAME_SCHEMA, ALL_SLOT_NAMES, SLOT2IDX, IDX2SLOT, NUM_SLOTS,
    SLOT_TO_ENTITY, SEMANTIC_MODEL_DIR, SIMCSE_CONFIG,
    GENRE_ALIASES,
)
from src.preprocessing import VietnamesePreprocessor
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================
#  Focal Loss (giúp tập trung vào samples khó)
# =============================================================
class FocalLoss(nn.Module):
    """Focal Loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
    Khi gamma=0 -> tương đương CrossEntropy."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


# =============================================================
#  Mixup trên embedding (data augmentation trong training)
# =============================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup: tạo convex combination của 2 samples.
    Trả về mixed_x, y_a, y_b, lam (lambda)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: lambda * L(pred, y_a) + (1-lambda) * L(pred, y_b)"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================
#  1. Frame Dataset
# =============================================================
class FrameDataset(Dataset):
    """Chuyen frame-annotated samples thanh tensor.
    Char-level argument spans -> token-level start/end positions.
    Position 0 ([CLS]) = slot khong co mat."""

    def __init__(self, samples, tokenizer, max_len,
                 num_slots=NUM_SLOTS, slot2idx=SLOT2IDX):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_slots = num_slots
        self.slot2idx = slot2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        arguments = item.get("arguments", {})
        label_id = item.get("label_id", 0)

        words = text.split()
        word_positions = []
        search_from = 0
        for w in words:
            start = text.index(w, search_from)
            word_positions.append((start, start + len(w)))
            search_from = start + len(w)

        all_token_ids = []
        word_token_spans = []
        for word in words:
            sub_tokens = self.tokenizer.tokenize(word)
            sub_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            if not sub_ids:
                sub_ids = [self.tokenizer.unk_token_id]
            first_idx = len(all_token_ids)
            all_token_ids.extend(sub_ids)
            last_idx = len(all_token_ids) - 1
            word_token_spans.append((first_idx, last_idx))

        max_content = self.max_len - 2
        all_token_ids = all_token_ids[:max_content]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        input_ids = [cls_id] + all_token_ids + [sep_id]
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_len - len(input_ids)
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len

        slot_starts = [0] * self.num_slots
        slot_ends = [0] * self.num_slots
        for slot_name, arg in arguments.items():
            if slot_name not in self.slot2idx:
                continue
            s_idx = self.slot2idx[slot_name]
            arg_cs, arg_ce = arg["start"], arg["end"]
            first_word = last_word = None
            for wi, (ws, we) in enumerate(word_positions):
                if ws >= arg_cs and we <= arg_ce:
                    if first_word is None:
                        first_word = wi
                    last_word = wi
            if first_word is not None and last_word is not None:
                ft = word_token_spans[first_word][0] + 1
                lt = word_token_spans[last_word][1] + 1
                if ft < self.max_len - 1 and lt < self.max_len - 1:
                    slot_starts[s_idx] = ft
                    slot_ends[s_idx] = lt

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "intent_label": torch.tensor(label_id, dtype=torch.long),
            "slot_starts": torch.tensor(slot_starts, dtype=torch.long),
            "slot_ends": torch.tensor(slot_ends, dtype=torch.long),
        }


# =============================================================
#  2. Semantic NLU Model (Shared Backbone + 3 Heads)
# =============================================================
class SemanticNLUModel(nn.Module):
    """PhoBERT shared backbone:
      Head 1 -- Intent:  [CLS] -> Dropout -> Dense(256) -> Linear(num_intents)
      Head 2 -- Slots:   seq -> Linear(H, num_slots) x2 (start + end)
      Head 3 -- Embed:   mean-pool -> MLP(H->H->proj) -> L2-norm
    """
    def __init__(self, model_name, num_intents, num_slots,
                 dropout=0.3, projection_dim=256):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(model_name, use_safetensors=True)
        H = self.bert.config.hidden_size
        self.num_slots = num_slots
        self.intent_drop1 = nn.Dropout(dropout)
        self.intent_dense = nn.Linear(H, 256)
        self.intent_drop2 = nn.Dropout(dropout)
        self.intent_out = nn.Linear(256, num_intents)
        self.slot_start = nn.Linear(H, num_slots)
        self.slot_end = nn.Linear(H, num_slots)
        self.projection = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Linear(H, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        cls = seq[:, 0, :]
        x = self.intent_drop1(cls)
        x = torch.relu(self.intent_dense(x))
        x = self.intent_drop2(x)
        intent_logits = self.intent_out(x)
        start_logits = self.slot_start(seq)
        end_logits = self.slot_end(seq)
        return intent_logits, start_logits, end_logits

    def get_embedding(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
        proj = self.projection(pooled)
        return nn.functional.normalize(proj, p=2, dim=-1)


# =============================================================
#  3. Semantic NLU Trainer (Joint Intent + Slot + FP16)
# =============================================================
class SemanticNLUTrainer:
    """L = alpha*L_intent + beta*L_slot
    Early stopping: score = 0.6*intent_f1 + 0.4*slot_acc"""

    def __init__(self):
        print(f"Loading SemanticNLUModel ({PHOBERT_MODEL})...")
        self.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
        self.model = SemanticNLUModel(
            PHOBERT_MODEL, len(INTENT_LABELS), NUM_SLOTS,
            dropout=TRAIN_CONFIG["dropout"],
            projection_dim=SIMCSE_CONFIG.get("projection_dim", 256),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.use_fp16 = (self.device == "cuda")
        tp = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"OK -- {self.device} | {tp:.1f}M | FP16: {self.use_fp16}")
        print(f"   Intent({len(INTENT_LABELS)}) + Slot({NUM_SLOTS}) + Embed({SIMCSE_CONFIG.get('projection_dim',256)}d)")

    @staticmethod
    def _fmt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0: return f"{h}h {m:02d}m {s:02d}s"
        if m > 0: return f"{m}m {s:02d}s"
        return f"{s}s"

    def _class_weights(self, data):
        counts = Counter(s.get("label_id", 0) for s in data)
        n, nc = len(data), len(INTENT_LABELS)
        return torch.tensor([n / (nc * counts.get(i, 1)) for i in range(nc)], dtype=torch.float32)

    def train(self, train_data, val_data, output_dir=SEMANTIC_MODEL_DIR):
        cfg = TRAIN_CONFIG
        use_focal = cfg.get("use_focal_loss", True)
        focal_gamma = cfg.get("focal_gamma", 2.0)
        use_rdrop = cfg.get("use_rdrop", True)
        rdrop_alpha = cfg.get("rdrop_alpha", 0.7)
        use_mixup = cfg.get("use_mixup", True)
        mixup_alpha = cfg.get("mixup_alpha", 0.2)
        train_ds = FrameDataset(train_data, self.tokenizer, NLU_MAX_LEN)
        val_ds = FrameDataset(val_data, self.tokenizer, NLU_MAX_LEN)
        tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
        vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        opt = AdamW(self.model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        ts = len(tl) * cfg["num_train_epochs"]
        ws = int(ts * cfg["warmup_ratio"])
        sched = get_linear_schedule_with_warmup(opt, ws, ts)

        cw = self._class_weights(train_data).to(self.device)
        cw = cw.clamp(max=10.0)  # Cap weights de tranh loss explosion
        if use_focal:
            i_crit = FocalLoss(weight=cw, gamma=focal_gamma,
                               label_smoothing=cfg.get("label_smoothing", 0.1))
            print(f"   Using FocalLoss (gamma={focal_gamma})")
        else:
            i_crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.get("label_smoothing", 0.1))
        s_crit = nn.CrossEntropyLoss()
        alpha = cfg.get("intent_loss_weight", 1.0)
        beta = cfg.get("slot_loss_weight", 0.5)
        scaler = GradScaler(enabled=self.use_fp16)

        best = 0.0
        pc = 0
        pat = cfg.get("early_stopping_patience", 5)
        ne = cfg["num_train_epochs"]

        print(f"\nJoint training ({ne} epochs, alpha={alpha}, beta={beta})...")
        print(f"   train: {len(train_data)} | val: {len(val_data)} | batch: {cfg['batch_size']}")
        print(f"   FocalLoss: {use_focal} | R-Drop: {use_rdrop} (alpha={rdrop_alpha}) | Mixup: {use_mixup}")
        print("=" * 75)
        train_start = time.time()   # <-- thêm dòng này
        for ep in range(ne):
            ep_start = time.time()
            self.model.train()
            tloss = 0.0
            ic = it = 0
            for b in tl:
                ids = b["input_ids"].to(self.device)
                msk = b["attention_mask"].to(self.device)
                il = b["intent_label"].to(self.device)
                ss = b["slot_starts"].to(self.device)
                se = b["slot_ends"].to(self.device)

                opt.zero_grad()
                with autocast(enabled=self.use_fp16):
                    ilog, sls, sle = self.model(ids, msk)
                    li = i_crit(ilog, il)

                    # R-Drop: forward lần 2, tính KL divergence
                    if use_rdrop:
                        ilog2, sls2, sle2 = self.model(ids, msk)
                        li2 = i_crit(ilog2, il)
                        p1 = F.log_softmax(ilog, dim=-1)
                        p2 = F.log_softmax(ilog2, dim=-1)
                        q1 = F.softmax(ilog, dim=-1)
                        q2 = F.softmax(ilog2, dim=-1)
                        kl = 0.5 * (F.kl_div(p1, q2, reduction="batchmean")
                                     + F.kl_div(p2, q1, reduction="batchmean"))
                        li = 0.5 * (li + li2) + rdrop_alpha * kl

                    # Mixup on intent logits
                    if use_mixup and random.random() < 0.3:
                        mixed_ilog, ya, yb, lam = mixup_data(ilog, il, mixup_alpha)
                        li_mix = mixup_criterion(i_crit, mixed_ilog, ya, yb, lam)
                        li = 0.5 * li + 0.5 * li_mix

                    ls = torch.tensor(0.0, device=self.device)
                    for si in range(NUM_SLOTS):
                        ls = ls + s_crit(sls[:, :, si], ss[:, si])
                        ls = ls + s_crit(sle[:, :, si], se[:, si])
                    ls = ls / (2 * NUM_SLOTS)
                    loss = alpha * li + beta * ls

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tloss += loss.item()
                preds = ilog.argmax(-1)
                ic += (preds == il).sum().item()
                it += il.size(0)

            ta = ic / it

            self.model.eval()
            vp, vlb = [], []
            sok = stot = 0
            with torch.no_grad():
                for b in vl:
                    ids = b["input_ids"].to(self.device)
                    msk = b["attention_mask"].to(self.device)
                    il = b["intent_label"].to(self.device)
                    ss = b["slot_starts"].to(self.device)
                    se = b["slot_ends"].to(self.device)
                    with autocast(enabled=self.use_fp16):
                        ilog, sls, sle = self.model(ids, msk)
                    vp.extend(ilog.argmax(-1).cpu().tolist())
                    vlb.extend(il.cpu().tolist())
                    for si in range(NUM_SLOTS):
                        ps = sls[:, :, si].argmax(-1)
                        pe = sle[:, :, si].argmax(-1)
                        sok += (ps == ss[:, si]).sum().item()
                        sok += (pe == se[:, si]).sum().item()
                        stot += 2 * ss.size(0)

            nc = len(INTENT_LABELS)
            pf, psu = [], []
            for c in range(nc):
                tp = sum(1 for p, t in zip(vp, vlb) if p == c and t == c)
                fp = sum(1 for p, t in zip(vp, vlb) if p == c and t != c)
                fn = sum(1 for p, t in zip(vp, vlb) if p != c and t == c)
                sup = tp + fn
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                pf.append(2 * prec * rec / max(prec + rec, 1e-8))
                psu.append(sup)
            tsu = sum(psu)
            vf1 = sum(f * s for f, s in zip(pf, psu)) / max(tsu, 1)
            va = sum(1 for p, t in zip(vp, vlb) if p == t) / max(len(vlb), 1)
            sa = sok / max(stot, 1)
            score = 0.6 * vf1 + 0.4 * sa
            al = tloss / len(tl)

            if score > best:
                best = score
                pc = 0
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(output_dir, "semantic_model.pt"))
                self.tokenizer.save_pretrained(output_dir)
                sv = " <- saved"
            else:
                pc += 1
                sv = f" ({pc}/{pat})"
            ep_elapsed = time.time() - ep_start
            total_elapsed = time.time() - train_start
            avg_ep_time = total_elapsed / (ep + 1)
            eta = avg_ep_time * (ne - ep - 1)

            print(f"Ep {ep+1:02d}/{ne} | Loss {al:.4f} | TrainAcc {ta:.3f} | "
                  f"ValF1 {vf1:.4f} Acc {va:.3f} SlotAcc {sa:.3f} Score {score:.4f}"
                  f" | {self._fmt_time(ep_elapsed)}/ep | ETA {self._fmt_time(eta)}{sv}")

            if pc >= pat:
                print(f"\nEarly stopping at epoch {ep+1}")
                break

        print("=" * 75)
        print(f"OK Joint training done! Best score: {best:.4f}")
        print(f"   Total time : {self._fmt_time(time.time() - train_start)}")
        print(f"   Saved -> {output_dir}")


# =============================================================
#  4. Semantic NLU Inference
# =============================================================
class SemanticNLUInference:
    """Load trained model, chay: intent + frame args + embedding.
    Output tuong thich DialogManager."""

    def __init__(self, model_path=SEMANTIC_MODEL_DIR, entities=None):
        print(f"Loading SemanticNLU from {model_path}...")
        model_file = os.path.join(model_path, "semantic_model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Not found: {model_file}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = SemanticNLUModel(
            PHOBERT_MODEL, len(INTENT_LABELS), NUM_SLOTS,
            dropout=TRAIN_CONFIG["dropout"],
            projection_dim=SIMCSE_CONFIG.get("projection_dim", 256),
        )
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device, weights_only=False)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocessor = VietnamesePreprocessor()
        self.entities_vocab = entities or {}
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        print(f"OK SemanticNLU ready ({self.device}) | Slots: {NUM_SLOTS}")

    def load_entities(self, entities):
        self.entities_vocab = entities

    def _tokenize_mapped(self, text):
        words = text.split()
        all_ids, wt = [], []
        for w in words:
            subs = self.tokenizer.tokenize(w)
            sub_ids = self.tokenizer.convert_tokens_to_ids(subs)
            if not sub_ids: sub_ids = [self.tokenizer.unk_token_id]
            f = len(all_ids)
            all_ids.extend(sub_ids)
            wt.append((f, len(all_ids) - 1))

        mc = NLU_MAX_LEN - 2
        all_ids = all_ids[:mc]
        ids = [self.tokenizer.cls_token_id] + all_ids + [self.tokenizer.sep_token_id]
        attn = [1] * len(ids)
        pl = NLU_MAX_LEN - len(ids)
        ids += [self.tokenizer.pad_token_id] * pl
        attn += [0] * pl

        wc = []
        sf = 0
        for w in words:
            s = text.index(w, sf)
            wc.append((s, s + len(w)))
            sf = s + len(w)
        return ids, attn, wt, wc, words

    def process(self, query):
        if not query or not query.strip():
            return {"error": "Input rong"}

        clean = self.preprocessor.preprocess(query)
        ids, attn, wt, wc, words = self._tokenize_mapped(clean)
        ids_t = torch.tensor([ids], dtype=torch.long).to(self.device)
        attn_t = torch.tensor([attn], dtype=torch.long).to(self.device)

        with torch.no_grad():
            ilog, sls, sle = self.model(ids_t, attn_t)
            probs = torch.softmax(ilog, dim=-1)
            pi = probs.argmax(-1).item()

        intent = ID2LABEL[pi]
        conf = round(probs[0, pi].item(), 4)
        if conf < self.confidence_threshold:
            intent = "out_of_scope"

        fi = FRAME_SCHEMA.get(intent, {"frame": intent.upper(), "slots": {}})
        frame_name = fi["frame"]
        valid_slots = fi["slots"]

        arguments = {}
        entities = {l: [] for l in FINAL_ENTITY_LABELS}

        for sn, et in valid_slots.items():
            if sn not in SLOT2IDX: continue
            si = SLOT2IDX[sn]
            sp = sls[0, :, si].argmax().item()
            ep = sle[0, :, si].argmax().item()
            if sp == 0 or ep == 0 or ep < sp: continue

            cs, ce = sp - 1, ep - 1
            fw = lw = None
            for wi, (ft, lt) in enumerate(wt):
                if ft <= cs <= lt: fw = wi
                if ft <= ce <= lt: lw = wi

            if fw is not None and lw is not None:
                c_s, c_e = wc[fw][0], wc[lw][1]
                val = clean[c_s:c_e]
                arguments[sn] = {"value": val, "start": c_s, "end": c_e}
                entities[et].append(val)

        # Regex fallback
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", clean)
        if years and not entities["YEAR"]:
            entities["YEAR"] = list(dict.fromkeys(years))
            if "year" in valid_slots and "year" not in arguments:
                for y in years:
                    i = clean.find(y)
                    if i >= 0:
                        arguments["year"] = {"value": y, "start": i, "end": i + len(y)}
                        break
        ratings = re.findall(r"\b(\d(?:\.\d)?/10|\d\.\d)\b", clean)
        entities["RATING"] = list(dict.fromkeys(ratings))

        # Vocab supplement
        vocab = self.entities_vocab
        tl = query.lower()
        found_genre = False
        found_person = False
        found_movie = False
        if not entities.get("GENRE") and vocab.get("genres"):
            for g in sorted(vocab["genres"], key=len, reverse=True):
                if g.lower() in tl:
                    entities["GENRE"] = [g]
                    found_genre = True
                    if "genre" in valid_slots and "genre" not in arguments:
                        i = tl.find(g.lower())
                        if i >= 0: arguments["genre"] = {"value": g, "start": i, "end": i + len(g)}
                    break
            # Also check GENRE_ALIASES
            if not found_genre:
                for alias, canonical in GENRE_ALIASES.items():
                    if alias.lower() in tl:
                        entities["GENRE"] = [canonical]
                        found_genre = True
                        if "genre" in valid_slots and "genre" not in arguments:
                            i = tl.find(alias.lower())
                            if i >= 0: arguments["genre"] = {"value": canonical, "start": i, "end": i + len(alias)}
                        break
        if not entities.get("PERSON"):
            persons = sorted(
                set(vocab.get("actors", []) + vocab.get("directors", [])),
                key=len, reverse=True,
            )
            for p in persons:
                if len(p) >= 2 and p.lower() in tl:
                    entities["PERSON"] = [p]
                    found_person = True
                    break
        if not entities.get("MOVIE_TITLE") and vocab.get("movies"):
            for m in sorted(vocab["movies"], key=len, reverse=True):
                if len(m) >= 2 and m.lower() in tl:
                    entities["MOVIE_TITLE"] = [m]
                    found_movie = True
                    break

        # ── Intent boosting dựa trên vocab + keyword ──
        # Nếu detect person/movie từ vocab → boost find_movie/movie_info
        _probs_np = probs[0].cpu().numpy()
        boosted = False

        if found_person and found_genre and intent == "genre_filter":
            # Có person + genre → find_movie thay vì genre_filter
            intent = "find_movie"
            boosted = True
        elif found_person and intent == "genre_filter":
            intent = "find_movie"
            boosted = True
        elif found_movie and intent == "person_info":
            # Có movie title → movie_info thay vì person_info
            intent = "movie_info"
            boosted = True

        # Keyword boosting cho recommendation khi câu mơ hồ
        _RECOMMEND_KEYWORDS = [
            "gợi ý", "recommend", "suggest", "đề xuất", "đề cử",
            "xem gì", "coi gì", "giải trí", "vui vui", "nhẹ nhàng",
            "chill", "hay hay", "giải khuây", "đỡ buồn", "thư giãn",
            "xem gì đó", "coi gì đó", "phim gì đó", "gì đó vui",
            "gì đó hay", "không biết chọn", "không biết xem",
        ]
        if intent == "out_of_scope" and any(kw in tl for kw in _RECOMMEND_KEYWORDS):
            intent = "recommendation"
            boosted = True

        # Keyword boosting cho movie_info khi có "về" + movie
        _INFO_KEYWORDS = ["thông tin", "nội dung", "biết về", "biết thêm về",
                          "kể về", "review", "đánh giá", "cho biết"]
        if found_movie and intent not in ("movie_info",) and any(kw in tl for kw in _INFO_KEYWORDS):
            intent = "movie_info"
            boosted = True

        if boosted:
            # Cập nhật frame theo intent mới
            fi = FRAME_SCHEMA.get(intent, {"frame": intent.upper(), "slots": {}})
            frame_name = fi["frame"]
            valid_slots = fi["slots"]
            # Re-populate arguments cho valid_slots mới
            for sn, et in valid_slots.items():
                if sn in arguments:
                    continue
                if et == "PERSON" and entities.get("PERSON"):
                    p = entities["PERSON"][0]
                    i = tl.find(p.lower())
                    if i >= 0: arguments[sn] = {"value": p, "start": i, "end": i + len(p)}
                elif et == "MOVIE_TITLE" and entities.get("MOVIE_TITLE"):
                    m = entities["MOVIE_TITLE"][0]
                    i = tl.find(m.lower())
                    if i >= 0: arguments[sn] = {"value": m, "start": i, "end": i + len(m)}
                elif et == "GENRE" and entities.get("GENRE"):
                    g = entities["GENRE"][0]
                    i = tl.find(g.lower())
                    if i >= 0: arguments[sn] = {"value": g, "start": i, "end": i + len(g)}

        entities = {k: v for k, v in entities.items() if v}
        hf = {}
        for l in ["PERSON", "YEAR", "GENRE"]:
            if entities.get(l): hf[l] = entities[l][0]

        with torch.no_grad():
            emb = self.model.get_embedding(ids_t, attn_t)
        qv = emb[0].cpu().float().numpy().tolist()

        return {
            "input": query,
            "intent": intent,
            "confidence": conf,
            "frame": frame_name,
            "arguments": arguments,
            "entities": entities,
            "query_vector": qv,
            "hard_filters": hf,
            "ready_for_chapter2": intent not in ["greeting", "goodbye", "out_of_scope"],
        }


print("OK semantic_nlu.py loaded")
print(f"   SemanticNLUModel | SemanticNLUTrainer | SemanticNLUInference")
print(f"   Slots ({NUM_SLOTS}): {ALL_SLOT_NAMES}")
