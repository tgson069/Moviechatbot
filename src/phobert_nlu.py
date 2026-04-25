# ============================================================
#  src/phobert_nlu.py  -- PhoBERT NLU Pipeline
#  Gom:
#    PhoBERTIntentClassifier  -- fine-tune intent classification
#    PhoBERTNLUTrainer        -- training loop cho intent
#    PhoBERTNERClassifier     -- fine-tune NER (BIO tagging) + CRF
#    NERDataset               -- dataset cho NER voi subword alignment
#    PhoBERTNERTrainer        -- training loop cho NER
#    PhoBERTNLUInference      -- inference: intent + NER (ML) + embedding
# ============================================================
import json, re, os, logging, random
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
from torchcrf import CRF
from seqeval.metrics import f1_score as seqeval_f1_score
from src.config import (
    PHOBERT_MODEL, NLU_MAX_LEN,
    INTENT_LABELS, NER_LABELS, FINAL_ENTITY_LABELS,
    INTENT_MODEL_DIR, LABEL2ID, ID2LABEL, TRAIN_CONFIG,
    CONFIDENCE_THRESHOLD,
    NER_BIO_LABELS, NER_BIO_LABEL2ID, NER_BIO_ID2LABEL,
    NER_MODEL_DIR, NER_TRAIN_CONFIG,
)
from src.preprocessing import VietnamesePreprocessor

logger = logging.getLogger(__name__)


# =============================================================
#  1. Intent Dataset
# =============================================================

class IntentDataset(Dataset):
    """Boc list dict {text, label_id} -> tensor cho DataLoader."""

    def __init__(self, samples: list, tokenizer, max_len: int):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc  = self.tokenizer(
            item["text"],
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(item["label_id"], dtype=torch.long),
        }


# =============================================================
#  2. Intent Model
# =============================================================

class PhoBERTIntentClassifier(nn.Module):
    """
    PhoBERT-base-v2 + Linear head cho intent classification.
    [CLS] token -> Dropout -> Dense -> Dropout -> Linear(num_labels)
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.bert.config.hidden_size
        self.dropout1   = nn.Dropout(dropout)
        self.dense      = nn.Linear(hidden_size, 256)
        self.dropout2   = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        x = self.dropout1(cls_output)
        x = torch.relu(self.dense(x))
        x = self.dropout2(x)
        return self.classifier(x)

    def get_embedding(self, input_ids, attention_mask) -> torch.Tensor:
        """Mean-pooling embedding (768 chieu) cho Chuong 2."""
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state              # (B, seq, 768)
        mask      = attention_mask.unsqueeze(-1).float()   # (B, seq, 1)
        pooled    = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return pooled                                      # (B, 768)


# =============================================================
#  3. Intent Trainer (Class Weights + Early Stopping by F1 + FP16)
# =============================================================

class PhoBERTNLUTrainer:
    def __init__(self):
        print(f"Loading {PHOBERT_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
        self.model     = PhoBERTIntentClassifier(
            PHOBERT_MODEL, len(INTENT_LABELS),
            dropout=TRAIN_CONFIG["dropout"],
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = self.model.to(self.device)
        self.use_fp16 = (self.device == "cuda")
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"OK Model loaded -- device: {self.device} | params: {total_params:.1f}M | FP16: {self.use_fp16}")

    def _compute_class_weights(self, train_data: list) -> torch.Tensor:
        """Tinh class weights tu distribution cua training data."""
        label_counts = Counter(s["label_id"] for s in train_data)
        n_samples = len(train_data)
        n_classes = len(INTENT_LABELS)
        weights = []
        for i in range(n_classes):
            count = label_counts.get(i, 1)
            weights.append(n_samples / (n_classes * count))
        w = torch.tensor(weights, dtype=torch.float32)
        print(f"   Class weights: {[f'{x:.2f}' for x in w.tolist()]}")
        return w

    def train(self, train_data: list, val_data: list,
              output_dir: str = INTENT_MODEL_DIR):

        train_ds     = IntentDataset(train_data, self.tokenizer, NLU_MAX_LEN)
        val_ds       = IntentDataset(val_data,   self.tokenizer, NLU_MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"],
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=TRAIN_CONFIG["batch_size"],
                                  shuffle=False, num_workers=0)

        optimizer    = AdamW(
            self.model.parameters(),
            lr           = TRAIN_CONFIG["learning_rate"],
            weight_decay = TRAIN_CONFIG["weight_decay"],
        )
        total_steps  = len(train_loader) * TRAIN_CONFIG["num_train_epochs"]
        warmup_steps = int(total_steps * TRAIN_CONFIG["warmup_ratio"])
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        
        # Class weights + label smoothing cho CrossEntropyLoss
        class_weights = self._compute_class_weights(train_data).to(self.device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # FP16 mixed precision
        scaler = GradScaler(enabled=self.use_fp16)
        
        best_val_f1 = 0.0
        patience_counter = 0
        patience = TRAIN_CONFIG.get("early_stopping_patience", 3)

        n_epochs = TRAIN_CONFIG["num_train_epochs"]
        print(f"\nBat dau fine-tuning PhoBERT ({n_epochs} epochs, early_stopping={patience})...")
        print(f"   train: {len(train_data)} | val: {len(val_data)} | "
              f"batch: {TRAIN_CONFIG['batch_size']} | lr: {TRAIN_CONFIG['learning_rate']}")
        print("=" * 65)

        for epoch in range(n_epochs):
            # -- Train ----------------------------------------
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                optimizer.zero_grad()
                with autocast(enabled=self.use_fp16):
                    logits = self.model(input_ids, attention_mask)
                    loss   = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                preds       = logits.argmax(dim=-1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)

            train_acc = correct / total

            # -- Validation (compute weighted F1) -------------
            self.model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels         = batch["label"].to(self.device)
                    with autocast(enabled=self.use_fp16):
                        logits  = self.model(input_ids, attention_mask)
                    preds   = logits.argmax(dim=-1)
                    val_preds.extend(preds.cpu().tolist())
                    val_labels.extend(labels.cpu().tolist())

            # Weighted F1 (sklearn-compatible manual computation)
            n_classes = len(INTENT_LABELS)
            per_class_f1 = []
            per_class_support = []
            for c in range(n_classes):
                tp = sum(1 for p, t in zip(val_preds, val_labels) if p == c and t == c)
                fp = sum(1 for p, t in zip(val_preds, val_labels) if p == c and t != c)
                fn = sum(1 for p, t in zip(val_preds, val_labels) if p != c and t == c)
                support = tp + fn
                prec = tp / max(tp + fp, 1)
                rec  = tp / max(tp + fn, 1)
                f1_c = 2 * prec * rec / max(prec + rec, 1e-8)
                per_class_f1.append(f1_c)
                per_class_support.append(support)
            
            total_support = sum(per_class_support)
            val_f1 = sum(f * s for f, s in zip(per_class_f1, per_class_support)) / max(total_support, 1)
            val_acc = sum(1 for p, t in zip(val_preds, val_labels) if p == t) / max(len(val_labels), 1)
            avg_loss = total_loss / len(train_loader)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                os.makedirs(output_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, "intent_model.pt"),
                )
                self.tokenizer.save_pretrained(output_dir)
                saved = " <- saved (best)"
            else:
                patience_counter += 1
                saved = f" (no improve {patience_counter}/{patience})"
            
            print(f"Epoch {epoch+1:02d}/{n_epochs} | Loss: {avg_loss:.4f} | "
                  f"Train: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}{saved}")

            # Early stopping by weighted F1
            if patience_counter >= patience:
                print(f"\nEarly stopping tai epoch {epoch+1} (khong cai thien F1 sau {patience} epochs)")
                break

        print("=" * 65)
        print(f"OK Fine-tuning xong! Best val weighted F1: {best_val_f1:.4f}")
        print(f"   Model saved -> {output_dir}")


# =============================================================
#  4. NER Dataset (BIO Tagging voi subword alignment)
# =============================================================

class NERDataset(Dataset):
    """Tokenize text va align character-level entity spans thanh BIO labels
    cho tung subword token cua PhoBERT."""

    def __init__(self, samples: list, tokenizer, max_len: int):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        entity_spans = item.get("entity_spans", [])

        words = text.split()

        # Build word character positions
        word_positions = []
        search_from = 0
        for w in words:
            start = text.index(w, search_from)
            word_positions.append((start, start + len(w)))
            search_from = start + len(w)

        # Assign BIO label to each word based on entity spans
        word_labels = []
        for w_start, w_end in word_positions:
            label = "O"
            for ent_start, ent_end, ent_type in entity_spans:
                if w_start >= ent_start and w_end <= ent_end:
                    label = f"B-{ent_type}" if w_start == ent_start else f"I-{ent_type}"
                    break
            word_labels.append(label)

        # Tokenize word by word -> subword tokens + expand labels
        all_token_ids = []
        all_label_ids = []

        for word, label in zip(words, word_labels):
            sub_tokens = self.tokenizer.tokenize(word)
            sub_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            if not sub_ids:
                sub_ids = [self.tokenizer.unk_token_id]

            label_id = NER_BIO_LABEL2ID[label]
            all_token_ids.extend(sub_ids)
            all_label_ids.append(label_id)

            # Subword continuation: B- -> I-, I- stays I-, O stays O
            for _ in sub_ids[1:]:
                if label.startswith("B-"):
                    all_label_ids.append(NER_BIO_LABEL2ID[f"I-{label[2:]}"])
                else:
                    all_label_ids.append(label_id)

        # Add special tokens: [CLS] content [SEP] [PAD...]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        max_content = self.max_len - 2
        all_token_ids = all_token_ids[:max_content]
        all_label_ids = all_label_ids[:max_content]

        input_ids      = [cls_id] + all_token_ids + [sep_id]
        labels         = [-100]   + all_label_ids + [-100]
        attention_mask  = [1] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        input_ids      += [pad_id] * pad_len
        labels         += [-100]   * pad_len
        attention_mask  += [0]     * pad_len

        return {
            "input_ids"     : torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels"        : torch.tensor(labels, dtype=torch.long),
        }


# =============================================================
#  5. NER Model (Token Classification + CRF)
# =============================================================

class PhoBERTNERClassifier(nn.Module):
    """PhoBERT + CRF head cho NER (BIO token classification).
    All tokens -> Dropout -> Linear(768, num_labels) -> CRF"""

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.num_labels = num_labels
        self.bert = RobertaModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.bert.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf        = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Returns:
          - Neu labels != None: tra ve -log_likelihood (loss) de training
          - Neu labels == None: tra ve emissions (logits) de decode
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state   # (B, seq, 768)
        x = self.dropout(sequence_output)
        emissions = self.classifier(x)               # (B, seq, num_labels)
        
        if labels is not None:
            # CRF can token dau tien trong moi chuoi phai duoc bat mask.
            # Vi labels dung -100 cho [CLS]/[SEP], ta train tren tat ca token
            # non-pad va ep cac vi tri dac biet ve nhan O.
            crf_mask = attention_mask.bool()                      # (B, seq)
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = NER_BIO_LABEL2ID["O"]
            loss = -self.crf(emissions, safe_labels, mask=crf_mask, reduction='mean')
            return loss
        
        return emissions

    def decode(self, input_ids, attention_mask):
        """Viterbi decode: tra ve best tag sequence cho moi sample."""
        emissions = self.forward(input_ids, attention_mask, labels=None)
        # Dung attention_mask lam CRF mask (bo PAD tokens)
        return self.crf.decode(emissions, mask=attention_mask.bool())


# =============================================================
#  6. NER Trainer (CRF + Span F1 + FP16 + Early Stopping)
# =============================================================

class PhoBERTNERTrainer:
    def __init__(self):
        print(f"Loading {PHOBERT_MODEL} for NER...")
        self.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
        self.model = PhoBERTNERClassifier(
            PHOBERT_MODEL, len(NER_BIO_LABELS),
            dropout=NER_TRAIN_CONFIG["dropout"],
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = self.model.to(self.device)
        self.use_fp16 = (self.device == "cuda")
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"OK NER Model loaded -- device: {self.device} | params: {total_params:.1f}M | CRF: ON | FP16: {self.use_fp16}")

    def _decode_bio_tags(self, tag_ids: list, label_map: dict) -> list:
        """Convert list of tag ids -> list of BIO label strings."""
        return [label_map.get(t, "O") for t in tag_ids]

    def train(self, train_data: list, val_data: list,
              output_dir: str = NER_MODEL_DIR):
        cfg = NER_TRAIN_CONFIG

        train_ds     = NERDataset(train_data, self.tokenizer, NLU_MAX_LEN)
        val_ds       = NERDataset(val_data,   self.tokenizer, NLU_MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=0)

        optimizer = AdamW(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        total_steps  = len(train_loader) * cfg["num_train_epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        # FP16 mixed precision
        scaler = GradScaler(enabled=self.use_fp16)

        best_val_f1 = 0.0
        patience_counter = 0
        patience = cfg.get("early_stopping_patience", 5)

        n_epochs = cfg["num_train_epochs"]
        print(f"\nBat dau fine-tuning NER ({n_epochs} epochs, early_stopping={patience})...")
        print(f"   train: {len(train_data)} | val: {len(val_data)} | "
              f"batch: {cfg['batch_size']} | lr: {cfg['learning_rate']}")
        print("=" * 65)

        for epoch in range(n_epochs):
            # -- Train ----------------------------------------
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                optimizer.zero_grad()
                with autocast(enabled=self.use_fp16):
                    loss = self.model(input_ids, attention_mask, labels=labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            # -- Validation (Span-level F1 via seqeval) -------
            self.model.eval()
            all_true_tags = []  # list of list of str
            all_pred_tags = []  # list of list of str
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels         = batch["labels"]  # keep on CPU
                    
                    # CRF Viterbi decode
                    pred_sequences = self.model.decode(input_ids, attention_mask)
                    
                    for i in range(labels.size(0)):
                        true_tags = []
                        pred_tags = []
                        label_seq = labels[i].tolist()
                        seq_len   = int(attention_mask[i].sum().item())
                        pred_seq  = pred_sequences[i][:seq_len]
                        
                        for j in range(seq_len):
                            if label_seq[j] != -100:
                                true_tags.append(NER_BIO_ID2LABEL.get(label_seq[j], "O"))
                                if j < len(pred_seq):
                                    pred_tags.append(NER_BIO_ID2LABEL.get(pred_seq[j], "O"))
                                else:
                                    pred_tags.append("O")
                        
                        if true_tags:
                            all_true_tags.append(true_tags)
                            all_pred_tags.append(pred_tags)

            # Span-level F1 (seqeval)
            val_f1 = seqeval_f1_score(all_true_tags, all_pred_tags, average='micro')
            avg_loss = total_loss / len(train_loader)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                os.makedirs(output_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, "ner_model.pt"),
                )
                self.tokenizer.save_pretrained(output_dir)
                saved = " <- saved (best)"
            else:
                patience_counter += 1
                saved = f" (no improve {patience_counter}/{patience})"

            print(f"Epoch {epoch+1:02d}/{n_epochs} | Loss: {avg_loss:.4f} | "
                  f"Span F1: {val_f1:.4f}{saved}")

            if patience_counter >= patience:
                print(f"\nEarly stopping tai epoch {epoch+1}")
                break

        print("=" * 65)
        print(f"OK NER fine-tuning xong! Best val Span F1: {best_val_f1:.4f}")
        print(f"   Model saved -> {output_dir}")


# =============================================================
#  7. Inference (Intent + NER + Embedding)
# =============================================================

class PhoBERTNLUInference:
    """
    Load model da fine-tune, chay:
      - Intent classification (PhoBERT) voi confidence threshold
      - NER: ML (PhoBERT BIO + CRF) neu co model, fallback vocab matching
      - Embedding (mean-pool PhoBERT, 768d)
    Output JSON tuong thich voi Chuong 2.
    """

    def __init__(self, model_path: str = INTENT_MODEL_DIR,
                 entities: dict = None,
                 ner_model_path: str = NER_MODEL_DIR):
        print(f"Loading NLU model tu {model_path}...")

        # -- Intent model --
        model_file = os.path.join(model_path, "intent_model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Khong tim thay model tai {model_file}. "
                "Hay chay training truoc (PhoBERTNLUTrainer.train())!"
            )

        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = PhoBERTIntentClassifier(
            PHOBERT_MODEL, len(INTENT_LABELS),
            dropout=TRAIN_CONFIG["dropout"],
        )
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device, weights_only=False)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # -- NER model (optional, load neu co) --
        self.ner_model = None
        ner_file = os.path.join(ner_model_path, "ner_model.pt")
        if os.path.exists(ner_file):
            print(f"Loading NER model tu {ner_model_path}...")
            self.ner_model = PhoBERTNERClassifier(
                PHOBERT_MODEL, len(NER_BIO_LABELS),
                dropout=NER_TRAIN_CONFIG["dropout"],
            )
            self.ner_model.load_state_dict(
                torch.load(ner_file, map_location=self.device, weights_only=False)
            )
            self.ner_model = self.ner_model.to(self.device)
            self.ner_model.eval()
            print("OK NER model (BIO + CRF) loaded!")
        else:
            print(f"[!] Khong tim thay NER model tai {ner_file} -> dung vocab matching fallback")

        self.preprocessor       = VietnamesePreprocessor()
        self.entities_vocab     = entities or {}
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        # Sets cho phan loai PERSON -> ACTOR / DIRECTOR
        self._actor_set    = set()
        self._director_set = set()
        if self.entities_vocab:
            self._actor_set    = {n.lower() for n in self.entities_vocab.get("actors", [])}
            self._director_set = {n.lower() for n in self.entities_vocab.get("directors", [])}

        print(f"OK NLU model san sang tren {self.device}!")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print(f"   NER mode: {'ML (BIO + CRF)' if self.ner_model else 'Vocab matching'}")

    def load_entities(self, entities: dict):
        """Nap vocab thuc the TMDB de dung cho NER matching."""
        self.entities_vocab = entities
        self._actor_set    = {n.lower() for n in entities.get("actors", [])}
        self._director_set = {n.lower() for n in entities.get("directors", [])}

    def _classify_person(self, name: str, intent: str = "") -> str:
        """Phan loai PERSON -> ACTOR hoac DIRECTOR.
        
        Uu tien:
        1. Tim trong entities vocab (actors / directors)
        2. Dua vao intent (actor_info -> ACTOR, ...)
        3. Mac dinh ACTOR
        """
        low = name.lower()
        in_actor    = low in self._actor_set
        in_director = low in self._director_set
        if in_actor and not in_director:
            return "ACTOR"
        if in_director and not in_actor:
            return "DIRECTOR"
        # Ca hai hoac khong co -> dua vao intent
        if intent in ("actor_info", "find_movie", "recommendation"):
            return "ACTOR"
        if intent in ("movie_info",):
            return "DIRECTOR"
        return "ACTOR"

    # ---------------------------------------------------------
    #  NER: ML-based (PhoBERT BIO tagging + CRF Viterbi)
    # ---------------------------------------------------------
    def extract_entities_ml(self, text: str, intent: str = "") -> dict:
        """NER bang PhoBERT BIO tagging + CRF Viterbi + regex fallback cho YEAR/RATING."""
        words = text.split()
        if not words:
            return {label: [] for label in FINAL_ENTITY_LABELS}

        # Tokenize word by word
        all_token_ids = []
        word_to_token = []  # list of (word_idx, is_first_subword)

        for wi, word in enumerate(words):
            sub_tokens = self.tokenizer.tokenize(word)
            sub_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            if not sub_ids:
                sub_ids = [self.tokenizer.unk_token_id]
            for si, sid in enumerate(sub_ids):
                all_token_ids.append(sid)
                word_to_token.append((wi, si == 0))

        # Build input tensors
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        max_content = NLU_MAX_LEN - 2
        all_token_ids = all_token_ids[:max_content]
        word_to_token = word_to_token[:max_content]

        input_ids = [cls_id] + all_token_ids + [sep_id]
        attention_mask = [1] * len(input_ids)
        pad_len = NLU_MAX_LEN - len(input_ids)
        input_ids      += [pad_id] * pad_len
        attention_mask  += [0]     * pad_len

        input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attn_t      = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            # CRF Viterbi decode
            pred_sequence = self.ner_model.decode(input_ids_t, attn_t)[0]

        # Content token predictions (skip CLS at index 0)
        content_preds = pred_sequence[1 : 1 + len(word_to_token)]

        # First-subword prediction per word
        word_preds = {}
        for ti, (wi, is_first) in enumerate(word_to_token):
            if is_first and ti < len(content_preds):
                word_preds[wi] = NER_BIO_ID2LABEL.get(content_preds[ti], "O")

        # Decode BIO -> entity dict
        entities = {label: [] for label in FINAL_ENTITY_LABELS}
        current_type  = None
        current_words = []

        for wi in range(len(words)):
            pred = word_preds.get(wi, "O")
            if pred.startswith("B-"):
                if current_type and current_words:
                    value = " ".join(current_words)
                    if current_type == "PERSON":
                        entities[self._classify_person(value, intent)].append(value)
                    else:
                        entities[current_type].append(value)
                current_type  = pred[2:]
                current_words = [words[wi]]
            elif pred.startswith("I-") and current_type == pred[2:]:
                current_words.append(words[wi])
            else:
                if current_type and current_words:
                    value = " ".join(current_words)
                    if current_type == "PERSON":
                        entities[self._classify_person(value, intent)].append(value)
                    else:
                        entities[current_type].append(value)
                current_type  = None
                current_words = []

        if current_type and current_words:
            value = " ".join(current_words)
            if current_type == "PERSON":
                entities[self._classify_person(value, intent)].append(value)
            else:
                entities[current_type].append(value)

        # Regex fallback cho YEAR va RATING (luon chinh xac hon ML)
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        entities["YEAR"] = list(dict.fromkeys(years))
        ratings = re.findall(r"\b(\d(?:\.\d)?/10|\d\.\d)\b", text)
        entities["RATING"] = list(dict.fromkeys(ratings))

        return {k: list(dict.fromkeys(v)) for k, v in entities.items()}

    # ---------------------------------------------------------
    #  NER: Vocab matching fallback
    # ---------------------------------------------------------
    def extract_entities(self, text: str, intent: str = "") -> dict:
        """NER fallback: vocabulary matching + regex (khi khong co NER model)."""
        text_norm = text.lower()
        entities  = {label: [] for label in FINAL_ENTITY_LABELS}
        vocab     = self.entities_vocab

        years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        entities["YEAR"] = list(dict.fromkeys(years))

        ratings = re.findall(r"\b(\d(?:\.\d)?/10|\d\.\d)\b", text)
        entities["RATING"] = list(dict.fromkeys(ratings))

        for genre in sorted(vocab.get("genres", []), key=len, reverse=True):
            if genre.lower() in text_norm:
                entities["GENRE"].append(genre)

        for actor in sorted(vocab.get("actors", []), key=len, reverse=True):
            if len(actor) >= 2 and actor.lower() in text_norm:
                entities["ACTOR"].append(actor)

        for director in sorted(vocab.get("directors", []), key=len, reverse=True):
            if len(director) >= 2 and director.lower() in text_norm:
                entities["DIRECTOR"].append(director)

        for movie in sorted(vocab.get("movies", []), key=len, reverse=True):
            if len(movie) >= 2 and movie.lower() in text_norm:
                entities["MOVIE_TITLE"].append(movie)

        return {k: list(dict.fromkeys(v)) for k, v in entities.items()}

    # ---------------------------------------------------------
    #  Embedding
    # ---------------------------------------------------------
    def get_embedding(self, query: str) -> np.ndarray:
        """Mean-pool PhoBERT -> vector 768 chieu (normalized)."""
        clean = self.preprocessor.preprocess(query)
        enc   = self.tokenizer(
            clean, max_length=NLU_MAX_LEN,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            vec = self.model.get_embedding(input_ids, attention_mask)
        vec  = vec[0].cpu().float().numpy()
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.zeros_like(vec)
        return vec / norm

    # ---------------------------------------------------------
    #  Process (main entry point)
    # ---------------------------------------------------------
    def process(self, query: str) -> dict:
        if not query or not query.strip():
            return {"error": "Input rong"}

        clean_query = self.preprocessor.preprocess(query)

        # Intent
        enc = self.tokenizer(
            clean_query, max_length=NLU_MAX_LEN,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).item()

        intent     = ID2LABEL[pred_idx]
        confidence = round(probs[0, pred_idx].item(), 4)

        if confidence < self.confidence_threshold:
            intent = "out_of_scope"

        # NER: dung ML neu co, fallback vocab matching
        if self.ner_model is not None:
            all_ents = self.extract_entities_ml(clean_query, intent)
        else:
            all_ents = self.extract_entities(clean_query, intent)

        # --- Hybrid post-processing ---
        # Buoc 1: Fix MOVIE_TITLE bi nham thanh PERSON (vi du: "Tom Cruise")
        #   NER ML doi khi predict B-MOVIE_TITLE cho ten nguoi.
        #   Kiem tra nguoc: neu title co trong actor_set / director_set -> chuyen sang dung type.
        misclassified = []
        for title in list(all_ents.get("MOVIE_TITLE", [])):
            t_low = title.lower()
            if t_low in self._actor_set:
                all_ents["ACTOR"]      = list(dict.fromkeys(all_ents.get("ACTOR", []) + [title]))
                misclassified.append(title)
            elif t_low in self._director_set:
                all_ents["DIRECTOR"]   = list(dict.fromkeys(all_ents.get("DIRECTOR", []) + [title]))
                misclassified.append(title)
        if misclassified:
            all_ents["MOVIE_TITLE"] = [t for t in all_ents["MOVIE_TITLE"] if t not in misclassified]

        # Buoc 2: Vocab-supplement -- bo sung entity type ma ML bo sot (rong)
        #   Chi bo sung, khong ghi de ket qua ML da co.
        vocab      = self.entities_vocab
        text_lower = query.lower()
        if not all_ents.get("GENRE") and vocab.get("genres"):
            for g in sorted(vocab["genres"], key=len, reverse=True):
                if g.lower() in text_lower:
                    all_ents["GENRE"] = [g]
                    break
        if not all_ents.get("ACTOR") and vocab.get("actors"):
            for a in sorted(vocab["actors"], key=len, reverse=True):
                if len(a) >= 2 and a.lower() in text_lower:
                    all_ents["ACTOR"] = [a]
                    break
        if not all_ents.get("DIRECTOR") and vocab.get("directors"):
            for d in sorted(vocab["directors"], key=len, reverse=True):
                if len(d) >= 2 and d.lower() in text_lower:
                    all_ents["DIRECTOR"] = [d]
                    break
        if not all_ents.get("MOVIE_TITLE") and vocab.get("movies"):
            for m in sorted(vocab["movies"], key=len, reverse=True):
                if len(m) >= 2 and m.lower() in text_lower:
                    all_ents["MOVIE_TITLE"] = [m]
                    break

        entities = {k: v for k, v in all_ents.items() if v}

        # Hard filters
        hard_filters = {}
        for label in ["ACTOR", "DIRECTOR", "YEAR", "GENRE"]:
            if entities.get(label):
                hard_filters[label] = entities[label][0]

        return {
            "input"             : query,
            "intent"            : intent,
            "confidence"        : confidence,
            "entities"          : entities,
            "query_vector"      : self.get_embedding(query).tolist(),
            "hard_filters"      : hard_filters,
            "ready_for_chapter2": intent not in ["greeting", "goodbye", "out_of_scope"],
        }
