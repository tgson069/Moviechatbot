# src/preprocessing.py -- Tien xu ly van ban tieng Viet
import re, unicodedata

_SEG, _SEG_NAME = None, "none"
try:
    from underthesea import word_tokenize as _f
    _SEG = lambda t: _f(t, format="text")
    _SEG_NAME = "underthesea"
except ImportError:
    try:
        from pyvi import ViTokenizer
        _SEG = ViTokenizer.tokenize
        _SEG_NAME = "pyvi"
    except ImportError:
        _SEG = lambda t: t
        _SEG_NAME = "none (fallback)"

print(f"[Preprocessing] Segmenter: {_SEG_NAME}")


class VietnamesePreprocessor:
    _EMOJI = re.compile(
        "[\U00010000-\U0010ffff\U0001F300-\U0001F9FF\u2700-\u27BF\u2600-\u26FF]",
        flags=re.UNICODE)

    def normalize(self, text):
        text = unicodedata.normalize("NFC", text)
        return re.sub(r"\s+", " ", self._EMOJI.sub("", text)).strip()

    def clean(self, text):
        text = self.normalize(text)
        return re.sub(r"\s+", " ",
               re.sub(r"[^\w\s.,?!;:()'\"\\-]", " ", text)).strip()

    def segment(self, text):
        return _SEG(self.clean(text))

    def preprocess(self, text):
        """Lam sach van ban -- dau vao cho PhoBERT."""
        return self.clean(text)

    @staticmethod
    def segmenter_info():
        return _SEG_NAME
