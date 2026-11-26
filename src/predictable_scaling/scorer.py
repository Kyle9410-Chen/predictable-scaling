import re

import fasttext
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_SENT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

def _mtld_pass(tokens, ttr_thershold=0.72):
    factors, types_, count = 0.0, set(), 0
    for token in tokens:
        types_.add(token)
        count += 1
        if len(types_) / count < ttr_thershold:
            factors += 1
            types_, count = set(), 0
    if count > 0:
        factors += (1 - len(types_) / count) / (1 - ttr_thershold)
    return len(tokens) / factors if factors > 0 else float("nan")

def mtld(text: str, ttr_thershold=0.72) -> float:
    """
    Score Interpretation
    
    • Higher is more diverse
    • Lower is less diverse
    """
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    forward = _mtld_pass(tokens, ttr_thershold)
    backward = _mtld_pass(tokens[::-1], ttr_thershold)
    return (forward + backward) / 2

def _count_syllables(word: str):
    w = word.lower()
    if len(w) <= 3:
        return 1
    
    # drop a silent trailing e ("make", "notes") but keep "-le" ("table")
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]e|ed)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(_VOWEL_GROUP_RE.findall(w)))

def flesch_reading_ease(text: str) -> float:
    """
    Score Interpretation

    • 90-100: Very easy (5th grade)
    • 70-89: Easy (6th-8th grade)
    • 60-69: Plain English / Standard (High school)
    • 50-59: Fairly difficult (College)
    • 30-49: Difficult (College graduate)
    • 0-29: Very difficult (University graduate / professional)
    • < 0: Extremely dense, containing very long sentences or a high concentration of multisyllabic words
    """
    words = _WORD_RE.findall(text)
    if not words:
        return float("nan")
    total_sentence = max(1, len(_SENT_RE.findall(text)))
    total_words = len(words)
    total_syllables = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (total_words / total_sentence) - 84.6 * (total_syllables / total_words)

def dolma3_fasttext_quality_classifier(text: str) -> float:
    """
    Labels: ['__label__0', '__label__1']

    __label__1 is high quality (positive class) and __label__0 is low quality (negative class)
    """
    model_path = hf_hub_download("allenai/dolma3-fasttext-quality-classifier", "model.bin")
    model = fasttext.load_model(model_path)
    label, probability = model.predict([text])
    return label[0], probability[0]

FINEWEB_EDU = "HuggingFaceFW/fineweb-edu-classifier"

class FineWebEduScorer:
    """
    Score Interpretation (regression, ~0-5)

    • 0-1: no / little educational value
    • 3+ : threshold FineWeb-Edu uses to keep a document
    • 5  : highly educational
    """
    def __init__(self, device="cuda", batch_size=256, max_length=512,
                 max_chars=6000, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(FINEWEB_EDU)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(FINEWEB_EDU, torch_dtype=dtype)
                      .to(device).eval())
        self.device, self.bs = device, batch_size
        self.max_length, self.max_chars = max_length, max_chars

    @torch.inference_mode()
    def __call__(self, texts: list[str]) -> np.ndarray:
        # 512 tokens ≈ 2-3k chars; cutting long docs first makes tokenization much cheaper
        texts = [t[:self.max_chars] for t in texts]
        ids = self.tok(texts, truncation=True, max_length=self.max_length)["input_ids"]
        order = np.argsort([len(x) for x in ids])          # length-sort -> minimal padding
        out = np.empty(len(texts), dtype=np.float32)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            enc = self.tok.pad({"input_ids": [ids[i] for i in idx]},
                               return_tensors="pt").to(self.device)
            out[idx] = self.model(**enc).logits.squeeze(-1).float().cpu().numpy()
        return out

def fineweb_edu_int(score: np.ndarray) -> np.ndarray:
    """Same rounding as the model card: int(round(clamp(score, 0, 5)))."""
    return np.clip(np.round(score), 0, 5).astype(np.int8)