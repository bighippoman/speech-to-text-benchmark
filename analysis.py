"""
Analysis module: WER, punctuation scoring, capitalization, statistics.
"""

import re
import string
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Word Error Rate ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for WER comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate using jiwer.
    Returns WER as a float (0.0 = perfect, 1.0 = 100% errors).
    Falls back to simple implementation if jiwer unavailable.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    try:
        import jiwer
        return jiwer.wer(ref_norm, hyp_norm)
    except ImportError:
        # Fallback: simple Levenshtein on words
        return _simple_wer(ref_norm, hyp_norm)


def _simple_wer(ref: str, hyp: str) -> float:
    """Simple WER via word-level edit distance."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    n = len(ref_words)
    m = len(hyp_words)

    # Dynamic programming edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m] / n if n > 0 else 0.0


def compute_wer_details(reference: str, hypothesis: str) -> dict:
    """Compute detailed WER metrics: substitutions, deletions, insertions."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    try:
        import jiwer
        output = jiwer.process_words(ref_norm, hyp_norm)
        return {
            "wer": output.wer,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "hits": output.hits,
            "ref_word_count": len(ref_norm.split()),
            "hyp_word_count": len(hyp_norm.split()),
        }
    except ImportError:
        wer = _simple_wer(ref_norm, hyp_norm)
        return {
            "wer": wer,
            "ref_word_count": len(ref_norm.split()),
            "hyp_word_count": len(hyp_norm.split()),
        }


# ── Punctuation Analysis ───────────────────────────────────────────────────

def score_punctuation(text: str) -> dict:
    """
    Analyze punctuation quality. Returns counts and a normalized score.
    Higher score = more punctuation = generally better formatted output.
    """
    if not text:
        return {"score": 0.0, "periods": 0, "commas": 0, "apostrophes": 0,
                "question_marks": 0, "exclamation_marks": 0, "total_marks": 0,
                "marks_per_word": 0.0}

    periods = text.count(".")
    commas = text.count(",")
    apostrophes = text.count("'") + text.count("\u2019")
    question_marks = text.count("?")
    exclamation_marks = text.count("!")
    colons = text.count(":") + text.count(";")

    total = periods + commas + apostrophes + question_marks + exclamation_marks + colons
    words = len(text.split())
    marks_per_word = total / words if words > 0 else 0.0

    # Score: ratio of punctuation marks to words (0-1 scale)
    # Typical English has ~0.15 marks per word
    score = min(marks_per_word / 0.15, 1.0)

    return {
        "score": score,
        "periods": periods,
        "commas": commas,
        "apostrophes": apostrophes,
        "question_marks": question_marks,
        "exclamation_marks": exclamation_marks,
        "total_marks": total,
        "marks_per_word": marks_per_word,
    }


# ── Capitalization Analysis ─────────────────────────────────────────────────

def score_capitalization(text: str) -> dict:
    """
    Analyze capitalization quality.
    Checks sentence starts, proper capitalization patterns.
    """
    if not text:
        return {"score": 0.0, "sentence_starts_capped": 0,
                "total_sentences": 0, "all_caps_words": 0, "all_lower_words": 0}

    # Split into sentences (approximate)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    total_sentences = len(sentences)
    starts_capped = 0
    for s in sentences:
        if s and s[0].isupper():
            starts_capped += 1

    words = text.split()
    all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    all_lower = sum(1 for w in words if w.islower())

    # Score: proportion of sentences properly capitalized
    score = starts_capped / total_sentences if total_sentences > 0 else 0.0

    return {
        "score": score,
        "sentence_starts_capped": starts_capped,
        "total_sentences": total_sentences,
        "all_caps_words": all_caps,
        "all_lower_words": all_lower,
    }


# ── Statistical Aggregation ────────────────────────────────────────────────

@dataclass
class RunStats:
    """Statistical summary of multiple benchmark runs."""
    values: list[float] = field(default_factory=list)
    median: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p25: float = 0.0
    p75: float = 0.0

    def compute(self):
        if not self.values:
            return
        arr = np.array(self.values)
        self.median = float(np.median(arr))
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))
        self.min_val = float(np.min(arr))
        self.max_val = float(np.max(arr))
        self.p25 = float(np.percentile(arr, 25))
        self.p75 = float(np.percentile(arr, 75))


def compute_stats(values: list[float]) -> RunStats:
    """Compute stats from a list of values."""
    stats = RunStats(values=values)
    stats.compute()
    return stats


# ── Word Overlap / Similarity ──────────────────────────────────────────────

def word_overlap(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts (word-level)."""
    if not text_a or not text_b:
        return 0.0
    words_a = set(normalize_text(text_a).split())
    words_b = set(normalize_text(text_b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


# ── Consensus Reference ────────────────────────────────────────────────────

def compute_consensus_reference(transcripts: list[str]) -> str:
    """
    Given multiple transcripts from different providers, pick the one
    with highest average similarity to all others (majority-vote proxy).
    """
    if not transcripts:
        return ""
    if len(transcripts) == 1:
        return transcripts[0]

    best_idx = 0
    best_avg = -1.0

    for i, t in enumerate(transcripts):
        similarities = []
        for j, other in enumerate(transcripts):
            if i != j:
                similarities.append(word_overlap(t, other))
        avg = np.mean(similarities)
        if avg > best_avg:
            best_avg = avg
            best_idx = i

    return transcripts[best_idx]


# ── Diff Highlighting ───────────────────────────────────────────────────────

def compute_word_diff(reference: str, hypothesis: str) -> list[dict]:
    """
    Compute word-level diff for transcript comparison.
    Returns list of {word, status} where status is 'match', 'error', 'missing', 'extra'.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    # Use dynamic programming to align
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack to get alignment
    result = []
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            ops.append(("match", hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", hyp_words[j - 1]))
            j -= 1
        else:
            ops.append(("del", ref_words[i - 1]))
            i -= 1

    ops.reverse()
    for op, word in ops:
        status_map = {"match": "match", "sub": "error", "ins": "extra", "del": "missing"}
        result.append({"word": word, "status": status_map[op]})

    return result


# ── RTFx Calculation ────────────────────────────────────────────────────────

def compute_rtfx(audio_duration_seconds: float, api_elapsed_seconds: float) -> float:
    """
    Compute Real-Time Factor (RTFx).
    RTFx > 1 means faster than real-time.
    E.g., RTFx = 10 means 10x faster than audio duration.
    Note: includes network latency, not pure inference speed.
    """
    if api_elapsed_seconds <= 0:
        return 0.0
    return audio_duration_seconds / api_elapsed_seconds
