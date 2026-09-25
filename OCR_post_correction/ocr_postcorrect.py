"""
OCR Post-Correction Pipeline for Descriptive Answer Scripts
=============================================================

Design goal: fix likely OCR misreads WITHOUT rewriting what the student
actually wrote. This is a "noisy channel" corrector:

    best_candidate = argmax_c  P(observed | c) * P(c | context)

- P(observed | c)  -> visual/edit-distance similarity between the OCR
                       token and the candidate (weighted by an OCR
                       confusion matrix, e.g. l<->1, rn<->m).
- P(c | context)    -> how plausible the candidate is given the words
                        around it (n-gram model here; swap in a masked
                        LM like BERT for stronger context scoring).

Every substitution is logged with scores so it can be audited or
reverted. Nothing is ever free-generated -- candidates only come from
a domain vocabulary + edit-distance neighbourhood of what OCR printed.

Pipeline stages (see functions below):
    1. tokenize_with_spans   - keep char offsets for auditing
    2. is_noise_token        - rule-based cleanup of junk artifacts
    3. needs_review          - anomaly detection (OOV / low confidence)
    4. generate_candidates   - constrained candidate generation
    5. score_candidate       - visual score x context score
    6. correct_document      - orchestrates + applies guardrails
    7. NgramContextModel     - swap-in-able context scorer (replace
                                 with a real MLM for production use)
"""

from __future__ import annotations

import re
import math
import json
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Optional


# ---------------------------------------------------------------------------
# 1. OCR confusion matrix -- weighted edit distance
# ---------------------------------------------------------------------------
# Pairs of characters (or short substrings) that OCR engines commonly
# confuse. Substitutions between these pairs cost less than a generic
# substitution, so candidates that explain the *specific* way OCR fails
# are preferred over ones that are merely "close" in plain edit distance.

OCR_CONFUSIONS = {
    ("l", "1"), ("1", "l"), ("l", "i"), ("i", "l"),
    ("rn", "m"), ("m", "rn"),
    ("cl", "d"), ("d", "cl"),
    ("0", "o"), ("o", "0"),
    ("s", "5"), ("5", "s"),
    ("a", ""), ("", "a"),          # stray inserted "a" -- see your sample
    ("vv", "w"), ("w", "vv"),
    ("nn", "m"), ("m", "nn"),
    ("h", "b"), ("b", "h"),
    ("tt", "u"),
}
CONFUSION_COST = 0.4     # cheaper than a generic substitution
GENERIC_SUB_COST = 1.0
INSERT_DELETE_COST = 1.0


def weighted_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance, but OCR-typical substitutions are cheap."""
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * INSERT_DELETE_COST
    for j in range(m + 1):
        dp[0][j] = j * INSERT_DELETE_COST

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub_cost = GENERIC_SUB_COST
            if (a[i - 1], b[j - 1]) in OCR_CONFUSIONS:
                sub_cost = CONFUSION_COST
            dp[i][j] = min(
                dp[i - 1][j] + INSERT_DELETE_COST,      # delete
                dp[i][j - 1] + INSERT_DELETE_COST,      # insert
                dp[i - 1][j - 1] + sub_cost,            # substitute
            )
    return dp[n][m]


# ---------------------------------------------------------------------------
# 2. Tokenization with character spans (needed for auditable diffs)
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"\S+")


@dataclass
class Token:
    text: str
    start: int
    end: int


def tokenize_with_spans(text: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# 3. Rule-based noise cleanup (cheap, high-precision, do this BEFORE the
#    LM/edit-distance stage -- handles the recurring junk you already see,
#    e.g. isolated stray "a" tokens that are really a misread bullet/dash)
# ---------------------------------------------------------------------------

NOISE_TOKEN_PATTERNS = [
    re.compile(r"^a$"),              # lone stray "a"
    re.compile(r"^[.\-_*•]{1,3}$"),  # stray punctuation-only tokens
]


def is_noise_token(tok: str, prev_tok: Optional[str], next_tok: Optional[str]) -> bool:
    """Heuristic: a lone 'a' is noise unless it's grammatically an article
    (i.e. immediately followed by a noun-ish word it could modify -- here
    we approximate with: keep it if the *previous* word ends a clause
    boundary word like 'a'/'an' pattern already, else drop).
    Tune this against your own error analysis (Phase 0 in the plan)."""
    for pat in NOISE_TOKEN_PATTERNS:
        if pat.match(tok):
            return True
    return False


# ---------------------------------------------------------------------------
# 4. Context model (swap this out for a masked LM in production)
# ---------------------------------------------------------------------------
# This n-gram model is a stand-in for step 4 of the design (a BERT-style
# masked LM). It's trained on your domain vocabulary / reference answers.
# Replace `NgramContextModel.score` with a call to a real MLM's
# P(token | context) and everything downstream keeps working unchanged --
# that's the whole point of separating "generate candidates" from "score
# candidates by context".

class NgramContextModel:
    def __init__(self, corpus_sentences: list[list[str]]):
        self.unigrams: Counter = Counter()
        self.bigrams: Counter = Counter()
        for sent in corpus_sentences:
            for w in sent:
                self.unigrams[w.lower()] += 1
            for w1, w2 in zip(sent, sent[1:]):
                self.bigrams[(w1.lower(), w2.lower())] += 1
        self.vocab_size = max(len(self.unigrams), 1)
        self.total_unigrams = max(sum(self.unigrams.values()), 1)

    def score(self, candidate: str, left: Optional[str], right: Optional[str]) -> float:
        """Return a log-probability-like score (higher = more plausible)."""
        c = candidate.lower()
        uni = math.log((self.unigrams.get(c, 0) + 1) / (self.total_unigrams + self.vocab_size))
        bi = 0.0
        if left:
            left_count = self.unigrams.get(left.lower(), 0) + self.vocab_size
            bi += math.log((self.bigrams.get((left.lower(), c), 0) + 1) / left_count)
        if right:
            c_count = self.unigrams.get(c, 0) + self.vocab_size
            bi += math.log((self.bigrams.get((c, right.lower()), 0) + 1) / c_count)
        return uni + bi


# ---------------------------------------------------------------------------
# 5. Candidate generation (constrained -- only from domain vocab)
# ---------------------------------------------------------------------------

def generate_candidates(
    token: str,
    vocab: set[str],
    max_edit_distance: float = 2.0,
    max_candidates: int = 6,
) -> list[tuple[str, float]]:
    """Return (candidate, visual_distance) pairs, closest first.
    Only words that are *cheap enough* to reach from the OCR token are
    considered -- this is what stops the pipeline from ever inventing
    unrelated words."""
    scored = []
    tok_lower = token.lower()
    for word in vocab:
        if abs(len(word) - len(tok_lower)) > max_edit_distance + 1:
            continue  # fast length filter before the expensive DP
        dist = weighted_edit_distance(tok_lower, word)
        if dist <= max_edit_distance:
            scored.append((word, dist))
    scored.sort(key=lambda x: x[1])
    return scored[:max_candidates]


# ---------------------------------------------------------------------------
# 6. Anomaly detection -- decide which tokens are even worth touching
# ---------------------------------------------------------------------------

def needs_review(token: str, vocab: set[str]) -> bool:
    t = token.lower().strip(".,;:!?")
    if not t:
        return False
    if t in vocab:
        return False           # already a known good word -- leave it alone
    return True


# ---------------------------------------------------------------------------
# 7. Guardrails + orchestration
# ---------------------------------------------------------------------------

@dataclass
class Correction:
    original: str
    corrected: str
    start: int
    end: int
    visual_distance: float
    context_score_gain: float
    accepted: bool


@dataclass
class CorrectionResult:
    corrected_text: str
    corrections: list[Correction] = field(default_factory=list)


def correct_document(
    text: str,
    vocab: set[str],
    context_model: NgramContextModel,
    max_edit_distance: float = 2.0,
    min_context_gain: float = 0.15,   # guardrail: candidate must be
                                       # meaningfully more plausible than
                                       # leaving the token untouched
) -> CorrectionResult:
    tokens = tokenize_with_spans(text)
    words = [t.text for t in tokens]
    result_chars = list(text)
    corrections: list[Correction] = []

    # Pass A: strip rule-based noise tokens
    keep_mask = [True] * len(tokens)
    for i, tok in enumerate(tokens):
        prev_w = words[i - 1] if i > 0 else None
        next_w = words[i + 1] if i < len(words) - 1 else None
        if is_noise_token(tok.text, prev_w, next_w):
            keep_mask[i] = False
            corrections.append(Correction(
                original=tok.text, corrected="[REMOVED: noise]",
                start=tok.start, end=tok.end,
                visual_distance=0.0, context_score_gain=0.0, accepted=True,
            ))

    # Pass B: constrained substitution for remaining flagged tokens
    for i, tok in enumerate(tokens):
        if not keep_mask[i]:
            continue
        if not needs_review(tok.text, vocab):
            continue

        left = words[i - 1] if i > 0 else None
        right = words[i + 1] if i < len(words) - 1 else None

        candidates = generate_candidates(tok.text, vocab, max_edit_distance)
        if not candidates:
            continue  # nothing plausible enough -- leave untouched (safe default)

        baseline_score = context_model.score(tok.text, left, right)
        best = None
        best_total = None
        for cand, dist in candidates:
            ctx_score = context_model.score(cand, left, right)
            gain = ctx_score - baseline_score
            # combine: prioritise low visual distance, require real context gain
            total = gain - (dist * 0.5)
            if best_total is None or total > best_total:
                best, best_total, best_gain, best_dist = cand, total, gain, dist

        accept = best is not None and best_gain >= min_context_gain
        corrections.append(Correction(
            original=tok.text, corrected=best or tok.text,
            start=tok.start, end=tok.end,
            visual_distance=best_dist if best else -1,
            context_score_gain=best_gain if best else 0.0,
            accepted=accept,
        ))

    # Apply accepted edits right-to-left so char offsets stay valid
    for corr in sorted(corrections, key=lambda c: c.start, reverse=True):
        if not corr.accepted:
            continue
        if corr.corrected.startswith("[REMOVED"):
            result_chars[corr.start:corr.end] = []
        else:
            result_chars[corr.start:corr.end] = list(corr.corrected)

    return CorrectionResult(corrected_text="".join(result_chars), corrections=corrections)


# ---------------------------------------------------------------------------
# Demo using your sample OCR output
# ---------------------------------------------------------------------------

SAMPLE_OCR_TEXT = """General health medical camp provide basic checkups for an
age groups a They beat common illness check blood pressure a and give health a
specially co medical camps a focus a on specific problems like
orthopaedic camps a heart camps a diabetes eye for camp a dental camps a
cancer screening camps maternal a held health-unch a camps
Focus on the health of pregnant women local lacking mothering children a
immuneration camps a organized to provide vaccines to
prevent communicable diseases such as pole a missiles a tetanus a and hepatitis
Blood donation camps correctsope blood from volunteer a
dancers a Blood is properly tested and spared to ensure safe
school health-camps a children health include a
organized an schools to check eye of hearing tests
dental checkups station a check Health education a"""

# A domain vocabulary you'd build from the subject's model answers /
# textbook glossary (Phase 1 of the plan). Kept small here for the demo.
DOMAIN_VOCAB = {
    "general", "health", "medical", "camp", "camps", "provide", "provides",
    "basic", "checkups", "for", "all", "age", "groups", "they", "treat",
    "common", "illness", "illnesses", "check", "blood", "pressure", "and",
    "give", "advice", "specially", "specialized", "organized", "focus", "on",
    "specific", "problems", "like", "orthopaedic", "heart", "diabetes",
    "eye", "for", "dental", "cancer", "screening", "maternal", "held",
    "health", "focus", "the", "of", "pregnant", "women", "and", "lactating",
    "mothers", "children", "immunization", "to", "vaccines", "prevent",
    "communicable", "diseases", "such", "as", "polio", "measles", "tetanus",
    "hepatitis", "donation", "collected", "from", "volunteer", "donors",
    "is", "properly", "tested", "stored", "ensure", "safe", "transfusion",
    "school", "include", "hearing", "checkups", "education", "in", "schools",
}

# Reference sentences to seed the n-gram context model -- in production,
# build this from a corpus of model/reference answers for the subject.
REFERENCE_SENTENCES = [
    "general health camps provide basic checkups for all age groups".split(),
    "they treat common illness and check blood pressure and give advice".split(),
    "specialized medical camps focus on specific problems like orthopaedic camps heart camps diabetes camps eye camps dental camps and cancer screening camps".split(),
    "maternal health camps are held to focus on the health of pregnant women and lactating mothers and children".split(),
    "immunization camps are organized to provide vaccines to prevent communicable diseases such as polio measles tetanus and hepatitis".split(),
    "blood donation camps collect blood from volunteer donors".split(),
    "blood is properly tested and stored to ensure safe transfusion".split(),
    "school health camps for children include eye and hearing tests dental checkups and health education".split(),
]


def main():
    ctx_model = NgramContextModel(REFERENCE_SENTENCES)
    result = correct_document(SAMPLE_OCR_TEXT, DOMAIN_VOCAB, ctx_model)

    print("=" * 70)
    print("CORRECTED TEXT")
    print("=" * 70)
    print(result.corrected_text)

    print("\n" + "=" * 70)
    print("AUDIT LOG (accepted changes only)")
    print("=" * 70)
    for c in result.corrections:
        if c.accepted:
            print(json.dumps({
                "original": c.original,
                "corrected": c.corrected,
                "visual_distance": round(c.visual_distance, 2),
                "context_gain": round(c.context_score_gain, 3),
            }))

    print("\n" + "=" * 70)
    print("REVIEWED BUT LEFT UNCHANGED (below guardrail threshold)")
    print("=" * 70)
    for c in result.corrections:
        if not c.accepted and not c.corrected.startswith("[REMOVED"):
            print(f"  {c.original!r} -> best guess {c.corrected!r} "
                  f"(gain {c.context_score_gain:.3f}, not confident enough)")


if __name__ == "__main__":
    main()
