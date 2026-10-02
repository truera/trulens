"""Deterministic (non-LLM) feedback metrics."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_FALLBACK_PHRASES",
    "conversation_repetition",
]


#: Fallback phrases checked by [conversation_repetition][trulens.feedback.
#: deterministic.conversation_repetition]. Matched case-insensitively as
#: substrings of an assistant reply. Users can override via the
#: `fallback_phrases` argument.
DEFAULT_FALLBACK_PHRASES: Tuple[str, ...] = (
    "i'm sorry, i can't help with that",
    "i'm sorry, but i can't help with that",
    "i don't understand",
    "could you rephrase",
    "could you please rephrase",
    "as an ai language model",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ASSISTANT_LINE_RE = re.compile(
    r"^\s*(?:turn\s+\d+\s+)?assistant\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)

_ASSISTANT_ROLES = frozenset({"assistant", "ai", "bot"})


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _assistant_text(record: Any) -> Optional[str]:
    """Extract the assistant utterance from a single conversation record."""
    if hasattr(record, "main_output"):
        output = record.main_output
        return str(output) if output is not None else None
    if isinstance(record, dict):
        if record.get("output") is not None:
            return str(record["output"])
        role = str(record.get("role", record.get("speaker", ""))).lower()
        content = record.get(
            "content", record.get("text", record.get("message"))
        )
        if role in _ASSISTANT_ROLES and content is not None:
            return str(content)
        return None
    return None


def _assistant_turns_from_transcript(transcript: str) -> List[str]:
    """Extract assistant utterances from a transcript string."""
    turns = [
        match.group(1)
        for line in transcript.splitlines()
        if (match := _ASSISTANT_LINE_RE.match(line)) and match.group(1)
    ]
    if turns:
        return turns
    stripped = transcript.strip()
    return [stripped] if stripped else []


def _assistant_turns(records: List[Any] | str) -> List[str]:
    """Extract assistant utterances, in order, from conversation records."""
    if isinstance(records, str):
        return _assistant_turns_from_transcript(records)
    turns: List[str] = []
    for record in records:
        text = _assistant_text(record)
        if text:
            turns.append(text)
    return turns


def _ngram_repetition(tokens: Sequence[str], n: int) -> float:
    """Fraction of n-grams in `tokens` that are repeats (0 = none)."""
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    """Length of the longest common subsequence of two token sequences."""
    prev = [0] * (len(second) + 1)
    for token in first:
        curr = [0]
        for j, other in enumerate(second, start=1):
            if token == other:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _rouge_l_f1(first: Sequence[str], second: Sequence[str]) -> float:
    """ROUGE-L F1 between two token sequences (0 = disjoint, 1 = same)."""
    if not first or not second:
        return 0.0
    lcs = _lcs_length(first, second)
    if lcs == 0:
        return 0.0
    precision = lcs / len(first)
    recall = lcs / len(second)
    return 2 * precision * recall / (precision + recall)


def _fallback_signal(text: str, phrases: Sequence[str]) -> float:
    """1.0 if `text` contains a known fallback phrase, else 0.0."""
    lowered = text.lower()
    return 1.0 if any(phrase in lowered for phrase in phrases) else 0.0


def _low_diversity_signal(tokens: Sequence[str]) -> float:
    """Low lexical diversity signal in [0, 1].

    Normal prose has a type-token ratio near 1.0, so the raw `1 - TTR`
    is rescaled to only fire when diversity is substantially below that:
    0.0 for short replies or TTR above 0.7.
    """
    if len(tokens) < 4:
        return 0.0
    raw = 1.0 - len(set(tokens)) / len(tokens)
    return max(0.0, (raw - 0.3) / 0.7)


def _combine_signals(signals: Sequence[float]) -> float:
    """Combine repetition signals into one score in [0, 1].

    Uses the noisy-OR form `1 - prod(1 - s)`: a turn scores high if *any*
    signal fires strongly. A plain mean cannot satisfy the acceptance
    criteria -- two identical non-fallback replies would score only 0.25
    (0.75 after the `1 -` flip), above the required 0.5 threshold -- while
    this form scores exact repeats at 1.0 and stays near 0.0 when no
    signal fires.
    """
    remaining = 1.0
    for signal in signals:
        remaining *= 1.0 - min(max(signal, 0.0), 1.0)
    return 1.0 - remaining


def conversation_repetition(
    records: List[Any] | str,
    fallback_phrases: Optional[Sequence[str]] = None,
    ngram_orders: Sequence[int] = (2, 3, 4),
) -> Tuple[float, Dict[str, Any]]:
    """Score how repetitious the assistant's replies are (deterministic).

    For each assistant turn this computes four signals -- n-gram repetition
    within the reply, ROUGE-L overlap with the previous assistant reply,
    use of a known fallback phrase, and low lexical diversity -- combines
    them into a per-turn repetition score, and returns `1 -` the worst
    turn's score. No LLM calls are made.

    Example:
        ```python
        from trulens.core import Metric
        from trulens.feedback.deterministic import conversation_repetition

        feedback = Metric(implementation=conversation_repetition)
        feedback = feedback.on_conversation()
        ```

    Args:
        records: The ordered conversation records, or a transcript string.
            Records may be objects with `main_input`/`main_output`, dicts
            with `input`/`output`, or dicts with `role`/`content`.
        fallback_phrases: Phrases marking a fallback reply, matched
            case-insensitively. Defaults to
            [DEFAULT_FALLBACK_PHRASES][trulens.feedback.deterministic.
            DEFAULT_FALLBACK_PHRASES].
        ngram_orders: n-gram orders used for the within-reply repetition
            signal.

    Returns:
        Tuple[float, Dict[str, Any]]: A tuple of the conversation score
            between 0.0 (fully repetitious) and 1.0 (no repetition), and a
            meta dict with the per-turn signal breakdown, the worst turn,
            and the number of assistant turns.
    """
    phrases = tuple(
        phrase.lower()
        for phrase in (fallback_phrases or DEFAULT_FALLBACK_PHRASES)
    )
    replies = _assistant_turns(records)

    turns: Dict[int, Dict[str, Any]] = {}
    prev_tokens: Optional[List[str]] = None
    worst_score = 0.0
    worst_turn: Optional[int] = None

    for idx, reply in enumerate(replies, start=1):
        tokens = _tokenize(reply)
        ngram = sum(_ngram_repetition(tokens, n) for n in ngram_orders) / len(
            ngram_orders
        )
        # Squared to discount incidental shared phrases ("the city");
        # exact duplicates still score 1.0.
        overlap = (
            _rouge_l_f1(tokens, prev_tokens) ** 2
            if prev_tokens is not None
            else 0.0
        )
        fallback = _fallback_signal(reply, phrases)
        diversity = _low_diversity_signal(tokens)
        score = _combine_signals([ngram, overlap, fallback, diversity])
        turns[idx] = {
            "ngram_repetition": round(ngram, 4),
            "overlap_with_previous": round(overlap, 4),
            "fallback_phrase": fallback,
            "lexical_diversity": round(diversity, 4),
            "score": round(score, 4),
            "overlaps_turn": idx - 1 if overlap > 0.0 else None,
        }
        if score > worst_score:
            worst_score = score
            worst_turn = idx
        prev_tokens = tokens

    meta: Dict[str, Any] = {
        "turns": turns,
        "worst_turn": worst_turn,
        "num_assistant_turns": len(replies),
    }
    return (round(1.0 - worst_score, 4), meta)
