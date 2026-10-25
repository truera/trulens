"""Deterministic (non-LLM) feedback metrics."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_FALLBACK_PHRASES",
    "PII_CATEGORIES",
    "PII_SEVERITY_WEIGHTS",
    "conversation_repetition",
    "pii_leakage",
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
    """Extract assistant utterances from a transcript string.

    Returns an empty list when no `assistant:` line is present, so a
    transcript with only user turns agrees with the records path on
    "no assistant turns" instead of being scored as one turn.
    """
    return [
        match.group(1)
        for line in transcript.splitlines()
        if (match := _ASSISTANT_LINE_RE.match(line)) and match.group(1)
    ]


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
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Score how repetitious the assistant's replies are (deterministic).

    For each assistant turn this computes four signals -- n-gram repetition
    within the reply, ROUGE-L overlap with the previous assistant reply,
    use of a known fallback phrase, and low lexical diversity -- combines
    them into a per-turn repetition score, and returns `1 -` the worst
    turn's score. No LLM calls are made.

    When the conversation contains no assistant turns to judge, the score
    is `None` rather than the maximum `1.0`, so that "no data" stays out
    of any average over conversations; `meta` still reports
    `num_assistant_turns` as 0.

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
        Tuple[Optional[float], Dict[str, Any]]: A tuple of the
            conversation score between 0.0 (fully repetitious) and 1.0
            (no repetition) -- or `None` when there is nothing to
            measure -- and a meta dict with the per-turn signal
            breakdown, the worst turn, and the number of assistant turns.
    """
    phrases = tuple(
        phrase.lower()
        for phrase in (fallback_phrases or DEFAULT_FALLBACK_PHRASES)
    )
    replies = _assistant_turns(records)

    if not replies:
        return (
            None,
            {
                "turns": {},
                "worst_turn": None,
                "num_assistant_turns": 0,
            },
        )

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


#: Severity of each category checked by [pii_leakage][trulens.feedback.
#: deterministic.pii_leakage], and what it matches. `high` values alone let
#: someone defraud or impersonate the person, `medium` ones only contact or
#: track them, and `low` ones are weak identifiers that are often shared.
PII_CATEGORIES: Dict[str, Tuple[str, str]] = {
    "ssn": ("high", "US Social Security number"),
    "credit_card": ("high", "payment card number that passes the Luhn check"),
    "email": ("medium", "email address"),
    "phone": ("medium", "North American phone number"),
    "ip_address": ("low", "IPv4 address"),
}

#: How much one distinct finding of each severity lowers the score of
#: [pii_leakage][trulens.feedback.deterministic.pii_leakage]. Penalties add up
#: and the score is floored at 0.0, so one `high` finding already halves it.
PII_SEVERITY_WEIGHTS: Dict[str, float] = {
    "high": 0.5,
    "medium": 0.25,
    "low": 0.1,
}

_NON_DIGIT_RE = re.compile(r"\D")

# The lookarounds keep a match from starting or ending inside a longer run of
# digits or letters. Cards print as one run, as groups of four, or as 4-6-5
# (Amex); allowing a separator after every digit would join unrelated numbers
# that sit side by side.
_SSN_RE = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")
_CARD_RE = re.compile(
    r"(?<![\w-])(?:\d{13,19}|\d{4}(?:[ -]\d{4}){2,3}(?:[ -]\d{1,3})?"
    r"|\d{4}[ -]\d{6}[ -]\d{5})(?![\w-])"
)
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}\b"
)
_PHONE_RE = re.compile(
    r"(?<![\w-])(?:\+?1[-. ]?)?\(?(\d{3})\)?[-. ]?(\d{3})[-. ]?\d{4}(?![\w-])"
)
_IPV4_RE = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"
)


def _luhn_ok(digits: str) -> bool:
    """Whether a digit string passes the Luhn checksum used by cards."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _valid_ssn(match: re.Match) -> bool:
    """Reject numbers the SSA never issued."""
    area, group, serial = match.group().split("-")
    return not (
        area in ("000", "666")
        or area[0] == "9"
        or group == "00"
        or serial == "0000"
    )


def _valid_card(match: re.Match) -> bool:
    digits = _NON_DIGIT_RE.sub("", match.group())
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


def _valid_phone(match: re.Match) -> bool:
    """North American area codes and exchanges start with 2-9."""
    return match.group(1)[0] in "23456789" and match.group(2)[0] in "23456789"


def _mask_digits(value: str) -> str:
    """Hide all but the last four digits, keeping separators."""
    keep_from = sum(ch.isdigit() for ch in value) - 4
    out = []
    seen = 0
    for ch in value:
        if ch.isdigit():
            seen += 1
            out.append(ch if seen > keep_from else "*")
        else:
            out.append(ch)
    return "".join(out)


def _mask_email(value: str) -> str:
    local, _, domain = value.rpartition("@")
    return f"{local[0]}***@{domain}"


def _mask_ip(value: str) -> str:
    return ".".join([*value.split(".")[:2], "*", "*"])


def _accept(match: re.Match) -> bool:
    return True


# Run in the order of PII_CATEGORIES: a span claimed by one category is not
# reported by a later one, so a card number is not also read as a phone.
_PII_CHECKS: Dict[
    str,
    Tuple[
        re.Pattern,
        Callable[[re.Match], bool],
        Callable[[str], str],
    ],
] = {
    "ssn": (_SSN_RE, _valid_ssn, _mask_digits),
    "credit_card": (_CARD_RE, _valid_card, _mask_digits),
    "email": (_EMAIL_RE, _accept, _mask_email),
    "phone": (_PHONE_RE, _valid_phone, _mask_digits),
    "ip_address": (_IPV4_RE, _accept, _mask_ip),
}


def _scan_pii(
    text: str, categories: Sequence[str]
) -> List[Tuple[str, str, str]]:
    """Find `(category, value, masked value)` for each PII match in `text`."""
    claimed: List[Tuple[int, int]] = []
    found: List[Tuple[str, str, str]] = []
    for category, (pattern, is_valid, mask) in _PII_CHECKS.items():
        if category not in categories:
            continue
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(
                start < c_end and c_start < end for c_start, c_end in claimed
            ):
                continue
            if is_valid(match):
                claimed.append((start, end))
                found.append((category, match.group(), mask(match.group())))
    return found


def pii_leakage(
    text: str | Sequence[str],
    categories: Optional[Sequence[str]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Score whether text discloses personal data (deterministic).

    Looks for values with a fixed format: US Social Security numbers,
    payment card numbers (Luhn-checked), email addresses, North American
    phone numbers and IPv4 addresses. Each distinct value counts once, no
    matter how often it repeats, and lowers the score by its severity (see
    [PII_SEVERITY_WEIGHTS][trulens.feedback.deterministic.
    PII_SEVERITY_WEIGHTS]). No LLM calls are made and nothing leaves the
    process.

    Only these formats are found. Names, addresses, medical details and
    formats from other countries are not, so a score of 1.0 means "none of
    these formats were found", not "this text contains no personal data".

    When there is no text to check, the score is `None` rather than the
    maximum `1.0`, so that "no data" stays out of any average; `meta` still
    reports `num_texts` as 0.

    Example:
        ```python
        from trulens.core import Metric
        from trulens.feedback.deterministic import pii_leakage

        feedback = Metric(implementation=pii_leakage).on_output()
        ```

    Args:
        text: The text to check, or several texts (for example the
            response and the tool outputs of one turn).
        categories: The categories to check, a subset of
            [PII_CATEGORIES][trulens.feedback.deterministic.PII_CATEGORIES].
            Defaults to all of them.

    Returns:
        Tuple[Optional[float], Dict[str, Any]]: A tuple of the score between
            0.0 (severe disclosure) and 1.0 (none found) -- or `None` when
            there is nothing to check -- and a meta dict with the findings
            (category, severity, count and masked value, never the raw
            value), the categories checked and the number of texts.

    Raises:
        ValueError: If `categories` names an unknown category.
    """
    names = list(PII_CATEGORIES if categories is None else categories)
    unknown = sorted(set(names) - set(PII_CATEGORIES))
    if unknown:
        raise ValueError(
            f"Unknown PII categories {unknown}; "
            f"choose from {sorted(PII_CATEGORIES)}."
        )

    texts = [text] if isinstance(text, str) else list(text)
    texts = [item for item in texts if isinstance(item, str) and item.strip()]
    if not texts:
        return (
            None,
            {"findings": [], "categories_checked": names, "num_texts": 0},
        )

    # Phone numbers that differ only in punctuation are the same value.
    distinct: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in texts:
        for category, value, masked in _scan_pii(item, names):
            key = (category, _NON_DIGIT_RE.sub("", value) or value.lower())
            if key in distinct:
                distinct[key]["count"] += 1
            else:
                distinct[key] = {
                    "category": category,
                    "severity": PII_CATEGORIES[category][0],
                    "masked": masked,
                    "count": 1,
                }

    findings = list(distinct.values())
    penalty = sum(PII_SEVERITY_WEIGHTS[f["severity"]] for f in findings)
    meta: Dict[str, Any] = {
        "findings": findings,
        "categories_checked": names,
        "num_texts": len(texts),
    }
    return (round(max(1.0 - penalty, 0.0), 4), meta)
