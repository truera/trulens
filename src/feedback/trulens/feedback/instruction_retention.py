"""Scoring for the instruction retention conversation metric.

A standing instruction is in force from the turn it was given until the turn
it was revoked, inclusive. The score is the share of (turn, instruction)
pairs in force that were followed. The forgetting and correction ratios
follow Multi-IF (arXiv 2410.15553), over consecutive turns in force:

- forgetting ratio: followed in one turn, not followed in the next, as a
  share of all followed-then-next transitions;
- correction ratio: not followed in one turn, followed in the next, as a
  share of all missed-then-next transitions.

Each instruction is decided by exactly one source: a caller-supplied check,
or the judge. Revocations come from the caller, never from the judge.

Instructions are matched to `checks` and `revocations` by text that is
case- and punctuation-insensitive but not semantic: "respond using JSON"
does not match "Reply in JSON". Callers should key both by the words the
user gave; a judge-listed instruction worded differently from a check key is
judged as well, and one worded differently from a revocation key is not
revoked.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from trulens.feedback import output_schemas as feedback_output_schemas

Turns = Sequence[Tuple[Optional[str], Optional[str]]]

# The three outcomes `assess` reports in `meta["status"]`.
SCORED = "scored"
NO_INSTRUCTIONS = "no_instructions"
NOTHING_EVALUABLE = "nothing_evaluable"

_NOT_WORD = re.compile(r"[^\w\s]")
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_judge_answer(
    response: Optional[str],
) -> feedback_output_schemas.InstructionRetentionResponse:
    """The judge's answer as a schema, also read from inside a code fence.

    Providers without structured outputs often wrap the JSON in a markdown
    fence, sometimes after a line of prose; the first fenced block is read.

    Raises:
        ValueError: The answer is not JSON of the expected shape.
    """
    if not isinstance(response, str):
        raise ValueError("the judge returned no text")
    fenced = _CODE_FENCE.search(response)
    text = fenced.group(1) if fenced else response
    return feedback_output_schemas.InstructionRetentionResponse.model_validate_json(
        text
    )


def _normalize(text: str) -> str:
    """Lowercased words, punctuation removed: the key text is matched on."""
    return " ".join(_NOT_WORD.sub(" ", text.lower()).split())


def _turn_given(instruction: str, turns: Turns) -> int:
    """The first turn whose user message states `instruction`, else turn 1.

    The instruction must appear as whole words, so a check keyed "EU" does
    not start at a turn about "European" policy. When no user message states
    it (it was worded differently, or set before the transcript begins, as
    in a system prompt), it is in force from turn 1: a check the caller
    supplies is taken to apply to the whole conversation unless the
    conversation shows when it was set.
    """
    wanted = f" {_normalize(instruction)} "
    for number, (user, _) in enumerate(turns, start=1):
        if user and wanted in f" {_normalize(user)} ":
            return number
    return 1


def assess(
    judged: List[feedback_output_schemas.StandingInstruction],
    turns: Optional[Turns],
    checks: Dict[str, Callable[[str], bool]],
    revocations: Dict[str, int],
) -> Tuple[Optional[float], Dict]:
    """Score standing instructions from the judge and from checks.

    Args:
        judged: The standing instructions the judge listed, with verdicts.
            Any whose text matches a check is dropped: the check decides it.
        turns: The conversation as `(user, assistant)` turns, or None when
            only a transcript string was given.
        checks: Instruction text to a predicate on one assistant reply.
        revocations: Instruction text to the turn it was revoked in. A
            revocation before the turn an instruction was given is ignored,
            so a revocation cannot remove an instruction from the score.

    Returns:
        The share of in-force pairs followed, and a meta dict with
        `status`, the per-instruction verdicts, the first broken turn of
        each, which source decided it, and both ratios. The score is 1.0
        with status `no_instructions` when the user set none, and None
        with status `nothing_evaluable` when instructions were set but no
        assistant turn was in force for any of them. A ratio with no
        transitions to count is None.
    """
    checked = {_normalize(text) for text in checks}
    revoked = {_normalize(text): turn for text, turn in revocations.items()}

    decided: List[Dict] = []
    for item in judged:
        if _normalize(item.instruction) in checked:
            continue
        decided.append({
            "instruction": item.instruction,
            "turn_given": item.turn_given,
            "decided_by": "judge",
            "verdicts": {v.turn: (v.followed, v.reason) for v in item.verdicts},
        })
    for text, predicate in checks.items():
        given = _turn_given(text, turns or [])
        decided.append({
            "instruction": text,
            "turn_given": given,
            "decided_by": "check",
            "verdicts": {
                number: (bool(predicate(assistant)), "check")
                for number, (_, assistant) in enumerate(turns or [], start=1)
                if number >= given and assistant is not None
            },
        })

    pairs = followed = 0
    kept = forgotten = missed = corrected = 0
    results: List[Dict] = []
    for item in decided:
        revoked_turn = revoked.get(_normalize(item["instruction"]))
        ignored_revocation = (
            revoked_turn is not None and revoked_turn < item["turn_given"]
        )
        if ignored_revocation:
            revoked_turn = None
        in_force = sorted(
            turn
            for turn in item["verdicts"]
            if turn >= item["turn_given"]
            and (revoked_turn is None or turn <= revoked_turn)
        )
        outcomes = [item["verdicts"][turn][0] for turn in in_force]
        pairs += len(outcomes)
        followed += sum(outcomes)
        for now, after in zip(outcomes, outcomes[1:]):
            if now:
                kept += 1
                forgotten += not after
            else:
                missed += 1
                corrected += after
        results.append({
            "instruction": item["instruction"],
            "turn_given": item["turn_given"],
            "revoked_turn": revoked_turn,
            "revocation_ignored": ignored_revocation,
            "decided_by": item["decided_by"],
            "first_broken_turn": next(
                (turn for turn, ok in zip(in_force, outcomes) if not ok), None
            ),
            "verdicts": [
                {
                    "turn": turn,
                    "followed": item["verdicts"][turn][0],
                    "reason": item["verdicts"][turn][1],
                }
                for turn in in_force
            ],
        })

    if not decided:
        status, score = NO_INSTRUCTIONS, 1.0
    elif not pairs:
        status, score = NOTHING_EVALUABLE, None
    else:
        status, score = SCORED, followed / pairs
    meta = {
        "status": status,
        "instructions": results,
        "pairs_in_force": pairs,
        "pairs_followed": followed,
        "forgetting_ratio": forgotten / kept if kept else None,
        "correction_ratio": corrected / missed if missed else None,
    }
    return score, meta
