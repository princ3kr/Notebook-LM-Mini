import ast
import json
import re
from typing import Any, List, Tuple


# If the student sends this while a recall question is pending, treat as a new
# learning request instead of an answer (Streamlit default placeholder).
NEW_TOPIC_INTENT_RE = re.compile(
    r"^(i\s+want\s+to\s+learn\s+about\s+\S+|"
    r"teach\s+me\s+about\s+\S+|"
    r"i['’]d\s+like\s+to\s+learn\s+about\s+\S+|"
    r"help\s+me\s+learn\s+about\s+\S+)",
    re.IGNORECASE,
)


# After the lesson, student confirms they want the check-up questions.
PROCEED_TO_QUIZ_RE = re.compile(
    r"(want\s+to\s+move\s+further|move\s+further|ready\s+for(\s+the)?\s+quiz|"
    r"start(\s+the)?\s+quiz|begin(\s+the)?\s+quiz|i\s*['’]m\s+ready|"
    r"let\s*['’]s\s+continue|proceed|move\s+on|next\s+step|"
    r"continue\s+to(\s+the)?\s+questions|go\s+to(\s+the)?\s+questions)",
    re.IGNORECASE,
)

PROCEED_PROMPT = (
    "When you've read the lesson, type **I want to move further** "
    "(or say you're **ready for the quiz**) to start the check-up questions."
)


def is_new_topic_intent(text: str) -> bool:
    if not text:
        return False
    return NEW_TOPIC_INTENT_RE.match(text) is not None


def is_proceed_to_quiz(text: str) -> bool:
    if not text or text == "__empty__":
        return False
    short_ok = bool(
        re.fullmatch(r"(ok|yes|yeah|yep)\.?", text.strip(), re.IGNORECASE)
    )
    return PROCEED_TO_QUIZ_RE.search(text) is not None or short_ok


def _parse_equations_field(raw: Any) -> List[dict]:
    """
    Neo4j stores `equations` as str(list_of_dicts); normalize to list of dicts.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    s = str(raw).strip()
    if not s or s in ("[]", "{}", "(none)", "null", "None"):
        return []

    # Try JSON first
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback: GraphService stored str(list_of_dicts) -> parse back
    try:
        data = ast.literal_eval(s)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except (ValueError, SyntaxError, TypeError):
        pass

    return []


def format_equations_for_prompt(raw: Any) -> Tuple[str, bool]:
    """
    Returns (equations_block, has_any_equation).
    equations_block is an LLM-ready block (includes names + latex + context).
    """
    eqs = _parse_equations_field(raw)
    if not eqs:
        return "", False

    parts: List[str] = []
    for i, eq in enumerate(eqs, 1):
        name = (eq.get("name") or eq.get("title") or f"Equation {i}").strip()
        latex = (eq.get("latex") or eq.get("formula") or "").strip()
        ctx = (eq.get("context") or eq.get("meaning") or "").strip()

        block = f"**{name}**"
        if latex:
            block += f"\n- LaTeX: `{latex}`"
        if ctx:
            block += f"\n- Context / meaning: {ctx}"

        parts.append(block)

    return "\n\n".join(parts), True

