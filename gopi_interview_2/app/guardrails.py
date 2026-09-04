"""Input/output guardrails.

Input side: cheap regex checks that run BEFORE the LLM sees the message (prompt injection,
bulk-data / other-guest requests, SQL-ish requests, size limits). Off-topic detection is left to the
LLM, which routes such messages to the `out_of_scope` tool (see agent.py).

Output side: make sure no email other than the user's own ever appears in a reply.
"""
import re
from dataclasses import dataclass

from app import config
from app.pii import redact_emails

MAX_INPUT_CHARS = 2000

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+|everything\s+|your\s+)?(previous|prior|above|earlier)?\s*(instructions?|rules?)",
    r"(reveal|show|print|repeat|output|leak|display)\s+(me\s+)?(your\s+|the\s+)?(system|hidden|initial|original|secret)\s+(prompt|instructions?)",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+now\s+(a|an|in|the)\b",
    r"\b(developer|god|admin)\s+mode\b",
    r"\bjailbreak\b",
    r"\bpretend\s+(you\s+are|to\s+be)\b",
    r"\bact\s+as\s+(a|an|the)\s+(?!guest)",
    r"\bnew\s+instructions?\s*:",
    r"^\s*(system|assistant|developer)\s*:",
    r"\boverride\s+(your|the|all)\s+(rules|instructions|guardrails|safety)",
]

_DATA_NOUNS = (
    r"(reservations?|bookings?|customers?|users?|records?|emails?|"
    r"guest\s+(?:data|details|information|info|list|names|records)|"
    r"guests'?\s+(?:data|details|information|info|names|emails|records)|"
    r"list\s+of\s+(?:all\s+)?guests)"
)
_BULK_PATTERN = r"\b(all|every|entire|complete|full|whole)\s+(of\s+)?(the\s+|your\s+)?(list\s+of\s+)?" + _DATA_NOUNS
_LIST_PATTERN = r"\b(list|dump|export|show|give|print|display|fetch|retrieve|enumerate)\b[^.?!]*\b" + _DATA_NOUNS
_OWN_DATA_HINT = r"\b(my|mine|our|i\s+made|i\s+booked|i\s+have)\b"
_OTHER_GUEST_PATTERN = (
    r"\b(other|another|someone\s+else'?s?|somebody\s+else'?s?|other\s+people'?s?|different|a\s+friend'?s?|my\s+friend'?s?)"
    r"\s+(guest'?s?\s+|person'?s?\s+|customer'?s?\s+|user'?s?\s+)?(reservations?|bookings?|details|data|emails?|records?)"
)
_SQL_PATTERNS = [
    r"\bselect\b[^.?!]*\bfrom\b",
    r"\bdrop\s+table\b",
    r"\bdelete\s+from\b",
    r"\binsert\s+into\b",
    r"\b(database|sqlite|sql\s+query)\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _INJECTION_PATTERNS]
_BULK_RE = re.compile(_BULK_PATTERN, re.IGNORECASE)
_LIST_RE = re.compile(_LIST_PATTERN, re.IGNORECASE)
_OWN_DATA_RE = re.compile(_OWN_DATA_HINT, re.IGNORECASE)
_OTHER_GUEST_RE = re.compile(_OTHER_GUEST_PATTERN, re.IGNORECASE)
_SQL_RE = [re.compile(p, re.IGNORECASE) for p in _SQL_PATTERNS]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = "ok"
    message: str = ""


def contains_injection(text: str) -> bool:
    """True if the text looks like a prompt-injection attempt. Used for user input AND PDF chunks."""
    return any(p.search(text or "") for p in _INJECTION_RE)


def is_bulk_or_other_guest_request(text: str) -> bool:
    text = text or ""
    if _BULK_RE.search(text) or _OTHER_GUEST_RE.search(text):
        return True
    if any(p.search(text) for p in _SQL_RE):
        return True
    # "show me the bookings" is bulk; "show me my booking" is fine.
    return bool(_LIST_RE.search(text)) and not _OWN_DATA_RE.search(text)


def check_input(text: str) -> GuardrailResult:
    text = (text or "").strip()
    if not text:
        return GuardrailResult(False, "empty", "Please type a question about the hotel or a reservation request.")
    if len(text) > MAX_INPUT_CHARS:
        return GuardrailResult(False, "too_long", "That message is too long. Please keep it under 2000 characters.")
    if contains_injection(text):
        return GuardrailResult(False, "prompt_injection", config.INJECTION_MESSAGE)
    if is_bulk_or_other_guest_request(text):
        return GuardrailResult(False, "bulk_or_other_guest_data", config.PRIVACY_MESSAGE)
    return GuardrailResult(True)


def sanitize_output(text: str, allowed_emails: set[str]) -> str:
    """Final reply filter: never leak an email the current user did not provide, never return empty text."""
    text = (text or "").strip()
    if not text:
        return config.FALLBACK_MESSAGE
    return redact_emails(text, allowed_emails)
