"""PII helpers: validation and masking. Masked values are the only form that ever reaches the logs."""
import re

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_EMAIL_FIND_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

PII_KEYS = {"email", "guest_name", "name"}


def is_valid_email(email: str) -> bool:
    return isinstance(email, str) and len(email) <= 254 and bool(_EMAIL_RE.match(email.strip()))


def mask_email(email: str) -> str:
    """john.doe@example.com -> j***@example.com"""
    if not isinstance(email, str) or "@" not in email:
        return "***"
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def mask_name(name: str) -> str:
    """John Doe -> J*** D***"""
    if not isinstance(name, str) or not name.strip():
        return "***"
    return " ".join(f"{part[:1]}***" for part in name.split())


def mask_args(args: dict) -> dict:
    """Return a copy of tool arguments with PII fields masked (safe for logging)."""
    masked = {}
    for key, value in (args or {}).items():
        if key == "email":
            masked[key] = mask_email(str(value))
        elif key in PII_KEYS:
            masked[key] = mask_name(str(value))
        else:
            masked[key] = value
    return masked


def find_emails(text: str) -> set[str]:
    return {m.lower() for m in _EMAIL_FIND_RE.findall(text or "")}


def redact_emails(text: str, allowed: set[str]) -> str:
    """Mask every email in `text` except the ones the current user typed themselves."""
    return _EMAIL_FIND_RE.sub(lambda m: m.group(0) if m.group(0).lower() in allowed else mask_email(m.group(0)), text)
