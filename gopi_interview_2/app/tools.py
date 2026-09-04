"""Tool schemas exposed to the LLM (Anthropic tool-use shape) and a safe dispatcher.

The LLM never sees SQL; it only sees these five functions. `search_hotel_info` and
`out_of_scope` are handled by the agent itself, the reservation tools dispatch here.
"""
import logging

from app import reservations

log = logging.getLogger(__name__)

_EMAIL_PROP = {"type": "string", "description": "The guest's email address exactly as given by the guest."}
_ID_PROP = {"type": "integer", "description": "The numeric reservation ID given by the guest."}

TOOLS: list[dict] = [
    {
        "name": "search_hotel_info",
        "description": (
            "Look up factual information about the Grand Azure Bay Hotel (location, rooms, facilities, "
            "hygiene and safety, dining, menus, policies such as check-in/out, cancellation, pets, payment). "
            "ALWAYS use this for any question about the hotel instead of answering from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The guest's question, self-contained."}},
            "required": ["question"],
        },
    },
    {
        "name": "create_reservation",
        "description": "Book a room for the guest you are talking to. Only call it once ALL five fields are known.",
        "input_schema": {
            "type": "object",
            "properties": {
                "guest_name": {"type": "string", "description": "Full name of the guest."},
                "email": _EMAIL_PROP,
                "room_type": {"type": "string", "enum": ["standard", "deluxe", "suite"]},
                "check_in": {"type": "string", "description": "Check-in date, YYYY-MM-DD."},
                "check_out": {"type": "string", "description": "Check-out date, YYYY-MM-DD."},
            },
            "required": ["guest_name", "email", "room_type", "check_in", "check_out"],
        },
    },
    {
        "name": "view_reservation",
        "description": "Retrieve the guest's own reservation. Requires BOTH the reservation ID and the booking email.",
        "input_schema": {
            "type": "object",
            "properties": {"reservation_id": _ID_PROP, "email": _EMAIL_PROP},
            "required": ["reservation_id", "email"],
        },
    },
    {
        "name": "cancel_reservation",
        "description": "Cancel the guest's own reservation. Requires BOTH the reservation ID and the booking email.",
        "input_schema": {
            "type": "object",
            "properties": {"reservation_id": _ID_PROP, "email": _EMAIL_PROP},
            "required": ["reservation_id", "email"],
        },
    },
    {
        "name": "out_of_scope",
        "description": (
            "Use when the message is not about the Grand Azure Bay Hotel or the guest's own reservation: "
            "small talk, general knowledge, other businesses, coding, jokes, personal advice, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Short reason, e.g. 'general knowledge question'."}},
            "required": ["reason"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOLS}

_RESERVATION_HANDLERS = {
    "create_reservation": reservations.create_reservation,
    "view_reservation": reservations.view_reservation,
    "cancel_reservation": reservations.cancel_reservation,
}


def call_reservation_tool(name: str, args: dict) -> dict:
    """Validate the argument names, call the backend, and convert any crash into a safe error dict."""
    handler = _RESERVATION_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": "Unknown tool."}
    schema = next(t for t in TOOLS if t["name"] == name)["input_schema"]
    missing = [k for k in schema["required"] if k not in (args or {})]
    if missing:
        return {"ok": False, "error": f"Missing required field(s): {', '.join(missing)}."}
    clean_args = {k: args[k] for k in schema["properties"] if k in args}
    try:
        return handler(**clean_args)
    except Exception:  # noqa: BLE001 - never let a stack trace reach the LLM or the user
        log.exception("Tool %s failed", name)
        return {"ok": False, "error": "The reservation system hit an unexpected error. Please try again."}
