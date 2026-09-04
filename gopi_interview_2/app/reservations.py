"""Reservation backend: SQLite schema + the three tool functions the agent may call.

Every function validates its inputs before touching the DB, returns a plain dict
({"ok": True, ...} or {"ok": False, "error": "..."}), never raises to the caller for
user errors, and never puts PII into an error message or a log line.
"""
import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from app import config
from app.pii import is_valid_email, mask_email, mask_name

log = logging.getLogger(__name__)

ROOM_TYPES = {"standard", "deluxe", "suite"}
MAX_STAY_NIGHTS = 30
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{1,79}$")

# (type, nightly price, number of rooms). Assumption: the PDF does not dictate inventory.
ROOM_SEED = [("standard", 120.0, 4), ("deluxe", 190.0, 3), ("suite", 320.0, 2)]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT    NOT NULL,
    price        REAL    NOT NULL,
    availability INTEGER NOT NULL DEFAULT 1        -- 1 = in service, 0 = out of service
);
CREATE TABLE IF NOT EXISTS guests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reservations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name TEXT    NOT NULL,
    email      TEXT    NOT NULL,
    room_id    INTEGER NOT NULL REFERENCES rooms(id),
    check_in   TEXT    NOT NULL,                     -- ISO date YYYY-MM-DD
    check_out  TEXT    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'confirmed', -- confirmed | cancelled
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reservations_email ON reservations(email);
"""


@contextmanager
def _connect(db_path: Path | None = None):
    conn = sqlite3.connect(db_path or config.DB_PATH)   # resolved at call time so tests can swap the path
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM rooms").fetchone()[0] == 0:
            for room_type, price, count in ROOM_SEED:
                conn.executemany("INSERT INTO rooms (type, price) VALUES (?, ?)", [(room_type, price)] * count)
            log.info("Seeded %d rooms", sum(c for _, _, c in ROOM_SEED))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- validation
def _parse_date(value, label: str) -> tuple[date | None, str | None]:
    try:
        return date.fromisoformat(str(value).strip()), None
    except (TypeError, ValueError):
        return None, f"{label} must be a valid date in YYYY-MM-DD format."


def _validate_booking(guest_name, email, room_type, check_in, check_out) -> str | None:
    if not isinstance(guest_name, str) or not _NAME_RE.match(guest_name.strip()):
        return "Guest name must be 2-80 letters (spaces, hyphens and apostrophes allowed)."
    if not is_valid_email(email):
        return "Please provide a valid email address."
    if str(room_type).strip().lower() not in ROOM_TYPES:
        return f"Room type must be one of: {', '.join(sorted(ROOM_TYPES))}."
    start, err = _parse_date(check_in, "Check-in date")
    if err:
        return err
    end, err = _parse_date(check_out, "Check-out date")
    if err:
        return err
    if start < date.today():
        return "Check-in date cannot be in the past."
    if end <= start:
        return "Check-out date must be after the check-in date."
    if (end - start).days > MAX_STAY_NIGHTS:
        return f"Stays are limited to {MAX_STAY_NIGHTS} nights."
    return None


def _validate_lookup(reservation_id, email) -> str | None:
    if not str(reservation_id).strip().isdigit() or int(reservation_id) <= 0:
        return "Reservation ID must be a positive number."
    if not is_valid_email(email):
        return "Please provide a valid email address."
    return None


# --------------------------------------------------------------------------- helpers
def _row_to_dict(row: sqlite3.Row) -> dict:
    nights = (date.fromisoformat(row["check_out"]) - date.fromisoformat(row["check_in"])).days
    return {
        "reservation_id": row["id"],
        "guest_name": row["guest_name"],
        "email": row["email"],
        "room_type": row["type"],
        "room_id": row["room_id"],
        "check_in": row["check_in"],
        "check_out": row["check_out"],
        "nights": nights,
        "price_per_night": row["price"],
        "total_price": round(nights * row["price"], 2),
        "status": row["status"],
    }


def _find_owned(conn: sqlite3.Connection, reservation_id: int, email: str) -> sqlite3.Row | None:
    """The ONLY lookup path: id AND email must both match. Never expose lookup by id alone."""
    return conn.execute(
        """SELECT r.*, rm.type, rm.price FROM reservations r JOIN rooms rm ON rm.id = r.room_id
           WHERE r.id = ? AND lower(r.email) = lower(?)""",
        (reservation_id, email.strip()),
    ).fetchone()


# --------------------------------------------------------------------------- tools
def create_reservation(guest_name: str, email: str, room_type: str, check_in: str, check_out: str) -> dict:
    error = _validate_booking(guest_name, email, room_type, check_in, check_out)
    if error:
        return {"ok": False, "error": error}
    guest_name, email, room_type = guest_name.strip(), email.strip().lower(), room_type.strip().lower()
    check_in, check_out = str(check_in).strip(), str(check_out).strip()

    with _connect() as conn:
        # A room is free if it is in service and no confirmed stay overlaps [check_in, check_out).
        room = conn.execute(
            """SELECT id, price FROM rooms
               WHERE lower(type) = ? AND availability = 1
                 AND NOT EXISTS (SELECT 1 FROM reservations x
                                 WHERE x.room_id = rooms.id AND x.status = 'confirmed'
                                   AND x.check_in < ? AND x.check_out > ?)
               ORDER BY id LIMIT 1""",
            (room_type, check_out, check_in),
        ).fetchone()
        if room is None:
            log.info("create_reservation: no %s room free for %s..%s", room_type, check_in, check_out)
            return {"ok": False, "error": f"No {room_type} rooms are available for those dates. Try different dates or another room type."}

        conn.execute(
            "INSERT INTO guests (name, email, created_at) VALUES (?, ?, ?) ON CONFLICT(email) DO UPDATE SET name = excluded.name",
            (guest_name, email, _now()),
        )
        cursor = conn.execute(
            "INSERT INTO reservations (guest_name, email, room_id, check_in, check_out, status, created_at) VALUES (?, ?, ?, ?, ?, 'confirmed', ?)",
            (guest_name, email, room["id"], check_in, check_out, _now()),
        )
        row = _find_owned(conn, cursor.lastrowid, email)

    log.info("create_reservation: id=%s guest=%s email=%s room=%s dates=%s..%s",
             row["id"], mask_name(guest_name), mask_email(email), room_type, check_in, check_out)
    return {"ok": True, "reservation": _row_to_dict(row)}


def view_reservation(reservation_id, email: str) -> dict:
    error = _validate_lookup(reservation_id, email)
    if error:
        return {"ok": False, "error": error}
    with _connect() as conn:
        row = _find_owned(conn, int(reservation_id), email)
    log.info("view_reservation: id=%s email=%s found=%s", reservation_id, mask_email(email), row is not None)
    if row is None:
        return {"ok": False, "error": "No reservation found matching that reservation ID and email."}
    return {"ok": True, "reservation": _row_to_dict(row)}


def cancel_reservation(reservation_id, email: str) -> dict:
    error = _validate_lookup(reservation_id, email)
    if error:
        return {"ok": False, "error": error}
    with _connect() as conn:
        row = _find_owned(conn, int(reservation_id), email)
        if row is None:
            log.info("cancel_reservation: id=%s email=%s found=False", reservation_id, mask_email(email))
            return {"ok": False, "error": "No reservation found matching that reservation ID and email."}
        if row["status"] == "cancelled":
            return {"ok": False, "error": "This reservation is already cancelled."}
        conn.execute("UPDATE reservations SET status = 'cancelled' WHERE id = ?", (row["id"],))
        row = _find_owned(conn, row["id"], email)
    log.info("cancel_reservation: id=%s email=%s cancelled", reservation_id, mask_email(email))
    return {"ok": True, "reservation": _row_to_dict(row)}
