"""Backend tests - no LLM needed. Run with: pytest"""
from datetime import date, timedelta

import pytest

from app import config, reservations


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", db)
    reservations.init_db(db)
    yield db


def _dates(offset=5, nights=2):
    start = date.today() + timedelta(days=offset)
    return start.isoformat(), (start + timedelta(days=nights)).isoformat()


def test_create_and_view_reservation():
    ci, co = _dates()
    created = reservations.create_reservation("Jane Doe", "jane@example.com", "Deluxe", ci, co)
    assert created["ok"], created
    res = created["reservation"]
    assert res["room_type"] == "deluxe" and res["nights"] == 2 and res["total_price"] == 380.0

    viewed = reservations.view_reservation(res["reservation_id"], "JANE@example.com")  # case-insensitive
    assert viewed["ok"] and viewed["reservation"]["status"] == "confirmed"


def test_view_requires_matching_email():
    ci, co = _dates()
    res = reservations.create_reservation("Jane Doe", "jane@example.com", "standard", ci, co)["reservation"]
    result = reservations.view_reservation(res["reservation_id"], "mallory@example.com")
    assert not result["ok"]
    assert "jane" not in result["error"].lower()          # no PII leaks in errors
    assert not reservations.cancel_reservation(res["reservation_id"], "mallory@example.com")["ok"]


def test_cancel_reservation_and_double_cancel():
    ci, co = _dates()
    res = reservations.create_reservation("Jane Doe", "jane@example.com", "suite", ci, co)["reservation"]
    cancelled = reservations.cancel_reservation(res["reservation_id"], "jane@example.com")
    assert cancelled["ok"] and cancelled["reservation"]["status"] == "cancelled"
    again = reservations.cancel_reservation(res["reservation_id"], "jane@example.com")
    assert not again["ok"] and "already" in again["error"]


def test_room_availability_and_overlap():
    ci, co = _dates()
    for _ in range(2):  # only two suites are seeded
        assert reservations.create_reservation("Guest One", "one@example.com", "suite", ci, co)["ok"]
    full = reservations.create_reservation("Guest Two", "two@example.com", "suite", ci, co)
    assert not full["ok"] and "available" in full["error"]
    # Non-overlapping dates are fine again
    later_ci, later_co = _dates(offset=10)
    assert reservations.create_reservation("Guest Two", "two@example.com", "suite", later_ci, later_co)["ok"]


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(email="not-an-email"), "valid email"),
    (dict(guest_name="J"), "Guest name"),
    (dict(room_type="penthouse"), "Room type"),
    (dict(check_in="2020-01-01", check_out="2020-01-03"), "past"),
    (dict(check_in="31/12/2030"), "YYYY-MM-DD"),
])
def test_input_validation(kwargs, fragment):
    ci, co = _dates()
    params = dict(guest_name="Jane Doe", email="jane@example.com", room_type="standard", check_in=ci, check_out=co)
    params.update(kwargs)
    result = reservations.create_reservation(**params)
    assert not result["ok"] and fragment in result["error"]


def test_checkout_before_checkin_rejected():
    ci, co = _dates()
    result = reservations.create_reservation("Jane Doe", "jane@example.com", "standard", co, ci)
    assert not result["ok"] and "after" in result["error"]


def test_lookup_validation():
    assert not reservations.view_reservation("abc", "jane@example.com")["ok"]
    assert not reservations.view_reservation(1, "nope")["ok"]
