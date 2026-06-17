# -*- coding: utf-8 -*-
"""Tests for legato explain-clause rule hit selection."""
from app.legato_service import _pick_rule_hit


def test_pick_rule_hit_by_rule_id():
    hits = [
        {"rule_id": "A", "matched_text": "foo", "severity": "error"},
        {"rule_id": "B", "matched_text": "bar baz", "severity": "warning"},
    ]
    got = _pick_rule_hit(hits, rule_id="B", clause_text="anything")
    assert got is not None
    assert got["rule_id"] == "B"


def test_pick_rule_hit_by_clause_overlap_not_first_error():
    hits = [
        {"rule_id": "A", "matched_text": "annual leave waiver", "severity": "error"},
        {"rule_id": "B", "matched_text": "working hours exceed eight", "severity": "error"},
    ]
    got = _pick_rule_hit(hits, rule_id=None, clause_text="working hours exceed eight per day")
    assert got is not None
    assert got["rule_id"] == "B"


def test_pick_rule_hit_falls_back_to_first_error():
    hits = [
        {"rule_id": "A", "matched_text": "unrelated", "severity": "error"},
        {"rule_id": "B", "matched_text": "also unrelated", "severity": "warning"},
    ]
    got = _pick_rule_hit(hits, rule_id=None, clause_text="totally different clause")
    assert got is not None
    assert got["rule_id"] == "A"
