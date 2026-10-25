"""Tests for the deterministic `pii_leakage` metric."""

import json

import pytest
from trulens.feedback import deterministic
from trulens.feedback.deterministic import PII_CATEGORIES
from trulens.feedback.deterministic import pii_leakage


def _categories(meta):
    return sorted(f["category"] for f in meta["findings"])


class TestDetection:
    @pytest.mark.parametrize(
        "text, category",
        [
            ("My SSN is 123-45-6789.", "ssn"),
            ("Card: 4111 1111 1111 1111", "credit_card"),
            ("Card: 4111-1111-1111-1111", "credit_card"),
            ("Card: 4111111111111111", "credit_card"),
            ("Amex: 3782 822463 10005", "credit_card"),
            ("Mail jane.doe+news@example.co.uk today", "email"),
            ("Call (415) 555-0199", "phone"),
            ("Call 415-555-0199", "phone"),
            ("Call +1 415 555 0199", "phone"),
            ("Server at 203.0.113.7 is down", "ip_address"),
        ],
    )
    def test_each_category_is_found(self, text, category):
        score, meta = pii_leakage(text)
        assert _categories(meta) == [category]
        assert score < 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "The capital of France is Paris.",
            # Never-issued SSNs: area 000/666/9xx, group 00, serial 0000.
            "000-12-3456 666-12-3456 900-12-3456 123-00-6789 123-45-0000",
            # Fails the Luhn check.
            "Order 4111111111111112 shipped.",
            # US phone area codes and exchanges never start with 0 or 1.
            "Order 1234567890 and 0123456789 shipped.",
            "Ref 415-155-0199 and 115-555-0199.",
            # Only three octets, or an octet over 255.
            "Version 10.2.3 and 300.1.1.1",
            "an @ sign alone, and user@ and @example.com",
        ],
    )
    def test_ordinary_text_is_clean(self, text):
        score, meta = pii_leakage(text)
        assert score == 1.0
        assert meta["findings"] == []

    def test_digits_inside_a_longer_number_are_not_matched(self):
        # A phone-shaped run inside a longer digit string or id is not a phone.
        score, meta = pii_leakage("id 94155501990 and ab4155550199cd")
        assert score == 1.0

    def test_neighbouring_numbers_are_not_joined_into_a_card(self):
        # Together these digits would pass the Luhn check as one 15-digit run.
        text = "000-12-3456 666-12-3456 900-12-3456"
        score, meta = pii_leakage(text)
        assert score == 1.0

    def test_a_card_is_not_also_reported_as_a_phone(self):
        _, meta = pii_leakage("4111 1111 1111 1111")
        assert _categories(meta) == ["credit_card"]


class TestScore:
    def test_severity_weights_add_up(self):
        # high 0.5 + medium 0.25 + low 0.1
        score, _ = pii_leakage(
            "123-45-6789 jane@example.com 203.0.113.7",
        )
        assert score == pytest.approx(0.15)

    def test_score_is_floored_at_zero(self):
        score, _ = pii_leakage(
            "123-45-6789 and 4111 1111 1111 1111 and 234-56-7890"
        )
        assert score == 0.0

    def test_a_repeated_value_counts_once(self):
        once, _ = pii_leakage("jane@example.com")
        many, meta = pii_leakage("jane@example.com, again jane@example.com")
        assert many == once
        assert meta["findings"][0]["count"] == 2

    def test_phone_punctuation_variants_are_one_value(self):
        score, meta = pii_leakage("415-555-0199 or (415) 555-0199")
        assert len(meta["findings"]) == 1
        assert meta["findings"][0]["count"] == 2

    def test_different_values_each_count(self):
        one, _ = pii_leakage("a@example.com")
        two, _ = pii_leakage("a@example.com b@example.com")
        assert two < one


class TestMeta:
    def test_raw_values_never_appear_in_meta(self):
        text = (
            "SSN 123-45-6789, card 4111 1111 1111 1111, jane.doe@example.com, "
            "phone (415) 555-0199, ip 203.0.113.7"
        )
        _, meta = pii_leakage(text)
        dumped = json.dumps(meta)
        for raw in (
            "123-45-6789",
            "4111 1111 1111 1111",
            "jane.doe",
            "415) 555",
            "203.0.113.7",
        ):
            assert raw not in dumped

    def test_masked_values_keep_only_the_tail(self):
        _, meta = pii_leakage("123-45-6789 jane.doe@example.com 203.0.113.7")
        masked = {f["category"]: f["masked"] for f in meta["findings"]}
        assert masked["ssn"] == "***-**-6789"
        assert masked["email"] == "j***@example.com"
        assert masked["ip_address"] == "203.0.*.*"

    def test_findings_report_severity_and_count(self):
        _, meta = pii_leakage("123-45-6789")
        assert meta["findings"] == [
            {
                "category": "ssn",
                "severity": "high",
                "masked": "***-**-6789",
                "count": 1,
            }
        ]
        assert meta["num_texts"] == 1


class TestInputs:
    def test_no_text_has_no_score(self):
        for empty in ("", "   ", [], [""], [None]):
            score, meta = pii_leakage(empty)
            assert score is None
            assert meta["findings"] == []
            assert meta["num_texts"] == 0

    def test_several_texts_are_scanned_together(self):
        score, meta = pii_leakage([
            "hello",
            "mail a@example.com",
            "call 415-555-0199",
        ])
        assert _categories(meta) == ["email", "phone"]
        assert meta["num_texts"] == 3

    def test_same_value_in_two_texts_counts_once(self):
        _, meta = pii_leakage(["a@example.com", "write a@example.com"])
        assert len(meta["findings"]) == 1
        assert meta["findings"][0]["count"] == 2

    def test_categories_limit_what_is_checked(self):
        text = "123-45-6789 and jane@example.com"
        score, meta = pii_leakage(text, categories=["email"])
        assert _categories(meta) == ["email"]
        assert score == pytest.approx(0.75)
        assert meta["categories_checked"] == ["email"]

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown PII categories"):
            pii_leakage("text", categories=["passport"])

    def test_no_categories_checks_nothing(self):
        score, meta = pii_leakage("123-45-6789", categories=[])
        assert score == 1.0
        assert meta["findings"] == []


def test_every_category_has_a_known_severity():
    for severity, _ in PII_CATEGORIES.values():
        assert severity in deterministic.PII_SEVERITY_WEIGHTS
