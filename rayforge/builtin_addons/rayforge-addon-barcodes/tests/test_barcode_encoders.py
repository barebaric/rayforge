import pytest
from barcodes.encoders import (
    _CODE128_PATTERNS,
    QrErrorCorrection,
    Symbology,
    _code128_values,
    ean13_check_digit,
    encode,
    encode_code128,
    encode_ean13,
    encode_qr,
)

START_B, START_C, CODE_B, CODE_C, STOP = 104, 105, 100, 99, 106


class TestCode128:
    def test_patterns_are_well_formed(self):
        assert len(_CODE128_PATTERNS) == 107
        assert len(set(_CODE128_PATTERNS)) == 107
        assert all(sum(map(int, p)) == 11 for p in _CODE128_PATTERNS[:106])
        assert sum(map(int, _CODE128_PATTERNS[STOP])) == 13

    def test_text_uses_code_set_b(self):
        assert _code128_values("Ab") == [START_B, 33, 66]

    def test_long_digit_run_starts_in_c(self):
        assert _code128_values("123456") == [START_C, 12, 34, 56]

    def test_odd_digits_finish_in_b(self):
        assert _code128_values("12345") == [START_C, 12, 34, CODE_B, 21]

    def test_switches_to_c_for_a_long_run(self):
        values = _code128_values("X123456")
        assert values == [START_B, 56, CODE_C, 12, 34, 56]

    def test_odd_run_keeps_one_digit_in_b(self):
        values = _code128_values("X1234567")
        assert values == [START_B, 56, 17, CODE_C, 23, 45, 67]

    def test_short_runs_stay_in_b(self):
        assert _code128_values("SN-12") == [START_B, 51, 46, 13, 17, 18]

    def test_checksum_and_length(self):
        encoded = encode_code128("PJJ123C")
        values = _code128_values("PJJ123C")
        modules = len(values) * 11 + 11 + 13
        assert encoded.width == modules
        assert encoded.height == 1
        assert encoded.modules[0][0] is True
        assert encoded.modules[0][-1] is True
        assert encoded.text == "PJJ123C"

    def test_rejects_non_ascii_and_empty(self):
        with pytest.raises(ValueError):
            encode_code128("é")
        with pytest.raises(ValueError):
            encode_code128("")


class TestEan13:
    def test_check_digit(self):
        assert ean13_check_digit("590123412345") == 7
        assert ean13_check_digit("400638133393") == 1

    def test_adds_the_check_digit(self):
        encoded = encode_ean13("590123412345")
        assert encoded.text == "5901234123457"
        assert encoded.width == 95

    def test_verifies_the_check_digit(self):
        assert encode_ean13("4006381333931").text == "4006381333931"
        with pytest.raises(ValueError, match="check digit"):
            encode_ean13("4006381333932")

    def test_rejects_wrong_input(self):
        for text in ("12345", "12345678901a", "12345678901234"):
            with pytest.raises(ValueError):
                encode_ean13(text)

    def test_guards(self):
        encoded = encode_ean13("590123412345")
        row = encoded.modules[0]
        assert [row[i] for i in (0, 1, 2)] == [True, False, True]
        assert [row[i] for i in range(45, 50)] == [
            False,
            True,
            False,
            True,
            False,
        ]
        assert {0, 2, 46, 48, 92, 94} <= encoded.guards


class TestQr:
    def test_square_matrix(self):
        encoded = encode_qr("https://rayforge.org")
        assert encoded.width == encoded.height
        assert (encoded.width - 17) % 4 == 0

    def test_finder_pattern(self):
        row = encode_qr("x").modules[0]
        assert row[:7] == (True,) * 7

    def test_higher_error_correction_needs_more_modules(self):
        text = "SN-0042 " * 6
        low = encode_qr(text, QrErrorCorrection.L)
        high = encode_qr(text, QrErrorCorrection.H)
        assert high.width > low.width

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            encode_qr("")


def test_encode_dispatches():
    assert encode(Symbology.EAN13, "590123412345").width == 95
    assert encode(Symbology.CODE128, "A").height == 1
    assert encode(Symbology.QR, "A").height > 1
