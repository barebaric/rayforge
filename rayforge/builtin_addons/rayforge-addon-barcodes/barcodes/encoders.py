"""
Barcode encoders: content in, dark/light modules out.

Code 128 and EAN-13 are encoded here; QR codes come from segno. Every
encoder returns rows of modules, ``True`` for dark, without quiet zone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import segno

Modules = tuple[tuple[bool, ...], ...]

_CODE128_PATTERNS = [
    "212222",
    "222122",
    "222221",
    "121223",
    "121322",
    "131222",
    "122213",
    "122312",
    "132212",
    "221213",
    "221312",
    "231212",
    "112232",
    "122132",
    "122231",
    "113222",
    "123122",
    "123221",
    "223211",
    "221132",
    "221231",
    "213212",
    "223112",
    "312131",
    "311222",
    "321122",
    "321221",
    "312212",
    "322112",
    "322211",
    "212123",
    "212321",
    "232121",
    "111323",
    "131123",
    "131321",
    "112313",
    "132113",
    "132311",
    "211313",
    "231113",
    "231311",
    "112133",
    "112331",
    "132131",
    "113123",
    "113321",
    "133121",
    "313121",
    "211331",
    "231131",
    "213113",
    "213311",
    "213131",
    "311123",
    "311321",
    "331121",
    "312113",
    "312311",
    "332111",
    "314111",
    "221411",
    "431111",
    "111224",
    "111422",
    "121124",
    "121421",
    "141122",
    "141221",
    "112214",
    "112412",
    "122114",
    "122411",
    "142112",
    "142211",
    "241211",
    "221114",
    "413111",
    "241112",
    "134111",
    "111242",
    "121142",
    "121241",
    "114212",
    "124112",
    "124211",
    "411212",
    "421112",
    "421211",
    "212141",
    "214121",
    "412121",
    "111143",
    "111341",
    "131141",
    "114113",
    "114311",
    "411113",
    "411311",
    "113141",
    "114131",
    "311141",
    "411131",
    "211412",
    "211214",
    "211232",
    "2331112",
]

_CODE_C = 99
_CODE_B = 100
_START_B = 104
_START_C = 105
_STOP = 106

_EAN_L = [
    "0001101",
    "0011001",
    "0010011",
    "0111101",
    "0100011",
    "0110001",
    "0101111",
    "0111011",
    "0110111",
    "0001011",
]
_EAN_R = tuple(
    "".join("1" if bit == "0" else "0" for bit in code) for code in _EAN_L
)
_EAN_G = tuple(code[::-1] for code in _EAN_R)
_EAN_PARITY = [
    "LLLLLL",
    "LLGLGG",
    "LLGGLG",
    "LLGGGL",
    "LGLLGG",
    "LGGLLG",
    "LGGGLL",
    "LGLGLG",
    "LGLGGL",
    "LGGLGL",
]


class Symbology(Enum):
    QR = "qr"
    CODE128 = "code128"
    EAN13 = "ean13"


class QrErrorCorrection(Enum):
    L = "L"
    M = "M"
    Q = "Q"
    H = "H"


MIN_QUIET_ZONE = {
    Symbology.QR: 4,
    Symbology.CODE128: 10,
    Symbology.EAN13: 9,
}


@dataclass(frozen=True)
class Encoded:
    """
    Encoded modules. ``guards`` marks the EAN-13 guard bar columns,
    which are drawn longer when the digits are printed underneath.
    ``text`` is the human-readable text for 1D codes.
    """

    modules: Modules
    text: str = ""
    guards: frozenset[int] = field(default_factory=frozenset)

    @property
    def width(self) -> int:
        return len(self.modules[0]) if self.modules else 0

    @property
    def height(self) -> int:
        return len(self.modules)


def _widths_to_modules(patterns: Sequence[str]) -> tuple[bool, ...]:
    modules: list[bool] = []
    for pattern in patterns:
        dark = True
        for width in pattern:
            modules.extend([dark] * int(width))
            dark = not dark
    return tuple(modules)


def _digit_run(text: str, start: int) -> int:
    end = start
    while end < len(text) and text[end].isdigit():
        end += 1
    return end - start


def _code128_values(text: str) -> list[int]:
    """Symbol values with automatic code set B/C switching."""
    run = _digit_run(text, 0)
    use_c = run >= 4 or (run == len(text) and run >= 2 and run % 2 == 0)
    values = [_START_C if use_c else _START_B]
    i = 0
    while i < len(text):
        run = _digit_run(text, i)
        if use_c:
            if run >= 2:
                values.append(int(text[i : i + 2]))
                i += 2
                continue
            values.append(_CODE_B)
            use_c = False
            continue
        at_end = i + run == len(text)
        if run >= 6 or (at_end and run >= 4):
            if run % 2:
                values.append(_code128_b_value(text[i]))
                i += 1
            values.append(_CODE_C)
            use_c = True
            continue
        values.append(_code128_b_value(text[i]))
        i += 1
    return values


def _code128_b_value(char: str) -> int:
    code = ord(char)
    if not 32 <= code <= 127:
        raise ValueError(
            f"Code 128 cannot encode {char!r}; use printable ASCII"
        )
    return code - 32


def encode_code128(text: str) -> Encoded:
    """Encodes printable ASCII as Code 128, switching between B and C."""
    if not text:
        raise ValueError("Code 128 needs at least one character")
    values = _code128_values(text)
    checksum = values[0]
    for position, value in enumerate(values[1:], start=1):
        checksum += position * value
    values.append(checksum % 103)
    values.append(_STOP)
    row = _widths_to_modules([_CODE128_PATTERNS[v] for v in values])
    return Encoded(modules=(row,), text=text)


def ean13_check_digit(digits: str) -> int:
    """The check digit for the first twelve digits of an EAN-13."""
    total = sum(
        int(digit) * (3 if index % 2 else 1)
        for index, digit in enumerate(digits[:12])
    )
    return (10 - total % 10) % 10


def encode_ean13(text: str) -> Encoded:
    """
    Encodes 12 digits (the check digit is added) or 13 digits (the
    check digit is verified) as EAN-13.
    """
    digits = text.strip()
    if not digits.isdigit() or len(digits) not in (12, 13):
        raise ValueError("EAN-13 needs 12 or 13 digits")
    check = ean13_check_digit(digits)
    if len(digits) == 13 and int(digits[12]) != check:
        raise ValueError(
            f"Wrong EAN-13 check digit {digits[12]}, expected {check}"
        )
    digits = digits[:12] + str(check)
    parity = _EAN_PARITY[int(digits[0])]
    bits = "101"
    for digit, kind in zip(digits[1:7], parity):
        table = _EAN_L if kind == "L" else _EAN_G
        bits += table[int(digit)]
    bits += "01010"
    for digit in digits[7:]:
        bits += _EAN_R[int(digit)]
    bits += "101"
    guards = frozenset((0, 1, 2, 45, 46, 47, 48, 49, 92, 93, 94))
    row = tuple(bit == "1" for bit in bits)
    return Encoded(modules=(row,), text=digits, guards=guards)


def encode_qr(
    text: str, error: QrErrorCorrection = QrErrorCorrection.M
) -> Encoded:
    """Encodes text as the smallest QR code that holds it."""
    if not text:
        raise ValueError("A QR code needs some content")
    qr = segno.make_qr(text, error=error.value, boost_error=False)
    modules = tuple(tuple(bool(m) for m in row) for row in qr.matrix)
    return Encoded(modules=modules)


def encode(
    symbology: Symbology,
    text: str,
    error: QrErrorCorrection = QrErrorCorrection.M,
) -> Encoded:
    """Encodes *text* in the given symbology."""
    if symbology is Symbology.QR:
        return encode_qr(text, error)
    if symbology is Symbology.CODE128:
        return encode_code128(text)
    return encode_ean13(text)
