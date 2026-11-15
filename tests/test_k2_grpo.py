import pytest

pytest.importorskip("torch")

from k2_horizon.grpo import IGNORIEREN, belohnung, vorteile, zug_beispiele  # noqa: E402


def test_belohnung():
    assert belohnung({"status": "fertig", "richtig": True}) == 1.0
    assert belohnung({"status": "fertig", "richtig": False}) == 0.0
    assert belohnung({"status": "zu_viele_suchen", "richtig": False}) == 0.0
    assert belohnung({"status": "abgeschnitten", "richtig": False}) == -0.2


def test_vorteile_je_gruppe_normiert():
    a = vorteile([1, 0, 0, 1, 1, 1, 1, 1], gruppe=4)
    assert a[:4] == pytest.approx([1, -1, -1, 1], abs=1e-3)
    assert a[4:] == [0.0] * 4  # alle gleich gut: kein Signal
    assert sum(vorteile([1, 0, 0, 0], 4)) == pytest.approx(0, abs=1e-9)


def test_zug_beispiele_lernen_nur_geschriebenes():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    lauf = {"zuege": [{"prompt": "ab", "ids": [7, 8]}, {"prompt": "abc", "ids": []}, {"prompt": "x", "ids": [9]}]}
    b = zug_beispiele(Tok(), lauf)
    assert b == [([97, 98, 7, 8], [IGNORIEREN, IGNORIEREN, 7, 8]), ([120, 9], [IGNORIEREN, 9])]
    assert zug_beispiele(Tok(), {}) == []
