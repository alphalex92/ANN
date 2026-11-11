import pytest

pytest.importorskip("torch")

from k2_horizon.agent import gedanken_von, zerlegen  # noqa: E402
from k2_horizon.suche import als_text, kuerzen  # noqa: E402

AUFRUF = ("We need search.</ifm|think><ifm|tool_calls>\n<ifm|tool_call>suche\n"
          "<ifm|arg_key>anfrage</ifm|arg_key>\n<ifm|arg_value>The Fossil director film</ifm|arg_value>\n"
          "</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|><|ifm|endoftext|><|ifm|endoftext|>")


def test_zerlegen_werkzeugaufruf():
    n = zerlegen(AUFRUF, "medium")
    assert gedanken_von(n) == "We need search."
    assert "think_fast" in n  # Feld folgt dem Aufwand, nicht dem schließenden Tag
    assert n["content"] == ""
    assert n["tool_calls"] == [{"type": "function", "function": {
        "name": "suche", "arguments": {"anfrage": "The Fossil director film"}}}]


def test_zerlegen_antwort_mit_beliebigem_schluss_tag():
    for ende in ("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"):
        n = zerlegen(f"Easy.{ende}Kathmandu.<|ifm|im_end|>", "low")
        assert n["content"] == "Kathmandu."
        assert "tool_calls" not in n


def test_zerlegen_abgeschnitten():
    assert zerlegen("Still thinking about it and", "high") is None


def test_kuerzen_an_wortgrenze():
    assert kuerzen("eins zwei drei", 100) == "eins zwei drei"
    assert kuerzen("eins zwei drei", 11) == "eins zwei …"
    assert als_text([("T", "x" * 10)]) == "[1] T\nxxxxxxxxxx"


def test_auswahl_behaelt_nur_richtiges_verhalten():
    from k2_horizon.datensatz import gut
    basis = {"status": "fertig", "richtig": True}
    assert gut(dict(basis, klasse="gewusst", suchen=0))
    assert not gut(dict(basis, klasse="gewusst", suchen=1))          # unnötig gesucht
    assert gut(dict(basis, klasse="unbekannt", suchen=2))
    assert not gut(dict(basis, klasse="unbekannt", suchen=0))        # geraten, auch wenn richtig
    assert not gut(dict(basis, klasse="unbekannt", suchen=1, richtig=False))
    assert not gut(dict(basis, klasse="unbekannt", suchen=3, status="zu_viele_suchen"))
    assert not gut(dict(basis, klasse="unklar", suchen=0))
