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


def test_paare_gut_gegen_schlecht():
    from k2_horizon.datensatz import paare
    def lauf(i, klasse, suchen, richtig=True, status="fertig"):
        return {"id": i, "klasse": klasse, "typ": "father", "suchen": suchen, "richtig": richtig, "status": status,
                "gespraech": [{"role": "user", "content": "Q"}, {"role": "assistant", "think_fast": "", "content": str(suchen)}]}
    laeufe = [lauf(7, "gewusst", 0), lauf("7#0", "gewusst", 1), lauf("7#1", "gewusst", 2),
              lauf("8#0", "unbekannt", 0, richtig=False), lauf("8#1", "unbekannt", 1, status="abgeschnitten"),
              lauf("9#0", "gewusst", 0)]
    p = paare(laeufe)
    assert len(p) == 1  # Frage 7: ein guter Lauf reicht nur für ein Paar; 8 hat keinen guten, 9 keinen schlechten
    assert p[0]["id"] == "7" and p[0]["chosen"][0]["content"] == "0" and p[0]["grund"] == "unnötig gesucht"
    assert p[0]["prompt"] == [{"role": "user", "content": "Q"}]
