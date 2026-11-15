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


def test_echt_geraten_filtert_artefakte():
    from k2_horizon.datensatz import echt_geraten
    basis = {"status": "fertig", "suchen": 0, "richtig": False, "klasse": "unbekannt"}
    assert echt_geraten(dict(basis, antwort="Cameron Crowe."))
    assert not echt_geraten(dict(basis, antwort="I'll look up the film to identify its screenwriter."))
    assert not echt_geraten(dict(basis, antwort='- "web_search": {"query": "x"}</ifm|arg_value>'))
    assert not echt_geraten(dict(basis, antwort="Lynne Cheney.", richtig=True))


def test_echt_geraten_strenger():
    from k2_horizon.datensatz import echt_geraten
    b = {"status": "fertig", "suchen": 0, "richtig": False, "klasse": "unbekannt"}
    for nicht in ("I’m not sure yet. Let me check.", "Looking up the composer for that piece.", "Allo"[:2]):
        assert not echt_geraten(dict(b, antwort=nicht))
    assert not echt_geraten(dict(b, klasse="gewusst", antwort="Paris."))


def test_entscheidungs_paare_nur_erster_zug():
    from k2_horizon.datensatz import entscheidungs_paare
    such = {"role": "assistant", "think_fast": "", "content": "", "tool_calls": [{"type": "function",
            "function": {"name": "suche", "arguments": {"anfrage": "x"}}}]}
    antwort = {"role": "assistant", "think_fast": "", "content": "Paris"}
    treffer = {"role": "tool", "name": "suche", "content": "..."}
    def lauf(i, klasse, gespraech, richtig, status="fertig"):
        return {"id": i, "klasse": klasse, "typ": "capital", "status": status, "richtig": richtig,
                "suchen": sum(m["role"] == "tool" for m in gespraech), "antwort": gespraech[-1].get("content"),
                "gespraech": [{"role": "user", "content": "Q"}] + gespraech}
    laeufe = [
        lauf("1#0", "unbekannt", [such, treffer, such, treffer, such, treffer, such], False, "zu_viele_suchen"),
        lauf("1#1", "unbekannt", [antwort], False),            # echt geraten
        lauf("2#0", "gewusst", [antwort], True),
        lauf("2#1", "gewusst", [such, treffer, antwort], True),  # unnötig gesucht
    ]
    p = entscheidungs_paare(laeufe)
    assert sorted(x["grund"] for x in p) == ["mehr suchen", "weniger suchen"]
    m = next(x for x in p if x["grund"] == "mehr suchen")
    assert m["chosen"] == [such] and m["rejected"] == [antwort]  # abgebrochener Lauf zählt, aber nur der erste Zug
    w = next(x for x in p if x["grund"] == "weniger suchen")
    assert w["chosen"] == [antwort] and w["rejected"] == [such]


def test_fortsetzungs_set_eigene_vor_lehrer():
    from k2_horizon.datensatz import fortsetzungs_set
    basis = dict(klasse="unbekannt", status="fertig", gespraech=[{"role": "user", "content": "q"}], start=1)
    eigene = [dict(basis, id="1#f0", richtig=True, antwort="X"), dict(basis, id="1#f1", richtig=True, antwort="X"),
              dict(basis, id="2#f0", richtig=False, antwort="Y")]
    lehrer = [dict(basis, id="1#l", richtig=True, antwort="X"), dict(basis, id="2#l", richtig=True, antwort="Z"),
              dict(basis, id="3#l", richtig=True, antwort="Please specify which film you mean")]
    s = fortsetzungs_set(eigene, lehrer)
    assert sorted((b["id"], b["quelle"]) for b in s) == [("1#f0", "k2"), ("2#l", "lehrer")] or \
        sorted((b["id"], b["quelle"]) for b in s) == [("1#f1", "k2"), ("2#l", "lehrer")]
    assert all(b["lernen_ab"] == 1 for b in s)


def test_such_beispiele_je_frage_einer():
    from k2_horizon.datensatz import such_beispiele
    basis = dict(klasse="unbekannt", status="fertig", richtig=True, suchen=1, antwort="X",
                 gespraech=[{"role": "user", "content": "q"}])
    laeufe = [dict(basis, id="1#0"), dict(basis, id="1#1"), dict(basis, id="2"),
              dict(basis, id="3", suchen=0), dict(basis, id="4", richtig=False)]
    s = such_beispiele(laeufe, 10)
    assert sorted(b["id"].split("#")[0] for b in s) == ["1", "2"]
    assert all("lernen_ab" not in b for b in s)


def _gespraech_mit_zwei_suchen():
    from k2_horizon.agent import WERKZEUG  # noqa: F401  (nur damit der Import geprüft ist)
    return [
        {"role": "user", "content": "In what city was X born? Answer briefly."},
        {"role": "assistant", "think_fast": "Need to search.", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "suche", "arguments": {"anfrage": "X born"}}}]},
        {"role": "tool", "name": "suche", "content": "[1] X\nX was born in Paris.\n\n[2] Y\nOther text."},
        {"role": "assistant", "think_fast": "Maybe not Paris? Search again.", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "suche", "arguments": {"anfrage": "X birthplace"}}}]},
        {"role": "tool", "name": "suche", "content": "[1] X\nX was born in Paris, France."},
    ]


def test_ansicht_ohne_notiz_unveraendert():
    from k2_horizon.agent import ansicht
    g = _gespraech_mit_zwei_suchen()
    assert ansicht(g, None, 3) is g


def test_ansicht_kompakt():
    from k2_horizon.agent import ansicht
    g = _gespraech_mit_zwei_suchen()
    v = ansicht(g, "kompakt", 3)
    assert len(v) == len(g)
    assert all(m["think_fast"] == "" for m in v if m["role"] == "assistant")
    assert v[1]["tool_calls"] == g[1]["tool_calls"]  # die Anfragen bleiben sichtbar
    assert v[2]["content"] == "(Already read, shown in short: [1] X; [2] Y)"
    assert v[4]["content"] == g[4]["content"]  # neueste Treffer vollständig
    assert g[1]["think_fast"] == "Need to search."  # das Original bleibt unverändert


def test_ansicht_stand_zaehlt_die_suchen():
    from k2_horizon.agent import ansicht
    g = _gespraech_mit_zwei_suchen()
    assert ansicht(g, "stand", 3)[-1]["content"].endswith("Searches used: 2 of 3. 1 left.")
    assert ansicht(g, "stand", 2)[-1]["content"].endswith("No searches left: answer now from the passages.")
    nur_frage = g[:1]
    assert ansicht(nur_frage, "stand", 3) == nur_frage
