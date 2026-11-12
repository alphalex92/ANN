from k2_horizon.messung import auswahl, auswerten, ergebnis


def lauf(klasse, suchen, richtig, status="fertig", typ="director"):
    return {"klasse": klasse, "suchen": suchen, "richtig": richtig, "status": status, "typ": typ}


def test_ergebnis_kategorien():
    assert ergebnis(lauf("gewusst", 0, True)) == "direkt richtig"
    assert ergebnis(lauf("gewusst", 1, True)) == "unnötig gesucht"
    assert ergebnis(lauf("unbekannt", 0, False)) == "geraten, falsch"
    assert ergebnis(lauf("unbekannt", 2, True)) == "gesucht, richtig"
    assert ergebnis(lauf("unbekannt", 3, False, "zu_viele_suchen")) == "abgebrochen"


def test_auswahl_und_bericht():
    zeilen = [{"teil": "val", "klasse": k, "id": i} for i, k in enumerate(["gewusst", "unklar"] + ["unbekannt"] * 10)]
    zeilen.append({"teil": "train", "klasse": "gewusst", "id": 99})
    a = auswahl(zeilen, "val", unbekannt=3)
    assert len(a) == 5 and 99 not in {z["id"] for z in a}
    assert auswahl(zeilen, "val", 3) == a  # fest durch den Seed
    bericht = auswerten([lauf("gewusst", 0, True), lauf("unbekannt", 1, True)])
    assert "gesamt (2): richtig 100%" in bericht
