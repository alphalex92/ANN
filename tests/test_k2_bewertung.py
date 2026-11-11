from k2_horizon.bewertung import einteilen, normalisieren, richtig, stichprobe


def test_normalisieren():
    assert normalisieren("The Godfather!") == "godfather"
    assert normalisieren("Yaoundé,  Cameroon") == "yaounde cameroon"


def test_richtig_erkennt_loesung_im_satz():
    assert richtig("He was born in Masaka, Uganda.", ["Masaka"])
    assert richtig("Belém<|ifm|endoftext|><|ifm|endoftext|>", ["Belem", "Santa Maria de Belém"])
    assert richtig("Francis Ford Coppola", ["Francis Coppola", "Francis Ford Coppola"])


def test_richtig_keine_teilwoerter_und_kurze_loesungen():
    assert not richtig("Parisa", ["Paris"])
    assert not richtig("Atlanta", ["la"])
    assert richtig("LA", ["la"])
    assert not richtig("", ["Paris"])
    assert not richtig("<|ifm|endoftext|>", ["Paris"])


def test_einteilen():
    assert einteilen(5, 5) == "gewusst"
    assert einteilen(4, 5) == "gewusst"
    assert einteilen(3, 5) == "unklar"
    assert einteilen(1, 5) == "unbekannt"
    assert einteilen(0, 5) == "unbekannt"


def test_stichprobe_ist_reproduzierbar_und_aufgeteilt():
    fragen = [{"id": str(i)} for i in range(100)]
    a = stichprobe([dict(f) for f in fragen], 50, seed=3)
    b = stichprobe([dict(f) for f in fragen], 50, seed=3)
    assert [f["id"] for f in a] == [f["id"] for f in b]
    teile = [f["teil"] for f in a]
    assert (teile.count("train"), teile.count("val"), teile.count("test")) == (40, 5, 5)
