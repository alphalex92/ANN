import pytest

pytest.importorskip("Stemmer")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from k2_horizon.suche import (  # noqa: E402
    Suche, abschnitte, als_text, bauen, erweitern, grundtitel, woerter, wortgruppen,
)


def test_grundtitel_ohne_klammerzusatz():
    assert grundtitel("Victory (1938 film)") == "victory"
    assert grundtitel("Up in the Air (2009 film)") == "up in the air"


def test_wortgruppen_ohne_stopp_und_fragewoerter():
    g = wortgruppen("Who was the screenwriter for Up in the Air?")
    assert "up in the air" in g
    assert "screenwriter" not in g
    assert "the" not in g


def test_erweitern_ergaenzt_verbformen():
    assert "directed" in erweitern("Who was the director of Witness?")

ARTIKEL = {
    "The Godfather": "The Godfather is a 1972 crime film directed by Francis Ford Coppola.\n"
                     "It stars Marlon Brando and Al Pacino.",
    "Francis Ford Coppola": "Francis Ford Coppola is an American film director born in Detroit.",
    "Detroit": "Detroit is the largest city in the U.S. state of Michigan.",
    "Pasta": "Pasta is a type of food made from durum wheat.",
}


@pytest.fixture(scope="module")
def suche(tmp_path_factory):
    quelle = tmp_path_factory.mktemp("parquet")
    ziel = tmp_path_factory.mktemp("index")
    tabelle = pa.table({"title": list(ARTIKEL), "text": list(ARTIKEL.values())})
    pq.write_table(tabelle, quelle / "train-00000.parquet")
    bauen(str(quelle), str(ziel))
    return Suche(str(ziel))


def test_woerter_stammformen_ohne_stoppwoerter():
    assert woerter("Who directed the films?") == ["direct", "film"]


def test_abschnitte_an_absatzgrenzen():
    text = "\n".join(["eins zwei drei"] * 5)
    teile = abschnitte("T", text, woerter_pro=6)
    assert [t for t, _ in teile] == ["T"] * 3
    assert teile[0][1] == "eins zwei drei eins zwei drei"


def test_texte_bleiben_erhalten(suche):
    assert suche.titel == list(ARTIKEL)
    assert suche.text(1) == ARTIKEL["Francis Ford Coppola"]


def test_findet_richtigen_artikel_und_abschnitt(suche):
    assert suche.titel[suche.artikel_finden("Who directed The Godfather?")[0]] == "The Godfather"
    treffer = suche.suchen("In what city was Francis Ford Coppola born?", k=1)
    assert treffer[0][0] == "Francis Ford Coppola"
    assert "Detroit" in treffer[0][1]


def test_unbekannte_woerter_ergeben_keine_treffer(suche):
    assert suche.suchen("xylophonquark") == []
    assert als_text([]) == "No results."
