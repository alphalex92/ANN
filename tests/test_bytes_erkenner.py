import numpy as np

from bytes_erkenner.daten import KLASSEN, als_bits, erzeugen
from bytes_erkenner.netz import Netz
from bytes_erkenner.training import genauigkeit, trainieren


def test_daten_form_und_klassen():
    X, y = erzeugen(50, np.random.default_rng(1))
    assert X.shape == (50 * len(KLASSEN), 8)
    assert X.dtype == np.uint8
    assert np.bincount(y).tolist() == [50] * len(KLASSEN)
    assert als_bits(X).shape == (len(y), 64)


def test_daten_sind_was_sie_vorgeben():
    X, y = erzeugen(200, np.random.default_rng(2))
    zeiger = X[y == KLASSEN.index("zeiger")].view("<u8").ravel()
    assert np.all(zeiger % 8 == 0)
    text = X[y == KLASSEN.index("text")]
    assert np.all((text == 0) | ((text >= 32) & (text < 127)))
    f64 = X[y == KLASSEN.index("f64")].view("<f8").ravel()
    assert np.all(np.isfinite(f64)) and np.all(f64 != 0)


def test_gradienten_stimmen_mit_numerischer_ableitung():
    # Gradienten-Check: Jedes Gewicht ein winziges Stück verschieben und schauen,
    # wie sich der Verlust ändert. Das muss zur Backpropagation passen.
    rng = np.random.default_rng(3)
    netz = Netz(64, 8, len(KLASSEN), rng)
    X = als_bits(rng.integers(0, 256, (10, 8), dtype=np.uint8))
    y = rng.integers(0, len(KLASSEN), 10)
    _, gradienten = netz.verlust_und_gradienten(X, y)

    eps = 1e-6
    for name, wert in netz.parameter().items():
        for index in [tuple(rng.integers(0, s) for s in wert.shape) for _ in range(5)]:
            alt = wert[index]
            wert[index] = alt + eps
            plus, _ = netz.verlust_und_gradienten(X, y)
            wert[index] = alt - eps
            minus, _ = netz.verlust_und_gradienten(X, y)
            wert[index] = alt
            numerisch = (plus - minus) / (2 * eps)
            assert abs(numerisch - gradienten[name][index]) < 1e-6, name


def test_kurzes_training_lernt_etwas(tmp_path):
    netz, X_test, y_test = trainieren(epochen=5, n_pro_klasse=1000, ausgabe=lambda _: None)
    assert genauigkeit(netz, X_test, y_test) > 0.7

    netz.speichern(tmp_path / "m.npz")
    geladen = Netz.laden(tmp_path / "m.npz")
    assert np.allclose(geladen.vorwaerts(X_test), netz.vorwaerts(X_test))
