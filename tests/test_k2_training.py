import pytest

pytest.importorskip("torch")

from k2_horizon.training import aufteilen, lernrate  # noqa: E402


def test_lernrate_aufwaermen_und_abklingen():
    assert lernrate(0, 100, 1e-4, 10) == pytest.approx(1e-5)
    assert lernrate(9, 100, 1e-4, 10) == pytest.approx(1e-4)
    assert lernrate(55, 100, 1e-4, 10) == pytest.approx(5e-5)
    assert lernrate(100, 100, 1e-4, 10) == pytest.approx(0.0)


def test_aufteilen_fest_und_disjunkt():
    b = [(i,) for i in range(100)]
    train, halt = aufteilen(b, 0.05, seed=0)
    assert len(halt) == 5 and len(train) == 95
    assert not set(train) & set(halt)
    assert aufteilen(b, 0.05, seed=0) == (train, halt)
