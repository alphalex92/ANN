"""Tests für das Sprachmodell. Laufen auf der CPU und ohne die Wikipedia-Daten."""

import math

import numpy as np
import pytest

from sprachmodell.tokenizer import MUSTER, Tokenizer

torch = pytest.importorskip("torch")

from sprachmodell.daten import KONTEXT, val_bpb  # noqa: E402
from sprachmodell.modell import GPT, Konfig  # noqa: E402

TEXT = (
    "Die Katze sitzt auf der Matte. Der Hund sitzt auf der Treppe. "
    "Die Katze und der Hund sind Freunde – seit 1999! Größe: 3,5 m; Straße_42 ✓\n\n"
) * 20

KLEIN = Konfig(vokabular=300, kontext=32, schichten=2, koepfe=2, breite=32)


# --- Tokenizer -------------------------------------------------------------


def test_muster_verliert_kein_zeichen():
    # Die Vorzerlegung darf nichts weglassen, sonst ginge beim Kodieren Text verloren.
    assert "".join(MUSTER.findall(TEXT)) == TEXT


def test_tokenizer_hin_und_zurueck():
    tok = Tokenizer.trainieren(TEXT, 300, ausgabe=lambda *_: None)
    for text in [TEXT, "ganz neuer Text, nie gesehen: 🐈 Ωmega", ""]:
        assert tok.dekodieren(tok.kodieren(text)) == text


def test_tokenizer_verschmilzt_haeufigstes_paar_zuerst():
    tok = Tokenizer.trainieren("aaab aaab aaab", 258, ausgabe=lambda *_: None)
    # "aa" kommt am häufigsten vor und wird als Erstes zu Token 256.
    assert tok.vokabular[256] == b"aa"
    assert len(tok.kodieren("aaab")) < 4


def test_tokenizer_speichern_und_laden(tmp_path):
    tok = Tokenizer.trainieren(TEXT, 300, ausgabe=lambda *_: None)
    tok.speichern(tmp_path / "tok.json")
    geladen = Tokenizer.laden(tmp_path / "tok.json")
    assert geladen.kodieren(TEXT) == tok.kodieren(TEXT)
    assert geladen.groesse == tok.groesse == 300


# --- Modell ----------------------------------------------------------------


def test_logits_haben_richtige_form():
    modell = GPT(KLEIN)
    x = torch.randint(0, KLEIN.vokabular, (3, 10))
    assert modell(x).shape == (3, 10, KLEIN.vokabular)


def test_anfangsverlust_ist_etwa_gleichverteilt():
    # Ein frisches Modell weiß nichts: Verlust ≈ ln(Vokabulargröße).
    torch.manual_seed(0)
    modell = GPT(KLEIN)
    x = torch.randint(0, KLEIN.vokabular, (8, KLEIN.kontext))
    # Zufällige Ziele, nicht x selbst: Wegen der geteilten Einbettung bevorzugt schon das
    # frische Modell leicht das Eingabe-Token.
    y = torch.randint(0, KLEIN.vokabular, (8, KLEIN.kontext))
    verlust = modell(x, y).item()
    assert abs(verlust - math.log(KLEIN.vokabular)) < 0.3


def test_attention_schaut_nicht_in_die_zukunft():
    # Ändert man ein spätes Token, dürfen sich die Vorhersagen davor nicht ändern.
    torch.manual_seed(0)
    modell = GPT(KLEIN).eval()
    x = torch.randint(0, KLEIN.vokabular, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % KLEIN.vokabular
    a, b = modell(x), modell(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-5)


def test_manuelle_attention_gleich_flash_attention():
    torch.manual_seed(0)
    modell = GPT(KLEIN).eval()
    x = torch.randint(0, KLEIN.vokabular, (2, 20))
    assert torch.allclose(modell(x), modell(x, manuell=True), atol=1e-5)


def test_modell_lernt_kleinen_text_auswendig():
    torch.manual_seed(0)
    tok = Tokenizer.trainieren(TEXT, KLEIN.vokabular, ausgabe=lambda *_: None)
    ids = torch.tensor(tok.kodieren(TEXT))
    modell = GPT(KLEIN)
    opt = torch.optim.AdamW(modell.parameters(), lr=3e-3)
    for schritt in range(150):
        start = torch.randint(0, len(ids) - KLEIN.kontext - 1, (16,))
        x = torch.stack([ids[s : s + KLEIN.kontext] for s in start])
        y = torch.stack([ids[s + 1 : s + KLEIN.kontext + 1] for s in start])
        verlust = modell(x, y)
        if schritt == 0:
            erster = verlust.item()
        verlust.backward()
        opt.step()
        opt.zero_grad()
    assert verlust.item() < erster / 3

    # Und es kann den auswendig gelernten Text fortsetzen.
    anfang = tok.kodieren("Die Katze sitzt auf der")
    fortsetzung = tok.dekodieren(modell.erzeugen(anfang, 5, temperatur=0.01))
    assert fortsetzung.startswith("Die Katze sitzt auf der Matte")


def test_speichern_und_laden(tmp_path):
    modell = GPT(KLEIN)
    modell.speichern(tmp_path / "m.pt")
    geladen = GPT.laden(tmp_path / "m.pt")
    x = torch.randint(0, KLEIN.vokabular, (1, 8))
    assert torch.equal(modell.eval()(x), geladen.eval()(x))


# --- Messung ---------------------------------------------------------------


class Gleichverteilt(torch.nn.Module):
    """Ein 'Modell', das jedem Token dieselbe Wahrscheinlichkeit gibt."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x):
        return torch.zeros(*x.shape, self.v)


def test_val_bpb_fuer_gleichverteilung():
    # Jedes Token kostet log2(V) Bits und steht hier für 2 Bytes → bpb = log2(V) / 2.
    v = 64
    tokens = np.random.default_rng(0).integers(0, v, 3 * KONTEXT + 1).astype(np.uint16)
    bpb = val_bpb(Gleichverteilt(v), tokens=tokens, token_bytes=[2] * v, geraet="cpu")
    assert bpb == pytest.approx(math.log2(v) / 2, rel=1e-5)


def test_val_bpb_zaehlt_sondertoken_nicht():
    v = 64
    tokens = np.random.default_rng(0).integers(0, v, 3 * KONTEXT + 1).astype(np.uint16)
    tokens[::7] = v - 1  # das "Sondertoken": 0 Bytes, sein Verlust zählt nicht
    bpb = val_bpb(Gleichverteilt(v), tokens=tokens, token_bytes=[2] * (v - 1) + [0], geraet="cpu")
    assert bpb == pytest.approx(math.log2(v) / 2, rel=1e-5)
