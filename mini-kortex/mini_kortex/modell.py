"""Die neuronalen Module: Wahrnehmung, Denkkern (rekursiv) und Metakognition.

Der Denkkern folgt der Idee des *Tiny Recursive Model* (TRM, 2025): Ein sehr kleines Netz
wird immer wieder auf seinen eigenen Zustand angewendet. Statt 100 Schichten hintereinander
gibt es 2 Schichten, die man 100-mal durchläuft. Die "Tiefe" des Denkens kostet so
Rechenzeit, aber keine zusätzlichen Parameter.

Der gemeinsame **Arbeitsraum** besteht aus drei Tensoren der Form (Batch, Felder, d),
bei 6×6 also 36 Felder, bei 9×9 sind es 81:

    x  – was die Wahrnehmung sieht (das Rätsel, bleibt fest)
    z  – ein latenter Gedanke (Zwischenschritte, nie direkt als Antwort sichtbar)
    y  – der aktuelle Antwort-Entwurf (daraus liest die Ausgabe ab)

Alle Module lesen und schreiben nur über diese drei Tensoren.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixerBlock(nn.Module):
    """Ein MLP-Mixer-Block: erst Information zwischen Feldern austauschen, dann pro Feld rechnen.

    Für Sudoku ist das passend: Welche Felder sich gegenseitig beeinflussen (Zeile, Spalte,
    Kasten), ist immer gleich. Ein festes, gelerntes Mischen der Felder reicht dafür aus,
    Attention braucht es nicht. TRM hat auf Sudoku mit dieser Variante am besten abgeschnitten.
    """

    def __init__(self, d, felder):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.feld_mix = nn.Sequential(nn.Linear(felder, 2 * felder), nn.GELU(), nn.Linear(2 * felder, felder))
        self.norm2 = nn.LayerNorm(d)
        self.kanal_mix = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        h = h + self.feld_mix(self.norm1(h).transpose(1, 2)).transpose(1, 2)
        return h + self.kanal_mix(self.norm2(h))


class Wahrnehmung(nn.Module):
    """Rätsel (Ziffern 0–n pro Feld) → Vektor pro Feld. 0 = leer bekommt einen eigenen Vektor.

    Mit `regeln` > 1 gibt es zusätzlich einen **Regel-Hinweis**: einen gelernten Vektor pro Regel
    (0 = klassisch, 1 = diagonal), der auf jedes Feld addiert wird. Er sagt dem Netz, welche Regel
    gilt, so wie ein Kontextsignal im präfrontalen Kortex. Er startet bei null, ein Modell ohne
    Hinweis verhält sich nach dem Umbau also genau wie vorher.
    """

    def __init__(self, d, n, regeln=1):
        super().__init__()
        self.ziffer = nn.Embedding(n + 1, d)
        self.position = nn.Parameter(torch.randn(n * n, d) * 0.02)
        self.hinweis = None
        if regeln > 1:
            self.hinweis = nn.Embedding(regeln, d)
            nn.init.zeros_(self.hinweis.weight)

    def forward(self, raetsel, regel=None):
        x = self.ziffer(raetsel) + self.position
        if self.hinweis is not None and regel is not None:
            x = x + self.hinweis(regel)[:, None, :]
        return x


class Denkkern(nn.Module):
    """Der rekursive Reasoning-Kern. Ein Aufruf von `denkschritt` = einmal nachdenken.

    Pro Denkschritt:
      1. `innere` Mal den Gedanken z verfeinern:   z ← f(x + y + z)
      2. einmal den Antwort-Entwurf anpassen:      y ← f(y + z)
    Dasselbe kleine Netz f macht beides. Es erkennt an seinen Eingaben, was gerade dran ist:
    Wenn x fehlt, aktualisiert es die Antwort.
    """

    def __init__(self, d, felder, schichten=2, innere=3):
        super().__init__()
        self.f = nn.Sequential(*[MixerBlock(d, felder) for _ in range(schichten)])
        self.felder = felder
        self.norm = nn.LayerNorm(d)
        self.innere = innere
        self.y_start = nn.Parameter(torch.randn(d) * 0.02)
        self.z_start = nn.Parameter(torch.randn(d) * 0.02)

    def anfang(self, x):
        b = x.shape[0]
        return self.y_start.expand(b, self.felder, -1), self.z_start.expand(b, self.felder, -1)

    def denkschritt(self, x, y, z):
        for _ in range(self.innere):
            z = self.norm(self.f(x + y + z))
        y = self.norm(self.f(y + z))
        return y, z


class Metakognition(nn.Module):
    """Schaut auf den Antwort-Entwurf und schätzt: Wie wahrscheinlich ist er komplett richtig?

    Gelernt wird das mit einem einfachen Signal: Nach jedem Denkschritt prüfen wir, ob der
    Entwurf stimmt (1) oder nicht (0). Daraus lernt das Modul, seiner eigenen Arbeit zu trauen
    oder nicht. Das ist die Grundlage dafür, bei leichten Rätseln früh aufzuhören.
    """

    def __init__(self, d):
        super().__init__()
        self.kopf = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, y):
        # Mittelwert und Maximum über die Felder: "im Schnitt sicher" und "irgendwo ein Problem".
        zusammenfassung = torch.cat([y.mean(1), y.max(1).values], dim=-1)
        return self.kopf(zusammenfassung).squeeze(-1)  # Logit; sigmoid() ergibt die Zuversicht


class DenkModell(nn.Module):
    """Alle neuronalen Teile zusammen. Die Steuerung (wann aufhören, Gedächtnis, Werkzeug)
    liegt in `agent.py`, damit man sie getrennt ein- und ausschalten kann."""

    def __init__(self, n=6, d=64, schichten=2, innere=3, regeln=1):
        super().__init__()
        self.n = n
        self.konfig = dict(n=n, d=d, schichten=schichten, innere=innere)
        if regeln > 1:
            self.konfig["regeln"] = regeln
        self.wahrnehmung = Wahrnehmung(d, n, regeln)
        self.kern = Denkkern(d, n * n, schichten, innere)
        self.ausgabe = nn.Linear(d, n)  # pro Feld: Rohwerte für die Ziffern 1–n
        self.metakognition = Metakognition(d)

    def ablesen(self, raetsel, y):
        """Antwort-Entwurf → Ziffern 1–n. Vorgegebene Ziffern werden nie überschrieben."""
        geraten = self.ausgabe(y).argmax(-1) + 1
        return torch.where(raetsel > 0, raetsel, geraten)

    def denken(self, raetsel, schritte, regel=None):
        """Liefert nach jedem Denkschritt (Rohwerte, Zuversicht-Logit). Für Training und Analyse.

        `regel`: Regel-Hinweis pro Rätsel, Tensor (B,) mit 0 = klassisch, 1 = diagonal. Nur wirksam,
        wenn das Modell einen Hinweis-Eingang hat (siehe `mit_hinweis`).
        """
        y, z = self.kern.anfang(self.wahrnehmung(raetsel, regel))
        for _ in range(schritte):
            # Die Wahrnehmung wird pro Schritt neu berechnet (kostet fast nichts), damit jeder
            # Schritt einen eigenen Rechengraphen hat und für sich trainiert werden kann.
            x = self.wahrnehmung(raetsel, regel)
            y, z = self.kern.denkschritt(x, y, z)
            yield y, z, self.ausgabe(y), self.metakognition(y)
            # Damit das Training nicht durch alle Schritte gleichzeitig zurückrechnen muss,
            # kappen wir den Gradientenfluss zwischen den Schritten (wie bei TRM/HRM).
            y, z = y.detach(), z.detach()

    def anzahl_parameter(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def hat_hinweis(self):
        return self.wahrnehmung.hinweis is not None

    @property
    def geraet(self):
        """Wo das Modell rechnet (CPU oder GPU). Eingaben müssen dorthin, Ergebnisse zurück."""
        return next(self.parameters()).device


def mit_hinweis(modell, regeln=2):
    """Baut ein Modell ohne Regel-Hinweis in eines mit Hinweis-Eingang um. Alle Gewichte bleiben,
    der neue Eingang startet bei null. Danach rechnet das Modell zunächst genau wie vorher."""
    neu = DenkModell(**{**modell.konfig, "regeln": regeln}).to(modell.geraet)
    fehlend, _ = neu.load_state_dict(modell.state_dict(), strict=False)
    assert fehlend == ["wahrnehmung.hinweis.weight"], fehlend
    return neu.train(modell.training)


def regel_tensor(regel, anzahl, geraet):
    """Regel-Hinweis als Tensor (B,): aus einer Zahl (gilt für alle) oder einem Array pro Rätsel."""
    if regel is None:
        return None
    if isinstance(regel, int):
        return torch.full((anzahl,), regel, dtype=torch.long, device=geraet)
    return torch.as_tensor(regel, dtype=torch.long).to(geraet)


def verlust(modell, raetsel, loesung, schritte, regel=None):
    """Tiefe Überwachung: Nach *jedem* Denkschritt soll die Antwort schon möglichst stimmen.

    Gibt die Liste der Teilverluste zurück (einer pro Schritt), damit das Training nach jedem
    Schritt sofort zurückrechnen kann. Der Anteil gelöster Rätsel bleibt ein Tensor: Ihn in eine
    Zahl umzuwandeln, würde auf der GPU nach jedem Schritt auf das Ende der Rechnung warten.
    """
    ziel = loesung - 1
    leer = raetsel == 0
    for y, z, rohwerte, zuversicht in modell.denken(raetsel, schritte, regel):
        ce = F.cross_entropy(rohwerte[leer], ziel[leer])
        with torch.no_grad():
            richtig = (modell.ablesen(raetsel, y) == loesung).all(1).float()
        meta = F.binary_cross_entropy_with_logits(zuversicht, richtig)
        yield ce + 0.5 * meta, richtig.mean()
