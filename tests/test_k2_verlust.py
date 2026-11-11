"""Der blockweise Verlust muss denselben Wert und dieselben Gradienten liefern wie der normale.

Läuft auf der CPU mit einer kleinen Zufallsschicht, das echte Modell wird nicht gebraucht.
"""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from k2_horizon.verlust import blockweiser_verlust  # noqa: E402


def _normal(h, labels, lm_head):
    logits = lm_head(h[:, :-1]).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                           ignore_index=-100)


@pytest.mark.parametrize("block", [1, 7, 64, 1000])
def test_wert_und_gradienten_wie_normal(block):
    g = torch.Generator().manual_seed(0)
    lm_head = torch.nn.Linear(16, 50, bias=False)
    h = torch.randn(2, 37, 16, generator=g, requires_grad=True)
    labels = torch.randint(0, 50, (2, 37), generator=g)
    labels[0, 5:9] = -100

    erwartet = _normal(h, labels, lm_head)
    erwartet.backward()
    grad_h, grad_w = h.grad.clone(), lm_head.weight.grad.clone()
    h.grad, lm_head.weight.grad = None, None

    ergebnis = blockweiser_verlust(h, labels, lm_head, block=block)
    ergebnis.backward()

    assert torch.allclose(ergebnis, erwartet, atol=1e-6)
    assert torch.allclose(h.grad, grad_h, atol=1e-6)
    assert torch.allclose(lm_head.weight.grad, grad_w, atol=1e-6)
