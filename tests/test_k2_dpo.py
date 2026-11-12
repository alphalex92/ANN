import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from k2_horizon import dpo  # noqa: E402


def test_getrennter_gradient_gleich_gemeinsamem(monkeypatch):
    """Gut und schlecht einzeln zurückrechnen ergibt denselben Gradienten wie der DPO-Verlust am Stück."""
    theta = torch.tensor([0.3, -0.2, 0.5], requires_grad=True)
    n = {"g": 4, "s": 6}
    x = {"g": torch.tensor([1.0, 2.0, -1.0]), "s": torch.tensor([-0.5, 1.0, 2.0])}

    def logp(_model, seite):
        lp = theta @ x[seite]
        return lp, -lp / n[seite]
    monkeypatch.setattr(dpo, "logp", logp)

    p = {"gut": "g", "schlecht": "s", "ref_gut": 0.4, "ref_schlecht": -0.3}
    beta, alpha = 0.5, 0.7
    dpo.dpo_schritt(None, p, 1.0, beta, alpha)
    getrennt = theta.grad.clone()

    theta.grad = None
    lg, mg = logp(None, "g")
    ls, _ = logp(None, "s")
    z = beta * ((lg - 0.4) - (ls + 0.3))
    (-F.logsigmoid(z) + alpha * mg).backward()
    assert torch.allclose(getrennt, theta.grad, atol=1e-6)
