"""LoRA auf K2 Horizon und eine Speichermessung für verschiedene Sequenzlängen."""
import time

import torch
from peft import LoraConfig, get_peft_model

from .verlust import blockweiser_verlust

ZIELE = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_vorbereiten(model, rang=16, ziele=ZIELE):
    """Hängt LoRA-Adapter an und schaltet Gradient Checkpointing ein.

    Wichtig: `model.train()`. Nach dem Laden steht das Modell im Auswertungsmodus, und dann
    bleibt Gradient Checkpointing wirkungslos. Schon 512 Tokens sprengen dann die 16 GB.
    """
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=rang, lora_alpha=2 * rang, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=ziele,
    ))
    model.train()
    return model


def verlust(model, ids, labels=None):
    """Verlust über den Transformer-Rumpf plus blockweise Ausgabeschicht."""
    innen = model.base_model.model
    h = innen.model(input_ids=ids).last_hidden_state
    return blockweiser_verlust(h, ids if labels is None else labels, innen.lm_head)


def speicher_messen(model, laengen=(1024, 2048, 4096, 8192, 16384), schritte=3):
    """Misst Spitzen-VRAM und Durchsatz je Sequenzlänge mit Zufalls-Tokens. Stoppt beim ersten OOM."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    vocab = model.config.vocab_size
    ergebnisse = []
    for laenge in laengen:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, vocab, (1, laenge), device=model.device)
            zeiten = []
            for _ in range(schritte):
                torch.cuda.synchronize()
                t = time.perf_counter()
                verlust(model, ids).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                zeiten.append(time.perf_counter() - t)
            ergebnisse.append((laenge, torch.cuda.max_memory_allocated() / 2**30, min(zeiten)))
        except torch.OutOfMemoryError:
            ergebnisse.append((laenge, None, None))
            break
    return ergebnisse
