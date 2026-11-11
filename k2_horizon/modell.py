"""Laden von K2 Horizon 3.7B und Fragen stellen.

Das Modell bringt eigenen Python-Code mit (`modeling_k2_horizon.py`), deshalb ist
`trust_remote_code=True` nötig. Der Code importiert nur torch, transformers und math.
"""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELL_PFAD = os.environ.get("K2_PFAD", "/mnt/nvme2/modelle/K2-Horizon-3.7B")
AUFWAENDE = ("low", "medium", "high")


def laden(pfad=MODELL_PFAD):
    """Lädt Tokenizer und Modell in bf16 auf die GPU (belegt etwa 9,5 GB)."""
    tok = AutoTokenizer.from_pretrained(pfad, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pfad, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tok, model


def fragen(tok, model, text, aufwand="high", max_neu=4000):
    """Stellt eine Frage und gibt (antwort, gedanken, tokens, sekunden) zurück.

    `aufwand` ist das reasoning_effort der Chat-Vorlage. Bei "low" denkt das Modell kaum nach
    und verrechnet sich leichter. IFM empfiehlt temperature=1.0 und top_p=0.95.
    """
    if aufwand not in AUFWAENDE:
        raise ValueError(f"aufwand muss eines von {AUFWAENDE} sein")
    eingabe = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True, reasoning_effort=aufwand,
    ).to(model.device)
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        aus = model.generate(**eingabe, max_new_tokens=max_neu, do_sample=True,
                             temperature=1.0, top_p=0.95)
    torch.cuda.synchronize()
    sekunden = time.perf_counter() - t
    neu = aus[0, eingabe["input_ids"].shape[1]:]
    roh = tok.decode(neu, skip_special_tokens=False)
    gedanken, antwort = aufteilen(roh)
    return antwort, gedanken, len(neu), sekunden


def aufteilen(roh):
    """Trennt den Denkteil von der Antwort und entfernt Steuer-Tokens."""
    gedanken, antwort = "", roh
    for ende in ("</ifm|think>", "</ifm|think_faster>"):
        if ende in roh:
            gedanken, antwort = roh.split(ende, 1)
            break
    antwort = antwort.replace("<|ifm|im_end|>", "")
    return gedanken.strip(), antwort.strip()
