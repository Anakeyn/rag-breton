# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Paramètres par défaut, chemins de l'index, lecture de son état et petits
utilitaires.

Partagé par la préparation (qui écrit l'index) et la traduction (qui le lit).
"""

import json
import os
import sys


DOSSIER_INDEX = "index_breton"
LMSTUDIO_URL = "http://localhost:1234/v1"
MODELE_EMBEDDING = "lm-kit/text-embedding-bge-m3"
LOT_EMBEDDING = 64          # phrases envoyées par requête d'embedding
LOT_APERTIUM = 2000         # phrases envoyées par appel à lt-proc


def chemin_phrases(dossier):
    return os.path.join(dossier, "phrases.jsonl")


def charger_phrases(dossier):
    chemin = chemin_phrases(dossier)
    if not os.path.exists(chemin):
        sys.exit(f"Index introuvable ({chemin}). Lance d'abord :\n"
                 "  python3 rag_breton_preparation.py <fichier des phrases>")
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def chemins_vecteurs(dossier):
    """vecteurs.npy : une ligne par phrase, dans l'ordre de phrases.jsonl.
    vecteurs.json : état de l'indexation (nombre de lignes déjà calculées)."""
    return os.path.join(dossier, "vecteurs.npy"), os.path.join(dossier, "vecteurs.json")


def lire_etat(dossier):
    _, chemin = chemins_vecteurs(dossier)
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def empreinte(phrases):
    """Empreinte du texte des phrases : des vecteurs ne valent que pour ces phrases-là."""
    import hashlib
    h = hashlib.sha256()
    for p in phrases:
        h.update(p["br"].encode("utf-8") + b"\n")
    return h.hexdigest()[:16]


def etat_compatible(etat, phrases, modele):
    """L'état correspond-il à ce corpus et à ce modèle d'embedding ?
    (l'empreinte n'est vérifiée que si l'état en contient une : les index
    construits avant son ajout restent utilisables)"""
    return bool(
        etat and etat.get("n") == len(phrases) and etat.get("modele") == modele
        and etat.get("premier") == phrases[0]["id"] and etat.get("dernier") == phrases[-1]["id"]
        and etat.get("empreinte", empreinte(phrases)) == empreinte(phrases)
    )


def duree(secondes):
    """12,3 s  ou  4 min 05 s"""
    if secondes < 60:
        return f"{secondes:.1f} s".replace(".", ",")
    m, s = divmod(int(round(secondes)), 60)
    return f"{m} min {s:02d} s"
