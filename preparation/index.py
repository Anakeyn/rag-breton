# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Construction de l'index : phrases.jsonl (phrases et lemmes, pour BM25) et
vecteurs.npy (bge-m3), avec reprise après interruption et recalcul partiel
des seules phrases modifiées.
"""

import json
import os

import numpy as np

from commun.apertium import lemmatiser
from commun.bge_m3 import embeddings, normaliser_vecteurs
from commun.config import (
    LOT_EMBEDDING,
    chemin_phrases,
    chemins_vecteurs,
    empreinte,
    etat_compatible,
    lire_etat,
)


def textes_indexes(dossier):
    """Textes bretons de l'index existant (None s'il n'y en a pas)."""
    chemin = chemin_phrases(dossier)
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(l)["br"] for l in f]


def ecrire_index_lexical(dossier, phrases, automorf):
    """Lemmatise les phrases avec Apertium et écrit phrases.jsonl."""
    for p, lemmes in zip(phrases, lemmatiser([p["br"] for p in phrases], automorf)):
        p["lemmes"] = lemmes
    os.makedirs(dossier, exist_ok=True)
    with open(chemin_phrases(dossier), "w", encoding="utf-8") as f:
        for p in phrases:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def ecrire_etat(dossier, etat):
    """Écriture atomique : un état à moitié écrit ne peut pas survivre à une coupure."""
    _, chemin = chemins_vecteurs(dossier)
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        json.dump(etat, f, ensure_ascii=False, indent=1)
    os.replace(chemin + ".tmp", chemin)


def meme_structure(etat, phrases, modele):
    """Même nombre de phrases, mêmes identifiants, même modèle (texte non vérifié)."""
    return bool(
        etat and etat.get("n") == len(phrases) and etat.get("modele") == modele
        and etat.get("premier") == phrases[0]["id"] and etat.get("dernier") == phrases[-1]["id"]
    )


def etendre_vecteurs(dossier, etat, phrases):
    """Phrases AJOUTÉES à la fin d'un corpus déjà vectorisé : agrandit
    vecteurs.npy en gardant les vecteurs existants ; seules les nouvelles
    phrases restent à calculer (elles reprennent là où le calcul en était)."""
    from numpy.lib.format import open_memmap
    npy, _ = chemins_vecteurs(dossier)
    ancien_n, total = etat["n"], len(phrases)
    print(f"  {total - ancien_n} phrase(s) ajoutée(s) à la fin du corpus : les {ancien_n} "
          "vecteurs existants sont gardés.")
    ancien = open_memmap(npy, mode="r")
    provisoire = npy + ".agrandi"
    V = open_memmap(provisoire, mode="w+", dtype=ancien.dtype, shape=(total, ancien.shape[1]))
    for i in range(0, ancien_n, 10000):                 # copie par tranches (peu de mémoire)
        j = min(i + 10000, ancien_n)
        V[i:j] = ancien[i:j]
    V.flush()
    del V, ancien
    os.replace(provisoire, npy)
    etat.update({"n": total, "dernier": phrases[-1]["id"], "empreinte": empreinte(phrases)})
    etat.pop("origine", None)
    ecrire_etat(dossier, etat)


def indexer_embeddings(phrases, args, anciens=None):
    """Calcule les vecteurs par lots et les écrit directement dans vecteurs.npy
    (fichier projeté en mémoire). Reprise automatique après interruption.
    anciens : textes de l'ancien phrases.jsonl ; si seules quelques phrases ont
    changé (corpus corrigé), seules celles-là sont recalculées."""
    from numpy.lib.format import open_memmap
    npy, _ = chemins_vecteurs(args.index)
    total = len(phrases)
    etat = lire_etat(args.index)
    nouvelle = empreinte(phrases)
    if (meme_structure(etat, phrases, args.modele_embedding) and os.path.exists(npy)
            and etat.get("empreinte") != nouvelle and anciens and len(anciens) == total):
        V = open_memmap(npy, mode="r+")
        changees = [i for i in range(etat["fait"]) if anciens[i] != phrases[i]["br"]]
        print(f"  {len(changees)} phrase(s) modifiée(s) depuis le dernier calcul : "
              "recalcul de celles-là seulement.")
        for k in range(0, len(changees), LOT_EMBEDDING):
            idx = changees[k:k + LOT_EMBEDDING]
            V[idx] = normaliser_vecteurs(embeddings([phrases[i]["br"] for i in idx], args))
            V.flush()
            print(f"\r  recalcul : {k + len(idx)}/{len(changees)}", end="", flush=True)
        if changees:
            print()
        etat["empreinte"] = nouvelle
        etat.pop("origine", None)
        ecrire_etat(args.index, etat)
    if (etat and etat.get("modele") == args.modele_embedding and os.path.exists(npy)
            and etat.get("n", 0) < total and anciens and len(anciens) >= etat["n"]
            and anciens[:etat["n"]] == [p["br"] for p in phrases[:etat["n"]]]):
        etendre_vecteurs(args.index, etat, phrases)
    if etat_compatible(etat, phrases, args.modele_embedding) and os.path.exists(npy):
        fait = etat["fait"]
        V = open_memmap(npy, mode="r+")
        if "empreinte" not in etat:
            print("  (index sans empreinte : texte des phrases non vérifié)")
    else:
        fait = 0
        premier = normaliser_vecteurs(embeddings([phrases[0]["br"]], args))
        V = open_memmap(npy, mode="w+", dtype=np.float32, shape=(total, premier.shape[1]))
        etat = {"modele": args.modele_embedding, "n": total, "dim": int(premier.shape[1]),
                "premier": phrases[0]["id"], "dernier": phrases[-1]["id"],
                "empreinte": empreinte(phrases), "fait": 0}
        ecrire_etat(args.index, etat)
    if fait >= total:
        print(f"  embeddings : déjà complets ({total}/{total}).")
        return
    for i in range(fait, total, LOT_EMBEDDING):
        lot = phrases[i:i + LOT_EMBEDDING]
        V[i:i + len(lot)] = normaliser_vecteurs(embeddings([p["br"] for p in lot], args))
        V.flush()
        etat["fait"] = i + len(lot)
        ecrire_etat(args.index, etat)
        print(f"\r  embeddings : {etat['fait']}/{total} ({100 * etat['fait'] // total} %)",
              end="", flush=True)
    print()
