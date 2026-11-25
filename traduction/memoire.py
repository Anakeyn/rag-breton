# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Mémoire de traduction : des phrases du corpus déjà traduites par un humain
(paires « breton<TAB>français », comme celles de l'Office public de la langue
bretonne) servent d'exemples à Gemma pendant la traduction d'une leçon.

Gemma ne voit toujours aucun breton : chaque exemple lui est donné sous la forme
« mot à mot français → traduction humaine ». Deux usages :
- exemples proches : les phrases traduites qui partagent le plus de mots
  (lemmes) avec la phrase à traduire, pour le vocabulaire et les tournures ;
- mots inconnus d'Apertium : une phrase traduite qui contient le mot, pour
  que Gemma en devine le sens d'après la traduction humaine.
"""

import json
import os
import re
import sys
import time

import numpy as np

from commun.apertium import lemmatiser, normaliser
from commun.config import chemin_phrases
from traduction.recherche import BM25

MOTS_MAX = 30          # exemples trop longs écartés (contexte plus court, plus lisible)
# lemmes trop fréquents pour indiquer une vraie ressemblance entre deux phrases
LEMMES_VIDES = {"an", "ar", "al", "un", "ur", "ul", "bezañ", "kaout", "ober", "ha", "hag",
                "e", "a", "da", "eus", "gant", "war", "ne", "ket", "mont", "se", "hon",
                "ho", "o", "he", "ma", "va", "evit", "en", "ez", "met", "pe", "pa", "ma",
                "prpers", "indirect", "direct", "holl", "ivez", "bet"}
JACCARD_MIN = 0.4
_MEMOIRE = {}


def charger_memoire(dossier):
    """Paires breton–français de l'index (phrases.jsonl) et leur index BM25."""
    if dossier not in _MEMOIRE:
        paires = []
        chemin = chemin_phrases(dossier)
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                for ligne in f:
                    if '"fr": ""' in ligne:     # phrase sans traduction : lecture rapide
                        continue
                    p = json.loads(ligne)
                    if p.get("fr") and len(p["br"].split()) <= MOTS_MAX:
                        paires.append(p)
        bm25 = BM25().fit([p["lemmes"] for p in paires]) if paires else None
        formes = {}                     # forme exacte -> paire la plus courte qui la contient
        for p in sorted(paires, key=lambda p: len(p["br"]), reverse=True):
            for m in RE_MOT.findall(normaliser(p["br"]).lower()):
                formes[m] = p
        _MEMOIRE[dossier] = (paires, bm25, formes)
    return _MEMOIRE[dossier]


def mots_pleins(lemmes):
    """Lemmes significatifs : ni mots-outils, ni ponctuation (« ? » commun à deux
    questions ne les rend pas semblables)."""
    return {l for l in lemmes if any(c.isalpha() for c in l)} - LEMMES_VIDES


def exemples_proches(lemmes, paires, bm25, k):
    """Les k paires qui partagent au moins deux lemmes significatifs avec la phrase."""
    if not bm25 or k <= 0:
        return []
    pleins = mots_pleins(lemmes)
    if len(pleins) < 2:
        return []                       # phrase trop courte : un exemple serait du bruit
    # ressemblance exigée : au moins 2 lemmes pleins en commun, et un indice de
    # Jaccard d'au moins JACCARD_MIN sur les lemmes pleins (l'exemple ne doit pas
    # parler d'autre chose : un ou deux mots communs dans une longue phrase ne suffisent pas)
    scores = bm25.transform(lemmes)
    choisis = []
    for i in np.argsort(-scores)[: 20 * k]:
        if scores[i] <= 0:
            break
        autres = mots_pleins(paires[i]["lemmes"])
        communs = len(pleins & autres)
        if communs >= 2 and communs / len(pleins | autres) >= JACCARD_MIN:
            choisis.append(paires[i])
        if len(choisis) == k:
            break
    return choisis


RE_MOT = re.compile(r"[\w']+")


RE_INCONNU = re.compile(r"«([^»]+)» \(inconnu\)")


def chercher(textes, mots_a_mot, args, automorf):
    """Pour chaque phrase : (exemples proches, {mot inconnu: exemple})."""
    t0 = time.monotonic()
    paires, bm25, formes = charger_memoire(args.index)
    if not paires:
        return None
    k = getattr(args, "exemples", 2)
    resultats = []
    tous_lemmes = lemmatiser(textes, automorf)      # un seul appel à Apertium
    for lemmes, mam in zip(tous_lemmes, mots_a_mot):
        proches = exemples_proches(lemmes, paires, bm25, k)
        inconnus = {}
        for mot in dict.fromkeys(RE_INCONNU.findall(mam)):
            if mot[:1].isupper():       # nom propre probable (Mireille, Beatriz) : rien à apprendre
                continue
            ex = formes.get(normaliser(mot).lower())
            if ex and ex not in proches:
                inconnus[mot] = ex
        resultats.append((proches, inconnus))
    if not getattr(args, "_memoire_annoncee", False):     # une fois, pas à chaque lot
        args._memoire_annoncee = True
        print(f"(mémoire : {len(paires)} paires traduites, recherche en "
              f"{time.monotonic() - t0:.1f} s)", file=sys.stderr)
    return resultats
