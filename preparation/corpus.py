# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Lecture du corpus : fichier *-sentences.txt de la Leipzig Corpora Collection
(« numéro<TAB>phrase »), texte brut ou « breton<TAB>français », en UTF-8 ou UTF-16,
avec l'article Wikipédia d'origine de chaque phrase quand il est connu.
"""

import os
import re

from commun.apertium import normaliser


RE_LEIPZIG = re.compile(r"^\d+\t")       # « 1234<TAB>phrase » (Leipzig Corpora Collection)


def encodage(fichier):
    """UTF-8 (avec ou sans BOM) ou UTF-16 (export d'un tableur ou d'un éditeur Windows)."""
    with open(fichier, "rb") as f:
        debut = f.read(4)
    if debut.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return "utf-8-sig"


def sources_leipzig(fichier):
    """Pour un fichier *-sentences.txt de Leipzig : {numéro de phrase: (adresse, date)},
    d'après les fichiers -inv_so.txt et -sources.txt rangés à côté (vide s'ils manquent)."""
    base = fichier[: -len("-sentences.txt")] if fichier.endswith("-sentences.txt") else None
    if not base or not (os.path.exists(base + "-inv_so.txt") and os.path.exists(base + "-sources.txt")):
        return {}
    adresses = {}
    with open(base + "-sources.txt", encoding="utf-8") as f:
        for ligne in f:
            so_id, _, reste = ligne.rstrip("\n").partition("\t")
            adresse, _, date = reste.partition("\t")
            adresses[so_id] = (adresse, date)
    sources = {}
    with open(base + "-inv_so.txt", encoding="utf-8") as f:
        for ligne in f:
            so_id, _, s_id = ligne.strip().partition("\t")
            if so_id in adresses:
                sources[s_id] = adresses[so_id]
    return sources


def lire_corpus(fichier, maximum=None):
    """Trois formats reconnus, une phrase par ligne :
    - texte brut : « phrase » ;
    - corpus bilingue : « breton<TAB>français » ;
    - fichier *-sentences.txt de Leipzig : « numéro<TAB>phrase » (numéro ignoré).
    Le format Leipzig est reconnu sur la première ligne non vide."""
    phrases = []
    leipzig = None
    with open(fichier, encoding=encodage(fichier)) as f:
        for num, ligne in enumerate(f, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            if leipzig is None:
                leipzig = bool(RE_LEIPZIG.match(ligne))
                if leipzig:
                    print("Format Leipzig reconnu : « numéro<TAB>phrase », numéros ignorés.")
                    sources = sources_leipzig(fichier)
                    if sources:
                        print(f"Sources Leipzig : article Wikipédia connu pour {len(sources)} phrases.")
            source = None
            if leipzig:
                s_id, _, br = ligne.partition("\t")
                fr = ""
                source = sources.get(s_id)
            else:
                br, _, fr = ligne.partition("\t")
            # identifiant = numéro de ligne (l1, l2…), identique au numéro de phrase de Leipzig
            p = {"id": f"l{num}", "br": br.strip(), "fr": fr.strip()}
            if source:
                p["source"], p["date"] = source
            phrases.append(p)
            if maximum and len(phrases) >= maximum:
                break
    return phrases


def cle_doublon(texte):
    """Forme comparée pour repérer les doublons : apostrophes, casse et espaces unifiés."""
    return " ".join(normaliser(texte).lower().split())


def lire_corpus_multiples(fichiers, maximum=None):
    """Lit plusieurs fichiers de phrases à la suite.
    Le premier garde ses identifiants (l1, l2…) : un index déjà calculé pour lui
    reste valable, et seules les phrases ajoutées seront à vectoriser.
    Les suivants prennent des identifiants propres (2l1, 2l2… pour le 2e fichier),
    et leurs phrases déjà présentes plus haut sont écartées."""
    phrases, vues = [], set()
    for rang, fichier in enumerate(fichiers, start=1):
        lues = lire_corpus(fichier)
        if rang == 1:
            gardees = lues
        else:
            gardees = []
            for p in lues:
                cle = cle_doublon(p["br"])
                if cle in vues:
                    continue
                vues.add(cle)
                p["id"] = f"{rang}{p['id']}"
                p.setdefault("corpus", os.path.basename(fichier))
                gardees.append(p)
            print(f"{fichier} : {len(lues)} phrases lues, {len(lues) - len(gardees)} doublons "
                  f"écartés, {len(gardees)} ajoutées.")
        vues.update(cle_doublon(p["br"]) for p in gardees if rang == 1)
        phrases.extend(gardees)
        if maximum and len(phrases) >= maximum:
            return phrases[:maximum]
    return phrases
