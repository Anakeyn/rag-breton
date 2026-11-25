#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
#
# Ce programme est un logiciel libre : vous pouvez le redistribuer et/ou le
# modifier selon les termes de la GNU General Public License publiée par la
# Free Software Foundation, version 3 de la licence ou (à votre choix) toute
# version ultérieure. Il est distribué dans l'espoir qu'il sera utile, mais
# SANS AUCUNE GARANTIE. Voir <https://www.gnu.org/licenses/>.
#
# Apertium (apertium, apertium-br-fr) est une dépendance installée à part,
# sous sa propre licence (GPL 2).
"""
rag_breton_preparation.py — construire l'index du corpus breton
================================================================

À lancer une seule fois, ou quand le corpus change. Écrit dans index_breton/ :
  phrases.jsonl   les phrases, leurs lemmes Apertium (recherche par les mots, BM25)
                  et l'article Wikipédia d'origine
  vecteurs.npy    un vecteur bge-m3 par phrase (recherche par le sens), facultatif
  vecteurs.json   l'état du calcul : reprise après coupure, empreinte du texte

Fichier d'entrée : texte UTF-8 (ou UTF-16), une phrase par ligne ; « breton<TAB>
français » ; ou le fichier de la Leipzig Corpora Collection tel quel
(« numéro<TAB>phrase », numéro ignoré, sources lues dans -sources.txt et -inv_so.txt).

Exemples :
  python3 rag_breton_preparation.py bre_wikipedia_2021_100K/bre_wikipedia_2021_100K-sentences.txt
                       # complet : lemmes + vecteurs (des heures en CPU, interruptible)
  python3 rag_breton_preparation.py bre_wikipedia_2021_100K/bre_wikipedia_2021_100K-sentences.txt --sans-embeddings
                       # recherche par les mots seulement : quelques secondes, sans bge-m3
  python3 rag_breton_preparation.py bre_wikipedia_2021_100K/bre_wikipedia_2021_100K-sentences.txt --max 200 --index essai
                       # essai sur 200 phrases, dans un autre dossier

Pendant le calcul des vecteurs : LM Studio ouvert, bge-m3 chargé. Relancer la même
commande après une coupure reprend où le calcul s'était arrêté ; si seules quelques
phrases ont changé, seules celles-là sont recalculées.

Ajouter des phrases sans tout recalculer : donner l'ancien fichier EN PREMIER,
puis le ou les nouveaux. Les phrases déjà présentes sont écartées, les vecteurs
existants gardés, et seules les nouvelles phrases sont vectorisées :
  python3 rag_breton_preparation.py bre_wikipedia_2021_100K/bre_wikipedia_2021_100K-sentences.txt wikipedia_br_2024.txt

La traduction et la recherche se font ensuite avec rag_breton.py.
"""

import argparse
import os
import shutil
import sys

from commun.apertium import trouver_automorf
from commun.config import DOSSIER_INDEX, LMSTUDIO_URL, MODELE_EMBEDDING, chemin_phrases
from preparation.corpus import lire_corpus_multiples
from preparation.index import ecrire_index_lexical, indexer_embeddings, textes_indexes


def preparer(args, automorf):
    phrases = lire_corpus_multiples(args.fichiers, args.max)
    print(f"{len(phrases)} phrases lues.")
    existant = chemin_phrases(args.index)
    if args.max and os.path.exists(existant):
        with open(existant, encoding="utf-8") as f:
            deja = sum(1 for _ in f)
        if deja > len(phrases):
            sys.exit(f"{args.index} contient déjà {deja} phrases : un essai --max {args.max} "
                     "l'écraserait.\nPour un essai, utilise un autre dossier : "
                     f"python3 rag_breton_preparation.py {' '.join(args.fichiers)} --max {args.max} "
                     "--index essai")
    anciens = textes_indexes(args.index)    # pour un recalcul partiel des vecteurs
    print("Lemmatisation" + (" avec Apertium…" if automorf else " SANS Apertium (mode de secours)…"))
    ecrire_index_lexical(args.index, phrases, automorf)
    print(f"Index lexical écrit dans {chemin_phrases(args.index)}")
    if not args.sans_embeddings:
        print(f"Embeddings avec {args.modele_embedding} (interruptible, reprise automatique)…")
        indexer_embeddings(phrases, args, anciens)
    print("Terminé.")


def main():
    ap = argparse.ArgumentParser(description="Construire l'index du corpus breton "
                                             "(lemmes Apertium et vecteurs bge-m3)")
    ap.add_argument("fichiers", nargs="+", metavar="fichier",
                    help="phrases du corpus, par exemple le fichier *-sentences.txt de Leipzig ; "
                         "plusieurs fichiers possibles : le premier garde ses vecteurs, les "
                         "suivants sont ajoutés (doublons écartés)")
    ap.add_argument("--sans-embeddings", action="store_true",
                    help="index par les mots seulement, sans bge-m3 (quelques secondes)")
    ap.add_argument("--max", type=int, default=None, help="limiter à N phrases (essais)")
    ap.add_argument("--index", default=DOSSIER_INDEX,
                    help=f"dossier de l'index (défaut : {DOSSIER_INDEX})")
    ap.add_argument("--lmstudio", default=LMSTUDIO_URL, help="URL de l'API LM Studio")
    ap.add_argument("--modele-embedding", default=MODELE_EMBEDDING)
    ap.add_argument("--automorf", default=None, help="chemin de br-fr.automorf.bin")
    args = ap.parse_args()
    automorf = trouver_automorf(args.automorf)
    if not automorf or not shutil.which("lt-proc"):
        print("Attention : Apertium introuvable, lemmatisation de secours "
              "(sudo apt install apertium apertium-br-fr).", file=sys.stderr)
        automorf = None
    preparer(args, automorf)


if __name__ == "__main__":
    main()
