#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Fabrique un fichier « à valider » (pour integrer_validations.py) à partir des
mots qui restent INCONNUS d'Apertium dans l'analyse d'un texte.

Usage :
  python3 rag_breton.py phrase "$(cat TEXTE.txt)" --sans-llm \\
      | python3 generer_a_valider.py > a_valider_2.tsv

Pour chaque mot : nombre d'occurrences, forme d'origine possible s'il semble
muté (bourc'hadig pour vourc'hadig), et « nom propre ? » s'il commence par une
majuscule. Les mots déjà présents dans complements.tsv ou variantes.tsv sont
écartés. Les colonnes « proposition » et « valide » restent à remplir.
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from commun.apertium import formes_d_origine, normaliser  # noqa: E402

RE_INCONNU = re.compile(r"^\s*(.+?) : INCONNU d'Apertium")


def cles(chemin):
    if not os.path.exists(chemin):
        return set()
    with open(chemin, encoding="utf-8") as f:
        return {normaliser(l.split("\t")[0]).lower().strip()
                for l in f if "\t" in l and not l.lstrip().startswith("#")}


def main():
    connus = cles("complements.tsv") | cles("variantes.tsv")
    compte, forme_vue = collections.Counter(), {}
    for ligne in sys.stdin:
        m = RE_INCONNU.match(ligne)
        if m:
            forme = normaliser(m.group(1))
            cle = forme.lower()
            compte[cle] += 1
            forme_vue.setdefault(cle, forme)
    print("fichier\tbreton\tproposition\tvalide\tremarque")
    # noms propres probables à la fin, puis du plus fréquent au plus rare
    ordre = sorted(compte, key=lambda c: (forme_vue[c][:1].isupper(), -compte[c], c))
    for cle in ordre:
        if cle in connus:
            continue
        remarques = [f"{compte[cle]} fois"]
        origines = [o for o in formes_d_origine(cle) if o in connus]
        if origines:
            remarques.append(f"forme mutée de {', '.join(origines)} (déjà connu)")
        elif formes_d_origine(cle):
            remarques.append(f"mutée ? origine possible : {' / '.join(formes_d_origine(cle))}")
        if forme_vue[cle][:1].isupper():
            remarques.append("nom propre ?")
        print(f"complements\t{forme_vue[cle]}\t\t\t{' ; '.join(remarques)}")


if __name__ == "__main__":
    main()
