#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Ajoute à complements.tsv et variantes.tsv les lignes validées d'un fichier
« à valider » (colonnes : fichier, breton, proposition, valide, remarque).

Seules les lignes dont la colonne « valide » vaut oui (ou o, x, 1) sont
prises, et jamais une proposition « ? ». Une forme déjà présente dans le
fichier cible n'est pas ajoutée deux fois.

Usage :
  python3 integrer_validations.py a_valider_martoloded.tsv
  python3 integrer_validations.py a_valider_martoloded.tsv --essai   # sans rien écrire
"""
import argparse
import datetime
import os
import sys

CIBLES = {"complements": "complements.tsv", "variantes": "variantes.tsv"}
OUI = {"oui", "o", "x", "1", "yes", "y"}


def cle(mot):
    return " ".join(mot.replace("’", "'").lower().split())


def deja_presents(chemin):
    if not os.path.exists(chemin):
        return set()
    with open(chemin, encoding="utf-8") as f:
        return {cle(l.split("\t")[0]) for l in f if "\t" in l and not l.lstrip().startswith("#")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fichier")
    p.add_argument("--essai", action="store_true", help="afficher sans écrire")
    a = p.parse_args()

    ajouts = {nom: [] for nom in CIBLES}
    ignorees = 0
    with open(a.fichier, encoding="utf-8") as f:
        for num, ligne in enumerate(f, start=1):
            champs = ligne.rstrip("\n").split("\t") + [""] * 5
            fichier, breton, proposition, valide, remarque = (c.strip() for c in champs[:5])
            if num == 1 and fichier == "fichier":
                continue                                   # en-tête
            if not breton or fichier.startswith("#"):
                continue
            if fichier not in CIBLES:
                print(f"ligne {num} : fichier « {fichier} » inconnu (complements ou variantes)",
                      file=sys.stderr)
                continue
            if valide.lower() not in OUI or not proposition or proposition == "?":
                ignorees += 1
                continue
            ajouts[fichier].append((breton, proposition, remarque))

    date = datetime.date.today().isoformat()
    for nom, lignes in ajouts.items():
        chemin = CIBLES[nom]
        presents = deja_presents(chemin)
        nouvelles = [l for l in lignes if cle(l[0]) not in presents]
        doublons = len(lignes) - len(nouvelles)
        print(f"{chemin} : {len(nouvelles)} ajout(s)" + (f", {doublons} déjà présent(s)" if doublons else ""))
        for breton, proposition, remarque in nouvelles:
            print(f"   {breton} -> {proposition}")
        if not nouvelles or a.essai:
            continue
        with open(chemin, "a", encoding="utf-8", newline="\n") as out:
            out.write(f"#\n# Ajouts validés ({os.path.basename(a.fichier)}, {date})\n")
            for breton, proposition, remarque in nouvelles:
                if nom == "variantes" and remarque:
                    out.write(f"{breton}\t{proposition}\t{remarque}\n")   # 3e colonne : commentaire
                else:
                    out.write(f"{breton}\t{proposition}\n")            # complements : 2 colonnes
    print(f"{ignorees} ligne(s) non validée(s) ou sans proposition, laissées de côté.")
    if a.essai:
        print("(essai : rien n'a été écrit)")


if __name__ == "__main__":
    main()
