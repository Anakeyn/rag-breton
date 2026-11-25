# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Données libres de l'Office public de la langue bretonne (licence ODbL),
téléchargées sur https://www.fr.brezhoneg.bzh/212-donnees-libres-de-droits.htm :

- le corpus de phrases alignées, livré en deux fichiers (une phrase par ligne,
  la ligne n du fichier breton correspond à la ligne n du fichier français),
  devient un fichier « breton<TAB>français » que rag_breton_preparation.py sait lire ;
- la base de toponymie KerOfis (CSV séparé par des points-virgules) devient
  toponymes.tsv : « nom breton<TAB>nom français<TAB>commune<TAB>type de lieu »,
  que rag_breton.py utilise pour reconnaître les noms de lieux.

    python3 -m preparation.donnees_oplb corpus Korpus/Korpus-frazennou-brezhonek.csv \\
            Korpus/Korpus-frazennou-gallek.csv Korpus/korpus_oplb.tsv
    python3 -m preparation.donnees_oplb kerofis donnees/corpus/Korpus/toponymie_lechanvadurezh_20260925.csv donnees/dictionnaires/toponymes.tsv

Les fichiers produits restent sous licence ODbL : à citer (« Office public de la
langue bretonne, ODbL ») et à redistribuer sous la même licence.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict

from commun.apertium import normaliser

MENTION = "Source : Office public de la langue bretonne (KerOfis), licence ODbL"

# en cas d'homonymie, le nom français d'une commune ou d'un quartier l'emporte
PRIORITE_TYPE = {"Kumun": 0, "Karter": 1, "Enez": 2, "Beg": 2, "Karreg": 2, "Dour": 3}


def lignes(fichier):
    """Lignes d'un fichier UTF-8 (avec ou sans BOM), fins de ligne Windows comprises."""
    with open(fichier, encoding="utf-8-sig", newline="") as f:
        return [l.rstrip("\r\n") for l in f]


def aligner(breton, francais, sortie):
    """Deux fichiers alignés ligne à ligne -> « breton<TAB>français ».
    Écarte les lignes vides d'un côté ou de l'autre et les doublons exacts."""
    br, fr = lignes(breton), lignes(francais)
    if len(br) != len(fr):
        sys.exit(f"Les deux fichiers n'ont pas le même nombre de lignes ({len(br)} et {len(fr)}) : "
                 "ils ne sont pas alignés.")
    vues, gardees, vides = set(), [], 0
    for b, f in zip(br, fr):
        b, f = " ".join(b.replace("\t", " ").split()), " ".join(f.replace("\t", " ").split())
        if not b or not f:
            vides += 1
            continue
        if (b, f) in vues:
            continue
        vues.add((b, f))
        gardees.append(f"{b}\t{f}")
    with open(sortie, "w", encoding="utf-8") as s:
        s.write("\n".join(gardees) + "\n")
    print(f"{len(br)} lignes lues, {vides} vides écartées, "
          f"{len(br) - vides - len(gardees)} doublons écartés : {len(gardees)} paires dans {sortie}")


def cle_toponyme(nom):
    """Forme de comparaison d'un nom de lieu : apostrophes unifiées, minuscules,
    sans espaces, traits d'union ni apostrophes (Beuzeg-ar-C'hab = beuzeg ar c'hab)."""
    return "".join(c for c in normaliser(nom).lower() if c not in " -'")


def kerofis(entree, sortie):
    """KerOfis -> toponymes.tsv, une ligne par nom breton. Quand un même nom
    breton correspond à plusieurs lieux, on garde le nom français d'une commune ou
    d'un quartier s'il y en a un, sinon le nom français le plus fréquent."""
    with open(entree, encoding="utf-8-sig", newline="") as f:
        lecteur = csv.DictReader(f, delimiter=";")
        entrees = defaultdict(list)
        for r in lecteur:
            breton = " ".join((r.get("kss_dibab") or "").split())
            nom = " ".join((r.get("kk_anv") or "").split())
            if not breton or not nom:
                continue
            article = (r.get("kk_stagadenn") or "").strip()
            francais = (article + ("" if article.endswith("'") else " ") + nom).strip() if article else nom
            entrees[breton].append((francais, r.get("kkb_kumun_bre", ""), r.get("kr_rummad_bre", "")))
    # un nom breton écrit de deux façons proches (majuscules, trait d'union) : une seule ligne ;
    # article facultatif entre parenthèses : « (An) Enez Wenn » = An Enez Wenn et Enez Wenn
    par_cle = {}
    for breton, liste in entrees.items():
        formes = {breton}
        if "(" in breton:
            formes = {" ".join(re.sub(r"[()]", "", breton).split()),
                      " ".join(re.sub(r"\([^)]*\)", "", breton).split())} - {""}
        for forme in formes:
            par_cle.setdefault(cle_toponyme(forme), (forme, []))[1].extend(liste)
    resultat = []
    for _, (breton, liste) in sorted(par_cle.items()):
        frequence = Counter(fr for fr, _, _ in liste)
        meilleur = min(liste, key=lambda e: (PRIORITE_TYPE.get(e[2], 9), -frequence[e[0]]))
        communes = sorted({c for _, c, _ in liste if c})
        commune = communes[0] if len(communes) == 1 else f"{len(communes)} communes"
        # toutes les formes françaises, commune par commune, pour choisir celle de
        # la commune dont parle le texte (option --commune de rag_breton.py)
        par_commune = sorted({f"{c}={fr}" for fr, c, _ in liste if c and "|" not in fr})
        resultat.append(f"{breton}\t{meilleur[0]}\t{commune}\t{meilleur[2]}\t{'|'.join(par_commune)}")
    with open(sortie, "w", encoding="utf-8") as s:
        s.write(f"# {MENTION}\n# nom breton\tnom français\tcommune\ttype de lieu"
                "\tformes par commune (commune=nom français|…)\n")
        s.write("\n".join(resultat) + "\n")
    print(f"{sum(len(l) for l in entrees.values())} lieux lus : {len(resultat)} noms bretons "
          f"distincts écrits dans {sortie}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="commande", required=True)
    c = sp.add_parser("corpus", help="deux fichiers alignés -> breton<TAB>français")
    c.add_argument("breton")
    c.add_argument("francais")
    c.add_argument("sortie")
    k = sp.add_parser("kerofis", help="base KerOfis -> toponymes.tsv")
    k.add_argument("entree")
    k.add_argument("sortie")
    args = p.parse_args()
    if args.commande == "corpus":
        aligner(args.breton, args.francais, args.sortie)
    else:
        kerofis(args.entree, args.sortie)


if __name__ == "__main__":
    main()
