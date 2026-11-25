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
rag_breton.py — traduire et chercher : RAG sur un corpus de phrases bretonnes
=============================================================================

Recherche lexicale BM25 sur les lemmes (Apertium, mutations ramenées au lemme :
benn / fenn -> penn) + recherche sémantique bge-m3 (vecteurs dans un fichier
numpy index_breton/vecteurs.npy, recherche exacte par produit scalaire),
fusionnées par Reciprocal Rank Fusion. Génération par le modèle chargé dans
LM Studio via son API compatible OpenAI.

L'index du corpus se construit à part, une seule fois, avec
rag_breton_preparation.py. Ce programme-ci ne fait que le lire ; la commande
phrase (leçons) n'en a même pas besoin.

Les options générales (--modele-chat, --sans-vecteurs, --index…) se placent
avant ou après le nom de la commande.

Commandes :
  python3 rag_breton.py cherche "morlu" --sans-vecteurs      # recherche par les mots seulement
  python3 rag_breton.py mot penn                    # concordance : toutes les phrases avec le lemme
  python3 rag_breton.py cherche "ar vro hag ar mor"  # recherche hybride, sans LLM
  python3 rag_breton.py demande "Que disent les phrases sur la marine ?"  # recherche + réponse du LLM
  python3 rag_breton.py demande "Traduis et commente ces phrases" --cherche "morlu"
                                                    # recherche sur « morlu », question posée au LLM à part
  python3 rag_breton.py demande --explique --cherche "morlu" -n 3
                                                    # traduction + explication mot à mot (lemme, mutation, grammaire)
  python3 rag_breton.py demande "…" --voir-contexte   # affiche aussi ce qui est envoyé au LLM
  python3 rag_breton.py phrase "Me a gar ar mor."     # une phrase de son choix, hors corpus
  python3 rag_breton.py phrase "…" "…" --sans-llm     # plusieurs phrases, analyse seule (instantané)
  python3 rag_breton.py phrase "…" "…" --une-par-une  # une requête par phrase (dialogues)
  python3 rag_breton.py phrase "$(cat lecon.txt)" --par-lots 5   # long texte, lots de 5
  python3 rag_breton.py phrase "$(cat lecon.txt)" --par-lots 5 --sortie lecon.html
                                                    # + fiche de relecture à imprimer
  python3 rag_breton.py phrase "$(cat lecon.txt)" --sans-llm --sortie lecon.xlsx
  python3 rag_breton.py phrase "$(cat lecon.txt)" --par-lots 5 --memoire   # + exemples traduits du corpus
                                                    # tableau Excel seul (aussi : --sortie lecon.html lecon.xlsx)
  python3 rag_breton.py fiche lecon.html lecon.xlsx  # convertit une fiche HTML déjà faite
  python3 rag_breton.py demande "…" --sans-gloses     # sans l'aide d'Apertium

Aide à la compréhension (commande demande) : pour chaque phrase retrouvée,
le programme ajoute une traduction automatique Apertium (br -> fr) et des
gloses mot à mot (forme -> lemme -> équivalent français, via le dictionnaire
bilingue br-fr.autobil.bin). Le LLM n'a ainsi plus à deviner le sens des
mots bretons : il commente et traduit en français à partir de ces aides.
Avec --explique, les gloses sont remplacées par l'analyse complète d'Apertium,
mot par mot : lemme, catégorie, temps, personne, genre, nombre, traduction,
mutation déduite, toutes les analyses d'un mot ambigu, et « INCONNU » pour les
mots absents du dictionnaire. Cette analyse est affichée directement, sans LLM.
Le LLM, lui, ne voit plus aucun breton : il reçoit un mot à mot français établi
depuis l'analyse, la traduction automatique et des notes (fautes, tournures),
et n'a qu'à rédiger une phrase française correcte.

Les vecteurs (bge-m3) sont FACULTATIFS : ils ne servent qu'à la recherche par
le sens dans le corpus (cherche, demande). Sans eux, la recherche se fait par
les mots (BM25 sur les lemmes). La commande phrase (leçons, phrases de son
choix) n'utilise ni l'index ni les vecteurs : Apertium et le modèle de chat
suffisent.

Organisation du code :
  rag_breton_preparation.py   préparation : construire l'index (une seule fois)
  rag_breton.py               traduction et recherche (ce fichier : commandes et options)
  commun/        outils partagés par les deux phases
    config.py      paramètres par défaut, chemins et état de l'index
    apertium.py    lemmatisation, analyse, dictionnaire bilingue (Apertium br-fr)
    bge_m3.py      un texte -> un vecteur (bge-m3 dans LM Studio)
  preparation/   une seule fois, ou quand le corpus change (rag_breton_preparation.py)
    corpus.py      lecture du fichier de Leipzig, encodages, sources Wikipédia
    index.py       phrases.jsonl, vecteurs.npy, empreinte, recalcul partiel
  traduction/    à chaque utilisation (rag_breton.py : mot, cherche, demande, phrase, fiche)
    grammaire.py   mutations, règles, complements.tsv, mot à mot
    recherche.py   BM25, recherche par le sens, fusion RRF
    gemma.py       prompts, serveur LM Studio, réponse, timing
    fiches.py      fiches de relecture HTML, Excel, TSV
La préparation écrit l'index, la traduction ne fait que le lire ; commun/ ne
dépend d'aucun des deux autres dossiers.

Dépendances :
  sudo apt install apertium apertium-br-fr      # analyseur morphologique breton (lt-proc)
  pip install numpy requests scikit-learn
  pip install openpyxl                          # facultatif : fiches de relecture .xlsx
"""

import argparse
import re
import shutil
import sys
import time

from commun.apertium import lemmatiser, trouver_automorf
from commun.config import (
    DOSSIER_INDEX,
    LMSTUDIO_URL,
    MODELE_EMBEDDING,
    charger_phrases,
    duree,
)
from traduction.fiches import ecrire_fiche, ecrire_tableau, lire_fiche_html
from traduction.gemma import (
    afficher_timing,
    contexte,
    generer,
    lire_traductions,
    modele_chat,
    verifier_serveur,
)
from traduction.grammaire import MOTS_MAX, RE_FIN_PHRASE, decouper_longue
from traduction.recherche import afficher, recherche_hybride


def cmd_mot(args, automorf):
    phrases = charger_phrases(args.index)
    cibles = set(lemmatiser([args.mot], automorf)[0])
    trouves = [i for i, p in enumerate(phrases) if cibles & set(p["lemmes"])]
    print(f"Lemme(s) {sorted(cibles)} : {len(trouves)} phrase(s).")
    afficher(phrases, trouves[: args.n])


def cmd_cherche(args, automorf):
    phrases = charger_phrases(args.index)
    t0 = time.monotonic()
    indices = recherche_hybride(phrases, args.requete, args.n, args, automorf)
    afficher(phrases, indices)
    print(f"\n⏱ recherche : {duree(time.monotonic() - t0)}", file=sys.stderr)


def cmd_demande(args, automorf):
    t0 = time.monotonic()
    phrases = charger_phrases(args.index)
    verifier_serveur(args)
    # --cherche : mots bretons pour la recherche, distincts de la question posée
    requete = args.cherche or args.question
    t_recherche = time.monotonic()
    indices = recherche_hybride(phrases, requete, args.n, args, automorf)
    t_recherche = time.monotonic() - t_recherche
    if not indices:
        sys.exit("Aucune phrase pertinente trouvée.")
    print("=== Phrases retrouvées (texte exact du corpus) ===")
    afficher(phrases, indices)
    t_aides = time.monotonic()
    documents, analyses = contexte(phrases, indices, args, automorf)
    t_aides = time.monotonic() - t_aides
    if analyses:
        print("\n=== Analyse grammaticale (Apertium, sans LLM) ===")
        for r, a in enumerate(analyses, start=1):
            print(f"[{r}]")
            for ligne in a["lignes"].split("\n"):
                print("  " + ligne)
            print(f"  Mot à mot : {a['mot_a_mot']}")
    if args.voir_contexte:
        print("\n=== Contexte envoyé au modèle ===")
        print(documents)
    print("\n=== Réponse du modèle ===")
    question = args.question
    if args.explique:
        question = ("Pour chaque document, écris une traduction en bon français, fidèle "
                    "au mot à mot et conforme aux notes, sans rien ajouter ni retrancher. "
                    "Puis, en une ligne, signale ce qui reste incertain (mots inconnus)."
                    + ("" if args.question == "Traduis ces phrases." else
                       "\nQuestion de l'utilisateur : " + args.question))
    m = {}
    generer(question, documents, args, m)           # affiche la réponse au fil de l'eau
    afficher_timing(t_recherche, t_aides, m, time.monotonic() - t0)


def cmd_fiche(args, automorf):
    """Convertit une fiche de relecture HTML déjà écrite en .xlsx ou .tsv,
    sans refaire l'analyse ni la traduction."""
    lignes = lire_fiche_html(args.html)
    if not lignes:
        sys.exit(f"Aucune ligne de tableau trouvée dans {args.html}.")
    ecrire_tableau(args.sortie, lignes)
    print(f"{len(lignes)} lignes écrites dans {args.sortie}")


# abréviation faite de lettres isolées suivies d'un point (d.l.e., h.a., k.o.)
RE_ABREVIATION = re.compile(r"[(\"«“]?(?:[^\W\d_]\.){2,}[,;:)\"»”]?")
# abréviations courantes, développées avant l'analyse (Apertium ne les connaît pas)
ABREVIATIONS = {"d.l.e.": "da lavarout eo", "h.a.": "hag all"}


def developper_abreviation(mot):
    """d.l.e. -> da lavarout eo ; une abréviation inconnue garde ses points."""
    for abr, forme in ABREVIATIONS.items():
        if abr in mot.lower():
            return mot.lower().replace(abr, forme)
    return mot


def cmd_phrase(args, automorf):
    """Analyse une ou plusieurs phrases tapées par l'utilisateur (sans passer
    par le corpus) : analyse Apertium, mot à mot, puis traduction par le LLM."""
    t0 = time.monotonic()
    # chaque ligne est une phrase (utile pour coller un dialogue) ; avec
    # --une-par-une, une ligne contenant plusieurs phrases est aussi découpée
    lignes = [l for t in args.texte for l in t.splitlines() if l.strip()]
    # syllabes séparées pour la prononciation : U.gent -> Ugent, sko.li.di -> skolidi
    # (point suivi d'une minuscule) ; mais « daol-gont.Bras » = espace oublié
    # après le point : daol-gont. Bras
    # sauf dans une abréviation faite de lettres isolées : d.l.e., h.a.
    lignes = [" ".join(developper_abreviation(m) if RE_ABREVIATION.fullmatch(m)
                       else re.sub(r"(?<=[^\W\d_])\.(?=[a-zà-öø-ÿ])", "", m)
                       for m in l.split(" ")) for l in lignes]
    lignes = [re.sub(r"(?<=[a-zà-öø-ÿ])\.(?=[A-ZÀ-ÖØ-Þ])", ". ", l) for l in lignes]
    # Ꝃ (k barré) : abréviation traditionnelle de « Ker » (Ꝃlien = Kerlien)
    lignes = [l.replace("Ꝃ", "Ker").replace("ꝃ", "ker") for l in lignes]
    # « Pironig…Porzh Laouen », « Pentraezh….Gouzout » : espace oublié entre un mot,
    # les points de suspension et une majuscule (pas dans « naontek…ha me »)
    lignes = [re.sub(r"(?<=[^\W\d_])…\.?(?=[A-ZÀ-ÖØ-Þ])", "… ", l) for l in lignes]
    if args.une_par_une:
        args.par_lots = 1
    if args.par_lots:
        lignes = [p for l in lignes for p in RE_FIN_PHRASE.split(l) if p.strip()]
    # une ligne de plus de MOTS_MAX mots est d'abord coupée en phrases, puis
    # chaque phrase trop longue aux « ; » et aux « … »
    lignes = [m for l in lignes
              for p in ([l] if len(l.split()) <= MOTS_MAX else RE_FIN_PHRASE.split(l))
              if p.strip() for m in decouper_longue(p)]
    phrases = [{"id": f"saisie{k}", "br": " ".join(t.split()), "fr": ""}
               for k, t in enumerate(lignes, start=1)]
    if not phrases:
        sys.exit("Aucune phrase à analyser.")
    if not automorf:
        sys.exit("Apertium est nécessaire pour la commande phrase.")
    args.explique, args.sans_gloses = True, False
    t_aides = time.monotonic()
    documents, analyses = contexte(phrases, list(range(len(phrases))), args, automorf)
    t_aides = time.monotonic() - t_aides
    print("=== Analyse grammaticale (Apertium, sans LLM) ===")
    for r, (p, a) in enumerate(zip(phrases, analyses), start=1):
        print(f"[{r}] {p['br']}")
        for ligne in a["lignes"].split("\n"):
            print("  " + ligne)
        print(f"  Mot à mot : {a['mot_a_mot']}")
    if args.voir_contexte:
        print("\n=== Contexte envoyé au modèle ===")
        print(documents)
    if args.sans_llm:
        print(f"\n(analyse en {duree(time.monotonic() - t0)}, sans LLM)", file=sys.stderr)
        if args.sortie:
            for chemin in args.sortie:
                ecrire_fiche(chemin, phrases, analyses, {}, args, time.monotonic() - t0)
        return
    verifier_serveur(args)
    print("\n=== Réponse du modèle ===")
    question = ("Pour chaque document, écris une traduction en bon français, fidèle "
                "au mot à mot et conforme aux notes, sans rien ajouter ni retrancher.")
    if any("(inconnu)" in a["mot_a_mot"] for a in analyses):
        question += " Puis, en une ligne, signale ce qui reste incertain (mots inconnus)."
    else:
        question += " Réponds uniquement par la traduction."
    if args.par_lots == 1:                  # les INCONNUS sont déjà affichés par le script
        question = ("Traduis ce document en bon français, fidèle au mot à mot et "
                    "conforme aux notes, sans rien ajouter ni retrancher. "
                    "Réponds uniquement par la traduction.")
    elif args.par_lots:
        question = ("Traduis chaque document séparément, en bon français, fidèle à son "
                    "propre mot à mot et conforme aux notes, sans rien ajouter ni "
                    "retrancher. Réponds uniquement par les traductions, une par ligne, "
                    "chacune précédée de son numéro entre crochets.")
    if not args.par_lots or len(phrases) <= args.par_lots:
        m = {}
        texte = generer(question, documents, args, m)
        traductions = lire_traductions(texte, 1, len(phrases))
        afficher_timing(0.0, t_aides, m, time.monotonic() - t0)
        if args.sortie:
            for chemin in args.sortie:
                ecrire_fiche(chemin, phrases, analyses, traductions, args, time.monotonic() - t0)
        return
    # --une-par-une : une requête par phrase, pour qu'une phrase courte ne soit
    # pas « harmonisée » avec ses voisines (le début du prompt, identique,
    # reste en cache dans LM Studio : seule la phrase elle-même est relue)
    total = {"lecture": 0.0, "reflexion": 0.0, "reponse": 0.0,
             "morceaux_reflexion": 0, "morceaux_reponse": 0, "usage": {}}
    n = args.par_lots
    print(f"(modèle : {modele_chat(args)} ; "
          + ("une requête par phrase…)" if n == 1 else f"lots de {n} phrases…)"),
          file=sys.stderr, flush=True)
    args._discret = True
    traductions = {}
    for i in range(0, len(phrases), n):
        lot = list(range(i, min(i + n, len(phrases))))
        document, _ = contexte(phrases, lot, args, automorf, debut=i + 1)
        if n == 1:
            print(f"[{i + 1}] ", end="", flush=True)
        m = {}
        texte = generer(question, document, args, m)
        if n == 1:
            traductions[i + 1] = texte.strip()
        else:
            traductions.update(lire_traductions(texte, i + 1, len(lot)))
        for cle in ("lecture", "reflexion", "reponse", "morceaux_reflexion", "morceaux_reponse"):
            total[cle] += m.get(cle, 0)
        for cle, valeur in (m.get("usage") or {}).items():
            if isinstance(valeur, (int, float)):
                total["usage"][cle] = total["usage"].get(cle, 0) + valeur
    afficher_timing(0.0, t_aides, total, time.monotonic() - t0)
    if args.sortie:
        for chemin in args.sortie:
            ecrire_fiche(chemin, phrases, analyses, traductions, args, time.monotonic() - t0)


def main():
    def options_generales(parser, defauts):
        """Options communes à toutes les commandes. Déclarées deux fois : avant la
        commande (avec leurs valeurs par défaut) et après (sans défaut, pour ne pas
        écraser une valeur donnée avant)."""
        d = (lambda v: v) if defauts else (lambda v: argparse.SUPPRESS)
        parser.add_argument("--index", default=d(DOSSIER_INDEX), help="dossier de l'index")
        parser.add_argument("--lmstudio", default=d(LMSTUDIO_URL), help="URL de l'API LM Studio")
        parser.add_argument("--modele-embedding", default=d(MODELE_EMBEDDING))
        parser.add_argument("--modele-chat", default=d(None),
                            help="par défaut : premier modèle de chat chargé")
        parser.add_argument("--automorf", default=d(None), help="chemin de br-fr.automorf.bin")
        parser.add_argument("--sans-vecteurs", action="store_true", default=d(False),
                            help="recherche par les mots seulement (BM25), sans bge-m3")

    ap = argparse.ArgumentParser(description="Traduire et chercher dans un corpus breton "
                                             "(l'index se construit avec rag_breton_preparation.py)")
    options_generales(ap, defauts=True)
    generales = argparse.ArgumentParser(add_help=False)
    options_generales(generales, defauts=False)
    sp = ap.add_subparsers(dest="commande", required=True)
    _add = sp.add_parser
    sp.add_parser = lambda *a, **k: _add(*a, parents=[generales], **k)

    p = sp.add_parser("mot", help="concordance d'un mot (toutes ses formes mutées)")
    p.add_argument("mot")
    p.add_argument("-n", type=int, default=30)

    p = sp.add_parser("cherche", help="recherche hybride sans LLM")
    p.add_argument("requete")
    p.add_argument("-n", type=int, default=10)

    p = sp.add_parser("demande", help="recherche hybride + réponse du LLM")
    p.add_argument("question", nargs="?", default="Traduis ces phrases.",
                   help="question posée au modèle (facultative avec --explique)")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--complements", default="complements.tsv",
                   help="fichier « breton<TAB>traduction » prioritaire sur Apertium "
                        "(mots inconnus, expressions) ; défaut : complements.tsv")
    p.add_argument("--variantes", default="variantes.tsv",
                   help="graphies dialectales -> breton unifié, appliquées avant Apertium")
    p.add_argument("--toponymes", default="donnees/dictionnaires/toponymes.tsv",
                   help="noms de lieux bretons -> français, tirés de KerOfis "
                        "(python3 -m preparation.donnees_oplb kerofis …) ; défaut : donnees/dictionnaires/toponymes.tsv")
    p.add_argument("--commune", default=None,
                   help="commune dont parle le texte (ex. Douarnenez) : pour un nom de lieu "
                        "porté par plusieurs lieux, la forme française de cette commune")
    p.add_argument("--explique", action="store_true",
                   help="ajoute la consigne : traduction puis explication mot à mot")
    p.add_argument("--cherche", default=None,
                   help="mots bretons utilisés pour la recherche (par défaut : la question)")
    p.add_argument("--sans-gloses", action="store_true",
                   help="ne pas ajouter la traduction et les gloses Apertium")
    p.add_argument("--voir-contexte", action="store_true",
                   help="afficher le contexte complet envoyé au modèle")
    p.add_argument("--voir-reflexion", action="store_true",
                   help="afficher la réflexion du modèle (Gemma 4…) au lieu d'un compteur")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="longueur maximale de la réponse, réflexion comprise (0 = sans limite)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = réponse la plus probable, sans fantaisie (défaut) ; "
                        "monter vers 0,7 pour plus de variété")
    p.add_argument("--delai", type=int, default=600,
                   help="abandon si LM Studio reste muet plus de N secondes (défaut 600)")

    p = sp.add_parser("phrase", help="analyser et traduire une phrase de son choix (hors corpus)")
    p.add_argument("texte", nargs="+", help="une ou plusieurs phrases, chacune entre guillemets")
    p.add_argument("--sans-llm", action="store_true",
                   help="analyse Apertium et mot à mot seulement, sans traduction par le LLM")
    p.add_argument("--une-par-une", action="store_true",
                   help="une requête au LLM par phrase : plus fiable pour les dialogues "
                        "de phrases courtes, mais lent sur un long texte")
    p.add_argument("--sortie", default=None, metavar="FICHIER", nargs="+",
                   help="fiche(s) de relecture : .html (navigateur, impression), .xlsx "
                        "(Excel, LibreOffice Calc) ou .tsv ; plusieurs fichiers possibles")
    p.add_argument("--par-lots", type=int, default=0, metavar="N",
                   help="requêtes de N phrases (ex. 5) : compromis entre fiabilité et "
                        "vitesse pour un long texte")
    p.add_argument("--complements", default="complements.tsv")
    p.add_argument("--variantes", default="variantes.tsv",
                   help="graphies dialectales -> breton unifié, appliquées avant Apertium "
                        "(défaut : variantes.tsv)")
    p.add_argument("--toponymes", default="donnees/dictionnaires/toponymes.tsv",
                   help="noms de lieux bretons -> français, tirés de KerOfis "
                        "(python3 -m preparation.donnees_oplb kerofis …) ; défaut : donnees/dictionnaires/toponymes.tsv")
    p.add_argument("--commune", default=None,
                   help="commune dont parle le texte (ex. Douarnenez) : pour un nom de lieu "
                        "porté par plusieurs lieux, la forme française de cette commune")
    p.add_argument("--memoire", action="store_true",
                   help="donner au LLM des phrases du corpus traduites par un humain "
                        "(paires de l'Office public) : phrases proches et mots inconnus ; "
                        "contexte plus long, donc traduction plus lente")
    p.add_argument("--exemples", type=int, default=2, metavar="N",
                   help="avec --memoire : N phrases proches par phrase (défaut 2)")
    p.add_argument("--voir-contexte", action="store_true")
    p.add_argument("--voir-reflexion", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--delai", type=int, default=600)

    p = sp.add_parser("fiche", help="convertir une fiche de relecture HTML en .xlsx ou .tsv")
    p.add_argument("html", help="fiche HTML écrite par « phrase … --sortie fiche.html »")
    p.add_argument("sortie", help="fichier à écrire : .xlsx (Excel) ou .tsv")

    args = ap.parse_args()
    automorf = trouver_automorf(args.automorf)
    if not automorf or not shutil.which("lt-proc"):
        print("Attention : Apertium introuvable, lemmatisation de secours "
              "(sudo apt install apertium apertium-br-fr).", file=sys.stderr)
        automorf = None
    {"mot": cmd_mot, "cherche": cmd_cherche, "demande": cmd_demande,
     "phrase": cmd_phrase, "fiche": cmd_fiche}[args.commande](args, automorf)


if __name__ == "__main__":
    main()
