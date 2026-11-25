# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Dialogue avec le modèle de chat (Gemma) de LM Studio : prompts, contexte
envoyé (mot à mot français, sans breton), réponse au fil de l'eau et timing.
"""

import json
import re
import subprocess
import sys
import time

import requests

from commun.apertium import demuter_inconnus, gloses, traduction_apertium, trouver_autobil
from commun.config import duree
from traduction.memoire import chercher
from traduction.grammaire import (
    LEGENDE_MOT_A_MOT,
    analyse_complete,
    appliquer_variantes,
    charger_complements,
    charger_toponymes,
    charger_variantes,
)


# Prompt volontairement court : Gemma 4 récite chaque consigne dans sa
# réflexion avant de travailler, et en CPU chaque mot coûte cher.
PROMPT_SYSTEME = (
    "Réponds en français, brièvement. Tu peux citer les phrases et les mots "
    "bretons fournis, mais n'invente aucun mot breton. Les lignes « Apertium » et "
    "« Gloses » sont des aides automatiques, parfois fausses : traduis en bon "
    "français sans les recopier."
)

# Prompt du mode --explique : le modèle ne voit que du français.
PROMPT_SYSTEME_FRANCAIS = (
    "Tu es traducteur. Tu transformes un mot à mot fiable en phrase française "
    "correcte et naturelle. Le mot à mot suit l'ordre des mots breton, qui place "
    "en tête l'élément mis en relief : remets les mots dans l'ordre naturel du "
    "français, mais laisse en tête les mots d'ouverture (oui, non, interjections). "
    "Réponds brièvement, en français, uniquement par la traduction demandée, même "
    "si le document ne contient qu'un mot."
)


def verifier_serveur(args):
    """Arrête proprement si le serveur LM Studio ne répond pas."""
    try:
        requests.get(f"{args.lmstudio}/models", timeout=10).raise_for_status()
    except requests.RequestException:
        sys.exit(
            f"Le serveur LM Studio ne répond pas ({args.lmstudio}).\n"
            "Démarre-le : onglet Developer -> Start Server, ou « lms server start »,\n"
            "puis charge le modèle de chat, et bge-m3 si tu utilises la recherche\n"
            "par le sens (« lms ps » pour vérifier)."
        )


def modele_chat(args):
    if args.modele_chat:
        return args.modele_chat
    modeles = requests.get(f"{args.lmstudio}/models", timeout=30).json()["data"]
    chats = [m["id"] for m in modeles if "embed" not in m["id"].lower()]
    if not chats:
        sys.exit("Aucun modèle de chat disponible dans LM Studio.")
    return chats[0]


def contexte(phrases, indices, args, automorf, debut=1):
    """Bloc « Documents » envoyé au LLM : phrases exactes + aides Apertium.
    debut : numéro du premier document (traitement par lots)."""
    textes = [phrases[i]["br"] for i in indices]
    # graphies dialectales ou anciennes ramenées au breton unifié (variantes.tsv)
    variantes = charger_variantes(getattr(args, "variantes", None))
    remplacements = [[] for _ in textes]
    if variantes:
        resultats = [appliquer_variantes(t, variantes) for t in textes]
        textes = [t for t, _ in resultats]
        remplacements = [r for _, r in resultats]
    complements = charger_complements(getattr(args, "complements", None))
    # mots inconnus d'Apertium dont la forme non mutée est connue d'Apertium ou
    # de complements.tsv : zivoull -> divoull, c'hevelloù -> kevelloù
    demutes = [[] for _ in textes]
    if automorf and not args.sans_gloses:
        try:
            textes, trouves = demuter_inconnus(
                textes, automorf, {c for c in complements if " " not in c})
            for k, t in enumerate(textes):
                mots = {m.lower() for m in re.findall(r"[^\W\d_]+(?:'h[^\W\d_]*)*", t)}
                demutes[k] = [f"{m} → {c}" for m, c in sorted(trouves.items()) if c in mots]
        except (subprocess.SubprocessError, OSError) as e:
            print(f"(démutation des mots inconnus indisponible : {e})", file=sys.stderr)
    complete = getattr(args, "explique", False)       # --explique : analyse mot à mot
    if args.sans_gloses or not automorf:
        lignes_gloses = [""] * len(textes)
        traductions = [""] * len(textes)
        complete = False
    else:
        try:
            if complete:
                lignes_gloses = analyse_complete(
                    textes, automorf, trouver_autobil(automorf),
                    complements,
                    charger_toponymes(getattr(args, "toponymes", None), getattr(args, "commune", None)))
            else:
                lignes_gloses = gloses(textes, automorf, trouver_autobil(automorf))
            if complete:
                for g, r, d in zip(lignes_gloses, remplacements, demutes):
                    if d:
                        g["lignes"] = (f"DÉMUTATIONS (mots inconnus d'Apertium) : {', '.join(d)}\n"
                                       + g["lignes"])
                    if r:
                        g["lignes"] = f"VARIANTES (variantes.tsv) : {', '.join(r)}\n" + g["lignes"]
        except (subprocess.SubprocessError, OSError) as e:
            print(f"(gloses indisponibles : {e})", file=sys.stderr)
            lignes_gloses = [""] * len(textes)
            complete = False
        # pas de traduction automatique si le corpus fournit déjà la traduction
        a_traduire = [k for k, i in enumerate(indices) if not phrases[i]["fr"]]
        traductions = [""] * len(textes)
        for k, t in zip(a_traduire, traduction_apertium([textes[k] for k in a_traduire])):
            traductions[k] = t
    memoire = None
    if complete and getattr(args, "memoire", False):
        memoire = aides_memoire(textes, lignes_gloses, args, automorf)
    blocs = []
    if complete:
        # --explique : le LLM ne voit AUCUN breton, seulement du français
        # (ses fausses « connaissances » du breton l'emportaient sur l'analyse).
        for r, (i, g, t) in enumerate(zip(indices, lignes_gloses, traductions), start=debut):
            bloc = f"[{r}] Mot à mot : {g['mot_a_mot']}"
            if phrases[i]["fr"]:
                bloc += f"\n    Traduction du corpus : {phrases[i]['fr']}"
            if t:
                bloc += f"\n    Traduction automatique : {t}"
            if g["notes"]:
                bloc += "\n    Notes : " + " ".join(g["notes"])
            if memoire:
                bloc += memoire[r - debut]
            blocs.append(bloc)
        legende = LEGENDE_MOT_A_MOT + (LEGENDE_MEMOIRE if memoire else "")
        return legende + "\n" + "\n".join(blocs), lignes_gloses
    for r, (i, g, t) in enumerate(zip(indices, lignes_gloses, traductions), start=debut):
        bloc = f"[{r}] {phrases[i]['br']}"
        if phrases[i]["fr"]:
            bloc += f"\n    Traduction (corpus) : {phrases[i]['fr']}"
        if t:
            bloc += f"\n    Apertium : {t}"
        if g:
            bloc += f"\n    Gloses : {g}"
        blocs.append(bloc)
    return "\n".join(blocs), None


LEGENDE_MEMOIRE = (
    " Certains documents sont suivis d'« Exemples » : d'autres phrases, données "
    "elles aussi en mot à mot, avec leur traduction faite par un humain. Ils "
    "servent seulement à comprendre le vocabulaire et les tournures : ne les "
    "traduis pas et n'en recopie pas le contenu."
)


def aides_memoire(textes, analyses, args, automorf):
    """Texte ajouté à chaque document : exemples traduits par un humain, sous
    forme de mot à mot (Gemma ne voit pas de breton). Ajoute aussi des lignes
    « MÉMOIRE » à l'analyse affichée pour l'utilisateur."""
    trouves = chercher(textes, [a["mot_a_mot"] for a in analyses], args, automorf)
    if trouves is None:
        print("(--memoire : aucune phrase traduite dans l'index ; indexe par exemple "
              "ofis_br_fr.tsv avec rag_breton_preparation.py)", file=sys.stderr)
        return None
    # mot à mot des exemples, en une seule analyse
    exemples = [e for proches, inconnus in trouves for e in proches + list(inconnus.values())]
    mam = {}
    if exemples:
        textes_ex = list(dict.fromkeys(e["br"] for e in exemples))
        res = analyse_complete(textes_ex, automorf, trouver_autobil(automorf),
                               charger_complements(getattr(args, "complements", None)),
                               charger_toponymes(getattr(args, "toponymes", None), getattr(args, "commune", None)))
        mam = {t: a["mot_a_mot"] for t, a in zip(textes_ex, res)}
    ajouts = []
    for a, (proches, inconnus) in zip(analyses, trouves):
        texte, affichage = "", []
        if proches:
            texte += "\n    Exemples (ne pas traduire) :"
            for e in proches:
                texte += f"\n      - Mot à mot : {mam[e['br']]}\n        Traduction humaine : {e['fr']}"
                affichage.append(f"MÉMOIRE : {e['br']}  →  {e['fr']}")
        for mot, e in inconnus.items():
            texte += (f"\n    Exemple pour «{mot}» (ne pas traduire) : Mot à mot : {mam[e['br']]}"
                      f"\n        Traduction humaine : {e['fr']}")
            affichage.append(f"MÉMOIRE «{mot}» : {e['br']}  →  {e['fr']}")
        if affichage:
            a["lignes"] += "\n" + "\n".join(affichage)
        ajouts.append(texte)
    return ajouts


def generer(question, documents, args, mesures=None):
    """Réponse du modèle en streaming : le texte s'affiche au fur et à mesure.
    Pas de limite globale de durée ; on n'abandonne que si LM Studio reste
    muet plus de --delai secondes d'affilée (lecture du prompt comprise).
    La « réflexion » des modèles qui raisonnent (Gemma 4…) est signalée par un
    compteur, ou affichée avec --voir-reflexion."""
    corps = {
        "model": modele_chat(args),
        "temperature": args.temperature,
        "stream": True,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEME_FRANCAIS
             if getattr(args, "explique", False) else PROMPT_SYSTEME},
            {"role": "user", "content": f"Documents :\n{documents}\n\nQuestion : {question}"},
        ],
    }
    if args.max_tokens:
        corps["max_tokens"] = args.max_tokens
    corps["stream_options"] = {"include_usage": True}   # nombre de tokens en fin de flux
    mesures = {} if mesures is None else mesures
    t_envoi = time.monotonic()
    t_premier = t_reponse = None
    if not getattr(args, "_discret", False):
        print(f"(modèle : {corps['model']} ; lecture du contexte, la réponse s'affichera ensuite…)",
              file=sys.stderr, flush=True)
    texte, reflexion = [], 0
    try:
        with requests.post(f"{args.lmstudio}/chat/completions", json=corps, stream=True,
                           timeout=(30, args.delai)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"                # sinon les accents arrivent abîmés
            for ligne in r.iter_lines(chunk_size=1, decode_unicode=True):
                if not ligne or not ligne.startswith("data:"):
                    continue
                donnees = ligne[5:].strip()
                if donnees == "[DONE]":
                    break
                try:
                    paquet = json.loads(donnees)
                except ValueError:
                    continue
                if paquet.get("usage"):
                    mesures["usage"] = paquet["usage"]
                choix = paquet.get("choices") or []
                if not choix:
                    continue
                delta = choix[0].get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")
                        or delta.get("reasoning")) and t_premier is None:
                    t_premier = time.monotonic()
                pensee = delta.get("reasoning_content") or delta.get("reasoning")
                if pensee:
                    reflexion += 1
                    if args.voir_reflexion:
                        print(pensee, end="", file=sys.stderr, flush=True)
                    elif reflexion % 25 == 1:
                        print(f"\r(le modèle réfléchit… {reflexion} morceaux)", end="",
                              file=sys.stderr, flush=True)
                contenu = delta.get("content")
                if contenu:
                    if t_reponse is None:
                        t_reponse = time.monotonic()
                    if reflexion and not texte:
                        print("\n", file=sys.stderr, flush=True)
                    print(contenu, end="", flush=True)
                    texte.append(contenu)
    except requests.exceptions.ReadTimeout:
        print(f"\n(LM Studio n'a rien envoyé pendant {args.delai} s : abandon. "
              "Essaie -n 3, ou --delai 1200.)", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n(interrompu)", file=sys.stderr)
    print()
    t_fin = time.monotonic()
    mesures.update({
        "lecture": (t_premier or t_fin) - t_envoi,
        "reflexion": ((t_reponse or t_fin) - t_premier) if (t_premier and reflexion) else 0.0,
        "reponse": (t_fin - t_reponse) if t_reponse else 0.0,
        "morceaux_reflexion": reflexion, "morceaux_reponse": len(texte),
    })
    if reflexion and not texte:
        print("(le modèle n'a produit que de la réflexion, sans réponse : augmente "
              "--max-tokens ou désactive le raisonnement dans LM Studio)", file=sys.stderr)
    return "".join(texte)


RE_NUMERO = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def lire_traductions(texte, debut, nombre):
    """« [3] Une salle de classe ? » … -> {3: "Une salle de classe ?"}.
    Une ligne sans numéro complète la traduction précédente. Si le modèle
    n'a mis aucun numéro et qu'il n'y avait qu'un document, tout le texte
    lui revient."""
    traductions, courant = {}, None
    for ligne in texte.splitlines():
        m = RE_NUMERO.match(ligne)
        if m and debut <= int(m.group(1)) < debut + nombre:
            courant = int(m.group(1))
            traductions[courant] = m.group(2).strip()
        elif courant is not None and ligne.strip():
            traductions[courant] += " " + ligne.strip()
    if not traductions and nombre == 1 and texte.strip():
        traductions[debut] = texte.strip()
    return traductions


def afficher_timing(t_recherche, t_aides, m, total):
    """Bilan des durées, avec les nombres de tokens si LM Studio les fournit."""
    u = m.get("usage") or {}
    lignes = [
        f"recherche (BM25 + vecteurs)      {duree(t_recherche)}",
        f"aides Apertium                   {duree(t_aides)}",
        f"lecture du contexte par le LLM   {duree(m.get('lecture', 0))}"
        + (f"  ({u['prompt_tokens']} tokens)" if u.get("prompt_tokens") else ""),
    ]
    if m.get("morceaux_reflexion"):
        lignes.append(f"réflexion                        {duree(m['reflexion'])}"
                      f"  ({m['morceaux_reflexion']} morceaux)")
    lignes.append(f"rédaction de la réponse          {duree(m.get('reponse', 0))}"
                  f"  ({m.get('morceaux_reponse', 0)} morceaux)")
    generation = m.get("reflexion", 0) + m.get("reponse", 0)
    if u.get("completion_tokens") and generation > 0:
        lignes.append(f"vitesse de génération            "
                      f"{u['completion_tokens'] / generation:.1f} tokens/s".replace(".", ",", 1)
                      + f"  ({u['completion_tokens']} tokens)")
    lignes.append(f"TOTAL                            {duree(total)}")
    print("\n=== Timing ===", file=sys.stderr)
    for l in lignes:
        print("  " + l, file=sys.stderr)
