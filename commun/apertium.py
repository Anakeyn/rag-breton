# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Appels à Apertium (br-fr) : lemmatisation, analyse morphologique,
désambiguïsation, dictionnaire bilingue et traduction automatique.

Sert à l'indexation (lemmes des 100 000 phrases) comme à l'analyse des questions
et des leçons.
"""

import glob
import os
import re
import shlex
import shutil
import subprocess
import sys

from commun.config import LOT_APERTIUM


# Catégories glosées (les mots-outils et les noms propres sont ignorés)
CATEGORIES_GLOSEES = {"n": "n.", "adj": "adj.", "adv": "adv.", "vblex": "v."}
RE_UNITE = re.compile(r"\^((?:\\.|[^$\\])*)\$")
RE_SLASH = re.compile(r"(?<!\\)/")
RE_MOT = re.compile(r"[^\W\d_]+(?:'h[^\W\d_]*)*", re.UNICODE)   # garde « c'h »
CARACTERES_RESERVES = re.compile(r"([\[\]\\^$/@<>{}])")


# apostrophes typographiques et signes approchants (prime ′, accent ´, ‘, `)
RE_APOSTROPHES = re.compile("[’ʼ′‘`´]")


def normaliser(texte):
    """Apostrophes typographiques -> apostrophe droite (c’h, c′h -> c'h)."""
    return RE_APOSTROPHES.sub("'", texte)


def trouver_automorf(chemin=None):
    """Cherche l'analyseur breton compilé (br-fr.automorf.bin)."""
    if chemin:
        return chemin
    candidats = glob.glob("/usr/share/apertium/apertium-br-fr/br-fr.automorf.bin")
    candidats += glob.glob("/usr/local/share/apertium/apertium-br-fr/br-fr.automorf.bin")
    return candidats[0] if candidats else None


def lemmes_secours(texte):
    """Sans Apertium : simples mots en minuscules (les mutations ne sont pas gérées)."""
    return RE_MOT.findall(normaliser(texte).lower())


def unites(segment):
    """Unités de la sortie de lt-proc, avec les mots en c'h inconnus recollés.
    lt-proc coupe un mot inconnu à l'apostrophe : « troc'hiñ » donne
    ^troc/*troc$^'/a<vpart>/…$^hiñ/*hiñ$. On reforme ^troc'hiñ/*troc'hiñ$
    pour ne pas indexer ni gloser des morceaux sans sens."""
    brutes = RE_UNITE.findall(segment)
    sortie, i = [], 0
    while i < len(brutes):
        morceaux = RE_SLASH.split(brutes[i])
        if (i + 2 < len(brutes) and len(morceaux) > 1 and morceaux[1].startswith("*")
                and morceaux[0].lower().endswith("c")
                and RE_SLASH.split(brutes[i + 1])[0] == "'"
                and RE_SLASH.split(brutes[i + 2])[0][:1].lower() == "h"):
            forme = morceaux[0] + "'" + RE_SLASH.split(brutes[i + 2])[0]
            sortie.append(f"{forme}/*{forme}")
            i += 3
            continue
        sortie.append(brutes[i])
        i += 1
    return sortie


def analyser_sortie(sortie, connus_seulement=False):
    """Transforme la sortie de lt-proc en liste de lemmes.
    ^benn/penn<n><m><sg>$  -> penn
    ^xyz/*xyz$             -> xyz (mot inconnu : forme de surface),
                              ou ignoré si connus_seulement
    """
    lemmes = []
    # RE_UNITE et non unites() : les lemmes doivent rester ceux de l'index BM25
    for unite in RE_UNITE.findall(sortie):
        morceaux = RE_SLASH.split(unite)
        surface, analyses = morceaux[0], morceaux[1:]
        if not analyses or analyses[0].startswith("*"):
            if connus_seulement:
                continue
            candidats = [surface]
        else:
            # première analyse ; les mots composés sont joints par « + »
            candidats = [m.split("<")[0] for m in analyses[0].split("+")]
        for c in candidats:
            c = c.replace("\\", "").lower()
            if RE_MOT.search(c):
                lemmes.append(c)
    return lemmes


def lemmatiser(textes, automorf, connus_seulement=False):
    """Lemmatise une liste de phrases. Un appel à lt-proc par lot ;
    l'option -z sépare les phrases par le caractère nul.
    connus_seulement : ne garder que les mots reconnus par l'analyseur breton
    (utile pour une question rédigée en français)."""
    if not automorf or not shutil.which("lt-proc"):
        return [lemmes_secours(t) for t in textes]
    resultats = []
    for i in range(0, len(textes), LOT_APERTIUM):
        lot = textes[i:i + LOT_APERTIUM]
        entree = "".join(
            CARACTERES_RESERVES.sub(r"\\\1", normaliser(t)) + "\0" for t in lot
        )
        sortie = subprocess.run(
            ["lt-proc", "-z", "-w", automorf],
            input=entree.encode("utf-8"), capture_output=True, check=True,
        ).stdout.decode("utf-8")
        segments = sortie.split("\0")[: len(lot)]
        segments += [""] * (len(lot) - len(segments))
        resultats.extend(analyser_sortie(s, connus_seulement) for s in segments)
    return resultats


# ---------------------------------------------------------------------------
# Démutation des mots inconnus d'Apertium
# ---------------------------------------------------------------------------
# Apertium ramène lui-même les formes mutées des mots qu'il connaît
# (va zad -> tad). Mais un mot qu'il ignore, il en ignore aussi les mutations :
# « zivoull » reste inconnu même si l'on ajoute « divoull » à complements.tsv.
# Pour chaque mot inconnu, on essaie donc ses formes d'origine possibles ;
# la première connue d'Apertium ou des compléments remplace le mot muté.

# initiale observée -> initiales d'origine possibles (préfixes longs d'abord)
DEMUTATIONS = [
    ("c'h", ["k", "g"]),   # spirante k -> c'h ; adoucissante/mixte g -> c'h
    ("kw",  ["gw"]),       # durcissante gw -> kw
    ("w",   ["gw"]),       # adoucissante/mixte gw -> w
    ("v",   ["b", "m"]),   # adoucissante/mixte b -> v, m -> v
    ("f",   ["p"]),        # spirante p -> f
    ("z",   ["t", "d"]),   # spirante t -> z ; adoucissante d -> z
    ("g",   ["k"]),        # adoucissante k -> g
    ("b",   ["p"]),        # adoucissante p -> b
    ("d",   ["t"]),        # adoucissante t -> d
    ("p",   ["b"]),        # durcissante b -> p
    ("t",   ["d"]),        # durcissante/mixte d -> t
    ("k",   ["g"]),        # durcissante g -> k
]


def formes_d_origine(mot):
    """« zivoull » -> ['tivoull', 'divoull'] ; [] si l'initiale ne mute pas."""
    for prefixe, origines in DEMUTATIONS:
        if mot.startswith(prefixe) and len(mot) > len(prefixe) + 1:
            return [o + mot[len(prefixe):] for o in origines]
    return []


def analyser_mots(mots, automorf):
    """Analyse de mots isolés : {mot: [analyses]} ou {mot: None} si inconnu.
    Un seul appel à lt-proc pour toute la liste."""
    mots = sorted(set(mots))
    if not mots:
        return {}
    entree = "".join(CARACTERES_RESERVES.sub(r"\\\1", m) + "\0" for m in mots)
    sortie = subprocess.run(
        ["lt-proc", "-z", "-w", automorf],
        input=entree.encode("utf-8"), capture_output=True, check=True,
    ).stdout.decode("utf-8")
    segments = sortie.split("\0")[: len(mots)]
    segments += [""] * (len(mots) - len(segments))
    resultat = {}
    for mot, segment in zip(mots, segments):
        u = unites(segment)
        morceaux = RE_SLASH.split(u[0]) if len(u) == 1 else []
        connu = len(morceaux) >= 2 and not morceaux[1].startswith("*")
        resultat[mot] = morceaux[1:] if connu else None
    return resultat


def mots_inconnus(mots, automorf):
    """Sous-ensemble des mots (en minuscules) que l'analyseur ne connaît pas."""
    return {m for m, a in analyser_mots(mots, automorf).items() if a is None}


def demuter_inconnus(textes, automorf, connus=frozenset()):
    """Remplace dans les textes les mots inconnus d'Apertium par leur forme
    démutée, si celle-ci est connue d'Apertium ou figure dans « connus »
    (les entrées bretonnes de complements.tsv, en minuscules).
    À appliquer après variantes.tsv et avant Apertium.
    Renvoie (textes modifiés, {forme mutée: forme d'origine})."""
    if not automorf or not shutil.which("lt-proc"):
        return list(textes), {}
    textes = [normaliser(t) for t in textes]
    mots = {m.lower() for t in textes for m in RE_MOT.findall(t)}
    inconnus = mots_inconnus(mots, automorf)
    candidats = {m: formes_d_origine(m) for m in inconnus}
    a_tester = {c for cs in candidats.values() for c in cs if c not in connus}
    refuses = mots_inconnus(a_tester, automorf)
    remplacements = {}
    for mot, cs in candidats.items():
        for c in cs:
            if c in connus or (c in a_tester and c not in refuses):
                remplacements[mot] = c
                break

    def remplacer(m):
        forme = m.group(0)
        origine = remplacements.get(forme.lower())
        if not origine:
            return forme
        return origine[0].upper() + origine[1:] if forme[0].isupper() else origine

    return [RE_MOT.sub(remplacer, t) for t in textes], remplacements


# ---------------------------------------------------------------------------
# Dérivés inconnus d'Apertium : lazhourien = lazh(añ) + -our + -ien
# ---------------------------------------------------------------------------
# Suffixes du plus long au plus court ; on en retire au plus deux, puis on
# cherche la base telle quelle ou complétée d'une finale verbale (lazh -> lazhañ).
SUFFIXES = [
    ("adigoù", "diminutif, pluriel"), ("adig", "diminutif"),
    ("erezed", "agents féminins, pluriel"), ("ourien", "agents, pluriel"),
    ("erien", "agents, pluriel"), ("adennoù", "actions ponctuelles, pluriel"),
    ("adenn", "action ponctuelle (une fois)"), ("erezh", "activité, métier"),
    ("erez", "agent féminin, ou machine"), ("entez", "nom de qualité"),
    ("igoù", "diminutif, pluriel"), ("adoù", "contenus, pluriel"),
    ("ennoù", "éléments, pluriel"), ("elloù", "instruments, pluriel"),
    ("ded", "nom de qualité"), ("der", "nom de qualité"), ("dez", "nom de qualité"),
    ("iek", "adjectif"), ("aat", "devenir, rendre"), ("ioù", "pluriel"),
    ("ien", "pluriel"), ("idi", "pluriel"), ("ier", "pluriel"),
    ("enn", "un élément (singulatif)"), ("ell", "instrument"), ("our", "agent"),
    ("ig", "diminutif"), ("ad", "contenu, quantité"), ("er", "agent"),
    ("ek", "adjectif (abondance)"), ("us", "adjectif"), ("iz", "habitants"),
    ("où", "pluriel"), ("ed", "pluriel"), ("iñ", "verbe"), ("añ", "verbe"),
    ("et", "participe passé"),
]
PREFIXES = [
    ("dreist", "au-dessus, sur-"), ("peur", "complètement"), ("gour", "sous, petit"),
    ("dis", "contraire"), ("diz", "contraire"), ("dam", "à demi"), ("rak", "avant, pré-"),
    ("ken", "avec, co-"), ("ad", "de nouveau, re-"), ("di", "sans, privatif"),
]
FINALES_VERBALES = ["añ", "iñ", "out", "a", "at"]
# suffixes qui se greffent sur un radical verbal (lazh-our <- lazhañ) : seuls
# ceux-là autorisent à compléter la base d'une finale verbale
SUFFIXES_DE_VERBE = {"erezed", "ourien", "erien", "adennoù", "adenn", "erezh", "erez",
                     "our", "er", "ell", "elloù", "et"}
RACINE_MIN = 3


def decompositions(mot):
    """Bases possibles d'un mot : [(base, [affixes])], les plus simples d'abord."""
    resultat = []

    def suffixes(radical, affixes, profondeur):
        for suffixe, sens in SUFFIXES:
            if radical.endswith(suffixe) and len(radical) - len(suffixe) >= RACINE_MIN:
                racine = radical[: -len(suffixe)]
                aff = [f"-{suffixe} ({sens})"] + affixes
                resultat.append((racine, aff))
                if suffixe in SUFFIXES_DE_VERBE:
                    resultat.extend((racine + f, aff) for f in FINALES_VERBALES)
                if profondeur > 1:
                    suffixes(racine, aff, profondeur - 1)

    suffixes(mot, [], 2)
    for prefixe, sens in PREFIXES:
        if mot.startswith(prefixe) and len(mot) - len(prefixe) >= RACINE_MIN:
            for reste in [mot[len(prefixe):]] + formes_d_origine(mot[len(prefixe):]):
                aff = [f"{prefixe}- ({sens})"]
                resultat.append((reste, aff))
                avant = len(resultat)
                suffixes(reste, [], 1)
                resultat[avant:] = [(b, aff + a) for b, a in resultat[avant:]]
    # moins d'affixes d'abord, puis la base la plus longue
    return sorted(dict.fromkeys((b, tuple(a)) for b, a in resultat),
                  key=lambda x: (len(x[1]), -len(x[0])))


def lemme_coherent(candidat, analyse):
    """Le lemme d'Apertium ressemble-t-il à la base essayée ? Au moins 3 lettres,
    et même début que la base ou qu'une de ses formes démutées : écarte
    raeed -> rae -> ober (rae est une forme de ober) et c'houziz -> « c »."""
    lemme = normaliser(analyse.split("+")[0].split("<")[0]).lower().replace("\\", "").replace("#", "")
    if len(lemme) < RACINE_MIN:
        return False
    return any(f[:3] == lemme[:3] for f in [candidat] + formes_d_origine(candidat))


def derivations(mots, automorf, connus=frozenset()):
    """Pour des mots inconnus d'Apertium : {mot: (base, analyses de la base ou
    None si elle vient de « connus », [affixes])}. La base doit être connue
    d'Apertium ou figurer dans « connus » (complements.tsv). Les mots mutés
    sont d'abord démutés (vourc'hadig -> bourc'hadig -> bourc'h + -ad + -ig)."""
    if not automorf or not shutil.which("lt-proc"):
        return {}
    candidats = {}
    for mot in {m.lower() for m in mots}:
        liste = []
        for rang, forme in enumerate([mot] + formes_d_origine(mot)):
            note = [] if forme == mot else [f"mutation de {forme}"]
            liste += [(len(a), rang, -len(b), b, note + list(a)) for b, a in decompositions(forme)]
        # le moins d'affixes d'abord, démuté ou non (goueled + -iñ avant c'houel + -ed + -iñ)
        candidats[mot] = [(b, a) for *_, b, a in sorted(liste)]
    a_tester = {b for liste in candidats.values() for b, _ in liste if b not in connus}
    analyses = analyser_mots(a_tester, automorf)
    resultat = {}
    for mot, liste in candidats.items():
        for base, affixes in liste:
            if base in connus:
                resultat[mot] = (base, None, affixes)
                break
            retenues = [a for a in (analyses.get(base) or []) if lemme_coherent(base, a)]
            if retenues:
                resultat[mot] = (base, retenues, affixes)
                break
    return resultat


RE_ETIQUETTES = re.compile(r"<([^>]+)>")


def trouver_autobil(automorf):
    """Le dictionnaire bilingue compilé est rangé à côté de l'analyseur."""
    if not automorf:
        return None
    chemin = os.path.join(os.path.dirname(automorf), "br-fr.autobil.bin")
    return chemin if os.path.exists(chemin) else None


def analyses_detaillees(textes, automorf, inclure_inconnus=False):
    """Pour chaque phrase : liste de (forme, [analyses]) où chaque analyse vaut
    par exemple « morlu<n><m><sg> » ; les mots composés (d'ar = da + ar) ont
    des analyses jointes par « + ». Mots inconnus d'Apertium : ignorés, ou
    rendus comme (forme, None) si inclure_inconnus.
    Toutes les analyses sont gardées : un mot ambigu (zo = bezañ ou teiñ)
    ne doit pas être glosé au hasard."""
    entree = "".join(CARACTERES_RESERVES.sub(r"\\\1", normaliser(t)) + "\0" for t in textes)
    sortie = subprocess.run(
        ["lt-proc", "-z", "-w", automorf],
        input=entree.encode("utf-8"), capture_output=True, check=True,
    ).stdout.decode("utf-8")
    segments = sortie.split("\0")[: len(textes)]
    segments += [""] * (len(textes) - len(segments))
    resultats = []
    for segment in segments:
        mots = []
        for unite in unites(segment):
            morceaux = RE_SLASH.split(unite)
            forme, analyses = morceaux[0].replace("\\", ""), morceaux[1:]
            if not analyses or analyses[0].startswith("*"):
                if inclure_inconnus and RE_MOT.search(forme):
                    mots.append((forme, None))
                continue
            mots.append((forme, analyses))
        resultats.append(mots)
    return resultats


def analyses_glosables(analyses):
    """Analyses à gloser pour un mot, ou [] si le mot doit être ignoré :
    - mot composé (+) ou mot-outil (article, particule, bezañ…) ;
    - mot ambigu dont UNE des analyses est un mot-outil : sans désambiguïsation,
      on ne sait pas laquelle est la bonne (zo : bezañ ou teiñ ?)."""
    retenues = []
    for a in analyses:
        etiquettes = RE_ETIQUETTES.findall(a)
        if "+" in a or not etiquettes or etiquettes[0] not in CATEGORIES_GLOSEES:
            return []
        if a not in retenues:
            retenues.append(a)
    return retenues


def traduire_lemmes(analyses, autobil):
    """Dictionnaire bilingue : « morlu<n><m><sg> » -> « marine ».
    Une analyse par ligne, un seul appel à lt-proc -b."""
    uniques = sorted(set(analyses))
    if not uniques:
        return {}
    entree = "".join(f"^{a}$\n" for a in uniques)
    sortie = subprocess.run(
        ["lt-proc", "-b", autobil],
        input=entree.encode("utf-8"), capture_output=True, check=True,
    ).stdout.decode("utf-8")
    traductions = {}
    for unite in RE_UNITE.findall(sortie):
        morceaux = RE_SLASH.split(unite)
        source, cibles = morceaux[0], morceaux[1:]
        francais = []
        for c in cibles:
            if c.startswith("@") or not c:
                continue                       # @ : absent du dictionnaire bilingue
            mot = c.split("<")[0].replace("\\", "").replace("_", " ").replace("#", "")
            if mot and mot not in francais:
                francais.append(mot)
        if francais:
            traductions[source] = " / ".join(francais[:3])
    return traductions


def gloses(textes, automorf, autobil):
    """Pour chaque phrase, une ligne « vorlu = morlu (n. m.) -> marine ; … »."""
    if not automorf or not autobil or not shutil.which("lt-proc"):
        return [""] * len(textes)
    toutes = analyses_detaillees(textes, automorf)
    dico = traduire_lemmes(
        [a for mots in toutes for _, analyses in mots for a in analyses_glosables(analyses)],
        autobil,
    )
    lignes = []
    for mots in toutes:
        vus, morceaux = set(), []
        for forme, analyses in mots:
            if len(forme) < 2 or forme.lower() in vus:
                continue                       # « c » de c'est, doublons
            sens = []                          # une entrée par lemme distinct
            for a in analyses_glosables(analyses):
                etiquettes = RE_ETIQUETTES.findall(a)
                lemme = a.split("<")[0].replace("\\", "").replace("_", " ")
                categorie = CATEGORIES_GLOSEES[etiquettes[0]]
                if etiquettes[0] == "n" and len(etiquettes) > 1 and etiquettes[1] in ("m", "f"):
                    categorie += f" {etiquettes[1]}."
                entree = (lemme, categorie, dico.get(a, "?"))
                if entree not in sens:
                    sens.append(entree)
            if not sens:
                continue
            vus.add(forme.lower())
            texte = " ou ".join(f"{l} ({c}) -> {f}" for l, c, f in sens)
            if len(sens) == 1 and sens[0][0].lower() == forme.lower():
                morceaux.append(texte)
            else:
                morceaux.append(f"{forme} = {texte}")
        lignes.append(" ; ".join(morceaux))
    return lignes


def analyses_desambiguisees(textes, automorf):
    """Une seule analyse par mot : la chaîne d'Apertium jusqu'à l'étiqueteur,
    comme pour sa propre traduction (lt-proc | cg-proc | apertium-tagger).
    Renvoie None si cette chaîne n'est pas disponible."""
    dossier = os.path.dirname(automorf)
    rlx = os.path.join(dossier, "br-fr.rlx.bin")
    prob = os.path.join(dossier, "br-fr.prob")
    if not (os.path.exists(rlx) and os.path.exists(prob)
            and shutil.which("cg-proc") and shutil.which("apertium-tagger")):
        return None
    commande = (f"lt-proc -w {shlex.quote(automorf)} | cg-proc {shlex.quote(rlx)} "
                f"| apertium-tagger -g -p {shlex.quote(prob)}")
    resultats = []
    for texte in textes:                        # une phrase par appel : ~0,2 s chacune
        sortie = subprocess.run(
            ["bash", "-c", "set -o pipefail; " + commande],
            input=CARACTERES_RESERVES.sub(r"\\\1", normaliser(texte)).encode("utf-8"),
            capture_output=True, check=True, timeout=60,
        ).stdout.decode("utf-8")
        mots = []
        for unite in unites(sortie):
            morceaux = RE_SLASH.split(unite)
            forme = morceaux[0].replace("\\", "")
            analyse = morceaux[1] if len(morceaux) > 1 else "*"
            # étiquettes syntaxiques de cg-proc (<@+FMAINV>, <@SUBJ→>…) : elles
            # contiennent parfois « + » et faussent l'analyse et le dictionnaire
            analyse = RE_FONCTION_CG.sub("", analyse)
            if analyse.startswith("*"):
                if RE_MOT.search(forme):
                    mots.append((forme, None))
                continue
            mots.append((forme, [analyse]))
        resultats.append(mots)
    return resultats


RE_FONCTION_CG = re.compile(r"<@[^>]*>")


def traduction_apertium(textes):
    """Traduction automatique br -> fr, une phrase par ligne (apertium -u :
    sans les marques * des mots inconnus). Renvoie des chaînes vides si
    Apertium n'est pas disponible."""
    if not shutil.which("apertium") or not textes:
        return [""] * len(textes)

    def appel(texte):
        return subprocess.run(
            ["apertium", "-u", "br-fr"], input=texte.encode("utf-8"),
            capture_output=True, check=True, timeout=120,
        ).stdout.decode("utf-8")

    try:
        lignes = appel("\n".join(normaliser(t) for t in textes) + "\n").strip("\n").split("\n")
        if len(lignes) != len(textes):           # découpage perdu : une phrase à la fois
            lignes = [appel(normaliser(t)).strip() for t in textes]
        return [l.strip() for l in lignes]
    except (subprocess.SubprocessError, OSError) as e:
        print(f"(traduction Apertium indisponible : {e})", file=sys.stderr)
        return [""] * len(textes)
