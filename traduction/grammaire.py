# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Grammaire bretonne appliquée à l'analyse d'Apertium : mutations, règles,
tournures, compléments personnels (complements.tsv) et mot à mot français.

C'est ici que s'ajoutent les nouvelles règles.
"""

import os
import re
import shutil
import subprocess
import sys

from commun.apertium import (
    RE_ETIQUETTES,
    analyses_desambiguisees,
    analyses_detaillees,
    derivations,
    formes_d_origine,
    normaliser,
    traduire_lemmes,
)


# Étiquettes Apertium -> français. Une étiquette absente est gardée telle quelle.
ETIQUETTES_FR = {
    "n": "nom", "np": "nom propre", "adj": "adjectif", "adv": "adverbe",
    "vblex": "verbe", "vbser": "verbe être", "vbhaver": "verbe avoir",
    "vbloc": "verbe être (situation, lieu)", "cpl": "complément personnel",
    "rgl": "règle du script",
    "vbmod": "verbe modal", "vaux": "auxiliaire", "vpart": "particule verbale",
    "pr": "préposition", "prn": "pronom", "det": "déterminant", "num": "numéral",
    "cnjcoo": "conjonction de coordination", "cnjsub": "conjonction de subordination",
    "cnjadv": "conjonction adverbiale", "ij": "interjection", "preadv": "pré-adverbe",
    "def": "défini", "ind": "indéfini", "dem": "démonstratif", "pos": "possessif",
    "itg": "interrogatif", "rel": "relatif", "pers": "personnel", "obj": "objet",
    "subj": "sujet", "ref": "réfléchi", "qnt": "quantificateur", "ord": "ordinal",
    "m": "masculin", "f": "féminin", "mf": "masculin ou féminin", "nt": "neutre",
    "sg": "singulier", "pl": "pluriel", "sp": "singulier ou pluriel", "du": "duel",
    "coll": "collectif", "sgv": "singulatif",
    "p1": "1re personne", "p2": "2e personne", "p3": "3e personne", "impers": "impersonnel",
    "inf": "infinitif", "pri": "présent", "pii": "imparfait", "past": "passé simple",
    "fti": "futur", "cni": "conditionnel", "prs": "subjonctif", "imp": "impératif",
    "prh": "présent d'habitude", "pih": "imparfait d'habitude",
    "pp": "participe passé", "ger": "gérondif", "hab": "habituel",
    "comp": "comparatif", "sup": "superlatif", "excl": "exclamatif",
    "top": "lieu", "ant": "prénom", "cog": "nom de famille", "org": "organisation",
    "al": "autre nom propre", "prot": "protégé, non traduit",
}
CATEGORIES_IGNOREES = {"sent", "cm", "lpar", "rpar", "lquot", "rquot", "guio", "apos", "num"}

# Mutation déduite de la première lettre : (initiale du lemme, initiale de la forme)
MUTATIONS = [
    ("gw", "kw", "durcissante"), ("gw", "w", "adoucissante ou mixte"),
    ("k", "c'h", "spirante"), ("k", "g", "adoucissante"),
    ("t", "z", "spirante"), ("t", "d", "adoucissante"),
    ("p", "f", "spirante"), ("p", "b", "adoucissante"),
    ("g", "c'h", "adoucissante ou mixte"), ("g", "k", "durcissante"),
    ("d", "z", "adoucissante"), ("d", "t", "durcissante ou mixte"),
    ("b", "v", "adoucissante ou mixte"), ("b", "p", "durcissante"),
    ("m", "v", "adoucissante ou mixte"),
]


def mutation(forme, lemme):
    """« spirante (k → c'h) » si la forme est le lemme muté, sinon "".
    Pour un mot composé (sal-glas / sal-klas), chaque élément est comparé :
    « adoucissante (k → g), dans le 2e élément (klas → glas) »."""
    f, l = normaliser(forme).lower(), normaliser(lemme).lower().replace("#", "")
    if f == l:
        return ""
    elements_f, elements_l = f.split("-"), l.split("-")
    if len(elements_f) > 1 and len(elements_f) == len(elements_l):
        trouvees = []
        for rang, (ef, el) in enumerate(zip(elements_f, elements_l), start=1):
            m = mutation_simple(ef, el)
            if m:
                trouvees.append(m if rang == 1 else
                                f"{m}, dans le {rang}e élément ({el} → {ef})")
        return " ; ".join(trouvees)
    return mutation_simple(f, l)


def mutation_simple(f, l):
    """Mutation de l'initiale d'un mot simple (formes déjà en minuscules)."""
    if f == l:
        return ""
    for initiale_l, initiale_f, nom in MUTATIONS:
        if l.startswith(initiale_l) and f.startswith(initiale_f):
            if f[len(initiale_f):][:3] == l[len(initiale_l):][:3]:
                return f"{nom} ({initiale_l} → {initiale_f})"
    return ""


def decrire(analyse, dico):
    """« ober<vblex><past><p3><sg> » -> « ober (verbe, passé simple, 3e personne,
    singulier) -> faire ». Les mots composés sont décrits partie par partie."""
    parties = []
    for partie in analyse.split("+"):
        etiquettes = RE_ETIQUETTES.findall(partie)
        lemme = partie.split("<")[0].replace("\\", "").replace("_", " ").replace("#", "")
        texte = lemme
        if etiquettes:
            texte += " (" + ", ".join(ETIQUETTES_FR.get(e, e) for e in etiquettes) + ")"
        if dico.get(partie):
            texte += f" -> {dico[partie]}"
        parties.append(texte)
    return " + ".join(parties)


VERBES_CONJUGUES = {"vblex", "vbser", "vbhaver", "vbmod", "vbloc"}
TEMPS = {"pri", "pii", "past", "fti", "cni", "prs", "prh", "pih"}


def premieres_etiquettes(analyses):
    return RE_ETIQUETTES.findall(analyses[0].split("+")[0]) if analyses else []


def alertes(mots):
    """Fautes probables repérables par la grammaire : un article (ar, an, al)
    juste devant un verbe conjugué, où l'on attend la particule « a ».
    Renvoie des dictionnaires : position du mot fautif, texte pour l'utilisateur
    (en citant le breton), texte pour le LLM (en français seulement)."""
    resultat = []
    for k, ((f1, a1), (f2, a2)) in enumerate(zip(mots, mots[1:])):
        if not a1 or not a2 or f1.lower() not in ("ar", "an", "al"):
            continue
        e2 = premieres_etiquettes(a2)
        if any(RE_ETIQUETTES.findall(a.split("+")[0])[:1] == ["det"] for a in a1) \
                and e2 and e2[0] in VERBES_CONJUGUES and TEMPS & set(e2):
            resultat.append({
                "ignorer": k,
                "br": f"ALERTE « {f1} {f2} » : article devant un verbe conjugué, inattendu ; "
                      f"faute de saisie probable pour « a {f2} » (particule verbale).",
                "fr": "L'original contient une faute de saisie (un article à la place "
                      "d'une particule verbale) : elle est déjà retirée du mot à mot.",
            })
    return resultat


# Titres suivis d'un nom en apposition (Aotrou Kelenner = Monsieur le Professeur)
TITRES = {"aotrou", "itron", "dimezell", "aotrounez", "itronezed"}

# Ordinaux, au cas où le dictionnaire bilingue ne les traduit pas
ORDINAUX = {"kentañ": "premier", "eil": "second", "trede": "troisième", "pevare": "quatrième",
            "pempvet": "cinquième", "c'hwec'hvet": "sixième", "seizhvet": "septième",
            "eizhvet": "huitième", "navet": "neuvième", "dekvet": "dixième"}
PRONOMS = {("p1", "sg"): "je", ("p2", "sg"): "tu", ("p3", "sg"): "il",
           ("p1", "pl"): "nous", ("p2", "pl"): "vous", ("p3", "pl"): "ils/elles"}
# « être » au présent, pour les notes sur l'ordre des mots
ETRE_PRESENT = {("p1", "sg"): "je suis", ("p2", "sg"): "tu es", ("p1", "pl"): "nous sommes",
                ("p2", "pl"): "vous êtes", ("p3", "pl"): "ils sont"}
# formes de bezañ qui signifient « il y a » : ez eus, 'z eus, n'eus (ket)
RE_IL_Y_A = re.compile(r"(?:ez |'?z )?eus")
# explication affichée pour chaque règle du script (analyse « …<rgl> »)
RAISONS_REGLES = {
    "Du<rgl>": "majuscule en milieu de phrase = mois",
    "e<rgl>": "e devant un nom = préposition",
    "e-lieu<rgl>": "e devant un nom de lieu = préposition",
    "rgl-pron": "pronom : complément du verbe, ou « de » + pronom après un nom",
    "tre<rgl-entrer>": "mont tre, dont tre = entrer",
    "da<rgl>": "da devant un nom propre = préposition",
    "hini<rgl>": "hini = pronom (celui, celle), pas l'adverbe « aucun »",
    "n'<rgl>": "n' devant un nom = article élidé (an), pas la négation",
    "e Du<rgl>": "Du devant une année = novembre",
    "ger<rgl>": "ger après un déterminant = le nom « mot », pas ker muté (cher)",
    "e-pos<rgl>": "e après d' (d'e) = possessif « son, sa », pas la préposition",
    "o-pos<rgl>": "o après d' (d'o) = possessif « leur », pas « en train de »",
    "rgl-refl": "en em + verbe = verbe pronominal (se …)",
    "rgl-que": "ken + adjectif … ha(g) = aussi … que",
    "rgl-proch": "a zeu après un jour ou une période = prochain",
    "rgl-revoir": "Ken en tête de phrase devant un article ou un moment = « à » (au revoir)",
    "'benn<rgl>": "'benn = a-benn (à, pour : 'benn arc'hoazh = à demain)",
    "'ta<rgl>": "'ta = eta (donc)",
    "deus<rgl-imp>": "Deus en tête, sans nom propre après = impératif de dont (viens)",
    "rgl-gerondif": "en ur + verbe = gérondif (en …ant : en ur zivoull = en démaillant)",
    "rgl-certains": "nom + 'zo = certains (kanaouennoù 'zo = certaines chansons)",
    "rgl-ilya": "durée + 'zo = il y a (ur bloaz 'zo = il y a un an)",
    "rgl-agent": "gant après un passif = par (agent)",
    "rgl-cause": "gant ma + verbe = parce que, du fait que (et non « avec mon »)",
    "rgl-part": "'z après une conjonction = particule verbale (ez), pas le pronom « te »",
    "rgl-les": "o + mutation spirante (t → z, k → c'h, p → f) = pronom objet « les »",
}
# verbes de mouvement ou d'état conjugués avec bezañ aux temps composés :
# oa aet = était allé, zo tremenet = est passé (pas un passif)
VERBES_AUX_BEZAN = {"mont", "dont", "erruout", "tremen", "distreiñ", "chom", "kouezhañ",
                    "mervel", "diskenn", "pignat", "degouezhout", "donet", "krogiñ"}
TEMPS_COMPOSES = {"pri": "passé composé (est allé…)", "pii": "plus-que-parfait (était allé…)",
                  "past": "passé antérieur (fut allé…)", "fti": "futur antérieur (sera allé…)",
                  "cni": "conditionnel passé (serait allé…)", "prh": "passé composé (est allé…)",
                  "pih": "plus-que-parfait (était allé…)"}


def lemme_de(analyses):
    """Lemme de la première analyse (« bezañ<vblex><pri>… » -> « bezañ »)."""
    return (analyses or [""])[0].split("+")[0].split("<")[0].replace("\\", "").lower()


def categorie(analyses):
    """Catégorie de la première analyse, sans la marque <cpl> des compléments
    personnels (sal-glas<cpl><n> -> n). Liste vide si inconnue."""
    e = premieres_etiquettes(analyses)
    return e[1:2] if e[:1] == ["cpl"] else e[:1]


def personne_nombre(etiquettes):
    return (next((t for t in etiquettes if t in ("p1", "p2", "p3")), ""),
            next((t for t in etiquettes if t in ("sg", "pl")), ""))


def precedent(mots, k):
    """Indice du mot qui précède k, en sautant les particules verbales (a, e) ;
    None en début de phrase ou après une ponctuation."""
    j = k - 1
    while j >= 0 and mots[j][1] and premieres_etiquettes(mots[j][1])[:1] == ["vpart"]:
        j -= 1
    if j < 0 or (mots[j][1] and premieres_etiquettes(mots[j][1])[:1]
                 and premieres_etiquettes(mots[j][1])[0] in CATEGORIES_IGNOREES):
        return None
    return j


# catégories qui ne font jamais partie d'un nom de lieu (Deus, E, Kement…)
HORS_TOPONYME = {"pr", "cnjcoo", "cnjsub", "cnjadv", "det", "prn", "vpart", "adv",
                 "ij", "num", "preadv", "cpl", "rgl"}
ARTICLES = {"ar", "an", "al"}


def a_majuscule(forme):
    """Penn, Benn, Milin : majuscule initiale suivie de minuscules
    (ni « E », ni un mot tout en capitales comme « HOR »)."""
    return forme[:1].isupper() and any(c.islower() for c in forme)


def regrouper_toponymes(mots, dico):
    """Une suite de mots à majuscule en milieu de phrase, éventuellement reliés
    par ar / an / al, est un nom de lieu à garder tel quel : Penn ar Men,
    Milin Aod, Santa Roz. Sans cette règle, chaque mot était traduit
    (« tête », « moulin », « côte »). Le premier mot de la phrase n'est jamais
    pris (majuscule de début de phrase), ni un titre (Aotrou Kelenner), ni une
    suite qui finit par un adjectif (Whisky Du = le whisky noir)."""
    def candidat(k, debut=True):
        # au début, catégories strictes ; à la suite d'un article (Penn ar Men),
        # tout mot à majuscule convient sauf préposition, conjonction, déterminant
        forme, analyses = mots[k]
        e = premieres_etiquettes(analyses) if analyses else []
        exclues = HORS_TOPONYME if debut else {"pr", "cnjcoo", "cnjsub", "det", "vpart",
                                               "prn", "cpl", "rgl"}
        return (a_majuscule(forme) and not (e[:1] and e[0] in exclues)
                and not (e[:1] and e[0] in CATEGORIES_IGNOREES))
    resultat, k = [], 0
    while k < len(mots):
        if k > 0 and candidat(k) and lemme_de(mots[k][1]) not in TITRES:
            j, fin = k + 1, k
            while j < len(mots):
                if candidat(j) or (j > k and mots[j - 1][0].lower() in ARTICLES
                                   and candidat(j, debut=False)):
                    fin, j = j, j + 1
                elif normaliser(mots[j][0]).lower() in ARTICLES and j + 1 < len(mots) \
                        and candidat(j + 1, debut=False):
                    j += 1
                else:
                    break
            # un adjectif final décrit le nom, ce n'est pas un nom de lieu :
            # Whisky Du = le whisky noir (titre de chanson)
            if fin > k and categorie(mots[fin][1]) == ["adj"]:
                fin = k
            if fin > k:
                forme = " ".join(f for f, _ in mots[k:fin + 1])
                analyse = f"{forme}<np><top>"
                dico[analyse] = forme
                resultat.append((forme, [analyse]))
                k = fin + 1
                continue
        resultat.append(mots[k])
        k += 1
    return resultat


def nom_de_lieu(mot):
    """Nom de lieu reconnu (Penn ar Men, Brest) ou mot inconnu à majuscule
    (Gorreloù, Rejumant) : l'article breton qui le précède ne se traduit pas."""
    forme, analyses = mot
    if analyses is None:
        return a_majuscule(forme)
    e = premieres_etiquettes(analyses)
    return e[:1] == ["np"] and "top" in e


# premier mot d'un nom de lieu composé (Porzh Laouen, Beg Leide, Toull ar Soner,
# Kostez Beuzeg, Ar Garreg Wenn), y compris après mutation (da Borzh Douarnenez)
MOTS_DE_LIEU = {"porzh", "borzh", "vorzh", "pors", "beg", "veg", "toull", "doull", "penn",
                "benn", "kostez", "gostez", "enez", "plas", "leur", "poull", "boull", "roz",
                "menez", "venez", "traon", "draon", "milin", "vilin", "lann", "aod", "karreg",
                "garreg", "c'harreg", "maen", "vaen", "koad", "goad"}


def capitales(forme):
    """AN, DU, LOSQ, PiCHAVANT : mot écrit (presque) tout en capitales,
    au moins deux lettres ; une seule minuscule tolérée (faute de frappe)."""
    lettres = [c for c in forme if c.isalpha()]
    majuscules = sum(c.isupper() for c in lettres)
    return len(lettres) >= 2 and majuscules >= 2 and majuscules >= len(lettres) - 1


def est_ponctuation(mot):
    e = premieres_etiquettes(mot[1]) if mot[1] else []
    return bool(e[:1]) and e[0] in CATEGORIES_IGNOREES


TOPONYME_MOTS_MAX = 5


def toponyme_kerofis(mots, k, toponymes):
    """Plus long nom de lieu de KerOfis commençant au mot k : (fin, nom français)
    ou None. Tous les mots, sauf les articles, doivent avoir une majuscule ; un nom
    d'un seul mot n'est pris qu'en milieu de phrase et s'il est inconnu d'Apertium
    ou analysé comme nom propre (pas « Tad », « Mor » en majuscule)."""
    if not toponymes or not a_majuscule(mots[k][0]):
        return None
    for fin in range(min(len(mots), k + TOPONYME_MOTS_MAX) - 1, k - 1, -1):
        groupe = mots[k:fin + 1]
        if any(est_ponctuation(m) for m in groupe):
            continue
        if not all(a_majuscule(f) or normaliser(f).lower() in ARTICLES or f in ("'", "’")
                   for f, _ in groupe):
            continue
        if fin == k and (k == 0 or (mots[k][1] and premieres_etiquettes(mots[k][1])[:1] != ["np"])):
            continue
        formes = [f for f, _ in groupe]
        for premier in [formes[0]] + demutations(formes[0]):
            francais = toponymes.get(cle_toponyme(" ".join([premier] + formes[1:])))
            if francais is not None:
                return fin, francais      # "" : nom de lieu sûr, nom français incertain
    return None


def demutations(forme):
    """Formes non mutées possibles d'un nom à majuscule : da Veg Leide -> Beg Leide,
    da Borzh Douarnenez -> Porzh Douarnenez (KerOfis ne donne que la forme de base).
    Table des mutations : DEMUTATIONS de commun/apertium.py."""
    return [o.capitalize() for o in formes_d_origine(normaliser(forme).lower())]


ARTICLES_FRANCAIS = {"Le", "La", "Les"}


def proteger_noms_propres(mots, dico, toponymes=None):
    """Noms propres à garder tels quels, avant les règles et complements.tsv :
    - nom de lieu de KerOfis (toponymes.tsv), avec son nom français :
      Beg Leide (Pointe de Leyde), Treboull (Tréboul) ;
    - prénom suivi d'un nom de famille à majuscule : Yann Gwenn, Rozenn Milin ;
    - prénom suivi d'un nom de famille en capitales : Jakez AN DU, Jean PiCHAVANT
      (sans cette règle, « DU » devenait « novembre ») ;
    - nom de lieu commençant par un mot de lieu, même en début de phrase et même
      s'il finit par un adjectif : Porzh Laouen (pas « Port joyeux »), Beg Leide,
      Toull ar Soner, Ar Garreg Wenn ;
    - liste d'au moins trois noms à majuscule séparés par des virgules : Belbeoc'h,
      Hirwazh, Noeig… (sans cette règle, Marc'heg devenait « chevalier »).
    Les capitales d'insistance restent traduites (HOR c'helenner, MA skolidi) :
    elles ne suivent pas un prénom."""
    def unite(debut, fin, sorte):
        forme = " ".join(f for f, _ in mots[debut:fin + 1])
        analyse = f"{forme}<np><{sorte}><prot>"
        dico[analyse] = forme
        return (forme, [analyse])

    groupes, k = [], 0
    while k < len(mots):
        forme, analyses = mots[k]
        # article français devant une majuscule : nom de bateau ou de lieu français
        # (Le Petit André, la Marguerite, le Marie-Thérèse) ; sans cette règle,
        # « Le » devenait le mot breton le (serment)
        if (forme in ARTICLES_FRANCAIS or (k > 0 and forme in {"le", "la", "les"})) \
                and k + 1 < len(mots) and a_majuscule(mots[k + 1][0]) \
                and not est_ponctuation(mots[k + 1]):
            j = k + 1
            while j + 1 < len(mots) and a_majuscule(mots[j + 1][0]) and not est_ponctuation(mots[j + 1]):
                j += 1
            groupes.append(unite(k, j, "al"))
            k = j + 1
            continue
        # nom de lieu connu de KerOfis : Beg Leide = Pointe de Leyde, Treboull = Tréboul
        trouve = toponyme_kerofis(mots, k, toponymes)
        if trouve:
            fin, francais = trouve
            m = unite(k, fin, "top")
            if francais and cle_toponyme(francais) != cle_toponyme(m[0]):
                dico[m[1][0]] = f"{m[0]} (en français : {francais})"
            groupes.append(m)
            k = fin + 1
            continue
        # prénom + nom de famille à majuscule : Yann Gwenn (pas « Jean blanc »),
        # Rozenn Milin, Padrig An Habask ; le prénom doit être un prénom pour
        # Apertium (<np><ant>) : un mot inconnu ne suffit pas (ar Wiski Du reste
        # traduit « le whisky noir »), et jamais un titre (Aotrou Kelenner)
        if (a_majuscule(forme) and not capitales(forme) and analyses
                and premieres_etiquettes(analyses)[:2] == ["np", "ant"]
                and normaliser(forme).lower() not in TITRES):
            j = k
            while j + 1 < len(mots):
                suivant = mots[j + 1]
                if (normaliser(suivant[0]).lower() in ("an", "ar", "al", "le", "la", "de", "du")
                        and j + 2 < len(mots) and a_majuscule(suivant[0])
                        and a_majuscule(mots[j + 2][0]) and not est_ponctuation(mots[j + 2])):
                    j += 2
                elif (a_majuscule(suivant[0]) and not est_ponctuation(suivant)
                      and not (suivant[1] and premieres_etiquettes(suivant[1])[:1] == ["np"]
                               and "top" in premieres_etiquettes(suivant[1]))):
                    j += 1
                else:
                    break
            if j > k:
                groupes.append(unite(k, j, "cog"))
                k = j + 1
                continue
        # prénom + nom de famille en capitales
        if (a_majuscule(forme) and k + 1 < len(mots) and capitales(mots[k + 1][0])
                and (analyses is None or premieres_etiquettes(analyses)[:1] == ["np"])):
            j = k + 1
            while j + 1 < len(mots) and capitales(mots[j + 1][0]):
                j += 1
            groupes.append(unite(k, j, "cog"))
            k = j + 1
            continue
        # nom de lieu : [Ar] + mot de lieu + mots à majuscule (ou article + majuscule)
        i = k + 1 if (normaliser(forme).lower() in ARTICLES and a_majuscule(forme)) else k
        if (i < len(mots) and a_majuscule(mots[i][0])
                and normaliser(mots[i][0]).lower() in MOTS_DE_LIEU):
            j, fin = i, None
            while j + 1 < len(mots):
                suivant = mots[j + 1]
                if a_majuscule(suivant[0]) and not est_ponctuation(suivant):
                    j = fin = j + 1
                elif (normaliser(suivant[0]).lower() in ARTICLES and j + 2 < len(mots)
                      and a_majuscule(mots[j + 2][0])):
                    j = fin = j + 2
                else:
                    break
            if fin is not None:
                groupes.append(unite(k, fin, "top"))
                k = fin + 1
                continue
        groupes.append(mots[k])
        k += 1

    # listes : Nom , Nom , Nom (au moins trois)
    def nom(m):
        return a_majuscule(m[0]) and lemme_de(m[1]) not in TITRES and not est_ponctuation(m)
    def virgule(m):
        return m[0] == ","
    dans_liste, k = set(), 0
    while k < len(groupes):
        if nom(groupes[k]):
            membres, j = [k], k
            while j + 2 < len(groupes) and virgule(groupes[j + 1]) and nom(groupes[j + 2]):
                j += 2
                membres.append(j)
            if len(membres) >= 3:
                dans_liste.update(membres)
            k = j + 1
        else:
            k += 1
    resultat = []
    for k, m in enumerate(groupes):
        deja = m[1] and "<prot>" in m[1][0]
        if k in dans_liste and not deja:
            analyse = f"{m[0]}<np><al><prot>"
            dico[analyse] = m[0]
            m = (m[0], [analyse])
        resultat.append(m)
    return resultat


def protege(mot):
    return bool(mot[1]) and "<prot>" in mot[1][0]


def complement_de_nom(mots, k, ignorer):
    """Le mot k (nom propre, ou mot inconnu à majuscule) complète-t-il le nom
    qui le précède, éventuellement par-dessus un adjectif ou un participe ?
    breur Katell = le frère DE Katell ; kelenner Magnolia = le professeur DE
    Magnolia (l'association). Jamais après un titre (Aotrou Yann)."""
    j = k - 1
    ej = premieres_etiquettes(mots[j][1]) if mots[j][1] else []
    # entre les deux : un adjectif, un participe (divarchet) ou un mot de complements.tsv
    if categorie(mots[j][1]) == ["adj"] or "pp" in ej or ej[:1] == ["cpl"]:
        j -= 1
    return (j >= 0 and j not in ignorer and categorie(mots[j][1]) == ["n"]
            and lemme_de(mots[j][1]) not in TITRES)


def temps_d_habitude(mots):
    """Présent et imparfait d'habitude (a vez, e veze…) : en français, un simple
    présent ou imparfait. Gemma les rendait parfois par un passé composé
    (Petra a vez evet … = qu'ont bu … au lieu de que boivent …)."""
    notes = []
    for t, francais in (("prh", "un présent simple (que boivent…, ils boivent…)"),
                        ("pih", "un imparfait (buvaient…)")):
        if any(a and t in premieres_etiquettes(a) for _, a in mots):
            nom = ETIQUETTES_FR[t]
            notes.append({"br": f"TEMPS : {nom} → en français, {francais.split(' (')[0]}.",
                          "fr": f"« [{nom}] » marque une habitude : le traduire par "
                                f"{francais}, jamais par un passé composé."})
    return notes


def propositions(mots):
    """Deux propositions séparées par « ; », chacune avec son verbe conjugué
    (kelenner Magnolia oc'h ; AR c'helenner oc'h) : la seconde, qui reprend
    la première en insistant, était parfois omise. Une liste « un, daou ; tri »
    sans verbe n'est pas concernée."""
    a_un_verbe = lambda segment: any(
        a and premieres_etiquettes(a)[:1] and premieres_etiquettes(a)[0] in VERBES_CONJUGUES
        and TEMPS & set(premieres_etiquettes(a)) for _, a in segment)
    coupures = [k for k, (f, _) in enumerate(mots) if f == ";"]
    segments, debut = [], 0
    for k in coupures + [len(mots)]:
        segments.append(mots[debut:k])
        debut = k + 1
    avec_verbe = [seg for seg in segments if a_un_verbe(seg)]
    if len(avec_verbe) < 2:
        return []
    n = len(avec_verbe)
    return [{"br": f"PROPOSITIONS : {n} propositions séparées par « ; », toutes à traduire.",
             "fr": f"La phrase compte {n} propositions séparées par « ; » : traduire "
                   "chacune, même si la seconde semble répéter la première (elle insiste)."}]


def ordre_et_passif(mots, dico):
    """Deux constructions que Gemma rendait mal, signalées par des notes :
    - l'attribut (ou le lieu) mis en tête, avant « être » conjugué :
      Ho skolidi omp = nous sommes vos élèves ; kelenner on = je suis professeur ;
    - le passif « bezañ + participe passé » (Petra zo gwelet ?), que le français
      rend souvent par « on » : qu'est-ce qu'on voit ? ; avec « gant » (par),
      le complément d'agent est le vrai sujet : … a vez evet gant ar seizh paotr
      = ce que boivent les sept garçons.
    Renvoie des dictionnaires « br » (pour l'utilisateur) et « fr » (pour le LLM)."""
    resultat, vus = [], set()
    for k, (forme, analyses) in enumerate(mots):
        if not analyses:
            continue
        e = premieres_etiquettes(analyses)
        if not e or e[0] not in VERBES_CONJUGUES or lemme_de(analyses) != "bezañ":
            continue
        # attribut en tête : être au présent, 1re/2e personne ou 3e du pluriel
        conj = ETRE_PRESENT.get(personne_nombre(e))
        j = precedent(mots, k)
        if conj and "pri" in e and j is not None and conj not in vus:
            cat_avant = categorie(mots[j][1])
            if mots[j][1] is None or cat_avant in (["n"], ["np"], ["adj"]) or (
                    premieres_etiquettes(mots[j][1])[:1] == ["cpl"] and not cat_avant):
                vus.add(conj)
                resultat.append({
                    "br": f"ORDRE « … {mots[j][0]} {forme} » : ce qui précède « {forme} » "
                          f"est mis en tête pour l'insister ; en français : « {conj} … ».",
                    "fr": f"Le verbe être [{conj.split()[0]}] vient après ce qu'il "
                          f"présente, mis en tête pour l'insister : en français, "
                          f"commencer par « {conj} … ».",
                })
        # passif : bezañ conjugué + participe passé
        if k + 1 < len(mots) and mots[k + 1][1]:
            ep = premieres_etiquettes(mots[k + 1][1])
            verbe = lemme_de(mots[k + 1][1])
            if ep[:1] == ["vblex"] and "pp" in ep and verbe in VERBES_AUX_BEZAN:
                temps = next((TEMPS_COMPOSES[t] for t in e if t in TEMPS_COMPOSES),
                             "temps composé")
                partie = mots[k + 1][1][0].split("+")[0]
                sens = (dico.get(partie) or verbe).split(",")[0].strip()
                resultat.append({
                    "br": f"TEMPS COMPOSÉ « {forme} {mots[k + 1][0]} » : bezañ + participe "
                          f"d'un verbe conjugué avec bezañ ({sens}) = {temps.split(' (')[0]} "
                          "actif, pas un passif.",
                    "fr": f"« être + {sens} [participe passé] » n'est pas un passif : "
                          f"c'est un {temps.split(' (')[0]} actif du verbe « {sens} » "
                          "(en français avec être ou avoir selon le verbe : est allé, "
                          "a commencé).",
                })
            elif ep[:1] == ["vblex"] and "pp" in ep and verbe != "bezañ":
                partie = mots[k + 1][1][0].split("+")[0]
                sens = (dico.get(partie) or dico.get(partie.replace("<pp>", "<inf>"))
                        or lemme_de(mots[k + 1][1])).split(",")[0].strip()
                agent = any(lemme_de(a) == "gant" for _, a in mots[k + 2:] if a)
                fr = (f"« être + {sens} [participe passé] » est un passif : en français, "
                      f"le rendre de préférence par « on » + verbe actif ({sens} → on …) "
                      "ou par le passif.")
                if agent:
                    fr = (f"« être + {sens} [participe passé] » est un passif ; le "
                          f"complément introduit par « par » fait l'action : en "
                          f"français, en faire le sujet de « {sens} » à l'actif.")
                resultat.append({
                    "br": f"PASSIF « {forme} {mots[k + 1][0]} » : bezañ + participe "
                          f"passé ({sens}) ; en français, souvent « on » + actif"
                          + (", ou l'agent (gant) en sujet." if agent else "."),
                    "fr": fr,
                })
    return resultat


def sujet_avant_zo(mots, k):
    """Pronom sujet placé avant « zo », éventuellement suivi d'une apposition
    (Me, Pêr zo… ; Ni zo… ; Ha c'hwi zo) : « je », « nous », « vous »… ou ""."""
    j = k - 1
    while j >= 0:
        e = premieres_etiquettes(mots[j][1]) if mots[j][1] else []
        if e[:1] == ["prn"] and "subj" in e:
            return PRONOMS.get(personne_nombre(e), "")
        if e[:1] in (["cm"], ["np"]):       # virgule, nom propre en apposition
            j -= 1
            continue
        return ""
    return ""


TONIQUES = {("p1", "sg"): "moi", ("p2", "sg"): "toi", ("p3", "sg"): "lui", ("p1", "pl"): "nous",
            ("p2", "pl"): "vous", ("p3", "pl"): "eux"}


def pronom_en_apposition(mots, k):
    """Pronom sujet suivi d'une virgule et d'un nom propre (Me, Pêr zo…) :
    forme tonique française (moi), ou "" si ce n'est pas le cas."""
    e = premieres_etiquettes(mots[k][1]) if mots[k][1] else []
    if not (e[:1] == ["prn"] and "subj" in e and k + 2 < len(mots)):
        return ""
    virgule = premieres_etiquettes(mots[k + 1][1]) if mots[k + 1][1] else []
    nom = premieres_etiquettes(mots[k + 2][1]) if mots[k + 2][1] else []
    if virgule[:1] == ["cm"] and nom[:1] == ["np"]:
        return TONIQUES.get(personne_nombre(e), "")
    return ""


def appositions(mots):
    """Note pour « Me, Pêr zo breur… » : le pronom et le nom propre désignent la
    même personne, sujet du verbe (en français : Moi, Pêr, je suis…)."""
    resultat = []
    for k, (forme, _) in enumerate(mots):
        tonique = pronom_en_apposition(mots, k)
        if tonique:
            sujet = PRONOMS.get(personne_nombre(premieres_etiquettes(mots[k][1])), "")
            nom = mots[k + 2][0]
            resultat.append({
                "br": f"APPOSITION « {forme}, {nom} » : même personne, sujet du verbe ; "
                      f"en français : « {tonique.capitalize()}, {nom}, {sujet} … ».",
                "fr": f"« {tonique} , {nom} » désigne une seule personne, sujet du verbe : "
                      f"traduire « {tonique.capitalize()}, {nom}, {sujet} … » "
                      f"({nom} n'est pas un autre sujet ni un complément).",
            })
    return resultat


def apres_particule_a(mots, k):
    """Le mot k suit-il directement la particule verbale « a » ?"""
    a = mots[k - 1][1]
    return bool(a) and premieres_etiquettes(a)[:1] == ["vpart"] and lemme_de(a) == "a"


def recoller_ch(mots):
    """Un mot inconnu contenant c'h arrive coupé en deux à l'apostrophe
    (kelc'hgeriadur -> kelc + ' + hgeriadur ; c'hoñchoù -> c + ' + hoñchoù) :
    on recolle les morceaux en un seul mot inconnu."""
    resultat, k = [], 0
    while k < len(mots):
        forme, analyses = mots[k]
        j = k + 1
        if j < len(mots) and mots[j][0] in ("'", "’") :
            j += 1
        # morceau gauche inconnu ou simple lettre « c » ; morceau droit inconnu
        # ou simple lettre « h » (c'hoñchoù -> c + hoñchoù ; kabac'h -> kabac + h)
        if (j < len(mots) and j > k + 1 and forme[-1:] in ("c", "C")
                and mots[j][0][:1] in ("h", "H")
                and (analyses is None or forme.lower() == "c")
                and (mots[j][1] is None or mots[j][0].lower() == "h")
                and (analyses is None or mots[j][1] is None)):
            resultat.append((forme + "'" + mots[j][0], None))
            k = j + 1
            continue
        resultat.append((forme, analyses))
        k += 1
    return resultat


def traits_verbe(etiquettes):
    """« passé simple, il/elle », « infinitif », « participe passé »…"""
    temps = next((ETIQUETTES_FR[t] for t in etiquettes if t in TEMPS), "")
    autres = [ETIQUETTES_FR[t] for t in etiquettes if t in ("inf", "pp", "imp", "ger")]
    personne = next((t for t in etiquettes if t in ("p1", "p2", "p3")), "")
    nombre = next((t for t in etiquettes if t in ("sg", "pl")), "")
    sujet = PRONOMS.get((personne, nombre), "")
    if (personne, nombre) == ("p3", "sg") and "f" in etiquettes:
        sujet = "elle"                # he deus, he doa : auxiliaire au féminin
    if "impers" in etiquettes:
        sujet = "on"
    return ", ".join(x for x in [temps] + autres + [sujet] if x)


def tournures(mots, dico):
    """Repère la construction « infinitif + a + ober conjugué » (Strivañ a reas
    = il s'efforça) : le verbe porteur de sens est l'infinitif mis en tête,
    ober ne sert qu'à porter le temps et la personne. La particule « a » peut
    manquer ou être mal écrite (ar, an, al)."""
    resultat = []
    for k, (forme, analyses) in enumerate(mots):
        e = premieres_etiquettes(analyses)
        if not (e and e[0] in ("vblex", "vbmod") and "inf" in e):
            continue
        j = k + 1
        if j < len(mots) and mots[j][0].lower() in ("a", "ar", "an", "al"):
            j += 1
        if j >= len(mots) or not mots[j][1]:
            continue
        forme_ober, analyses_ober = mots[j]
        eo = premieres_etiquettes(analyses_ober)
        if analyses_ober[0].split("<")[0] != "ober" or not (TEMPS & set(eo)):
            continue
        infinitif = analyses[0].split("+")[0]
        sens = dico.get(infinitif) or infinitif.split("<")[0]
        traits = traits_verbe(eo)
        resultat.append({
            "infinitif": k, "auxiliaire": j, "traits": traits,
            "br": f"TOURNURE « {forme} … {forme_ober} » : infinitif mis en tête + ober "
                  f"conjugué. Le verbe à traduire est « {sens} », conjugué comme ober : "
                  f"{traits}. Ober ne se traduit pas.",
            "fr": f"« {sens} » est à conjuguer : {traits}.",
        })
    return resultat


def mot_a_mot(mots, dico, ignorer, auxiliaires, a_conjuguer, derives=None):
    """Mot à mot entièrement en français, construit depuis l'analyse :
    « s'efforcer [passé simple, il/elle] · pour · développer [infinitif] · … ».
    Les mots inconnus restent entre guillemets, la ponctuation (? ! : . ,)
    est gardée, les nombres gardés tels quels. Un article
    défini entre deux noms devient « de le » : complément du nom breton par
    simple juxtaposition (distro-skol ar c'helenner = la rentrée DU professeur)."""
    morceaux = []
    for k, (forme, analyses) in enumerate(mots):
        if k in ignorer or k in auxiliaires:
            continue
        if analyses is None:
            # mot inconnu à majuscule après un nom : nom propre complément du nom
            de = "de " if (k > 0 and forme[:1].isupper()
                           and complement_de_nom(mots, k, ignorer)) else ""
            indice = (derives or {}).get(normaliser(forme).lower())
            morceaux.append(f"{de}«{forme}» (inconnu" + (f" ; {indice['fr']}" if indice else "") + ")")
            continue
        e = premieres_etiquettes(analyses)
        if e and e[0] == "num":
            partie = analyses[0].split("+")[0]
            morceaux.append(dico.get(partie) or ORDINAUX.get(normaliser(forme).lower(), forme))
            continue
        if e and e[0] in CATEGORIES_IGNOREES:
            # ponctuation forte gardée : une question ou un ordre doit le rester
            if forme in ("?", "!", ":", ";", ".", "…", ",") and morceaux:
                dernier = morceaux[-1]
                colle = dernier and dernier[-1] in "?!.…" and forme in "?!.…"
                morceaux[-1] = dernier + ("" if colle else " ") + forme
            continue
        if e and e[0] == "vpart":              # particules verbales a, e : non traduites
            if analyses[0].split("<")[0].lower() == "o":
                morceaux.append("en train de")
            continue
        # complément du nom par juxtaposition : nom + article défini + nom
        # (sal-glas ar brezhoneg = la salle de classe DU breton)
        # article devant un nom de lieu : d'ar Gorreloù = à Gorreloù (pas « à la »)
        devant_lieu = k + 1 < len(mots) and nom_de_lieu(mots[k + 1])
        if e and e[0] == "det" and "def" in e and devant_lieu:
            continue
        if e and e[0] == "det" and "def" in e and 0 < k < len(mots) - 1:
            avant = premieres_etiquettes(mots[k - 1][1])
            apres = premieres_etiquettes(mots[k + 1][1])
            avant = avant[1:] if avant[:1] == ["cpl"] else avant
            apres = apres[1:] if apres[:1] == ["cpl"] else apres
            if avant[:1] == ["n"] and apres[:1] in (["n"], ["np"]):
                morceaux.append("de le")
                continue
        # nom + nom sans article : le second complète le premier
        # (sal-glas vrezhoneg = salle de classe DE breton)
        prefixe = ""
        if k > 0 and (e[1:] if e[:1] == ["cpl"] else e)[:1] == ["n"]:
            avant = premieres_etiquettes(mots[k - 1][1])
            avant = avant[1:] if avant[:1] == ["cpl"] else avant
            lemme_avant = ((mots[k - 1][1] or [""])[0]).split("<")[0].lower()
            if (avant[:1] == ["n"] and (k - 1) not in ignorer
                    and lemme_avant not in TITRES):
                prefixe = "de "
        # nom propre complément d'un nom, éventuellement suivi d'un adjectif :
        # breur Katell = le frère DE Katell ; breur divarchet Katell = le frère
        # déboussolé DE Katell (pas après un titre : Aotrou Yann)
        if k > 0 and e[:1] == ["np"] and complement_de_nom(mots, k, ignorer):
            prefixe = "de "
        sens = []
        il_y_a = (lemme_de(analyses) == "bezañ"
                  and RE_IL_Y_A.fullmatch(normaliser(forme).lower()) is not None)
        for partie in analyses[0].split("+"):
            ep = RE_ETIQUETTES.findall(partie)
            lemme = partie.split("<")[0].replace("\\", "").replace("_", " ")
            if devant_lieu and ep[:1] == ["det"] and "def" in ep and "+" in analyses[0]:
                continue                          # d'ar, d'an, en : article omis devant un lieu
            if il_y_a:
                texte = "il y a"                  # ez eus, n'eus (ket) : existence
            elif pronom_en_apposition(mots, k):
                texte = pronom_en_apposition(mots, k)   # Me, Pêr… = moi, Pêr…
            elif ep and ep[0] == "vbloc":
                texte = "se trouver"              # bezañ de situation : emaon, emañ…
            elif lemme == "gant" and k > 0 and ("pp" in premieres_etiquettes(mots[k - 1][1]) or (
                    mots[k - 1][1] is None
                    and (derives or {}).get(normaliser(mots[k - 1][0]).lower(), {}).get("pp"))):
                texte = "par"                     # agent du passif : evet gant = bu par
            elif dico.get(partie) and "<prot>" in partie:
                texte = dico[partie]              # citation, nom propre : tel quel
            elif dico.get(partie):
                # « dans, à l'intérieur de » -> « dans / à l'intérieur de » :
                # la virgule est réservée à la ponctuation de la phrase
                texte = dico[partie].replace(", ", " / ")
                if "+" not in analyses[0]:
                    autres = autres_sens(analyses, dico, texte)
                    if autres:
                        texte += f" (ou : {' / '.join(autres)})"
            elif ep and ep[0] == "np":
                texte = lemme                     # nom propre : tel quel
            else:
                texte = f"«{lemme}» (non traduit)"
            if k in a_conjuguer:
                texte += f" [{a_conjuguer[k]}]"
            elif il_y_a:
                temps = next((ETIQUETTES_FR[t] for t in ep if t in TEMPS), "")
                if temps and temps != "présent":
                    texte += f" [{temps}]"
            elif ep and ep[0] in VERBES_CONJUGUES:
                traits = traits_verbe(ep)
                j = precedent(mots, k)
                sujet = premieres_etiquettes(mots[j][1]) if j is not None and mots[j][1] else []
                if normaliser(forme).lower() == "zo":
                    # zo : même forme pour toutes les personnes ; le sujet est ce qui
                    # précède : Me zo = je suis ; Me, Pêr zo = moi, Pêr, je suis
                    pronom = sujet_avant_zo(mots, k)
                    traits = "présent" + (f", {pronom}" if pronom else "")
                elif k > 0 and apres_particule_a(mots, k):
                    # Ar verourien a voe, 3 a voe dilennet : après la particule a, le
                    # verbe reste au singulier ; le sujet est ce qui précède
                    traits = traits_verbe([t for t in ep if t not in ("p1", "p2", "p3", "sg", "pl")])
                elif (normaliser(forme).lower() == "eo" and sujet[:1] == ["prn"]
                      and PRONOMS.get(personne_nombre(sujet))):
                    # c'hwi eo = vous êtes ; ni eo = nous sommes
                    traits = "présent, " + PRONOMS[personne_nombre(sujet)]
                elif normaliser(forme).lower() == "eo" and not any(
                        premieres_etiquettes(a)[:1] == ["prn"] for _, a in mots[:k] if a):
                    traits = "présent, il / c'est"  # Spontus eo = C'est effrayant
                if traits:
                    texte += f" [{traits}]"
            elif ep and ep[0] in ("n", "adj") and "pl" in ep:
                texte += " [pluriel]"
            sens.append(texte)
        morceaux.append(prefixe + " + ".join(sens))
    return " · ".join(morceaux)


_TOPONYMES = {}


def cle_toponyme(nom):
    """Comparaison des noms de lieux : minuscules, sans espaces, traits d'union
    ni apostrophes (Beuzeg-ar-C'hab = Beuzeg ar C'hab)."""
    return "".join(c for c in normaliser(nom).lower() if c not in " -'")


# types de lieux dont le nom français est sûr même si le nom breton est porté
# ailleurs par un lieu-dit (Treboull = Tréboul, Kemper = Quimper)
TYPES_SURS = {"Kumun", "Karter", "Kumun gozh"}


def charger_toponymes(chemin, commune=None):
    """toponymes.tsv (tiré de KerOfis par preparation/donnees_oplb.py) :
    « nom breton<TAB>nom français<TAB>commune<TAB>type<TAB>commune=français|… ».
    Renvoie {clé du nom breton: nom français, ou "" si le nom français est
    incertain}. Le nom français est :
    - celui de la commune choisie (--commune Douarnenez) si le lieu y existe ;
    - sinon celui du fichier s'il n'y a qu'une forme française, ou s'il s'agit
      d'une commune ou d'un quartier ;
    - sinon incertain : le nom est protégé, mais sans traduction (An Aod Vras
      est la Côte Sauvage à Quiberon, pas à Douarnenez).
    Fichier absent : dictionnaire vide."""
    if not chemin or not os.path.exists(chemin):
        return {}
    cle_cache = (chemin, (commune or "").lower())
    if cle_cache not in _TOPONYMES:
        toponymes = {}
        with open(chemin, encoding="utf-8") as f:
            for ligne in f:
                if ligne.startswith("#") or "\t" not in ligne:
                    continue
                champs = ligne.rstrip("\n").split("\t") + ["", "", ""]
                breton, francais, _, sorte, formes = champs[:5]
                par_commune = dict(x.split("=", 1) for x in formes.split("|") if "=" in x)
                distinctes = {cle_toponyme(x) for x in par_commune.values()} or {cle_toponyme(francais)}
                if commune and commune.lower() in {c.lower() for c in par_commune}:
                    choisi = next(v for c, v in par_commune.items() if c.lower() == commune.lower())
                elif len(distinctes) == 1 or sorte in TYPES_SURS:
                    choisi = francais
                else:
                    choisi = ""
                toponymes[cle_toponyme(breton)] = choisi.strip()
        _TOPONYMES[cle_cache] = toponymes
    return _TOPONYMES[cle_cache]


def charger_complements(chemin):
    """complements.tsv : « breton<TAB>traduction », une entrée par ligne, mot
    ou expression (« e dalc'h »). Lignes vides et lignes en # ignorées.
    Ces entrées ont priorité sur Apertium."""
    complements = {}
    if not chemin or not os.path.exists(chemin):
        return complements
    with open(chemin, encoding="utf-8") as f:
        for num, ligne in enumerate(f, start=1):
            ligne = ligne.rstrip("\n")
            if not ligne.strip() or ligne.lstrip().startswith("#"):
                continue
            breton, tab, francais = ligne.partition("\t")
            if not tab or not breton.strip() or not francais.strip():
                print(f"({chemin}, ligne {num} ignorée : il faut « breton<TAB>traduction »)",
                      file=sys.stderr)
                continue
            cle = " ".join(normaliser(breton).lower().split())
            complements[cle] = francais.strip()
    return complements


def charger_variantes(chemin):
    """variantes.tsv : « forme dialectale ou ancienne<TAB>forme unifiée », une
    par ligne (deus → eus, errued → erruet). Lignes vides et lignes en #
    ignorées. Appliqué au texte AVANT Apertium, qui ne connaît que le breton
    unifié (peurunvan)."""
    variantes = []
    if not chemin or not os.path.exists(chemin):
        return variantes
    with open(chemin, encoding="utf-8") as f:
        for num, ligne in enumerate(f, start=1):
            ligne = ligne.rstrip("\n")
            if not ligne.strip() or ligne.lstrip().startswith("#"):
                continue
            forme, tab, unifiee = ligne.partition("\t")
            unifiee = unifiee.split("\t")[0].strip()     # 3e colonne : commentaire
            if not tab or not forme.strip() or not unifiee:
                print(f"({chemin}, ligne {num} ignorée : il faut « forme<TAB>forme unifiée »)",
                      file=sys.stderr)
                continue
            motif = re.compile(r"(?<![\w'])" + re.escape(normaliser(forme.strip()))
                               + r"(?![\w'])", re.IGNORECASE)
            variantes.append((motif, unifiee))
    return variantes


def appliquer_variantes(texte, variantes):
    """Remplace les formes dialectales par la forme unifiée, en gardant la
    majuscule initiale. Renvoie le texte et la liste des remplacements faits."""
    texte = normaliser(texte)
    faits = []
    # les citations en français (« La honte et le châtiment ») ne sont pas
    # touchées : sans cela, la variante et → aet changeait le « et » français
    morceaux, debut = [], 0
    for d, f in zones_francaises(texte):
        morceaux += [(texte[debut:d], True), (texte[d:f], False)]
        debut = f
    morceaux.append((texte[debut:], True))
    resultat = []
    for morceau, breton in morceaux:
        if breton:
            for motif, unifiee in variantes:
                def remplacer(m):
                    nouveau = unifiee[:1].upper() + unifiee[1:] if m.group(0)[:1].isupper() else unifiee
                    faits.append(f"{m.group(0)} → {nouveau}")
                    return nouveau
                morceau = motif.sub(remplacer, morceau)
        resultat.append(morceau)
    return "".join(resultat), list(dict.fromkeys(faits))


# Expressions figées, traduites d'un bloc ; complements.tsv reste prioritaire
EXPRESSIONS = {
    **{f"en {p} sav": "debout" for p in ("ma", "da", "e", "he", "hor", "ho", "o")},
    "a-dizh": "en vitesse, à toute allure",
    "na tost": "loin de là (litt. « ni près »)",
    "tamm ebet": "pas du tout",
    "n'eus forzh": "peu importe",
    "forzh pegement": "énormément, tant et plus",
    "evel just": "bien sûr",
    "gant plijadur": "avec plaisir",
    "a-benn ar fin": "finalement",
    "da gentañ": "d'abord",
    "pell zo": "il y a longtemps",
    "abaoe pell": "depuis longtemps",
    # « 'zo 'tont » : prochain (d'ar Merc'her 'zo 'tont = mercredi prochain)
    "'zo 'tont": "prochain (litt. « qui vient »)",
    "zo 'tont": "prochain (litt. « qui vient »)",
    "zo o tont": "prochain (litt. « qui vient »)",
    # « n'eo ket nemet » : pas seulement (l'inverse de « n'eo nemet » = ce n'est que)
    "n'eo ket nemet": "pas seulement (ne … pas que)",
    "n'eo nemet": "ce n'est que (ne … que)",
    "mod all": "autrement, sinon",
    "evel boaz": "comme d'habitude",
    # tournures relevées sur l'extrait 5 de Kontañ kaoz
    "an eil egile": "l'un l'autre",
    "an eil hag egile": "l'un et l'autre",
    "an eil d'egile": "l'un à l'autre",
    "a-hend-all": "autrement, sinon (aussi : par ailleurs)",
    "skiant prenet": "expérience (litt. « science achetée »)",
    "chom etre daou": "hésiter (litt. « rester entre deux »)",
    "porzh-c'hoari": "cour de récréation",
    "porzh ar skol": "cour de l'école",
    "war wel": "en vue, visible (dont war wel = apparaître ; lakaat war wel = mettre en évidence)",
    "war-wel": "en vue, visible (dont war-wel = apparaître ; lakaat war-wel = mettre en évidence)",
    "reiñ en-dro": "rendre, redonner",
    "ober gant": "utiliser, se servir de (ou : faire avec)",
    "da lavarout eo": "c'est-à-dire",
    "hag all": "etc. (et ainsi de suite)",
    "an deiz war-lerc'h": "le lendemain (litt. « le jour d'après »)",
    "an deiz war-lerc’h": "le lendemain (litt. « le jour d'après »)",
    "an noz war-lerc'h": "la nuit suivante",
    "ar sizhun war-lerc'h": "la semaine suivante",
    "ar bloaz war-lerc'h": "l'année suivante",
    "da lavaret eo": "c'est-à-dire",
    "bez' e": "(mise en relief du verbe qui suit : ne se traduit pas, ce n'est pas « peut-être »)",
}
ANALYSES_EXPRESSIONS = set()       # analyses <cpl> venues d'EXPRESSIONS (affichage)

# pronoms « a » + personne : complément d'un verbe (gwelet anezho = les voir)
# ou « de » + pronom après un nom (ur maen tev anezho = une grosse pierre chacun)
PRONOMS_COMPLEMENTS = {
    "ac'hanon": "me / de moi", "ac'hanout": "te / de toi", "anezhañ": "le / de lui",
    "anezhi": "la / d'elle", "ac'hanomp": "nous / de nous", "ac'hanoc'h": "vous / de vous",
    "anezho": "les / d'eux, d'elles",
}
VERBES_ENTRER = {"mont", "dont", "mont a ober", "dont a ober"}


def appliquer_complements(mots, complements, dico, source=None):
    """Remplace les mots ou suites de mots présents dans complements.tsv par une
    seule unité « <cpl> » dont la traduction est celle du fichier. L'expression
    la plus longue l'emporte (6 mots au plus). Avec source (ensemble), les
    analyses créées y sont notées, et les unités <cpl> déjà là sont laissées."""
    if not complements:
        return mots
    # « merc'her » peut arriver coupé (merc + ' + her), « Arsav-Brezel » en deux :
    # on compare aussi les formes sans espaces, traits d'union ni apostrophes
    tasse = lambda t: re.sub(r"[\s'\-]", "", t)
    par_forme_tassee = {tasse(c): c for c in complements}
    formes = [" ".join(normaliser(f).lower().split()) for f, _ in mots]
    resultat, k = [], 0
    deja = (lambda m: bool(m[1]) and "<cpl>" in m[1][0]) if source is not None else (lambda m: False)
    while k < len(mots):
        if protege(mots[k]) or deja(mots[k]):
            resultat.append(mots[k])
            k += 1
            continue
        borne = min(len(mots), k + 8)
        borne = next((i for i in range(k, borne) if protege(mots[i]) or deja(mots[i])), borne)
        for j in range(borne, k, -1):
            cle = " ".join(formes[k:j])
            if cle not in complements:
                cle = par_forme_tassee.get(tasse("".join(formes[k:j])), cle)
            if cle in complements:
                forme = re.sub(r"\s*(['’])\s*", r"\1", " ".join(f for f, _ in mots[k:j]))
                analyse = f"{cle}<cpl>"
                # un mot seul garde sa catégorie (nom, verbe…) pour les règles
                # de grammaire : « sal-glas<cpl><n> »
                if j == k + 1 and mots[k][1]:
                    categorie = premieres_etiquettes(mots[k][1])[:1]
                    if categorie:
                        analyse += f"<{categorie[0]}>"
                dico[analyse] = complements[cle]
                if source is not None:
                    source.add(analyse)
                resultat.append((forme, [analyse]))
                k = j
                break
        else:
            resultat.append(mots[k])
            k += 1
    return resultat


PONCTUATION_DEBUT = {".", "!", "?", ":", ";", "…"}


def imperatif_probable(mots, k, deja):
    """Un verbe conjugué ne commence pas une phrase affirmative en breton
    (on dit « Krogiñ a reomp », « Ni a grog ») : au présent, à la 1re ou 2e
    personne du pluriel, en tête de phrase ou après « : », c'est presque
    toujours un impératif (krogomp = commençons, gwelit = voyez). Un second
    verbe relié par « ha » à un impératif l'est aussi (gwelit ha klevit).
    Renvoie l'analyse à l'impératif, ou None."""
    forme, analyses = mots[k]
    if not analyses or "+" in analyses[0]:
        return None
    e = RE_ETIQUETTES.findall(analyses[0])
    if not (e[:1] == ["vblex"] and "pri" in e and "pl" in e and ("p1" in e or "p2" in e)):
        return None
    en_tete = k == 0 or mots[k - 1][0] in PONCTUATION_DEBUT
    apres_imperatif = (k >= 2 and normaliser(mots[k - 1][0]).lower() in ("ha", "hag")
                       and len(deja) >= 2 and deja[-2][1]
                       and "<imp>" in deja[-2][1][0])
    if not (en_tete or apres_imperatif):
        return None
    return analyses[0].replace("<pri>", "<imp>")


def appliquer_regles(mots, dico):
    """Corrections sûres que l'étiqueteur ne fait pas toujours :
    « Du » avec une majuscule, hors début de phrase, est le mois (novembre),
    pas l'adjectif du (noir), quand il suit miz/viz ou précède un nombre ;
    un verbe en -et juste après bezañ (zo gwelet) est un participe passé ;
    « da » devant un nom propre est la préposition (da Gatell), pas « ton » ;
    « hini » est le pronom « celui, celle », pas l'adverbe « aucun » ;
    « ez » devant un verbe est la particule verbale, pas « tu vas » ;
    « n' » devant un nom ou un adjectif est l'article élidé, pas la négation ;
    « e » minuscule analysé comme la lettre e est la préposition (dans, en)."""
    resultat = []
    for k, (forme, analyses) in enumerate(mots):
        etiquettes = premieres_etiquettes(analyses)
        avant = normaliser(mots[k - 1][0]).lower() if k > 0 else ""
        apres = premieres_etiquettes(mots[k + 1][1]) if k + 1 < len(mots) else []
        lemme_avant = (mots[k - 1][1] or [""])[0].split("<")[0] if k > 0 else ""
        bas = normaliser(forme).lower()
        verbes_avant = [lemme_de(m[1]) for m in mots[max(0, k - 3):k] if m[1]]
        imperatif = imperatif_probable(mots, k, resultat)
        if imperatif:
            # Krogomp ! Gwelit ha klevit ! : verbe au présent (1re ou 2e pers. du
            # pluriel) en tête de phrase, sans particule = impératif (même forme)
            dico[imperatif] = dico.get(analyses[0], "")
            resultat.append((forme, [imperatif, analyses[0]] + analyses[1:]))
        elif (bas == "deus" and (k == 0 or est_ponctuation(mots[k - 1])
                                  or mots[k - 1][0] in PONCTUATION_DEBUT | {"«", '"', "“"})
              and not (k + 1 < len(mots) and a_majuscule(mots[k + 1][0]))):
            # « Deus buan ! » = viens vite ! (impératif de dont) ; mais
            # « Deus Penn ar Men » = de Penn ar Men (préposition, dialectal)
            analyse = "deus<rgl-imp>"
            dico[analyse] = "viens (impératif de dont)"
            resultat.append((forme, [analyse]))
        elif bas in PRONOMS_COMPLEMENTS:
            analyse = f"{bas}<rgl-pron>"
            dico[analyse] = PRONOMS_COMPLEMENTS[bas]
            resultat.append((forme, [analyse]))
        elif bas == "tre" and any(v in VERBES_ENTRER for v in verbes_avant):
            # mont tre, dont tre : entrer (tre = dedans, pas « très »)
            analyse = "tre<rgl-entrer>"
            dico[analyse] = "dedans (mont tre = entrer)"
            resultat.append((forme, [analyse]))
        elif forme == "Du" and k > 0 and (avant in ("miz", "viz") or apres[:1] == ["num"]):
            analyse = "Du<rgl>"
            dico[analyse] = "novembre"
            resultat.append((forme, [analyse]))
        elif (normaliser(forme).lower() == "e du" and forme.endswith("Du")
              and apres[:1] == ["num"]):
            # « E Du 2005 » (analysé d'un bloc « en noir / en novembre ») : en novembre
            analyse = "e Du<rgl>"
            dico[analyse] = "en novembre"
            resultat.append((forme, [analyse]))
        elif (normaliser(forme).lower() == "ger" and k > 0 and mots[k - 1][1]
              and premieres_etiquettes(mots[k - 1][1])[:1] == ["det"]):
            # ar ger, ur ger, ma ger : le nom « mot » (ker muté = cher n'a pas d'article)
            analyse = "ger<rgl>"
            dico[analyse] = "mot"
            resultat.append((forme, [analyse]))
        elif (normaliser(forme).lower() in ("e", "o") and k > 0
              and normaliser(mots[k - 1][0]).lower().endswith("d'")):
            # d'e emzalc'h = à son comportement ; d'o zud = à leurs parents
            mot = normaliser(forme).lower()
            analyse = f"{mot}-pos<rgl>"
            dico[analyse] = "son / sa (possessif)" if mot == "e" else "leur (possessif)"
            resultat.append((forme, [analyse]))
        elif (etiquettes[:1] == ["vblex"] and "inf" in etiquettes
              and normaliser(forme).lower().endswith("et") and lemme_avant == "bezañ"):
            # gwelet, klevet… après zo / eo : participe passé (passif), pas infinitif
            ancienne = analyses[0]
            nouvelle = ancienne.replace("<inf>", "<pp>", 1)
            if ancienne in dico:
                dico[nouvelle] = dico[ancienne]
            resultat.append((forme, [nouvelle] + analyses[1:]))
        elif (normaliser(forme).lower() == "da" and etiquettes[:1] == ["det"]
              and apres[:1] == ["np"]):
            # da Gatell : « à Katell » (breur da Gatell = le frère de Katell),
            # pas le possessif « ton »
            analyse = "da<rgl>"
            dico[analyse] = "à (préposition)"
            resultat.append((forme, [analyse]))
        elif (normaliser(forme).lower() == "ez" and lemme_de(analyses) == "mont"
              and apres[:1] and apres[0] in VERBES_CONJUGUES | {"vblex"}):
            # ez devant un verbe : particule verbale (ez erruet, ez eus),
            # pas « tu vas » (mont, 2e personne)
            analyse = "ez<vpart>"
            dico[analyse] = ""
            resultat.append((forme, [analyse]))
        elif (normaliser(forme).lower() == "n'" and etiquettes[:1] == ["adv"]
              and not (apres[:1] and apres[0] in VERBES_CONJUGUES | {"vblex", "vaux"})):
            # n'holl dud, n'Aotrou Doue : article « an » élidé (à l'oral, dans
            # les chansons) ; devant un verbe (n'eo, n'emañ, n'omp) : négation
            analyse = "n'<rgl>"
            dico[analyse] = "le (article)"
            resultat.append((forme, [analyse]))
        elif normaliser(forme).lower() == "hini" and etiquettes[:1] != ["prn"]:
            # an hini, d'an hini, hini kreñv : pronom « celui, celle »
            analyse = "hini<rgl>"
            dico[analyse] = "celui / celle"
            resultat.append((forme, [analyse]))
        elif (forme in ("e", "E") and apres[:1] == ["np"] and k + 1 < len(mots)
              and "top" in premieres_etiquettes(mots[k + 1][1])):
            # e Treboull, e Kemper, E Beg Leide (en début de phrase) : préposition
            # devant un nom de lieu, pas le pronom « le »
            analyse = "e-lieu<rgl>"
            dico[analyse] = "à / dans (préposition)"
            resultat.append((forme, [analyse]))
        elif forme == "e" and etiquettes[:1] == ["n"]:
            # « e » minuscule analysé comme la lettre e : c'est la préposition
            analyse = "e<rgl>"
            dico[analyse] = "dans, en (préposition)"
            resultat.append((forme, [analyse]))
        else:
            resultat.append((forme, analyses))
    return resultat


# catégories dont un autre sens mérite d'être signalé à Gemma : le mot choisi
# doit être un nom, un adjectif ou un adverbe (pas un verbe : ober « faire »
# ou « action », bet « eu » ou « été » ajoutaient du bruit sans aider)
SENS_PLEINS = {"n", "adj", "vblex", "adv"}
SENS_CHOISIS = {"n", "adj", "adv"}
# sens absents du dictionnaire d'Apertium, ajoutés comme autre sens possible
SENS_ABSENTS = {"gwer<adj>": "verre (nom)",
                "porzh<n>": "cour (d'école, de ferme)",
                "merc'h<n>": "femme", "merc’h<n>": "femme"}


def ajouter_autres_sens(choisies, toutes):
    """L'étiqueteur d'Apertium ne garde qu'une analyse par mot : gwer devient
    « vert » (adjectif) même quand c'est « verre » (nom). Pour les noms,
    adjectifs, verbes et adverbes, on remet à la suite de l'analyse choisie les
    autres analyses de nom, d'adjectif ou d'adverbe (au plus 3 ; pas de verbe :
    ya « oui » n'est pas « va »), pour que le mot à mot
    propose « vert (ou : verre) » et que Gemma tranche d'après le contexte.
    Phrase dont les mots ne s'alignent pas : laissée telle quelle."""
    resultat = []
    for mots_c, mots_t in zip(choisies, toutes):
        formes = lambda ms: [normaliser(f).lower() for f, _ in ms]
        if formes(mots_c) != formes(mots_t):
            resultat.append(mots_c)
            continue
        nouveaux = []
        for (forme, a), (_, a_toutes) in zip(mots_c, mots_t):
            if a and a_toutes and len(a) == 1 and "+" not in a[0]:
                cat = RE_ETIQUETTES.findall(a[0])[:1]
                if cat and cat[0] in SENS_CHOISIS:
                    autres = [x for x in dict.fromkeys(a_toutes)
                              if "+" not in x and x.split("<")[0] != "" and x != a[0]
                              and RE_ETIQUETTES.findall(x)[:1]
                              and RE_ETIQUETTES.findall(x)[0] in SENS_CHOISIS]
                    a = a + autres[:3]
            nouveaux.append((forme, a))
        resultat.append(nouveaux)
    return resultat


def autres_sens(analyses, dico, sens_choisi):
    """Traductions des autres analyses d'un mot, différentes du sens choisi."""
    vus, autres = {sens_choisi}, []
    if analyses:
        cle = analyses[0].split("<")[0].lower() + "".join(f"<{e}>" for e in RE_ETIQUETTES.findall(analyses[0])[:1])
        if cle in SENS_ABSENTS:
            vus.add(SENS_ABSENTS[cle])
            autres.append(SENS_ABSENTS[cle])
    for x in (analyses or [])[1:]:
        t = (dico.get(x) or "").replace(", ", " / ")
        if t and t not in vus and t not in sens_choisi:
            vus.add(t)
            autres.append(t)
    return autres[:2]


# jours et périodes suivis de « a zeu » (prochain) : d'ar Sadorn 8 a zeu
PERIODES = {"lun", "meurzh", "merc'her", "yaou", "gwener", "sadorn", "sul", "sizhun",
            "sizhunvezh", "miz", "bloaz", "bloavezh", "wech", "deiz", "devezh", "noz",
            "mintin", "abardaez", "goañv", "hañv", "nedeleg", "pask"}
SENS_REFLECHIS = {"lakaat": "mettre (en em lakaat da = se mettre à)"}
# durées suivies de « 'zo » : il y a (ur bloaz 'zo = il y a un an)
DUREES = {"bloaz", "bloavezh", "miz", "sizhun", "sizhunvezh", "deiz", "devezh", "amzer",
          "eur", "eurvezh", "munutenn", "noz", "wech", "mare", "prantad", "pellik", "kantved"}
# verbes dont le sens change devant un infinitif ou « da »
SENS_DEVANT_INFINITIF = {
    "kinnig": "proposer (kinnig + infinitif = proposer de)",
    "delc'her": "continuer (derc'hel da = continuer à)",
    "derc'hel": "continuer (derc'hel da = continuer à)",
}
# « Ken » de salutation : Ken warc'hoazh ! = À demain !
KEN_AU_REVOIR = {"warc'hoazh", "arc'hoazh", "bremaik", "bremañ", "tuchant", "tuchantik",
                 "bremaikig", "disul", "dilun", "dimeurzh", "dimerc'her", "diriaou",
                 "digwener", "disadorn", "goude", "neuze", "wech", "atav", "souden"}


# « en ur » (en ul devant l, en un devant une voyelle, n, d, t, h) + verbe : gérondif
GERONDIF = {"en ur", "en ul", "en un"}


def verbe_probable(mot):
    """Le mot peut-il être l'infinitif d'un gérondif (en ur zivoull) ?
    Infinitif d'Apertium, mot inconnu, ou mot de complements.tsv qui n'est
    ni un nom ni un adjectif (en ur vag = dans un bateau reste tel quel)."""
    forme, analyses = mot
    if analyses is None:
        return True
    e = premieres_etiquettes(analyses)
    if e[:1] == ["vblex"] and "inf" in e:
        return True
    return e[:1] == ["cpl"] and categorie(analyses) in ([], ["vblex"])


def constructions_courantes(mots, dico):
    """Tours que l'analyse mot par mot rend mal, repérés d'après la forme des
    mots (quel que soit le découpage d'Apertium) :
    - « en em » + verbe : verbe pronominal (en em welout = se voir ;
      en em lakaat da = se mettre à) ;
    - « ken » + adjectif … « ha(g) » : comparaison d'égalité (ken plijus hag
      an hini = aussi agréable que celle) ; « ha ken » ou « ha » + adjectif
      coordonne (ken plijus ha ken kentelius = aussi agréable et aussi
      instructif ; ken yac'h ha birvidik = aussi sain et vigoureux) ;
    - « a zeu » après un jour, un nombre ou une période : prochain ;
    - « 'benn » (a-benn) et « 'ta » (eta) en tête d'apostrophe ;
    - « en ur » + verbe : gérondif (en ur zivoull = en démaillant) ;
    - nom + « 'zo » : certains (kanaouennoù 'zo) ; durée + « 'zo » : il y a ;
    - kinnig, derc'hel + infinitif ou « da » : proposer, continuer ;
    - « gant » après un passif (bezañ + participe passé) : par."""
    tasse = lambda t: re.sub(r"[\s'’\-]", "", normaliser(t).lower())
    bas = [normaliser(f).lower().replace("’", "'") for f, _ in mots]
    resultat, k = [], 0
    while k < len(mots):
        forme, analyses = mots[k]
        if protege(mots[k]):
            resultat.append(mots[k]); k += 1
            continue
        # « o » + verbe à mutation spirante (k → c'h, t → z, p → f) : pronom
        # objet « les » (o zreuzkas = les transmettre), pas « en train de »,
        # qui provoque la mutation mixte (o tremen, o c'hoari… : t → d, k → g)
        if (bas[k] == "o" and k + 1 < len(mots) and mots[k + 1][1]
                and premieres_etiquettes(mots[k + 1][1])[:1] == ["vblex"]):
            suivant, lemme = bas[k + 1], lemme_de(mots[k + 1][1])
            if any(suivant.startswith(m) and lemme.startswith(o)
                   for m, o in (("c'h", "k"), ("z", "t"), ("f", "p"))):
                resultat.append((forme, ["o<rgl-les>"]))
                dico["o<rgl-les>"] = "les (pronom objet devant l'infinitif)"
                k += 1
                continue
        # « gant ma » + verbe : conjonction « parce que » (gant ma 'z int bet
        # disoñjet = parce qu'elles ont été oubliées) ; « 'z » qui suit n'est
        # que la particule verbale, pas le pronom « te »
        if bas[k] == "gant ma" and RE_GANT_MA.match(" " + " ".join(bas[k:k + 3]) + " "):
            resultat.append((forme, ["gant ma<rgl-cause>"]))
            dico["gant ma<rgl-cause>"] = "parce que, du fait que"
            if k + 1 < len(mots) and bas[k + 1] in ("'z", "z"):
                resultat.append((mots[k + 1][0], ["'z<rgl-part>"]))
                dico["'z<rgl-part>"] = "(particule verbale, ne se traduit pas)"
                k += 1
            k += 1
            continue
        # en em (un ou deux mots), 'n em
        n = 0
        if bas[k] in ("en em", "'n em"):
            n = 1
        elif bas[k] in ("en", "'n") and k + 1 < len(mots) and bas[k + 1] == "em":
            n = 2
        if n and k + n < len(mots) and mots[k + n][1]:
            resultat.append((" ".join(f for f, _ in mots[k:k + n]), ["en em<rgl-refl>"]))
            dico["en em<rgl-refl>"] = "se (réfléchi, avec le verbe qui suit)"
            verbe_forme, verbe = mots[k + n]
            lemme = lemme_de(verbe)
            if lemme in SENS_REFLECHIS and "+" not in verbe[0]:
                nouvelle = "en-em-" + verbe[0]
                dico[nouvelle] = SENS_REFLECHIS[lemme]
                resultat.append((verbe_forme, [nouvelle] + verbe[1:]))
            else:
                resultat.append(mots[k + n])
            k += n + 1
            continue
        # 'benn, 'ta : apostrophe collée ou détachée
        apostrophe = forme in ("'", "’") and k + 1 < len(mots)
        suite = bas[k + 1] if apostrophe else bas[k]
        if (apostrophe or bas[k] in ("'benn", "'ta")) and suite.lstrip("'") in ("benn", "ta"):
            mot = suite.lstrip("'")
            analyse = f"'{mot}<rgl>"
            dico[analyse] = "à / pour (a-benn)" if mot == "benn" else "donc (eta)"
            resultat.append(("'" + mot, [analyse]))
            k += 2 if apostrophe else 1
            continue
        # nom + 'zo : certains (kanaouennoù 'zo) ; durée + 'zo : il y a (ur bloaz 'zo)
        detache = forme in ("'", "’") and k + 1 < len(mots) and bas[k + 1] == "zo"
        if (bas[k] == "'zo" or detache) and resultat and resultat[-1][1]:
            avant = resultat[-1][1]
            if categorie(avant) == ["n"]:
                duree = lemme_de(avant) in DUREES
                analyse = "'zo<rgl-ilya>" if duree else "'zo<rgl-certains>"
                dico[analyse] = ("il y a (durée + 'zo : ur bloaz 'zo = il y a un an)" if duree
                                 else "certains (nom + 'zo : kanaouennoù 'zo = certaines chansons)")
                resultat.append(("'zo", [analyse]))
                k += 2 if detache else 1
                continue
        # Ken ar Sul ! Ken warc'hoazh ! : « à … » (au revoir), pas « si, plus »
        if (bas[k] == "ken" and (k == 0 or mots[k - 1][0] in PONCTUATION_DEBUT)
                and k + 1 < len(mots)):
            suivant = bas[k + 1]
            if (suivant.split(" ")[0] in ("ar", "an", "al", "ur", "un", "ul")
                    or suivant.split(" ")[0] in PERIODES or suivant in KEN_AU_REVOIR):
                resultat.append((forme, ["ken<rgl-revoir>"]))
                dico["ken<rgl-revoir>"] = "à (au revoir : ken ar Sul = à dimanche)"
                k += 1
                continue
        # en ur + verbe : gérondif (en ur zivoull anezho = en les démaillant),
        # pas « dans un » ; « en ur » arrive en un ou deux morceaux
        n = 0
        if bas[k] in GERONDIF:
            n = 1
        elif bas[k] == "en" and k + 1 < len(mots) and f"en {bas[k + 1]}" in GERONDIF:
            n = 2
        if n and k + n < len(mots) and verbe_probable(mots[k + n]):
            resultat.append((" ".join(f for f, _ in mots[k:k + n]), ["en ur<rgl-gerondif>"]))
            dico["en ur<rgl-gerondif>"] = "en (gérondif : en …ant)"
            k += n
            continue
        # a zeu après un jour, un nombre ou une période : prochain
        if tasse(forme) == "azeu" or (bas[k] == "a" and k + 1 < len(mots) and bas[k + 1] == "zeu"):
            n = 1 if tasse(forme) == "azeu" else 2
            avant = resultat[-1] if resultat else None
            queue = tasse("".join(f for f, _ in resultat[-3:]))
            if avant and (premieres_etiquettes(avant[1])[:1] == ["num"]
                          or any(queue.endswith(tasse(p)) for p in PERIODES)):
                resultat.append((" ".join(f for f, _ in mots[k:k + n]), ["a zeu<rgl-proch>"]))
                dico["a zeu<rgl-proch>"] = "prochain (litt. « qui vient »)"
                k += n
                continue
        resultat.append(mots[k])
        k += 1
    # ken + adjectif … ha(g) : « que » de la comparaison
    for k, (forme, analyses) in enumerate(resultat):
        if normaliser(forme).lower() != "ken" or k + 1 >= len(resultat):
            continue
        if premieres_etiquettes(resultat[k + 1][1])[:1] not in (["adj"], ["adv"]):
            continue
        for j in range(k + 2, min(len(resultat), k + 14)):
            fj = normaliser(resultat[j][0]).lower()
            if fj not in ("ha", "hag"):
                continue
            if j + 1 < len(resultat) and normaliser(resultat[j + 1][0]).lower() == "ken":
                break            # ha ken … : coordination, le « que » vient plus loin
            if j + 1 < len(resultat) and premieres_etiquettes(resultat[j + 1][1])[:1] == ["adj"]:
                continue         # ken yac'h ha birvidik : aussi sain ET vigoureux
            resultat[j] = (resultat[j][0], ["ha-que<rgl-que>"])
            dico["ha-que<rgl-que>"] = "que (comparaison « ken … ha »)"
            break
    # kinnig, derc'hel + (a ran) + infinitif, « da » ou « en em » : proposer, continuer
    for k, (forme, analyses) in enumerate(resultat):
        lemme = lemme_de(analyses).replace("’", "'")
        if lemme not in SENS_DEVANT_INFINITIF or not analyses[0].startswith(lemme_de(analyses)):
            continue
        j = k + 1
        while j < len(resultat) and resultat[j][1] and (
                categorie(resultat[j][1]) == ["vpart"]
                or (lemme_de(resultat[j][1]) == "ober"
                    and "inf" not in premieres_etiquettes(resultat[j][1]))):
            j += 1
        if j >= len(resultat):
            continue
        suivant, ej = normaliser(resultat[j][0]).lower(), premieres_etiquettes(resultat[j][1])
        if (suivant in ("da", "d'") or ej[:1] == ["rgl-refl"]
                or (ej[:1] == ["vblex"] and "inf" in ej)):
            reste = analyses[0][len(lemme_de(analyses)):]
            nouvelle = lemme.replace("'", "") + "-sens" + reste
            dico[nouvelle.split("+")[0]] = SENS_DEVANT_INFINITIF[lemme]
            resultat[k] = (forme, [nouvelle] + analyses[1:])
    # gant après un passif (bezañ + participe passé) : par (agent)
    for k, (forme, analyses) in enumerate(resultat):
        if lemme_de(analyses) != "gant" or not analyses[0].startswith("gant<pr>"):
            continue
        for j in range(k - 1, max(-1, k - 6), -1):
            fj, aj = resultat[j]
            if fj in PONCTUATION_DEBUT or fj == ",":
                break
            ej = premieres_etiquettes(aj)
            if (ej[:1] == ["vblex"] and "pp" in ej and lemme_de(aj) not in VERBES_AUX_BEZAN
                    and lemme_de(aj) != "bezañ"
                    and any(lemme_de(resultat[i][1]) == "bezañ" for i in range(max(0, j - 2), j))):
                nouvelle = "gant-agent<rgl-agent>" + analyses[0][len("gant<pr>"):]
                dico[nouvelle.split("+")[0]] = "par (agent du passif ; sinon : avec)"
                resultat[k] = (forme, [nouvelle] + analyses[1:])
                break
    return resultat


def notes_constructions(mots):
    """Notes pour Gemma sur les tours repérés par constructions_courantes et
    par les expressions « n'eo ket nemet », « 'zo 'tont »."""
    premieres = [(a or [""])[0] for _, a in mots]
    notes = []
    def note(br, fr):
        notes.append({"br": br, "fr": fr})
    if any(a.endswith("<rgl-refl>") for a in premieres):
        note("CONSTRUCTION « en em » + verbe : verbe pronominal.",
             "« en em » + verbe = verbe pronominal : en em welout = se voir, "
             "en em gavout = se retrouver, en em lakaat da = se mettre à.")
    if any(a.endswith("<rgl-gerondif>") for a in premieres):
        note("GÉRONDIF « en ur » + verbe : en …ant.",
             "« en (gérondif) » + verbe = gérondif français : traduire par « en » + "
             "participe présent (en ur zivoull anezho = en les démaillant ; en ur "
             "c'hoarzhin = en riant), pas par « dans un ».")
    if any(a.endswith("<rgl-revoir>") for a in premieres):
        note("SALUTATION « Ken … » : à …",
             "« Ken » en tête de phrase devant un moment = « à » pour se dire au revoir : "
             "Ken warc'hoazh = À demain ; Ken ar Sul = À dimanche ; Ken ar wech all = À la prochaine.")
    if any("<rgl-agent>" in a for a in premieres):
        note("PASSIF + « gant » : par.",
             "Après un passif (bezañ + participe passé), « gant » introduit celui qui fait "
             "l'action : « par », pas « avec » (savet gant Didier = composées par Didier ; "
             "dilezet gant ar Vretoned = abandonnée par les Bretons).")
    if any(a.endswith("<rgl-certains>") or a.endswith("<rgl-ilya>") for a in premieres):
        note("« 'zo » après un nom : certains / il y a.",
             "« 'zo » placé après un nom n'est pas le verbe être : nom + 'zo = « certains » "
             "(kanaouennoù 'zo = certaines chansons) ; durée + 'zo = « il y a » "
             "(ur bloaz 'zo = il y a un an).")
    if sum(1 for f, _ in mots if normaliser(f).lower() == "pe") >= 2:
        note("« pe … pe » : ou … ou.",
             "« pe … pe » = « ou … ou », « soit … soit » (e galleg pe e brezhoneg = "
             "en français ou en breton), jamais « et ».")
    if any(a.endswith("<rgl-que>") for a in premieres):
        note("COMPARAISON « ken … ha(g) » : aussi … que.",
             "« ken + adjectif … ha(g) … » = « aussi … que … » : le ha(g) qui suit "
             "l'adjectif introduit le second terme de la comparaison (« que »), pas « et ».")
    if any(a.startswith("n'eo ket nemet<cpl>") for a in premieres):
        note("NÉGATION « n'eo ket nemet » : pas seulement.",
             "« n'eo ket nemet X » = « ce n'est pas seulement X » : n'eo ket nemet "
             "kaozeal a raio = il ne fera pas que parler (il fera aussi autre chose). "
             "Ne pas confondre avec « n'eo nemet » = ce n'est que.")
    if any(a.endswith("<rgl-proch>") or (a.endswith("<cpl>") and "tont" in a) for a in premieres):
        note("TEMPS « 'zo 'tont » / « a zeu » : prochain.",
             "« 'zo 'tont », « zo o tont » ou « a zeu » après un jour ou une période = "
             "« prochain » (d'ar Merc'her 'zo 'tont = mercredi prochain), pas « il vient ».")
    # Apertium rend « he deus » ou « gant ma » en un seul mot : on cherche dans
    # le texte recollé plutôt que mot par mot
    texte = " " + " ".join(normaliser(f).lower().replace("’", "'") for f, _ in mots) + " "
    if RE_AUX_FEMININ.search(texte):
        note("« he deus », « he doa »… : sujet féminin (elle).",
             "« he » devant une forme de l'auxiliaire « avoir » (he deus, he doa, he devo, "
             "he defe…) marque la 3e personne du féminin : le sujet est « elle » "
             "(gouezet he deus = elle a su ; kroget he deus = elle a commencé). "
             "Accorder au féminin, ne pas écrire « il ».")
    if RE_GANT_MA.search(texte):
        note("« gant ma » + verbe : parce que, du fait que.",
             "« gant ma » suivi d'un verbe (gant ma 'z int, gant ma oa) = « parce que, "
             "du fait que, tant » (rouez gant ma 'z int bet disoñjet = rares parce qu'elles "
             "ont été oubliées), pas « avec mon ».")
    return notes


# « he » + auxiliaire « avoir » (3e personne du féminin) : he deus, he doa…
RE_AUX_FEMININ = re.compile(r" he (?:deus|deveus|doa|devoa|devo|defe|dije|dez|devez|deveze) ")
# « gant ma » + verbe : la conjonction (et non « avec mon » : gant ma zad)
RE_GANT_MA = re.compile(r" gant ma (?:'?z(?:'| |$)|ez |e |eo |oa |oant |int |vez |veze |vo |vefe "
                        r"|emañ |edo |omp |on |out |oc'h )")


# Paraphrase française de chaque affixe, appliquée de la base vers l'extérieur :
# tangwall + -et -> « participe passé d'un verbe formé sur « incendie » »
PARAPHRASES = {
    "adigoù": "petits {}", "adig": "petit {}", "igoù": "petits {}", "ig": "petit {}",
    "erezed": "femmes qui font l'action de {}", "ourien": "personnes qui font l'action de {}",
    "erien": "personnes qui font l'action de {}", "our": "personne qui fait l'action de {}",
    "er": "personne qui fait l'action de {}", "erez": "femme (ou machine) qui fait l'action de {}",
    "erezh": "activité ou métier lié à {}", "adenn": "action de {} (une fois)",
    "adennoù": "actions de {} (une fois chacune)", "entez": "qualité d'être {}",
    "ded": "qualité d'être {}", "der": "qualité d'être {}", "dez": "qualité d'être {}",
    "adoù": "contenus de {}", "ad": "contenu de {}", "ennoù": "éléments de {}",
    "enn": "un élément de {}", "elloù": "instruments pour {}", "ell": "instrument pour {}",
    "iek": "adjectif tiré de {}", "ek": "qui a beaucoup de {}", "us": "adjectif tiré de {}",
    "aat": "devenir ou rendre {}", "iz": "habitants de {}",
    "ioù": "pluriel de {}", "où": "pluriel de {}", "ien": "pluriel de {}", "idi": "pluriel de {}",
    "ier": "pluriel de {}", "ed": "pluriel de {}",
    "iñ": "verbe formé sur {}", "añ": "verbe formé sur {}",
    "et": "participe passé d'un verbe formé sur {}",
    "dreist": "sur-{}", "peur": "complètement {}", "gour": "sous-{} ou petit {}",
    "dis": "contraire de {}", "diz": "contraire de {}", "dam": "à demi {}", "rak": "{} d'avance",
    "ken": "{} en commun", "ad-": "{} de nouveau", "di": "sans {}",
}


def paraphrase(sens, affixes):
    """« incendie », ['-et (participe passé)'] -> « participe passé d'un verbe
    formé sur « incendie » ». Les notes de mutation sont ignorées."""
    texte = f"« {sens} »"
    for affixe in affixes:
        if affixe.startswith("mutation"):
            continue
        forme = affixe.split(" (")[0]
        cle = forme.strip("-") if forme.startswith("-") else forme.rstrip("-")
        if cle == "ad" and forme.endswith("-"):
            cle = "ad-"                       # préfixe ad- (de nouveau), pas le suffixe -ad
        modele = PARAPHRASES.get(cle)
        if modele:
            texte = modele.format(*([texte] * modele.count("{}")))
    return texte


def passifs_derives(mots, derives):
    """Participe passé deviné (dérivé en -et, inconnu d'Apertium) suivi de « gant » :
    tournure passive, l'agent est introduit par « par ». Sans cette note, Gemma
    faisait du bateau le sujet actif (« Le Petit André a incendié les Allemands »
    pour « tangwallet gant an alamaned » = incendié par les Allemands)."""
    notes = []
    for k, (forme, analyses) in enumerate(mots[:-1]):
        d = derives.get(normaliser(forme).lower()) if analyses is None else None
        if d and d.get("pp") and lemme_de(mots[k + 1][1]) == "gant":
            notes.append({
                "br": f"PASSIF « {forme} {mots[k + 1][0]} » : participe passé deviné + gant (par).",
                "fr": "Un participe passé suivi de « par » est un passif : ce qui précède "
                      "le participe SUBIT l'action, ce qui suit « par » la FAIT "
                      "(… a été [participe] par …). Ne pas en faire une phrase active.",
            })
    return notes


# Citations en français dans un texte breton : « La honte et le châtiment ».
# Sans protection, « le » devenait le mot breton le (serment), « et » restait
# inconnu… On les repère dans le texte brut : entre guillemets, avec au moins
# un mot-outil ou une lettre propre au français, et aucun mot-outil breton.
RE_CITATION = re.compile(r'[«"“]\s*([^«»"“”]{2,120}?)\s*[»"”]')
MOTS_FRANCAIS = {"et", "le", "les", "des", "du", "une", "est", "au", "aux", "pour", "dans",
                 "sur", "avec", "qui", "que", "ne", "pas", "mon", "ma", "mes", "son", "sa", "ses"}
MOTS_BRETONS = {"ar", "an", "al", "ha", "hag", "eo", "zo", "eus", "gant", "da", "war", "e",
                "a", "ur", "ul", "ez", "evit", "eme", "ket", "n'eo"}
LETTRES_FRANCAISES = set("éèàçëîïôœû")


def mots_de_citation(segment):
    """Mots d'une citation, en minuscules, si elle est en français ; sinon None."""
    mots = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", segment.lower())
    if len(mots) < 2:
        return None
    francais = any(w in MOTS_FRANCAIS for w in mots) or any(c in LETTRES_FRANCAISES for w in mots for c in w)
    breton = any(w in MOTS_BRETONS for w in mots) or any("c'h" in w or "zh" in w for w in mots)
    return mots if francais and not breton else None


def recoller_ponctuation(texte):
    """« châtiment . Imposer » -> « châtiment. Imposer » : points, virgules et
    points de suspension se recollent au mot qui précède dans une unité protégée
    (citation, nom propre) ; « : ; ! ? » gardent l'espace qui les précède en français."""
    return re.sub(r"\s+([.,…)])", r"\1", re.sub(r"([(])\s+", r"\1", texte))


def citations_francaises(texte):
    """Suites de mots entre guillemets qui sont du français, en minuscules."""
    return [m for m in (mots_de_citation(c.group(1)) for c in RE_CITATION.finditer(normaliser(texte))) if m]


def zones_francaises(texte):
    """Positions (début, fin) des citations françaises dans le texte normalisé."""
    return [c.span(1) for c in RE_CITATION.finditer(texte) if mots_de_citation(c.group(1))]


def proteger_citations_francaises(mots, citations, dico):
    """Regroupe chaque citation française en une unité protégée, non traduite."""
    if not citations:
        return mots
    bas = [normaliser(f).lower().split() for f, _ in mots]
    resultat, k = [], 0
    while k < len(mots):
        if est_ponctuation(mots[k]):                  # une citation commence par un mot
            resultat.append(mots[k])
            k += 1
            continue
        for citation in citations:
            vus, j = [], k
            while j < len(mots) and len(vus) < len(citation):
                if not mots[j][1] or not premieres_etiquettes(mots[j][1])[:1] \
                        or premieres_etiquettes(mots[j][1])[0] not in CATEGORIES_IGNOREES:
                    vus += [re.sub(r"^[^\w']+|[^\w']+$", "", w) for w in bas[j]]
                j += 1
            vus = [w for w in vus if w]
            if vus == citation:
                forme = recoller_ponctuation(" ".join(f for f, _ in mots[k:j]))
                analyse = f"{forme}<np><al><prot>"
                dico[analyse] = f"{forme} (en français dans le texte)"
                resultat.append((forme, [analyse]))
                k = j
                break
        else:
            resultat.append(mots[k])
            k += 1
    return resultat


# Noms de groupes, d'entreprises, d'associations entre guillemets, juste après
# le mot qui les présente : ar strollad "Kern", un embregerezh anvet "Bo Travail".
# Sans cette règle, Kern devenait « Cime » et Bo Travail « Il Sera Travail ».
# Comme pour les citations françaises, les noms sont repérés dans le texte brut
# (Apertium ne garde pas toujours les guillemets comme des mots à part).
PRESENTE_UN_NOM = {"anvet", "strollad", "strolladoù", "embregerezh", "kevredigezh",
                   "ti-embann", "gelaouenn", "kelaouenn", "skol", "kompagnunezh", "laz-kanañ",
                   "bagad", "kelc'h", "stal", "lec'hienn", "anv",
                   # titres d'émissions, de films, de livres, de fêtes : abadennoù "Kontañ kaoz"
                   "abadenn", "abadennoù", "film", "filmoù", "teulfilm", "teulfilmoù",
                   "levr", "levrioù", "gouel", "festival", "kanaouenn", "pennad"}
NOM_MOTS_MAX = 4
RE_NOM_GUILLEMETS = re.compile(
    r"(?<![\w'])(" + "|".join(re.escape(m) for m in sorted(PRESENTE_UN_NOM, key=len, reverse=True))
    + r")\s*[«\"“]\s*([^«»\"“”]{1,60}?)\s*[»\"”]", re.IGNORECASE)


def noms_entre_guillemets(texte):
    """Noms (4 mots au plus) entre guillemets après anvet, strollad, embregerezh…,
    en minuscules, mot par mot."""
    noms = []
    for m in RE_NOM_GUILLEMETS.finditer(normaliser(texte)):
        mots = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", m.group(2).lower())
        if 1 <= len(mots) <= NOM_MOTS_MAX:
            noms.append(mots)
    return noms


def proteger_noms_entre_guillemets(mots, noms, dico):
    """Regroupe chaque nom repéré par noms_entre_guillemets en une unité
    protégée, non traduite (Kern, Bo Travail)."""
    if not noms:
        return mots
    bas = [normaliser(f).lower().split() for f, _ in mots]
    resultat, k = [], 0
    while k < len(mots):
        if est_ponctuation(mots[k]) or protege(mots[k]):
            resultat.append(mots[k])
            k += 1
            continue
        for nom in noms:
            vus, j = [], k
            while j < len(mots) and len(vus) < len(nom):
                if not est_ponctuation(mots[j]):
                    vus += [re.sub(r"^[^\w']+|[^\w']+$", "", w) for w in bas[j]]
                j += 1
            vus = [w for w in vus if w]
            if vus == nom:
                forme = recoller_ponctuation(" ".join(f for f, _ in mots[k:j] if not est_ponctuation((f, _))))
                analyse = f"{forme}<np><org><prot>"
                dico[analyse] = forme
                resultat.append((forme, [analyse]))
                k = j
                break
        else:
            resultat.append(mots[k])
            k += 1
    return resultat


def recoller_toutes(toutes):
    return [recoller_ch(mots) for mots in toutes]


def analyse_de_base(analyses):
    """Analyse à gloser pour la base d'un dérivé : la première de nom, adjectif,
    verbe ou adverbe, sinon la première ; None si la base vient de complements.tsv."""
    if not analyses:
        return None
    for a in analyses:
        e = RE_ETIQUETTES.findall(a)
        if "+" not in a and e[:1] and e[0] in ("n", "adj", "vblex", "adv"):
            return a
    return analyses[0]


def analyse_complete(textes, automorf, autobil, complements=None, toponymes=None):
    """Pour chaque phrase, un dictionnaire :
    - « lignes » : une ligne par mot, en citant le breton, pour l'utilisateur
      (lemme, catégorie, traits, traduction, mutation, INCONNU, ALERTE, TOURNURE) ;
    - « mot_a_mot » : le mot à mot entièrement français, pour le LLM ;
    - « notes » : alertes et tournures reformulées en français, pour le LLM.
    L'analyse est désambiguïsée par l'étiqueteur d'Apertium quand il est
    installé, sinon toutes les analyses possibles sont listées."""
    vide = {"lignes": "", "mot_a_mot": "", "notes": []}
    if not automorf or not shutil.which("lt-proc"):
        return [dict(vide) for _ in textes]
    try:
        toutes = analyses_desambiguisees(textes, automorf)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"(désambiguïsation Apertium indisponible : {e})", file=sys.stderr)
        toutes = None
    if toutes is None:
        toutes = analyses_detaillees(textes, automorf, inclure_inconnus=True)
    else:
        toutes = ajouter_autres_sens(toutes, analyses_detaillees(textes, automorf, inclure_inconnus=True))
    a_traduire = [p for mots in toutes for _, analyses in mots if analyses
                  for a in analyses for p in a.split("+")]
    # mots inconnus : dérivé probable d'une base connue (lazhourien -> lazhañ)
    inconnus = {normaliser(f).lower() for mots in recoller_toutes(toutes)
                for f, a in mots if a is None and len(f) > 3}
    connus = {c for c in (complements or {}) if " " not in c}
    try:
        trouves = derivations(inconnus - connus, automorf, connus)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"(analyse des dérivés indisponible : {e})", file=sys.stderr)
        trouves = {}
    bases = {m: (b, analyse_de_base(an), aff) for m, (b, an, aff) in trouves.items()}
    a_traduire += [p for _, an, _ in bases.values() if an for p in an.split("+")]
    dico = traduire_lemmes(a_traduire, autobil) if autobil else {}
    derives = {}
    for mot, (base, analyse, affixes) in bases.items():
        if analyse is None:
            sens, cat = (complements or {}).get(base, ""), "complements.tsv"
        else:
            # lemme d'Apertium plutôt que la forme essayée (alamanañ -> alaman)
            base = normaliser(analyse.split("+")[0].split("<")[0].replace("\\", "")) or base
            sens = dico.get(analyse, "")
            e = RE_ETIQUETTES.findall(analyse)
            cat = ETIQUETTES_FR.get(e[0], e[0]) if e else ""
        affixes_fr = ", ".join(affixes)
        derives[mot] = {
            "br": f"DÉRIVÉ probable de « {base} » ({cat})" + (f" -> {sens}" if sens else "")
                  + f" ; {affixes_fr}",
            "fr": "sens probable : " + paraphrase(sens.split(",")[0].strip() if sens else base, affixes),
            "pp": bool(affixes) and affixes[-1].startswith("-et "),
        }
    resultats = []
    for texte, mots in zip(textes, toutes):
        mots = proteger_citations_francaises(recoller_ch(mots), citations_francaises(texte), dico)
        mots = proteger_noms_entre_guillemets(mots, noms_entre_guillemets(texte), dico)
        mots = appliquer_regles(proteger_noms_propres(mots, dico, toponymes), dico)
        mots = appliquer_complements(mots, complements or {}, dico)
        mots = appliquer_complements(mots, EXPRESSIONS, dico, source=ANALYSES_EXPRESSIONS)
        mots = constructions_courantes(mots, dico)
        mots = regrouper_toponymes(mots, dico)
        lignes, vus = [], set()
        for rang, (forme, analyses) in enumerate(mots):
            if forme.lower() in vus:
                continue
            if analyses is None:
                vus.add(forme.lower())
                note = ("peut-être un nom propre" if forme[:1].isupper() and rang > 0
                        else "lemme et sens incertains")
                derive = derives.get(normaliser(forme).lower())
                lignes.append(f"{forme} : INCONNU d'Apertium ({note})"
                              + (f" ; {derive['br']}" if derive else ""))
                continue
            premiere = premieres_etiquettes(analyses)
            if premiere and premiere[0] in CATEGORIES_IGNOREES:
                continue
            vus.add(forme.lower())
            if premiere[:1] == ["cpl"]:
                origine = ("expression usuelle" if analyses[0] in ANALYSES_EXPRESSIONS
                           else "complements.tsv")
                lignes.append(f"{forme} : {dico[analyses[0]]}  ({origine})")
                continue
            if premiere[:1] and premiere[0].startswith("rgl"):
                raison = (RAISONS_REGLES.get(analyses[0]) or RAISONS_REGLES.get(premiere[0])
                          or "règle du script")
                sens = dico.get(analyses[0]) or " + ".join(
                    dico.get(p) or p.split("<")[0] for p in analyses[0].split("+"))
                lignes.append(f"{forme} : {sens}  (règle : {raison})")
                continue
            distinctes = list(dict.fromkeys(analyses))[:4]
            texte = " | ".join(decrire(a, dico) for a in distinctes)
            if len(set(analyses)) > 4:
                texte += " | …"
            lemme = distinctes[0].split("+")[0].split("<")[0].replace("\\", "")
            mut = mutation(forme, lemme)
            lignes.append(f"{forme} : {texte}" + (f" ; mutation {mut}" if mut else ""))
        les_passifs = passifs_derives(mots, derives)
        les_alertes = alertes(mots)
        les_tournures = tournures(mots, dico)
        les_constructions = (les_passifs + appositions(mots) + ordre_et_passif(mots, dico) + propositions(mots)
                             + temps_d_habitude(mots) + notes_constructions(mots))
        lignes += ([a["br"] for a in les_alertes] + [t["br"] for t in les_tournures]
                   + [c["br"] for c in les_constructions])
        resultats.append({
            "lignes": "\n".join(lignes),
            "mot_a_mot": mot_a_mot(
                mots, dico,
                ignorer={a["ignorer"] for a in les_alertes},
                auxiliaires={t["auxiliaire"] for t in les_tournures},
                a_conjuguer={t["infinitif"]: t["traits"] for t in les_tournures},
                derives=derives,
            ),
            "notes": ([a["fr"] for a in les_alertes] + [t["fr"] for t in les_tournures]
                      + [c["fr"] for c in les_constructions]),
        })
    return resultats


LEGENDE_MOT_A_MOT = (
    "Chaque document est une phrase bretonne que tu ne vois pas, donnée sous trois "
    "formes : un « Mot à mot » français fiable, établi par un analyseur (entre "
    "crochets : temps, personne, nombre ; « inconnu » = mot absent du dictionnaire : "
    "s'il est suivi d'un « sens probable », traduis-le d'après ce sens, sinon "
    "garde-le tel quel), une "
    "« Traduction automatique » souvent maladroite, et des « Notes » à respecter."
)

# fin de phrase : ? ! … (éventuellement répétés), ou un point après une minuscule,
# un chiffre, un guillemet fermant ("… territoires".) ou un mot en capitales
# (LOSQ.), ou un guillemet fermant après la ponctuation (… territoires."), mais pas après une initiale
# (« M. Hupel »), puis espace et majuscule, éventuellement après un guillemet (« Deus)
_RE_FIN_PHRASE = re.compile(r"(?:(?<=[?!…])|(?<=[a-zà-ÿ0-9)\"»”]\.)|(?<=[A-ZÀ-ÖØ-Þ]{2}\.)"
                            r"|(?<=[.?!…][\"»”]))"
                           r"\s+(?=[«\"“]?\s*[A-ZÀ-ÖØ-Þ])")


def guillemets_ouverts(texte):
    """Le texte laisse-t-il des guillemets ouverts ? (« sans », “ sans ”,
    ou nombre impair de guillemets droits)"""
    return (texte.count("«") > texte.count("»") or texte.count("“") > texte.count("”")
            or texte.count('"') % 2 == 1)


def recoller_guillemets(morceaux, joint=" "):
    """Recolle les morceaux coupés à l'intérieur de guillemets :
    « La honte et le châtiment. Imposer le français… » reste d'un seul tenant
    (sans cela, la règle des citations françaises ne reconnaissait plus le titre)."""
    resultat = []
    for m in morceaux:
        # au-delà de 2 × MOTS_MAX mots, un guillemet jamais refermé (faute de
        # frappe) ne doit pas recoller tout le reste du texte
        if (resultat and guillemets_ouverts(resultat[-1])
                and len((resultat[-1] + joint + m).split()) <= 2 * MOTS_MAX):
            resultat[-1] += joint + m
        else:
            resultat.append(m)
    return resultat


class _FinDePhrase:
    """Comme l'expression régulière de fin de phrase, mais split() ne coupe
    jamais à l'intérieur de guillemets. Les autres méthodes (search, finditer…)
    sont celles de l'expression régulière."""
    def split(self, texte):
        return recoller_guillemets(_RE_FIN_PHRASE.split(texte))

    def __getattr__(self, nom):
        return getattr(_RE_FIN_PHRASE, nom)


RE_FIN_PHRASE = _FinDePhrase()

# au-delà de MOTS_MAX mots, une phrase est coupée aux points-virgules et aux
# points de suspension : Gemma traduit mal les très longues phrases
MOTS_MAX = 40
MOTS_MIN = 6          # jamais de morceau plus court (« ha me : ugent ! » reste attaché)
RE_COUPURE_LONGUE = re.compile(r"(?<=;)\s+|(?<=…)\s+|(?<=\.\.\.)\s+")


def decouper_longue(phrase, maximum=MOTS_MAX, minimum=MOTS_MIN):
    """Coupe une phrase de plus de « maximum » mots aux « ; » et aux « … »,
    en regroupant les morceaux pour qu'ils restent sous le maximum quand c'est
    possible, et sans jamais laisser un morceau de moins de « minimum » mots.
    Une phrase sans « ; » ni « … » reste entière."""
    nb = lambda t: len(t.split())
    if nb(phrase) <= maximum:
        return [phrase]
    morceaux = recoller_guillemets([m for m in RE_COUPURE_LONGUE.split(phrase) if m.strip()])
    resultat, courant = [], ""
    for m in morceaux:
        if courant and nb(courant) + nb(m) > maximum and nb(courant) >= minimum:
            resultat.append(courant)
            courant = m
        else:
            courant = f"{courant} {m}".strip()
    if courant:
        if resultat and nb(courant) < minimum:
            resultat[-1] += " " + courant
        else:
            resultat.append(courant)
    return resultat

