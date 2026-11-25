# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Recherche dans le corpus indexé : BM25 sur les lemmes d'Apertium, recherche
par le sens (vecteurs bge-m3, facultative) et fusion des deux classements (RRF).
"""

import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from commun.apertium import lemmatiser
from commun.bge_m3 import embeddings, normaliser_vecteurs
from commun.config import chemins_vecteurs, etat_compatible, lire_etat


def identite(lemmes):
    """Les phrases sont déjà lemmatisées par Apertium : pas de découpage à refaire."""
    return lemmes


class BM25(object):
    """BM25 Okapi vectorisé avec scikit-learn et SciPy.

    Même classe que dans l'article Anakeyn « Comparaison de similarités
    syntaxiques : Jaccard, Dice, TF-IDF, BM25 » (2021), elle-même adaptée du
    gist CC-0 de koreyou. Mêmes paramètres (k1 = 1,6 ; b = 0,75) et même IDF :
    ln(N / n), soit l'IDF de scikit-learn sans lissage moins 1.

    Adaptations pour le RAG breton :
    - entrée : listes de lemmes Apertium (analyzer=identite) au lieu de textes ;
    - la matrice des documents est calculée une seule fois dans fit() et
      conservée, au lieu d'être recalculée à chaque requête (100 000 phrases) ;
    - idf_ (attribut public) au lieu de _tfidf.idf_, retiré des versions récentes.
    """

    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(
            analyzer=identite, norm=None, smooth_idf=False, lowercase=False
        )
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """X : liste de documents (listes de lemmes)."""
        self.vectorizer.fit(X)
        # comptages bruts : transform() de CountVectorizer, sans la pondération TF-IDF
        self.X = super(TfidfVectorizer, self.vectorizer).transform(X).tocsc()
        self.len_X = np.asarray(self.X.sum(1)).ravel()
        self.avdl = self.len_X.mean()
        return self

    def transform(self, q):
        """q : requête (liste de lemmes). Renvoie le score BM25 de chaque document."""
        b, k1, avdl = self.b, self.k1, self.avdl
        q = super(TfidfVectorizer, self.vectorizer).transform([q])
        termes = np.unique(q.indices)          # termes connus, comptés une seule fois
        if termes.size == 0:
            return np.zeros(self.X.shape[0])
        X = self.X[:, termes].toarray()        # fréquences f(qi, D)
        denom = X + (k1 * (1 - b + b * self.len_X / avdl))[:, None]
        idf = self.vectorizer.idf_[None, termes] - 1.
        numer = X * idf * (k1 + 1)
        return (numer / denom).sum(1)


_BM25_CACHE = {}


def recherche_bm25(phrases, requete, n, automorf):
    if "bm25" not in _BM25_CACHE:
        _BM25_CACHE["bm25"] = BM25().fit([p["lemmes"] for p in phrases])
    # Question souvent rédigée en français : on ne garde que les mots reconnus
    # par l'analyseur breton, sinon « et », « les »… ramènent les phrases du
    # corpus qui contiennent du français. Repli sur tous les mots si aucun.
    lemmes = lemmatiser([requete], automorf, connus_seulement=True)[0]
    if not lemmes:
        lemmes = lemmatiser([requete], automorf)[0]
    scores = _BM25_CACHE["bm25"].transform(lemmes)
    rangs = np.argsort(-scores)[:n]
    return [int(i) for i in rangs if scores[i] > 0]


_VECTEURS_CACHE = {}


def charger_vecteurs(phrases, args):
    """Les lignes déjà calculées de vecteurs.npy, ou None (avec la raison)."""
    if "V" not in _VECTEURS_CACHE:
        npy, _ = chemins_vecteurs(args.index)
        etat = lire_etat(args.index)
        if not os.path.exists(npy) or not etat:
            return None, "pas de vecteurs : lance « rag_breton_preparation.py <fichier des phrases> »"
        if not etat_compatible(etat, phrases, args.modele_embedding):
            return None, ("vecteurs calculés pour un autre corpus ou avec "
                          f"« {etat.get('modele')} » : relance « rag_breton_preparation.py <fichier des phrases> »")
        V = np.load(npy, mmap_mode="r")[: etat["fait"]]
        _VECTEURS_CACHE["V"] = np.ascontiguousarray(V)    # chargé en RAM une fois (~400 Mo)
        if etat["fait"] < len(phrases):
            print(f"(recherche sémantique sur {etat['fait']}/{len(phrases)} phrases : "
                  "indexation incomplète)", file=sys.stderr)
    return _VECTEURS_CACHE["V"], None


def recherche_semantique(phrases, requete, n, args):
    V, raison = charger_vecteurs(phrases, args)
    if V is None or len(V) == 0:
        print(f"(recherche sémantique indisponible : {raison or 'aucun vecteur'})", file=sys.stderr)
        return []
    try:
        q = normaliser_vecteurs(embeddings([requete], args))[0]
    except Exception as e:                      # LM Studio éteint, bge-m3 non chargé…
        print(f"(recherche sémantique indisponible : {e})", file=sys.stderr)
        return []
    scores = V @ q                              # cosinus, vecteurs normalisés
    n = min(n, len(scores))
    meilleurs = np.argpartition(-scores, n - 1)[:n]
    return [int(i) for i in meilleurs[np.argsort(-scores[meilleurs])]]


def fusion_rrf(listes, n, k=60):
    """Reciprocal Rank Fusion : score = somme de 1 / (k + rang)."""
    scores = {}
    for liste in listes:
        for rang, idx in enumerate(liste):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rang + 1)
    return sorted(scores, key=scores.get, reverse=True)[:n]


def recherche_hybride(phrases, requete, n, args, automorf):
    lexical = recherche_bm25(phrases, requete, n * 3, automorf)
    if args.sans_vecteurs:                      # recherche par les mots seulement
        return lexical[:n]
    npy, _ = chemins_vecteurs(args.index)
    if not os.path.exists(npy):                 # index construit avec --sans-embeddings
        print("(pas de vecteurs : recherche par les mots seulement)", file=sys.stderr)
        return lexical[:n]
    semantique = recherche_semantique(phrases, requete, n * 3, args)
    return fusion_rrf([lexical, semantique], n)


def afficher(phrases, indices):
    for rang, i in enumerate(indices, start=1):
        p = phrases[i]
        print(f"[{rang}] ({p['id']}) {p['br']}")
        if p["fr"]:
            print(f"     → {p['fr']}")
        if p.get("source"):
            print(f"     ↳ {p['source']}" + (f" ({p['date']})" if p.get("date") else ""))
        elif p.get("corpus"):
            print(f"     ↳ corpus : {p['corpus']}")
