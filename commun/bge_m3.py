# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Appel au modèle d'embedding bge-m3 servi par LM Studio : un texte -> un vecteur
de 1 024 nombres, normalisé (longueur 1).
"""

import numpy as np
import requests


def embeddings(textes, args):
    r = requests.post(
        f"{args.lmstudio}/embeddings",
        json={"model": args.modele_embedding, "input": textes},
        timeout=600,
    )
    r.raise_for_status()
    donnees = sorted(r.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in donnees]


def normaliser_vecteurs(vecteurs):
    """Vecteurs float32 de norme 1 : le cosinus devient un simple produit scalaire."""
    v = np.asarray(vecteurs, dtype=np.float32)
    normes = np.linalg.norm(v, axis=-1, keepdims=True)
    normes[normes == 0] = 1.0
    return v / normes
