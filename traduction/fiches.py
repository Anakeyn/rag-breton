# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Fiches de relecture pour le professeur : HTML (impression), Excel (.xlsx)
ou TSV, et conversion d'une fiche HTML déjà écrite.
"""

import sys

from commun.config import duree


def points_a_verifier(analyse):
    """Lignes de l'analyse qui méritent l'attention du professeur."""
    points = []
    for ligne in analyse["lignes"].split("\n"):
        if "INCONNU" in ligne:
            points.append(ligne.split(" : ")[0] + " : mot inconnu d'Apertium")
        elif ligne.startswith(("ALERTE", "TOURNURE")):
            points.append(ligne)
        elif "(complements.tsv)" in ligne:
            mot, _, sens = ligne.partition(" : ")
            points.append(f"{mot} → {sens.replace('(complements.tsv)', '').strip()} (mes compléments)")
        elif "(règle :" in ligne or ligne.startswith("MÉMOIRE"):
            points.append(ligne)            # MÉMOIRE : exemple donné au LLM (--memoire)
    return points


def ecrire_fiche_excel(chemin, lignes):
    """Fiche de relecture au format Excel : le tableau seul, une ligne d'en-tête
    figée et filtrable, texte renvoyé à la ligne, mise en page A4 paysage."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    except ImportError:
        sys.exit("La sortie .xlsx demande openpyxl : pip install openpyxl")
    classeur = Workbook()
    feuille = classeur.active
    feuille.title = "Relecture"
    entetes = ["N°", "Breton", "Mot à mot", "Traduction proposée", "À vérifier", "Correction"]
    largeurs = [6, 45, 45, 45, 35, 40]
    feuille.append(entetes)
    for r, br, mam, tr, pts in lignes:
        feuille.append([r, br, mam, tr, "\n".join(pts), ""])
    trait = Side(style="thin", color="999999")
    bord = Border(left=trait, right=trait, top=trait, bottom=trait)
    for col, largeur in enumerate(largeurs, start=1):
        feuille.column_dimensions[chr(64 + col)].width = largeur
    for ligne in feuille.iter_rows():
        for case in ligne:
            case.alignment = Alignment(wrap_text=True, vertical="top")
            case.border = bord
    for case in feuille[1]:
        case.font = Font(bold=True)
        case.fill = PatternFill("solid", fgColor="EEEEEE")
    for case in feuille["B"][1:]:
        case.font = Font(bold=True)
    gris = Font(color="444444", size=9)
    for col in ("C", "E"):
        for case in feuille[col][1:]:
            case.font = gris
    feuille.freeze_panes = "A2"
    feuille.auto_filter.ref = feuille.dimensions
    feuille.page_setup.orientation = "landscape"
    feuille.page_setup.paperSize = feuille.PAPERSIZE_A4
    feuille.page_setup.fitToWidth, feuille.page_setup.fitToHeight = 1, 0
    feuille.sheet_properties.pageSetUpPr.fitToPage = True
    feuille.print_title_rows = "1:1"
    classeur.save(chemin)


def ecrire_fiche(chemin, phrases, analyses, traductions, args, duree_totale):
    """Fiche de relecture : breton, mot à mot, traduction proposée, points à
    vérifier et une colonne vide pour les corrections. HTML (s'ouvre dans
    LibreOffice ou un navigateur, s'imprime en paysage) ou TSV (tableur)."""
    import html
    from datetime import date
    lignes = []
    for r, (p, a) in enumerate(zip(phrases, analyses), start=1):
        lignes.append((r, p["br"], a["mot_a_mot"], traductions.get(r, ""), points_a_verifier(a)))
    if chemin.lower().endswith((".xlsx", ".tsv", ".csv", ".txt")):
        ecrire_tableau(chemin, lignes)
    else:
        e = html.escape
        corps = "".join(
            f"<tr><td>{r}</td><td class='br'>{e(br)}</td><td class='mam'>{e(mam)}</td>"
            f"<td>{e(tr)}</td><td class='pts'>{'<br>'.join(e(x) for x in pts)}</td><td></td></tr>\n"
            for r, br, mam, tr, pts in lignes)
        modele = "analyse seule (sans LLM)" if getattr(args, "sans_llm", False) else \
            f"traduction : {e(args.modele_chat or 'modèle de LM Studio')}"
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><title>Fiche de relecture</title>
<style>
@page {{ size: A4 landscape; margin: 12mm; }}
body {{ font-family: sans-serif; font-size: 10pt; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #999; padding: 4px 6px; vertical-align: top; }}
th {{ background: #eee; }}
td.br {{ font-weight: bold; }}
td.mam, td.pts {{ font-size: 8.5pt; color: #444; }}
td:last-child {{ width: 18%; }}
</style></head><body>
<h1>Fiche de relecture</h1>
<p>{date.today().strftime("%d/%m/%Y")} · {len(lignes)} phrases · {modele} · analyse : Apertium br-fr
 · durée : {duree(duree_totale)}</p>
<table>
<tr><th>N°</th><th>Breton</th><th>Mot à mot</th><th>Traduction proposée</th><th>À vérifier</th><th>Correction</th></tr>
{corps}</table></body></html>
""")
    print(f"\nFiche de relecture écrite : {chemin}", file=sys.stderr)


def lire_fiche_html(chemin):
    """Relit le tableau d'une fiche de relecture HTML :
    liste de (n°, breton, mot à mot, traduction, [points à vérifier])."""
    from html.parser import HTMLParser

    class Lecteur(HTMLParser):
        def __init__(self):
            super().__init__()
            self.lignes, self.ligne, self.case = [], None, None
        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.ligne = []
            elif tag in ("td", "th") and self.ligne is not None:
                self.case = []
            elif tag == "br" and self.case is not None:
                self.case.append("\n")
        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.case is not None:
                self.ligne.append("".join(self.case).strip())
                self.case = None
            elif tag == "tr" and self.ligne is not None:
                self.lignes.append(self.ligne)
                self.ligne = None
        def handle_data(self, data):
            if self.case is not None:
                self.case.append(data)

    lecteur = Lecteur()
    with open(chemin, encoding="utf-8") as f:
        lecteur.feed(f.read())
    lignes = [(int(l[0]), l[1], l[2], l[3], [x for x in l[4].split("\n") if x])
              for l in lecteur.lignes if len(l) >= 5 and l[0].isdigit()]
    return lignes


def ecrire_tableau(chemin, lignes):
    """Écrit les lignes d'une fiche en .xlsx (Excel) ou en .tsv (tableur)."""
    if chemin.lower().endswith(".xlsx"):
        ecrire_fiche_excel(chemin, lignes)
    else:
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("N°\tBreton\tMot à mot\tTraduction proposée\tÀ vérifier\tCorrection\n")
            for r, br, mam, tr, pts in lignes:
                f.write("\t".join([str(r), br, mam, tr, " ; ".join(pts), ""]) + "\n")
