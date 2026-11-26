# rag-breton

Rag-breton est un traducteur **breton → français** qui tourne entièrement en local, sans envoyer de texte sur Internet. Il a été testé sur un ordinateur portable de 2017 doté de 32 Go de mémoire.


Il associe deux outils libres :

- **[Apertium](https://www.apertium.org)**, un traducteur à base de règles, analyse le breton : il ramène chaque mot à son lemme en défaisant les mutations (*vorlu* → *morlu*), donne sa catégorie, son temps et sa personne, puis établit un mot à mot français ;
- **Gemma 4 26B A4B**, un modèle de langue à poids ouverts servi par [LM Studio](https://lmstudio.ai), rédige à partir de ce mot à mot une phrase en bon français. Gemma ne voit jamais le breton.

Le système fonctionne aussi en **RAG hybride** (*Retrieval-Augmented Generation*) sur un corpus d'environ 400 000 phrases bretonnes. Il fait une recherche par les mots (BM25 sur les lemmes d'Apertium) et une recherche par le sens (vecteurs [bge-m3](https://huggingface.co/BAAI/bge-m3)), fusionnées par *Reciprocal Rank Fusion*. Il sert à chercher dans le corpus, à expliquer un mot, ou à donner à Gemma des phrases déjà traduites par des humains comme exemples (option `--memoire`).

Le fonctionnement est expliqué en détail dans l'article [Traduire le breton en local : Apertium, Gemma 4 et un RAG hybride](https://www.anakeyn.com/2026/11/26/traduire-le-breton-en-local-apertium-gemma-4-et-un-rag-hybride/), sur le blog d'Anakeyn.

## Ce qu'il faut

| Élément | Version testée | Rôle |
|---|---|---|
| Linux (Ubuntu) | 26.04 LTS | |
| Python | 3.14 | |
| Apertium et apertium-br-fr | 3.9.12 et 0.5.1 (paquets Ubuntu) | analyse du breton, mot à mot |
| [LM Studio](https://lmstudio.ai) | 0.4.28 (Build 2) | serveur local des modèles |
| Gemma 4 26B A4B QAT (`google/gemma-4-26b-a4b-qat`) | | rédaction en français |
| bge-m3 (`lm-kit/text-embedding-bge-m3`) | | recherche par le sens |

## Matériel nécessaire

| Usage | Mémoire vive | Carte graphique |
|---|---|---|
| Traduire (Gemma 4 26B A4B) | 32 Go conseillés : le modèle occupe à lui seul environ 16 Go | inutile : Gemma tourne sur le processeur |
| Construire l'index (bge-m3) | 8 Go suffisent | facultative, mais beaucoup plus rapide : une carte NVIDIA de 4 Go (GTX 1050 Ti) suffit |

Configuration testée : portable Acer Aspire 7 (2017), Intel Core i5 à 4 cœurs, 32 Go de mémoire, GeForce GTX 1050 Ti 4 Go, Ubuntu 26.04 LTS.

**Durée de l'indexation.** bge-m3 calcule environ 36 phrases par seconde sur la carte graphique, contre 3 à 7 sur le processeur. Pour les quelque 400 000 phrases du corpus, il faut donc environ 3 heures sur la carte, et 16 à 37 heures sur le processeur. Sans carte graphique, l'option `--sans-embeddings` construit en quelques minutes un index de recherche par les mots seuls.

**Avec moins de mémoire**, on peut charger dans LM Studio un modèle plus petit et le désigner avec `--modele-chat`, au prix de la qualité de traduction. La commande `phrase`, qui traduit un texte, n'a besoin ni de l'index ni de bge-m3.
## Installation

```bash
sudo apt install python3-venv apertium apertium-br-fr
git clone https://github.com/Anakeyn/rag-breton.git
cd rag-breton
python3 -m venv ~/venv-rag
source ~/venv-rag/bin/activate
pip install -r requirements.txt
echo 'set +H' >> ~/.bashrc     # sinon bash interprète les « ! » des phrases bretonnes
```

Vérifications, environnement activé :

```bash
python3 -c "import numpy, requests, sklearn, openpyxl, fastapi"   # ne doit rien afficher
echo 'Demat' | apertium br-fr                                       # doit répondre « Bonjour »
```

Dans LM Studio, téléchargez Gemma 4 26B A4B QAT et bge-m3. Ensuite, démarrez le serveur et chargez les modèles :

```bash
lms server start
lms load bge-m3
lms load google/gemma-4-26b-a4b-qat
lms ps
```

Pour traduire (commande `phrase`), Gemma suffit. Pour chercher (`cherche`), bge-m3 suffit. `demande` utilise les deux.

## Traduire un texte

La commande `phrase` n'a pas besoin de l'index : on peut l'utiliser dès l'installation.

```bash
python3 rag_breton.py phrase "Ur sal-glas ? Sal-glas ar brezhoneg!"
python3 rag_breton.py phrase "$(cat lecon.txt)" --par-lots 5 --sortie lecon.html lecon.xlsx
```

| Option | Effet |
|---|---|
| `--sans-llm` | analyse Apertium et mot à mot seuls, instantanés, sans traduction |
| `--par-lots N` | traduit N phrases par requête (5 est un bon compromis) |
| `--une-par-une` | une requête par phrase : plus fidèle, mais lent sur un long texte |
| `--sortie fichier.html / .xlsx / .tsv` | fiche de relecture : breton, mot à mot, traduction, points à vérifier, colonne pour les corrections |
| `--memoire` (et `--exemples N`) | donne à Gemma des phrases proches déjà traduites par des humains (nécessite l'index) |
| `--toponymes fichier.tsv` | noms de lieux bretons avec leur nom français (voir « Données ») |
| `--commune Douarnenez` | commune dont parle le texte, pour choisir entre des lieux homonymes |
| `--complements`, `--variantes` | autres dictionnaires personnels que ceux du dépôt |
| `--modele-chat <identifiant>` | modèle de LM Studio à utiliser (identifiant donné par `lms ps`) |

Une fiche HTML déjà faite se convertit en Excel sans relancer la traduction : `python3 rag_breton.py fiche lecon.html lecon.xlsx`.

### Interface web

```bash
python3 interface_web.py      # puis ouvrir http://127.0.0.1:8000
```

ou, pour tout démarrer d'un coup (serveur LM Studio, Gemma, interface) : `source demarrerWeb.sh`.

L'interface lance simplement `rag_breton.py phrase` : elle donne les mêmes résultats et utilise les mêmes fichiers. Elle traduit un seul texte à la fois et n'écoute que la machine elle-même.

## Chercher dans le corpus

Ces commandes ont besoin de l'index (voir « Construire l'index »).

```bash
python3 rag_breton.py mot morlu                                 # toutes les phrases avec le lemme, toutes mutations
python3 rag_breton.py cherche "ar vro hag ar mor"               # recherche hybride, sans LLM
python3 rag_breton.py demande "Que disent les phrases sur la marine ?"
python3 rag_breton.py demande "Traduis et commente ces phrases" --cherche "morlu" -n 5
python3 rag_breton.py demande --explique --cherche "morlu" -n 3  # analyse grammaticale mot à mot
```

`--sans-vecteurs`, placé avant ou après la commande, force la recherche par les mots seuls.

## Données

**Aucun corpus n'est inclus dans ce dépôt.** Toutes les sources sont libres, mais volumineuses ; chacun les télécharge à leur source, puis construit son propre index. L'arborescence attendue est :

```
donnees/
├── corpus/          fichiers téléchargés
└── dictionnaires/   ofis_br_fr.tsv, tatoeba_br_fr.tsv, toponymes.tsv
```

| Source | Licence | Téléchargement | Fichier pour l'index |
|---|---|---|---|
| Wikipédia en breton, 100 000 phrases (Leipzig Corpora Collection, 2021) | CC BY | [wortschatz.uni-leipzig.de](https://wortschatz.uni-leipzig.de/en/download/bre), archive `bre_wikipedia_2021_100K` | `bre_wikipedia_2021_100K-sentences.txt`, lu tel quel |
| Wikipédia en breton, 258 229 phrases (gweltou, mars 2024) | jeu de données Apache 2.0, textes CC BY-SA | [huggingface.co/datasets/gweltou/wikipedia-br-20240325](https://huggingface.co/datasets/gweltou/wikipedia-br-20240325) | `data.txt`, renommé `wikipedia_br_2024.txt` |
| Paires breton–français de l'Office public de la langue bretonne | CC BY 4.0 | [github.com/Ofis-publik-ar-brezhoneg/breton-french-corpus](https://github.com/Ofis-publik-ar-brezhoneg/breton-french-corpus) | `ofis_br_fr.tsv` (colonnes à inverser, voir ci-dessous) |
| Tatoeba breton–français, via OPUS | CC BY 2.0 FR | [opus.nlpl.eu](https://opus.nlpl.eu/corpora-search/fr&br), format Moses | `tatoeba_br_fr.tsv` (voir ci-dessous) |
| Corpus aligné et base de toponymie KerOfis de l'Office | ODbL | [fr.brezhoneg.bzh, données libres](https://www.fr.brezhoneg.bzh/212-donnees-libres-de-droits.htm) | `korpus_oplb.tsv` et `toponymes.tsv` (voir ci-dessous) |

**Mise en forme des paires.** Le programme lit les paires au format `breton<TAB>français`.

```bash
# Office public : le fichier est en français<TAB>breton, on inverse les colonnes
awk -F'\t' 'NF>=2 {print $2"\t"$1}' fr-br-corpus.tsv > donnees/dictionnaires/ofis_br_fr.tsv
# Tatoeba (format Moses : deux fichiers alignés ligne à ligne)
paste Tatoeba.br-fr.br Tatoeba.br-fr.fr > donnees/dictionnaires/tatoeba_br_fr.tsv
# Données libres de l'Office : corpus aligné et KerOfis
python3 -m preparation.donnees_oplb corpus Korpus-frazennou-brezhonek.csv Korpus-frazennou-gallek.csv korpus_oplb.tsv
python3 -m preparation.donnees_oplb kerofis toponymie_lechanvadurezh_AAAAMMJJ.csv donnees/dictionnaires/toponymes.tsv
```

Vérifiez la première ligne de `ofis_br_fr.tsv` : si le fichier de l'Office commence par une ligne d'en-têtes, supprimez-la. Les noms des fichiers Tatoeba dépendent de la version téléchargée sur OPUS.

`toponymes.tsv` est lu par défaut dans `donnees/dictionnaires/`. S'il manque, les noms de lieux sont seulement devinés d'après leur majuscule.

## Construire l'index

L'index (`index_breton/`) contient les phrases, leurs lemmes Apertium et un vecteur bge-m3 par phrase. On le construit une fois, en donnant tous les fichiers du corpus, toujours dans le même ordre :

```bash
systemd-inhibit --what=sleep:idle python3 rag_breton_preparation.py \
  bre_wikipedia_2021_100K/bre_wikipedia_2021_100K-sentences.txt \
  wikipedia_br_2024.txt ofis_br_fr.tsv tatoeba_br_fr.tsv korpus_oplb.tsv
```

- Les doublons entre sources sont écartés.
- Pour ajouter un corpus, on le met **à la fin** de la liste. Seuls les vecteurs des nouvelles phrases sont calculés.
- Si le calcul est interrompu, il suffit de relancer la même commande : il reprend où il s'était arrêté.
- `--sans-embeddings` construit seulement l'index par les mots, en quelques minutes, sans bge-m3.
- `--max 200 --index essai` construit un petit index d'essai dans un autre dossier.

**Durée.** Calculer les vecteurs de 400 000 phrases prend du temps : sur le processeur, bge-m3 traite 3 à 7 phrases par seconde. Sur une carte graphique, même modeste (GTX 1050 Ti, 4 Go), il en traite environ 36. Il suffit pour cela de choisir le runtime CUDA dans LM Studio le temps de l'indexation. L'index complet occupe environ 1,6 Go.

## Dictionnaires personnels

- **`complements.tsv`** contient des sens et des expressions absents d'Apertium ou mal traduits par lui (`breton<TAB>français`). Ses entrées ont priorité sur Apertium.
- **`variantes.tsv`** ramène des graphies dialectales ou anciennes au breton unifié (*peurunvan*) avant l'analyse.

Les deux fichiers du dépôt ont été établis à partir des textes de mise au point du projet. Vous pouvez les compléter, ou en donner d'autres avec `--complements` et `--variantes`.

Les scripts de `outils/complements/` aident à les enrichir. `generer_a_valider.py` liste les mots qu'Apertium ne connaît pas dans un texte, et `integrer_validations.py` ajoute les propositions validées par un relecteur.

## Organisation du code

```
rag_breton_preparation.py   construire ou compléter l'index
rag_breton.py               traduire et chercher : mot, cherche, demande, phrase, fiche
interface_web.py            interface web locale (FastAPI) pour la commande phrase
demarrerWeb.sh              démarre LM Studio, Gemma et l'interface
complements.tsv             dictionnaire personnel : sens et expressions
variantes.tsv               graphies dialectales → breton unifié
commun/                     paramètres, appels à Apertium, vecteurs bge-m3
preparation/                lecture des corpus, écriture de l'index, données de l'Office
traduction/                 règles de grammaire, recherche, mémoire, prompts de Gemma, fiches
outils/complements/         enrichir complements.tsv et variantes.tsv
```

Les règles de grammaire sont dans `traduction/grammaire.py`, les prompts de Gemma dans `traduction/gemma.py`, les paramètres par défaut (adresse de LM Studio, modèle d'embedding) dans `commun/config.py`.

## Autres traducteurs breton → français en local

D'autres projets ont fait un autre choix : réentraîner un petit modèle de traduction sur des paires de phrases breton–français. Ces modèles comptent 418 millions de paramètres (contre 26 milliards pour Gemma 4 26B A4B, dont environ 4 milliards actifs pour chaque mot) et occupent environ 2 Go de mémoire. Ils tournent sur un portable courant, sans carte graphique.

N'hésitez pas à faire vos propres comparaisons :

- **Kellag** (breton → français), du projet [GweLLM](https://github.com/blackccpie/GweLLM) d'Albert Murienne : [amurienne/kellag-m2m100](https://huggingface.co/amurienne/kellag-m2m100). Le même projet propose aussi **Gallek**, pour l'autre sens (français → breton) ;
- **m2m100_418M_br_fr** de Loïc Grobol (breton → français) : [lgrobol/m2m100_418M_br_fr](https://huggingface.co/lgrobol/m2m100_418M_br_fr).

Il existe aussi des traducteurs en ligne, comme celui de l'[Office public de la langue bretonne](https://niverel.brezhoneg.bzh/fr/troer/) ou celui d'[Apertium](https://www.apertium.org/index.fra.html#?dir=bre-fra) (breton → français seulement, le même moteur que celui utilisé ici pour l'analyse), mais le texte à traduire est alors envoyé sur leurs serveurs.

## Projet lié

[Apertium2iramuteq-breton](https://github.com/Anakeyn/Apertium2iramuteq-breton) : un dictionnaire breton pour le logiciel d'analyse de textes IRaMuTeQ, tiré lui aussi d'Apertium.

## Licence

Code et dictionnaires personnels : **GPL-3.0-or-later**, © 2026 Anakeyn – Pierre Rouarch. Voir [LICENSE](LICENSE).

Apertium et apertium-br-fr (GPL 2), LM Studio, Gemma 4 (Apache 2.0) et bge-m3 (MIT) sont des dépendances installées à part, sous leurs propres licences. Les fichiers tirés des données de l'Office (`korpus_oplb.tsv`, `toponymes.tsv`) restent sous licence ODbL.

## Sources et citations

- Leipzig Corpora Collection (2021) : *Breton Wikipedia corpus based on material from 2021*. D. Goldhahn, T. Eckart et U. Quasthoff (2012), *Building Large Monolingual Dictionaries at the Leipzig Corpora Collection: From 100 to 200 Languages*, LREC'12.
- gweltou (Gweltaz DG), [wikipedia-br-20240325](https://huggingface.co/datasets/gweltou/wikipedia-br-20240325), Hugging Face.
- Office public de la langue bretonne : [breton-french-corpus](https://github.com/Ofis-publik-ar-brezhoneg/breton-french-corpus) (CC BY 4.0), [données libres](https://www.fr.brezhoneg.bzh/212-donnees-libres-de-droits.htm) (ODbL).
- Tatoeba, via OPUS : J. Tiedemann (2012), *Parallel Data, Tools and Interfaces in OPUS*, LREC'12.
- [Apertium](https://www.apertium.org) et [apertium-br-fr](https://github.com/apertium/apertium-br-fr).
