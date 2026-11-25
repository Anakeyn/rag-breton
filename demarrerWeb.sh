# demarrerWeb.sh — démarre LM Studio, charge Gemma et lance l'interface web.
# À lancer depuis le dossier du projet avec : source demarrerWeb.sh
# L'environnement Python est cherché dans ~/venv-rag ; pour un autre chemin :
#   VENV=~/mon-venv source demarrerWeb.sh
source "${VENV:-$HOME/venv-rag}/bin/activate"
set +H
lms server start
lms ps | grep -q gemma-4-26b-a4b-qat || lms load google/gemma-4-26b-a4b-qat
lms ps
free -h
[ "$(free -m | awk '/^(Échange|Swap)/{print $3}')" -gt 500 ] && echo "Attention : échange (swap) utilisé, les traductions seront plus lentes."
python3 interface_web.py
