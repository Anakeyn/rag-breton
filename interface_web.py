# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anakeyn – Pierre Rouarch
"""
Interface web locale pour la commande « phrase » de rag_breton.py.

    pip install fastapi uvicorn
    python3 interface_web.py            # puis ouvrir http://127.0.0.1:8000

Trois zones : le texte breton à traduire, le résultat (traductions), et le
déroulement (tout ce qu'affiche rag_breton.py, en direct). L'interface ne fait
que lancer rag_breton.py dans un sous-processus : le programme en ligne de
commande reste la référence, avec les mêmes options et les mêmes fichiers
(complements.tsv, variantes.tsv, index_breton…).

Une seule traduction à la fois : Gemma occupe déjà tout le processeur.
Le serveur n'écoute que sur 127.0.0.1 (cette machine), pas sur le réseau.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

DOSSIER = os.path.dirname(os.path.abspath(__file__))
PROGRAMME = os.path.join(DOSSIER, "rag_breton.py")
FICHES = os.path.join(tempfile.gettempdir(), "rag_breton_fiches")

app = FastAPI(title="RAG breton")
travaux = {}                 # id -> {"proc", "file", "dossier", "fini"}
verrou = asyncio.Lock()


class Demande(BaseModel):
    texte: str
    par_lots: int = 5
    une_par_une: bool = False
    sans_llm: bool = False
    memoire: bool = False
    exemples: int = 2


async def lire(flux, nom, file):
    """Recopie un flux du sous-processus dans la file, morceau par morceau
    (pas ligne par ligne : la réponse de Gemma arrive mot à mot)."""
    while True:
        morceau = await flux.read(512)
        if not morceau:
            break
        await file.put((nom, morceau.decode("utf-8", errors="replace")))


async def executer(tid, commande):
    t = travaux[tid]
    async with verrou:
        t["proc"] = await asyncio.create_subprocess_exec(
            *commande, cwd=DOSSIER,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        await t["file"].put(("info", "Lancement : " + " ".join(
            c if len(c) < 60 else c[:57] + "…" for c in commande[1:]) + "\n"))
        await asyncio.gather(lire(t["proc"].stdout, "out", t["file"]),
                             lire(t["proc"].stderr, "err", t["file"]))
        code = await t["proc"].wait()
    fiches = sorted(f for f in os.listdir(t["dossier"])) if os.path.isdir(t["dossier"]) else []
    await t["file"].put(("fin", json.dumps({"code": code, "fiches": fiches})))


@app.post("/lancer")
async def lancer(d: Demande):
    if not d.texte.strip():
        raise HTTPException(400, "Texte vide.")
    if verrou.locked():
        raise HTTPException(409, "Une traduction est déjà en cours.")
    tid = uuid.uuid4().hex[:12]
    dossier = os.path.join(FICHES, tid)
    os.makedirs(dossier, exist_ok=True)
    commande = [sys.executable, "-u", PROGRAMME, "phrase", d.texte]
    if d.sans_llm:
        commande.append("--sans-llm")
    elif d.une_par_une:
        commande.append("--une-par-une")
    elif d.par_lots > 0:
        commande += ["--par-lots", str(d.par_lots)]
    if d.memoire:
        commande += ["--memoire", "--exemples", str(d.exemples)]
    commande += ["--sortie", os.path.join(dossier, "fiche.html"), os.path.join(dossier, "fiche.xlsx")]
    travaux[tid] = {"proc": None, "file": asyncio.Queue(), "dossier": dossier}
    asyncio.create_task(executer(tid, commande))
    return {"id": tid}


@app.get("/flux/{tid}")
async def flux(tid: str):
    """Déroulement en direct (Server-Sent Events)."""
    if tid not in travaux:
        raise HTTPException(404)
    file = travaux[tid]["file"]

    async def evenements():
        while True:
            nom, texte = await file.get()
            yield f"event: {nom}\ndata: {json.dumps(texte)}\n\n"
            if nom == "fin":
                break
    return StreamingResponse(evenements(), media_type="text/event-stream")


@app.post("/arreter/{tid}")
async def arreter(tid: str):
    t = travaux.get(tid)
    if t and t["proc"] and t["proc"].returncode is None:
        t["proc"].terminate()
    return {"ok": True}


@app.get("/fiche/{tid}/{nom}")
async def fiche(tid: str, nom: str):
    if nom not in ("fiche.html", "fiche.xlsx"):
        raise HTTPException(404)
    chemin = os.path.join(FICHES, tid, nom)
    if not os.path.exists(chemin):
        raise HTTPException(404)
    return FileResponse(chemin, filename=f"relecture_{tid}.{nom.split('.')[-1]}")


@app.get("/", response_class=HTMLResponse)
async def page():
    return PAGE


PAGE = r"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>RAG breton</title>
<style>
  :root { --fond:#f6f5f1; --carte:#fff; --bord:#d9d6cc; --texte:#222; --doux:#6b675d; --accent:#1f5f8b; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.45 system-ui, sans-serif; background:var(--fond); color:var(--texte); }
  header { padding:12px 20px; border-bottom:1px solid var(--bord); background:var(--carte);
           display:flex; align-items:baseline; gap:16px; }
  header h1 { font-size:18px; margin:0; }
  header span { color:var(--doux); font-size:13px; }
  main { display:grid; grid-template-columns:1fr 1fr;
         grid-template-rows:minmax(0, 1fr) minmax(0, 1fr); gap:14px;
         padding:14px 20px; height:calc(100vh - 50px); }
  section { background:var(--carte); border:1px solid var(--bord); border-radius:8px;
            display:flex; flex-direction:column; min-height:0; }
  section h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--doux);
               margin:0; padding:8px 12px; border-bottom:1px solid var(--bord);
               display:flex; justify-content:space-between; align-items:center; }
  #zone-breton { grid-row:1 / span 2; }
  textarea { flex:1; border:0; padding:12px; font:15px/1.5 Georgia, serif; resize:none; outline:none; }
  .options { padding:10px 12px; border-top:1px solid var(--bord); display:flex; flex-wrap:wrap;
             gap:10px 18px; align-items:center; font-size:14px; }
  .options input[type=number] { width:52px; }
  button { font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--accent);
           background:var(--accent); color:#fff; cursor:pointer; }
  button.second { background:#fff; color:var(--accent); }
  button:disabled { opacity:.45; cursor:default; }
  #resultat { flex:1; min-height:0; overflow:auto; padding:6px 12px; }
  #resultat table { border-collapse:collapse; width:100%; }
  #resultat td { padding:5px 6px; border-bottom:1px solid #eee; vertical-align:top; }
  #resultat td.num { color:var(--doux); width:2.5em; text-align:right; }
  #resultat td.br { color:var(--doux); font-style:italic; width:45%; }
  #deroulement { flex:1; overflow:auto; margin:0; padding:10px 12px; font:12.5px/1.4 ui-monospace, monospace;
                 white-space:pre-wrap; background:#1e1f22; color:#d6d6d6; border-radius:0 0 8px 8px; }
  #deroulement .err { color:#e8b86d; } #deroulement .info { color:#7fb5e0; }
  #etat { font-size:12px; text-transform:none; letter-spacing:0; }
  .liens a { margin-left:10px; font-size:13px; text-transform:none; letter-spacing:0; }
  .vide { color:var(--doux); padding:10px 0; }
</style></head>
<body>
<header><h1>RAG breton</h1><span>Apertium + Gemma (LM Studio) — traduction breton → français</span></header>
<main>
  <section id="zone-breton">
    <h2>Breton <label class="second" style="text-transform:none;letter-spacing:0;cursor:pointer">
      ouvrir un .txt <input type="file" id="fichier" accept=".txt" hidden></label></h2>
    <textarea id="texte" placeholder="Colle ici le texte breton (une ou plusieurs phrases)…"></textarea>
    <div class="options">
      <label>Mode
        <select id="mode">
          <option value="lots" selected>par lots</option>
          <option value="une">une phrase par requête</option>
          <option value="tout">tout en une requête</option>
          <option value="sansllm">analyse seule (sans LLM)</option>
        </select></label>
      <label id="lab-lots">taille <input type="number" id="lots" value="5" min="1" max="50"></label>
      <label><input type="checkbox" id="memoire"> mémoire (corpus traduit)</label>
      <label id="lab-ex" hidden>exemples <input type="number" id="exemples" value="2" min="1" max="5"></label>
      <span style="flex:1"></span>
      <button id="arreter" class="second" disabled>Arrêter</button>
      <button id="traduire">Traduire</button>
    </div>
  </section>
  <section>
    <h2>Résultat <span class="liens" id="liens"></span></h2>
    <div id="resultat"><div class="vide">Les traductions s'afficheront ici.</div></div>
  </section>
  <section>
    <h2>Déroulement <span id="etat"></span></h2>
    <pre id="deroulement"></pre>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let tid = null, sortie = "", debut = 0, minuteur = null;

$("fichier").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  const r = new FileReader(); r.onload = () => { $("texte").value = r.result.replace(/^﻿/, ""); };
  r.readAsText(f, "utf-8");
};
$("mode").onchange = () => { $("lab-lots").hidden = $("mode").value !== "lots"; };
$("memoire").onchange = () => { $("lab-ex").hidden = !$("memoire").checked; };

function ajouter(texte, classe) {
  const d = $("deroulement"), bas = d.scrollTop + d.clientHeight >= d.scrollHeight - 30;
  const s = document.createElement("span"); if (classe) s.className = classe;
  s.textContent = texte; d.appendChild(s);
  if (bas) d.scrollTop = d.scrollHeight;
}

const echap = t => t.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

// Rapproche les lignes « [n] … » de l'analyse (breton) et de la réponse (français)
function afficherResultat() {
  const i = sortie.indexOf("=== Réponse du modèle ===");
  const breton = {};
  const analyse = i >= 0 ? sortie.slice(0, i) : sortie;
  for (const m of analyse.matchAll(/^\[(\d+)\] (.*)$/gm)) breton[m[1]] = m[2].replace(/^﻿/, "");
  let lignes = [];
  if (i >= 0) {
    let rep = sortie.slice(i + 25); const j = rep.indexOf("=== Timing");
    if (j >= 0) rep = rep.slice(0, j);
    for (const m of rep.matchAll(/^\[(\d+)\]\s*(.*)$/gm)) lignes.push([m[1], m[2]]);
  } else if ($("mode").value === "sansllm") {
    for (const m of analyse.matchAll(/^\[(\d+)\][^\n]*\n(?:  [^\n]*\n)*?  Mot à mot : (.*)$/gm))
      lignes.push([m[1], "mot à mot : " + m[2]]);
  }
  if (!lignes.length) return;
  $("resultat").innerHTML = "<table>" + lignes.map(([n, fr]) =>
    `<tr><td class="num">${n}</td><td class="br">${echap(breton[n] || "")}</td><td>${echap(fr)}</td></tr>`
  ).join("") + "</table>";
}

function etat(t) { $("etat").textContent = t; }
function chrono() {
  const s = Math.round((Date.now() - debut) / 1000);
  etat(`en cours… ${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`);
}

$("traduire").onclick = async () => {
  const mode = $("mode").value;
  const corps = {
    texte: $("texte").value,
    par_lots: mode === "lots" ? parseInt($("lots").value) || 5 : 0,
    une_par_une: mode === "une", sans_llm: mode === "sansllm",
    memoire: $("memoire").checked, exemples: parseInt($("exemples").value) || 2,
  };
  const r = await fetch("/lancer", {method: "POST", headers: {"Content-Type": "application/json"},
                                    body: JSON.stringify(corps)});
  if (!r.ok) { const e = await r.json(); alert(e.detail || "Erreur"); return; }
  tid = (await r.json()).id; sortie = "";
  $("deroulement").textContent = ""; $("liens").innerHTML = "";
  $("resultat").innerHTML = '<div class="vide">Analyse en cours…</div>';
  $("traduire").disabled = true; $("arreter").disabled = false;
  debut = Date.now(); chrono(); minuteur = setInterval(chrono, 1000);

  const es = new EventSource(`/flux/${tid}`);
  es.addEventListener("out", e => { const t = JSON.parse(e.data); sortie += t; ajouter(t); afficherResultat(); });
  es.addEventListener("err", e => ajouter(JSON.parse(e.data), "err"));
  es.addEventListener("info", e => ajouter(JSON.parse(e.data), "info"));
  es.addEventListener("fin", e => {
    es.close(); clearInterval(minuteur);
    const f = JSON.parse(JSON.parse(e.data));
    const s = Math.round((Date.now() - debut) / 1000);
    etat((f.code === 0 ? "terminé" : `arrêté (code ${f.code})`) + ` en ${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`);
    $("liens").innerHTML = f.fiches.map(n =>
      `<a href="/fiche/${tid}/${n}" target="_blank">${n.endsWith("xlsx") ? "fiche Excel" : "fiche HTML"}</a>`).join("");
    $("traduire").disabled = false; $("arreter").disabled = true;
    afficherResultat();
  });
  es.onerror = () => { if (es.readyState === 2) { clearInterval(minuteur); $("traduire").disabled = false; } };
};

$("arreter").onclick = () => { if (tid) fetch(`/arreter/${tid}`, {method: "POST"}); };
</script>
</body></html>
"""


if __name__ == "__main__":
    shutil.rmtree(FICHES, ignore_errors=True)       # fiches de la session précédente
    print("Interface : http://127.0.0.1:8000   (Ctrl+C pour arrêter)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
