"""API de demostración sistema de recomendación dual."""

import json, pickle
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from tokenizers import Tokenizer

CKPT = "experiments/mejor_modelo_dual.pt"
PROC = Path("data/processed")
ASSET = Path("demo_assets")

#Pesos del checkpoint
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(sd, dict):
    for k in ("model_state_dict","state_dict","model"):
        if k in sd and isinstance(sd[k], dict): sd = sd[k]; break
EF, ET = sd["emb_frozen.weight"].float(), sd["emb_trainable.weight"].float()
G0w,G0b = sd["gate.0.weight"].float(), sd["gate.0.bias"].float()
G2w,G2b = sd["gate.2.weight"].float(), sd["gate.2.bias"].float()
PW,PB   = sd["projection.weight"].float(), sd["projection.bias"].float()
CONV = {k:(sd[f"conv{k}.0.weight"].float(), sd[f"conv{k}.0.bias"].float(),
           sd[f"conv{k}.1.running_mean"].float(), sd[f"conv{k}.1.running_var"].float(),
           sd[f"conv{k}.1.weight"].float(), sd[f"conv{k}.1.bias"].float()) for k in (3,5,7)}

# Datos: nombres, mapeo de IDs, tabla de tokens, usuarios muestra
nombres = pickle.load(open(PROC/"productos_nombres.pkl","rb"))
producto_ids = np.load(PROC/"producto_ids.npy")
pid2idx = {int(p): i+1 for i, p in enumerate(producto_ids)}      # +1: 0 es padding
idx2nombre = {i+1: nombres.get(int(producto_ids[i]), "?") for i in range(len(producto_ids))}
TOKVEC = torch.tensor(np.load(ASSET/"token_vectors.npy"))        # (8000,300)
tok = Tokenizer.from_file(str(PROC/"bpe_tokenizer.json"))
import pandas as pd
_hist_df = pickle.load(open(PROC/"historial_usuarios.pkl","rb"))
_hist_df = _hist_df.set_index("user_id")          # índice por user_id para búsqueda O(1)
def _historial_usuario(uid):
    if uid not in _hist_df.index: return None
    return list(_hist_df.loc[uid, "train"])[-10:]  # últimos 10 del historial de entrenamiento

@torch.no_grad()
def dual(idx):                                   # idx: LongTensor -> embeddings duales
    f, t = EF[idx], ET[idx]
    h = torch.relu(F.linear(torch.cat([f,t],-1), G0w, G0b))
    a = torch.sigmoid(F.linear(h, G2w, G2b))
    return a*f + (1-a)*t                          # gate sobre la rama semántica (tu forward)

@torch.no_grad()
def perfil(seq):                                  # seq: (1,T) -> perfil (1,300) L2
    x = dual(seq).transpose(1,2)                  # (1,300,T)
    outs=[]
    for k,(w,b,rm,rv,bw,bb) in CONV.items():
        h = F.conv1d(x, w, b, padding=(k-1)//2)   # padding same
        h = torch.relu(F.batch_norm(h, rm, rv, bw, bb, training=False, eps=1e-5))
        outs.append(h.mean(dim=2))                # GAP
    return F.normalize(F.linear(torch.cat(outs,1), PW, PB), dim=1)

# Catálogo de embeddings duales, normalizado (padding fuera)
CAT = F.normalize(dual(torch.arange(EF.shape[0])), dim=1); CAT[0]=0

def vector_cold(nombre):                          # nombre nuevo -> vector semántico (media de tokens)
    ids = tok.encode(nombre.lower()).ids
    if not ids: return None, []
    v = TOKVEC[ids].mean(dim=0)                    # MEDIA (igual que construir_embeddings_productos)
    return v, tok.encode(nombre.lower()).tokens

app = FastAPI(title="Recomendador Dual — Demo")

@app.get("/")
def home(): return FileResponse("index.html")

import random as _random
_ids_validos = json.load(open(ASSET/"ids_validos.json"))
@app.get("/api/usuario-aleatorio")
def usuario_aleatorio():
    return {"user_id": _random.choice(_ids_validos)}

@app.get("/api/recomendaciones/{uid}")
def recomendar(uid: int, k: int = 10):
    _h = _historial_usuario(uid)
    if _h is None: raise HTTPException(404, "usuario no existe en el dataset")
    hist_pids = [int(p) for p in _h]
    seq_idx = [pid2idx.get(p,0) for p in hist_pids]
    seq = torch.tensor([[0]*(10-len(seq_idx))+seq_idx], dtype=torch.long)
    s = (perfil(seq) @ CAT.T)[0]
    s[list(set(seq_idx))] = -1e9                  # no repetir lo ya visto
    top = torch.topk(s, k)
    historial = [nombres.get(p, "?") for p in hist_pids]
    return {"historial": historial,
            "recomendaciones":[{"nombre": idx2nombre.get(int(i),"?"), "score": float(v)}
                               for v,i in zip(top.values, top.indices)]}

class ColdReq(BaseModel):
    nombre_producto: str
    user_id: Optional[int] = None
    k_vecinos: int = 5

@app.post("/api/cold-start")
def cold_start(req: ColdReq):
    v, tokens = vector_cold(req.nombre_producto)
    if v is None: raise HTTPException(400, "nombre vacío o sin tokens")
    t = torch.zeros_like(v)                        # rama colaborativa ausente
    h = torch.relu(F.linear(torch.cat([v,t],-1), G0w, G0b))
    a = float(torch.sigmoid(F.linear(h, G2w, G2b)))
    e = F.normalize(a*v + (1-a)*t, dim=0)
    sims = CAT @ e; top = torch.topk(sims, req.k_vecinos)
    out = {"tokens_bpe": tokens, "alpha_gate": round(a,4),
           "vecinos_semanticos":[{"nombre": idx2nombre.get(int(i),"?"), "similitud": float(s)}
                                 for s,i in zip(top.values, top.indices)],
           "ranking_usuario": None}
    _h = _historial_usuario(req.user_id) if req.user_id is not None else None
    if _h is not None:
        seq_idx = [pid2idx.get(int(p),0) for p in _h]
        seq = torch.tensor([[0]*(10-len(seq_idx))+seq_idx], dtype=torch.long)
        p = perfil(seq)[0]; score = float(p @ e)
        pos = int((CAT @ p > score).sum())+1
        out["ranking_usuario"] = {"posicion": pos, "score": round(score,4),
                                  "entra_top10": pos<=10, "total_catalogo": int(CAT.shape[0]-1)}
    return out
