"""
Cold-start COMPLETO: evalua el modelo dual Y todos los baselines en el
MISMO protocolo (particion 90/10 warm/cold, contexto train_warm, negativos fijos).

A diferencia de cold_start_dual.py, este NO tiene numeros hardcodeados:
calcula TF-IDF, Item-CF y POP de verdad sobre la misma particion.
Asi la Tabla 5.2 es 100% comparable y reproducible.

Uso:
    python cold_start_dual_completo.py
"""
import sys, pickle, json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import (HISTORIAL_PATH, PRODUCT_IDS_PATH, EMBEDDINGS_PATH,
                    DATA_PROC, RESULTS, MODELO_BEST, SEQ_LEN, TOP_K, SEED,
                    EMB_DIM, N_FILTERS, DROPOUT, N_NEG_EVAL, COMBINE_MODE)
from model_dual import DualEmbeddingCNN


def metricas(hits, mrr, ndcg, total):
    return {"HR@10": hits/max(total,1), "MRR@10": mrr/max(total,1),
            "NDCG@10": ndcg/max(total,1), "n": total}


def main():
    print("COLD-START COMPLETO: DUAL + BASELINES (mismo protocolo)")
    random.seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datos
    historial = pd.read_pickle(HISTORIAL_PATH)
    producto_ids = np.load(PRODUCT_IDS_PATH)
    products = pd.read_csv(DATA_PROC / "products_filtrado.csv")
    pid2idx = {int(pid): i+1 for i, pid in enumerate(producto_ids)}
    pid2name = dict(zip(products['product_id'], products['product_name']))

    # Particion warm/cold (misma seed)
    pl = list(producto_ids); random.shuffle(pl)
    n_cold = int(len(pl)*0.10)
    productos_cold = set(pl[:n_cold]); productos_warm = set(pl[n_cold:])
    historial['test_es_cold'] = historial['test'].apply(lambda p: p in productos_cold)
    historial['train_warm'] = historial['train'].apply(
        lambda l: [p for p in l if p in productos_warm])
    historial['n_tw'] = historial['train_warm'].apply(len)
    h = historial[historial['n_tw'] >= 3].reset_index(drop=True)
    print(f"Cold: {len(productos_cold):,} | Warm: {len(productos_warm):,}")
    print(f"Eval - cold: {h['test_es_cold'].sum():,} | warm: {(~h['test_es_cold']).sum():,}")

    # Negativos fijos (compartidos por TODOS los modelos)
    print("\nGenerando negativos fijos...")
    todos = set(pid2idx.keys()); negs_d = {}
    for _, row in tqdm(h.iterrows(), total=len(h)):
        uid=int(row['user_id']); pos=row['test']
        if pos not in pid2idx: continue
        excl = set(row['train_warm']) | {pos}
        if pd.notna(row.get('val')): excl.add(row['val'])
        rng = random.Random(SEED + 77777 + uid)
        negs_d[uid] = rng.sample(list(todos-excl), min(N_NEG_EVAL, len(todos-excl)))

    resultados = {}

    # 1. MODELO DUAL
    print("\n[1/4] Modelo DUAL...")
    model = DualEmbeddingCNN(len(producto_ids), EMB_DIM, N_FILTERS, DROPOUT,
                             SEQ_LEN, COMBINE_MODE)
    model.init_frozen_embeddings(np.load(EMBEDDINGS_PATH))
    ckpt = torch.load(MODELO_BEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    hc=mc=nc=tc=hw=mw=nw=tw=0
    with torch.no_grad():
        for _, row in tqdm(h.iterrows(), total=len(h)):
            uid=int(row['user_id']); pos=row['test']; cold=row['test_es_cold']
            if pos not in pid2idx or uid not in negs_d: continue
            seq = row['train_warm'][-SEQ_LEN:]; seq=[0]*(SEQ_LEN-len(seq))+seq
            seq_t = torch.tensor([[pid2idx.get(p,0) for p in seq]],dtype=torch.long).to(device)
            perfil = model(seq_t)
            cand = list(negs_d[uid])+[pos]
            ci = [pid2idx[p]-1 for p in cand if p in pid2idx]
            if len(ci)<len(cand): continue
            ct = torch.tensor(ci,dtype=torch.long).to(device)+1
            cm = torch.zeros(len(ci),dtype=torch.bool).to(device)
            if cold: cm[-1]=True
            emb = F.normalize(model.combinar(ct.unsqueeze(0),cm.unsqueeze(0)).squeeze(0),dim=-1)
            sc = (emb @ perfil.squeeze(0)).cpu().numpy()
            rank = int((sc>=sc[-1]).sum())
            if cold:
                tc+=1
                if rank<=TOP_K: hc+=1; mc+=1/rank; nc+=1/np.log2(rank+1)
            else:
                tw+=1
                if rank<=TOP_K: hw+=1; mw+=1/rank; nw+=1/np.log2(rank+1)
    resultados["Dual"] = {"warm": metricas(hw,mw,nw,tw), "cold": metricas(hc,mc,nc,tc)}
    del model; torch.cuda.empty_cache()
    print(f"  Dual: warm={resultados['Dual']['warm']['HR@10']:.4f} cold={resultados['Dual']['cold']['HR@10']:.4f}")

    # 2. TF-IDF
    print("\n[2/4] TF-IDF...")
    tfidf = TfidfVectorizer(max_features=10000, lowercase=True,
                            stop_words='english', ngram_range=(1,2))
    nombres = products['product_name'].str.lower().tolist()
    tfm = normalize(tfidf.fit_transform(nombres).toarray().astype(np.float32), axis=1)
    pid2tf = {row['product_id']: i for i, row in products.iterrows()}
    hc=mc=nc=tc=hw=mw=nw=tw=0
    for _, row in tqdm(h.iterrows(), total=len(h)):
        uid=int(row['user_id']); pos=row['test']; cold=row['test_es_cold']
        if pos not in pid2idx or uid not in negs_d or pos not in pid2tf: continue
        seq = row['train_warm'][-SEQ_LEN:]
        vecs = [tfm[pid2tf[p]] for p in seq if p in pid2tf]
        if not vecs: continue
        perfil = np.mean(vecs,axis=0); perfil/=(np.linalg.norm(perfil)+1e-8)
        cand = list(negs_d[uid])+[pos]
        ci = [pid2tf[p] for p in cand if p in pid2tf]
        if len(ci)<len(cand): continue
        sc = tfm[ci] @ perfil
        rank = int((sc>=sc[-1]).sum())
        if cold:
            tc+=1
            if rank<=TOP_K: hc+=1; mc+=1/rank; nc+=1/np.log2(rank+1)
        else:
            tw+=1
            if rank<=TOP_K: hw+=1; mw+=1/rank; nw+=1/np.log2(rank+1)
    resultados["TF-IDF"] = {"warm": metricas(hw,mw,nw,tw), "cold": metricas(hc,mc,nc,tc)}
    print(f"  TF-IDF: warm={resultados['TF-IDF']['warm']['HR@10']:.4f} cold={resultados['TF-IDF']['cold']['HR@10']:.4f}")

    # 3. ITEM-CF
    print("\n[3/4] Item-CF...")
    ul, il = [], []
    for ui,(_,row) in enumerate(h.iterrows()):
        for pid in set(row['train_warm']):
            if pid in pid2idx: ul.append(ui); il.append(pid2idx[pid]-1)
    n_items = max(pid2idx.values())
    M = csr_matrix((np.ones(len(ul),dtype=np.float32),(ul,il)),shape=(len(h),n_items))
    Mn = normalize(M,axis=0,norm='l2').tocsc()
    hc=mc=nc=tc=hw=mw=nw=tw=0
    for _, row in tqdm(h.iterrows(), total=len(h)):
        uid=int(row['user_id']); pos=row['test']; cold=row['test_es_cold']
        if pos not in pid2idx or uid not in negs_d: continue
        seq = row['train_warm'][-SEQ_LEN:]
        si = [pid2idx[p]-1 for p in seq if p in pid2idx]
        if not si: continue
        perfil = np.asarray(Mn[:,si].sum(axis=1)).flatten()
        cand = list(negs_d[uid])+[pos]
        ci = [pid2idx[p]-1 for p in cand if p in pid2idx]
        if len(ci)<len(cand): continue
        sc = (Mn[:,ci].T @ perfil).astype(np.float32)
        rank = int((sc>=sc[-1]).sum())
        if cold:
            tc+=1
            if rank<=TOP_K: hc+=1; mc+=1/rank; nc+=1/np.log2(rank+1)
        else:
            tw+=1
            if rank<=TOP_K: hw+=1; mw+=1/rank; nw+=1/np.log2(rank+1)
    resultados["Item-CF"] = {"warm": metricas(hw,mw,nw,tw), "cold": metricas(hc,mc,nc,tc)}
    print(f"  Item-CF: warm={resultados['Item-CF']['warm']['HR@10']:.4f} cold={resultados['Item-CF']['cold']['HR@10']:.4f}")

    # 4. POP
    print("\n[4/4] POP...")
    freq = defaultdict(int)
    for _, row in h.iterrows():
        for pid in row['train_warm']: freq[pid]+=1
    hc=mc=nc=tc=hw=mw=nw=tw=0
    for _, row in tqdm(h.iterrows(), total=len(h)):
        uid=int(row['user_id']); pos=row['test']; cold=row['test_es_cold']
        if pos not in pid2idx or uid not in negs_d: continue
        cand = list(negs_d[uid])+[pos]
        sc = np.array([freq[c] for c in cand],dtype=np.float32)
        rank = int((sc>=sc[-1]).sum())
        if cold:
            tc+=1
            if rank<=TOP_K: hc+=1; mc+=1/rank; nc+=1/np.log2(rank+1)
        else:
            tw+=1
            if rank<=TOP_K: hw+=1; mw+=1/rank; nw+=1/np.log2(rank+1)
    resultados["POP"] = {"warm": metricas(hw,mw,nw,tw), "cold": metricas(hc,mc,nc,tc)}
    print(f"  POP: warm={resultados['POP']['warm']['HR@10']:.4f} cold={resultados['POP']['cold']['HR@10']:.4f}")

    #TABLA 5.2
    print("TABLA 5.2 — COMPARACION FINAL (mismo protocolo, warm + cold)")
    print(f"\n{'Modelo':<20} {'Warm HR@10':>11} {'Warm MRR@10':>12} {'Cold HR@10':>11} {'Cold MRR@10':>12}")
    orden = ["Item-CF", "Dual", "POP", "TF-IDF"]
    for m in orden:
        r = resultados[m]
        print(f"{m:<20} {r['warm']['HR@10']:>11.4f} {r['warm']['MRR@10']:>12.4f} "
              f"{r['cold']['HR@10']:>11.4f} {r['cold']['MRR@10']:>12.4f}")

    print("\nNota: todos evaluados con particion 90/10, contexto train_warm,")
    print("99 negativos fijos por usuario (misma seed). Protocolo identico.")

    out = RESULTS / "tabla_5_2_completa.json"
    with open(out, 'w') as f:
        json.dump(resultados, f, indent=2, default=float)
    print(f"Guardado: {out}")
    return resultados


if __name__ == "__main__":
    main()
