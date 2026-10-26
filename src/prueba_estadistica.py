"""
Prueba de significancia estadistica para comparar el modelo dual contra baselines.

Tests implementados:
  - McNemar: compara dos modelos en aciertos/fallos pareados por usuario (HR@10).
             Apropiado para metricas binarias (acierto si rank<=10).

Se comparan en cold-start y warm-start:
  - Dual vs Item-CF
  - Dual vs TF-IDF
  - Dual vs POP

Genera matriz de contingencia, estadistico, p-valor e interpretacion.

Uso:
    python prueba_estadistica.py
"""
import sys
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import (
    HISTORIAL_PATH, PRODUCT_IDS_PATH, EMBEDDINGS_PATH, DATA_PROC, RESULTS,
    MODELO_BEST, SEQ_LEN, TOP_K, SEED, EMB_DIM, N_FILTERS, DROPOUT,
    N_NEG_EVAL, COMBINE_MODE
)
from model_dual import DualEmbeddingCNN


def acierto_y_rank(scores):
    """Dado el array de scores (ultimo = positivo), devuelve (acierto_bin, rank)."""
    rank = int((scores >= scores[-1]).sum())
    return (1 if rank <= TOP_K else 0), rank


def main():
    print("PRUEBA DE SIGNIFICANCIA ESTADISTICA: DUAL vs BASELINES")
    random.seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Cargar datos
    print("Cargando datos")
    historial = pd.read_pickle(HISTORIAL_PATH)
    producto_ids = np.load(PRODUCT_IDS_PATH)
    products = pd.read_csv(DATA_PROC / "products_filtrado.csv")
    pid2idx = {int(pid): i+1 for i, pid in enumerate(producto_ids)}

    #Particion warm/cold (misma seed que cold_start_dual)
    productos_lista = list(producto_ids); random.shuffle(productos_lista)
    n_cold = int(len(productos_lista) * 0.10)
    productos_cold = set(productos_lista[:n_cold])
    productos_warm = set(productos_lista[n_cold:])
    historial['test_es_cold'] = historial['test'].apply(lambda p: p in productos_cold)
    historial['train_warm'] = historial['train'].apply(
        lambda lst: [p for p in lst if p in productos_warm])
    historial['n_tw'] = historial['train_warm'].apply(len)
    h = historial[historial['n_tw'] >= 3].reset_index(drop=True)

    #Negativos fijos
    todos = set(pid2idx.keys()); negs_d = {}
    for _, row in tqdm(h.iterrows(), total=len(h)):
        uid = int(row['user_id']); pos = row['test']
        if pos not in pid2idx: continue
        excl = set(row['train_warm']) | {pos}
        if pd.notna(row.get('val')): excl.add(row['val'])
        rng = random.Random(SEED + 77777 + uid)
        negs_d[uid] = rng.sample(list(todos - excl), min(N_NEG_EVAL, len(todos-excl)))

    #Recolectar aciertos por usuario para cada modelo
    #Estructura: aciertos[modelo][escenario][uid] = (bin, rank)
    aciertos = defaultdict(lambda: {"warm": {}, "cold": {}})

    #MODELO DUAL
    model = DualEmbeddingCNN(len(producto_ids), EMB_DIM, N_FILTERS, DROPOUT,
                             SEQ_LEN, COMBINE_MODE)
    model.init_frozen_embeddings(np.load(EMBEDDINGS_PATH))
    ckpt = torch.load(MODELO_BEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    with torch.no_grad():
        for _, row in tqdm(h.iterrows(), total=len(h)):
            uid=int(row['user_id']); pos=row['test']; es_cold=row['test_es_cold']
            if pos not in pid2idx or uid not in negs_d: continue
            seq = row['train_warm'][-SEQ_LEN:]
            seq = [0]*(SEQ_LEN-len(seq)) + seq
            seq_t = torch.tensor([[pid2idx.get(p,0) for p in seq]], dtype=torch.long).to(device)
            perfil = model(seq_t)
            cand = list(negs_d[uid]) + [pos]
            cand_idx = [pid2idx[p]-1 for p in cand if p in pid2idx]
            if len(cand_idx) < len(cand): continue
            cand_t = torch.tensor(cand_idx, dtype=torch.long).to(device) + 1
            cmask = torch.zeros(len(cand_idx), dtype=torch.bool).to(device)
            if es_cold: cmask[-1] = True
            emb = F.normalize(model.combinar(cand_t.unsqueeze(0), cmask.unsqueeze(0)).squeeze(0), dim=-1)
            scores = (emb @ perfil.squeeze(0)).cpu().numpy()
            esc = "cold" if es_cold else "warm"
            aciertos["Dual"][esc][uid] = acierto_y_rank(scores)
    del model; torch.cuda.empty_cache()

    #ITEM-CF
    user_list, item_list = [], []
    for ui, (_, row) in enumerate(h.iterrows()):
        for pid in set(row['train_warm']):
            if pid in pid2idx:
                user_list.append(ui); item_list.append(pid2idx[pid]-1)
    n_items = max(pid2idx.values())
    M = csr_matrix((np.ones(len(user_list), dtype=np.float32), (user_list, item_list)),
                   shape=(len(h), n_items))
    M_norm = normalize(M, axis=0, norm='l2').tocsc()
    for ui, (_, row) in enumerate(tqdm(h.iterrows(), total=len(h))):
        uid=int(row['user_id']); pos=row['test']; es_cold=row['test_es_cold']
        if pos not in pid2idx or uid not in negs_d: continue
        seq = row['train_warm'][-SEQ_LEN:]
        seq_items = [pid2idx[p]-1 for p in seq if p in pid2idx]
        if not seq_items: continue
        perfil = np.asarray(M_norm[:, seq_items].sum(axis=1)).flatten()
        cand = list(negs_d[uid]) + [pos]
        cand_idx = [pid2idx[p]-1 for p in cand if p in pid2idx]
        if len(cand_idx) < len(cand): continue
        scores = (M_norm[:, cand_idx].T @ perfil).astype(np.float32)
        esc = "cold" if es_cold else "warm"
        aciertos["ItemCF"][esc][uid] = acierto_y_rank(scores)

    #TF-IDF
    tfidf = TfidfVectorizer(max_features=10000, lowercase=True,
                            stop_words='english', ngram_range=(1,2))
    nombres = products['product_name'].str.lower().tolist()
    tfm = normalize(tfidf.fit_transform(nombres).toarray().astype(np.float32), axis=1)
    pid2tf = {row['product_id']: i for i, row in products.iterrows()}
    for _, row in tqdm(h.iterrows(), total=len(h)):
        uid=int(row['user_id']); pos=row['test']; es_cold=row['test_es_cold']
        if pos not in pid2idx or uid not in negs_d or pos not in pid2tf: continue
        seq = row['train_warm'][-SEQ_LEN:]
        vecs = [tfm[pid2tf[p]] for p in seq if p in pid2tf]
        if not vecs: continue
        perfil = np.mean(vecs, axis=0); perfil /= (np.linalg.norm(perfil)+1e-8)
        cand = list(negs_d[uid]) + [pos]
        cand_idx = [pid2tf[p] for p in cand if p in pid2tf]
        if len(cand_idx) < len(cand): continue
        scores = tfm[cand_idx] @ perfil
        esc = "cold" if es_cold else "warm"
        aciertos["TFIDF"][esc][uid] = acierto_y_rank(scores)

    #PRUEBA ESTADISTICA
    print("RESULTADO DE LA PRUEBA ESTADISTICA")

    comparaciones = [("Dual", "ItemCF"), ("Dual", "TFIDF")]
    resultados = {}

    for esc in ["cold", "warm"]:
        print(f"ESCENARIO: {esc.upper()}")
        for m1, m2 in comparaciones:
            # Usuarios comunes evaluados por ambos modelos
            comunes = set(aciertos[m1][esc].keys()) & set(aciertos[m2][esc].keys())
            if len(comunes) < 10:
                print(f"\n{m1} vs {m2}: muy pocos usuarios comunes ({len(comunes)})")
                continue

            # McNemar sobre aciertos binarios (HR@10)
            # Tabla 2x2: [m1_acierta][m2_acierta]
            b01 = b10 = b00 = b11 = 0
            for uid in comunes:
                a1 = aciertos[m1][esc][uid][0]
                a2 = aciertos[m2][esc][uid][0]
                if a1 and a2: b11 += 1
                elif a1 and not a2: b10 += 1
                elif not a1 and a2: b01 += 1
                else: b00 += 1

            tabla = [[b11, b10], [b01, b00]]
            # McNemar: usa b10 y b01 (discordantes)
            res_mc = mcnemar(tabla, exact=(b10+b01 < 25))
            stat_mc = res_mc.statistic
            p_mc = res_mc.pvalue

            hr1 = sum(aciertos[m1][esc][uid][0] for uid in comunes) / len(comunes)
            hr2 = sum(aciertos[m2][esc][uid][0] for uid in comunes) / len(comunes)

            print(f"{m1} vs {m2} (n={len(comunes):,})")
            print(f"  HR@10: {m1}={hr1:.4f}, {m2}={hr2:.4f} (dif {(hr1-hr2)*100:+.2f} pp)")
            print(f"  Tabla discordancia: {m1} solo acierta={b10}, {m2} solo acierta={b01}")
            print(f"  McNemar:  stat={stat_mc:.3f}, p-valor={p_mc:.2e}")
            sig = "SI (p<0.05)" if p_mc < 0.05 else "NO (p>=0.05)"
            print(f"  Diferencia significativa (McNemar): {sig}")

            resultados[f"{esc}_{m1}_vs_{m2}"] = {
                "n": len(comunes), "HR_m1": hr1, "HR_m2": hr2,
                "discordancia_m1_solo": b10, "discordancia_m2_solo": b01,
                "mcnemar_stat": float(stat_mc), "mcnemar_p": float(p_mc),
                "significativo": bool(p_mc < 0.05)
            }

    out = RESULTS / "prueba_estadistica.json"
    with open(out, 'w') as f:
        json.dump(resultados, f, indent=2, default=float)
    print(f"Guardado: {out}")
    return resultados


if __name__ == "__main__":
    main()
