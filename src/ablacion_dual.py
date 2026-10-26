"""
Ablación del modelo DUAL para el Objetivo 5.

Entrena y evalúa 4 variantes para aislar la contribución de cada componente:
  A. Completo:        gate + cold_dropout (p=0.4)   [modelo completo]
  B. Sin cold dropout: gate + p_cold=0              [prueba que cold dropout importa]
  C. Sin congelada:   solo entrenable + cold dropout [prueba que FastText importa]
  D. Modo sum:        sum + cold dropout            [justifica el gate]

Cada variante: entrena, evalúa warm y cold, guarda métricas.
Reutiliza training_dual.py y cold_start_dual.py.

Uso:
    python ablacion_dual.py                # corre las 4
    python ablacion_dual.py --solo-clave   # corre solo A y B (rápido)
"""
import sys
import json
import time
import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import (
    HISTORIAL_PATH, EMBEDDINGS_PATH, PRODUCT_IDS_PATH,
    NEGATIVOS_VAL, NEGATIVOS_TEST, RESULTS, EXPERIMENTS, DATA_PROC,
    EMB_DIM, N_FILTERS, DROPOUT, SEQ_LEN,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    N_NEG_TRAIN, N_NEG_EVAL, EPOCHS, PATIENCE, EVAL_EVERY, TOP_K, SEED
)
from model_dual import DualEmbeddingCNN
import training_dual as TD
import pickle


# Configuración de las 4 variantes
VARIANTES = {
    "A_completo":        {"combine": "gate", "p_cold": 0.40, "usar_congelada": True},
    "B_sin_colddropout": {"combine": "gate", "p_cold": 0.00, "usar_congelada": True},
    "C_sin_congelada":   {"combine": "gate", "p_cold": 0.40, "usar_congelada": False},
    "D_modo_sum":        {"combine": "sum",  "p_cold": 0.40, "usar_congelada": True},
}


def entrenar_variante(nombre, cfg, historial, emb_matrix, producto_ids,
                       pid2idx, neg_val, dataset, device, max_epochs=40):
    """Entrena una variante y devuelve el mejor modelo."""
    print(f"VARIANTE {nombre}: combine={cfg['combine']}, "
          f"p_cold={cfg['p_cold']}, congelada={cfg['usar_congelada']}")

    set_seed = TD.set_seed
    set_seed(SEED)

    model = DualEmbeddingCNN(
        n_productos=len(producto_ids), emb_dim=EMB_DIM, n_filters=N_FILTERS,
        dropout=DROPOUT, seq_len=SEQ_LEN, combine_mode=cfg['combine']
    )

    # Tabla congelada: con FastText (normal) o con ruido (ablación C)
    if cfg['usar_congelada']:
        model.init_frozen_embeddings(emb_matrix)
    else:
        # Ablación: tabla congelada = ruido aleatorio (anula la semántica)
        ruido = np.random.RandomState(SEED).normal(0, 0.01, emb_matrix.shape).astype(np.float32)
        model.init_frozen_embeddings(ruido)
        print("Tabla congelada = RUIDO (ablación: sin semántica FastText)")

    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5)

    mejor_hr = 0.0
    paciencia = 0
    mejor_state = None
    p_cold = cfg['p_cold']

    for epoch in range(1, max_epochs+1):
        model.train()
        total_loss = 0
        for seq, pos, negs in tqdm(dataloader, desc=f"  Ep{epoch:2d}", leave=False):
            optimizer.zero_grad()
            # Usar el bpr_loss_dual con el p_cold de esta variante
            loss = TD.bpr_loss_dual(model, seq, pos, negs, device, p_cold=p_cold)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            m = TD.evaluar(model, historial, 'val', neg_val, pid2idx, device)
            if m["HR@10"] > mejor_hr:
                mejor_hr = m["HR@10"]
                paciencia = 0
                mejor_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                paciencia += EVAL_EVERY
                if paciencia >= PATIENCE:
                    print(f" Early stop ep{epoch} (val HR@10={mejor_hr:.4f})")
                    break
        if epoch % 10 == 0:
            print(f"  Ep{epoch:2d} | loss {total_loss/len(dataloader):.4f} | "
                  f"val HR@10 {mejor_hr:.4f}")

    # Cargar mejor estado
    model.load_state_dict(mejor_state)
    model.eval()
    return model, mejor_hr


def evaluar_cold_warm_variante(model, historial, productos_cold, pid2idx,
                                negs_d, device):
    """Evalúa warm + cold para una variante (mismo protocolo que cold_start_dual)."""
    hc=mc=nc=tc=0; hw=mw=nw=tw=0
    with torch.no_grad():
        for _, row in historial.iterrows():
            uid=int(row['user_id']); pos=row['test']; es_cold=row['test_es_cold']
            if pos not in pid2idx or uid not in negs_d: continue
            seq = row['train_warm'][-SEQ_LEN:]
            seq = [0]*(SEQ_LEN-len(seq)) + seq
            seq_idx = [pid2idx.get(p,0) for p in seq]
            seq_t = torch.tensor([seq_idx], dtype=torch.long).to(device)
            perfil = model(seq_t)

            cand = list(negs_d[uid]) + [pos]
            cand_idx = [pid2idx[p]-1 for p in cand if p in pid2idx]
            if len(cand_idx) < len(cand): continue
            cand_t = torch.tensor(cand_idx, dtype=torch.long).to(device) + 1

            cmask = torch.zeros(len(cand_idx), dtype=torch.bool).to(device)
            if es_cold:
                cmask[-1] = True  # positivo cold = solo semántica

            emb = model.combinar(cand_t.unsqueeze(0), cmask.unsqueeze(0)).squeeze(0)
            emb = F.normalize(emb, dim=-1)
            scores = (emb @ perfil.squeeze(0)).cpu().numpy()
            rank = int((scores >= scores[-1]).sum())

            if es_cold:
                tc+=1
                if rank<=TOP_K: hc+=1; mc+=1/rank; nc+=1/np.log2(rank+1)
            else:
                tw+=1
                if rank<=TOP_K: hw+=1; mw+=1/rank; nw+=1/np.log2(rank+1)
    return {
        "warm": {"HR@10": hw/max(tw,1), "MRR@10": mw/max(tw,1),
                 "NDCG@10": nw/max(tw,1), "n": tw},
        "cold": {"HR@10": hc/max(tc,1), "MRR@10": mc/max(tc,1),
                 "NDCG@10": nc/max(tc,1), "n": tc},
    }


def main(solo_clave=False):
    print("ABLACIÓN DEL MODELO DUAL (Objetivo 5)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); np.random.seed(SEED)

    # Cargar datos comunes
    print("\nCargando datos...")
    historial = pd.read_pickle(HISTORIAL_PATH)
    emb_matrix = np.load(EMBEDDINGS_PATH)
    producto_ids = np.load(PRODUCT_IDS_PATH)
    pid2idx = {int(pid): i+1 for i, pid in enumerate(producto_ids)}
    with open(NEGATIVOS_VAL, 'rb') as f: neg_val = pickle.load(f)

    # Dataset de entrenamiento (compartido por todas las variantes)
    print("Construyendo dataset de entrenamiento...")
    dataset = TD.InstacartDataset(historial, pid2idx)

    # Preparar partición cold/warm para evaluación (misma seed)
    productos_lista = list(producto_ids); random.shuffle(productos_lista)
    n_cold = int(len(productos_lista) * 0.10)
    productos_cold = set(productos_lista[:n_cold])
    productos_warm = set(productos_lista[n_cold:])
    historial['test_es_cold'] = historial['test'].apply(lambda p: p in productos_cold)
    historial['train_warm'] = historial['train'].apply(
        lambda lst: [p for p in lst if p in productos_warm])
    historial['n_tw'] = historial['train_warm'].apply(len)
    h_eval = historial[historial['n_tw'] >= 3].reset_index(drop=True)

    # Negativos para cold-start
    todos = set(pid2idx.keys()); negs_d = {}
    for _, row in h_eval.iterrows():
        uid = int(row['user_id']); pos = row['test']
        if pos not in pid2idx: continue
        excl = set(row['train_warm']) | {pos}
        if pd.notna(row.get('val')): excl.add(row['val'])
        rng = random.Random(SEED + 77777 + uid)
        negs_d[uid] = rng.sample(list(todos - excl), min(N_NEG_EVAL, len(todos-excl)))

    # Seleccionar variantes a correr
    variantes = VARIANTES
    if solo_clave:
        variantes = {k: VARIANTES[k] for k in ["A_completo", "B_sin_colddropout"]}
        print(f"\n⚡ Modo solo-clave: corriendo {list(variantes.keys())}")

    # Correr cada variante
    resultados = {}
    for nombre, cfg in variantes.items():
        t0 = time.time()
        model, val_hr = entrenar_variante(
            nombre, cfg, historial, emb_matrix, producto_ids,
            pid2idx, neg_val, dataset, device)
        res = evaluar_cold_warm_variante(model, h_eval, productos_cold,
                                          pid2idx, negs_d, device)
        res["val_HR10"] = val_hr
        res["config"] = cfg
        res["tiempo_min"] = round((time.time()-t0)/60, 1)
        resultados[nombre] = res
        print(f"\n  >>> {nombre}: warm={res['warm']['HR@10']:.4f} | "
              f"cold={res['cold']['HR@10']:.4f} | {res['tiempo_min']}min")
        del model
        torch.cuda.empty_cache()

    #Tabla resumen
    print("RESUMEN DE ABLACIÓN")
    print(f"\n{'Variante':<22} {'Warm HR@10':>11} {'Cold HR@10':>11} {'Val HR@10':>10}")
    for nombre, res in resultados.items():
        print(f"{nombre:<22} {res['warm']['HR@10']:>11.4f} "
              f"{res['cold']['HR@10']:>11.4f} {res['val_HR10']:>10.4f}")

    # Interpretación automática
    print("\n--- INTERPRETACIÓN ---")
    if "A_completo" in resultados and "B_sin_colddropout" in resultados:
        cold_A = resultados["A_completo"]["cold"]["HR@10"]
        cold_B = resultados["B_sin_colddropout"]["cold"]["HR@10"]
        delta = (cold_A - cold_B) * 100
        print(f"Efecto del cold-start dropout en cold HR@10: {delta:+.2f} pp")
        if delta > 3:
            print("El cold dropout MEJORA sustancialmente el cold-start.")
            print("     Confirma que es el componente que habilita productos nuevos.")
        elif delta > 0:
            print("El cold dropout ayuda levemente.")
        else:
            print("El cold dropout NO ayuda. Revisar.")

    if "A_completo" in resultados and "C_sin_congelada" in resultados:
        cold_A = resultados["A_completo"]["cold"]["HR@10"]
        cold_C = resultados["C_sin_congelada"]["cold"]["HR@10"]
        delta = (cold_A - cold_C) * 100
        print(f"Efecto de la tabla congelada (FastText) en cold HR@10: {delta:+.2f} pp")

    # Guardar
    out = RESULTS / "ablacion_dual.json"
    with open(out, 'w') as f:
        json.dump(resultados, f, indent=2, default=float)
    print(f"Guardado: {out}")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-clave", action="store_true",
                        help="Corre solo A y B (rápido)")
    args = parser.parse_args()
    main(solo_clave=args.solo_clave)
