"""
Entrenamiento del modelo DualEmbeddingCNN con cold-start dropout.

CLAVE: cold-start dropout
  Durante el entrenamiento, con probabilidad P_COLD apagamos la tabla
  ENTRENABLE de cada producto del historial (lo tratamos como "nuevo").
  Esto fuerza al modelo a aprender a construir perfiles útiles usando
  SOLO la semántica FastText cuando el colaborativo no está disponible.
  Sin esto, el colaborativo dominaría y el modelo fallaría en cold-start.

  - SIN leakage de val (full_seq = train puro)
  - Contexto train+val en evaluación de test
  - BPR loss sobre señales implícitas
  - Augmentación temporal por ventanas deslizantes
"""
import pickle
import random
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    HISTORIAL_PATH, EMBEDDINGS_PATH, PRODUCT_IDS_PATH,
    NEGATIVOS_VAL, NEGATIVOS_TEST, MODELO_BEST, RESULTS,
    EMB_DIM, N_FILTERS, DROPOUT, SEQ_LEN, COMBINE_MODE,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    N_NEG_TRAIN, N_NEG_EVAL, EPOCHS, PATIENCE, EVAL_EVERY, TOP_K, SEED
)
from model_dual import DualEmbeddingCNN

#Probabilidad de cold-start dropout durante entrenamiento
P_COLD_DROPOUT = 0.40  # 30% de los productos del historial se tratan como cold


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InstacartDataset(Dataset):
    """Dataset con augmentación temporal. Hereda el fix de NO-leakage."""
    def __init__(self, historial, pid2idx, seq_len=SEQ_LEN, n_neg=N_NEG_TRAIN,
                 max_muestras_por_usuario=20):
        self.seq_len = seq_len
        self.n_neg = n_neg
        self.pid2idx = pid2idx
        self.todos_pids = list(pid2idx.keys())
        self.muestras = []

        for _, row in historial.iterrows():
            train_seq = row['train']
            full_seq = list(train_seq)  # SIN val (fix de leakage)
            if len(full_seq) < 2:
                continue
            items_usuario = set(full_seq) | {row['val'], row['test']}

            n_pos = len(full_seq) - 1
            if n_pos <= max_muestras_por_usuario:
                target_indices = range(1, len(full_seq))
            else:
                target_indices = range(len(full_seq) - max_muestras_por_usuario, len(full_seq))

            for t in target_indices:
                start = max(0, t - seq_len)
                window = full_seq[start:t]
                if len(window) < seq_len:
                    window = [0] * (seq_len - len(window)) + window
                seq_idx = [pid2idx.get(p, 0) for p in window]
                target_pid = full_seq[t]
                if target_pid not in pid2idx:
                    continue
                self.muestras.append((seq_idx, pid2idx[target_pid], items_usuario))

        print(f"Muestras (con augmentación): {len(self.muestras):,}")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        seq_idx, pos_idx, items_usuario = self.muestras[idx]
        negs = []
        while len(negs) < self.n_neg:
            neg = random.choice(self.todos_pids)
            if neg not in items_usuario:
                negs.append(self.pid2idx[neg])
        return (torch.tensor(seq_idx, dtype=torch.long),
                torch.tensor(pos_idx, dtype=torch.long),
                torch.tensor(negs, dtype=torch.long))


def bpr_loss_dual(model, seq, pos, negs, device, p_cold=P_COLD_DROPOUT):
    """
    BPR loss con cold-start dropout en los CANDIDATOS (fix v2).
    v1 apagaba la tabla entrenable del HISTORIAL (habilidad equivocada).
    v2 apaga la tabla entrenable del POSITIVO con prob p_cold, simulando
      que el producto a recomendar es NUEVO (solo tiene semantica congelada).
      Esto entrena la habilidad REAL de cold-start: rankear alto un candidato
      que solo dispone de su embedding semantico.

    El perfil del usuario se construye del historial completo (warm).
    """
    seq = seq.to(device)
    pos = pos.to(device)
    negs = negs.to(device)

    # Perfil del usuario: historial WARM completo (sin apagar nada)
    perfil = model(seq)  # (B, emb_dim)

    B = pos.shape[0]

    # Cold-start dropout en el POSITIVO:
    # con prob p_cold, el positivo se trata como nuevo (solo tabla congelada)
    cold_pos = (torch.rand(B, device=device) < p_cold)  # (B,) bool

    def emb_item_con_mask(idx, cold_mask_1d):
        """Embedding de items (B,) con mascara cold por item."""
        # idx: (B,) -> (B,1) para combinar
        cm = cold_mask_1d.unsqueeze(1) if cold_mask_1d is not None else None
        e = model.combinar(idx.unsqueeze(1), cm)  # (B,1,emb_comb)
        e = e.squeeze(1)
        return F.normalize(e, dim=-1)

    # Positivo: aplica cold_pos (algunos solo usan semantica)
    emb_pos = emb_item_con_mask(pos, cold_pos)  # (B, emb_dim)

    # Negativos: tambien aplicamos cold dropout a una fraccion, para que
    # el modelo no aprenda a distinguir "cold = positivo" trivialmente.
    # negs: (B, n_neg)
    nn_ = negs.shape[1]
    cold_negs = (torch.rand(B, nn_, device=device) < p_cold) & (negs != 0)
    emb_negs = F.normalize(model.combinar(negs, cold_negs), dim=-1)  # (B,n_neg,emb_dim)

    score_pos = (perfil * emb_pos).sum(dim=-1)              # (B,)
    score_neg = torch.bmm(emb_negs, perfil.unsqueeze(-1)).squeeze(-1)  # (B,n_neg)

    diff = score_pos.unsqueeze(1) - score_neg
    loss = -F.logsigmoid(diff).mean()
    return loss


def evaluar(model, historial, columna, negativos_dict, pid2idx, device,
            K=TOP_K, batch_size=2048):
    """Evaluación leave-one-out. Para test usa contexto train+val."""
    model.eval()
    seqs_list, pos_list, negs_list, recompra_list = [], [], [], []

    for _, row in historial.iterrows():
        user_id = int(row['user_id'])
        positivo = row[columna]
        if positivo not in pid2idx or user_id not in negativos_dict:
            continue
        if columna == 'test':
            historia = list(row['train']) + [row['val']]
        else:
            historia = list(row['train'])
        seq = historia[-SEQ_LEN:]
        seq = [0] * (SEQ_LEN - len(seq)) + seq
        seqs_list.append([pid2idx.get(p, 0) for p in seq])
        pos_list.append(pid2idx[positivo])
        negs_list.append([pid2idx[n] for n in negativos_dict[user_id]])
        recompra_list.append(positivo in set(row['train']))

    n_total = len(seqs_list)
    if n_total == 0:
        return {"HR@10": 0, "MRR@10": 0, "NDCG@10": 0, "n_usuarios": 0,
                "HR@10_no_recompra": 0, "n_no_recompra": 0}

    seqs_t = torch.tensor(seqs_list, dtype=torch.long)
    pos_t = torch.tensor(pos_list, dtype=torch.long)
    negs_t = torch.tensor(negs_list, dtype=torch.long)
    recompra = np.array(recompra_list, dtype=bool)
    total_nr = (~recompra).sum()

    hits = mrr = ndcg = 0.0
    hits_nr = 0.0
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            j = min(i + batch_size, n_total)
            bs = seqs_t[i:j].to(device)
            bp = pos_t[i:j].to(device)
            bn = negs_t[i:j].to(device)
            br = recompra[i:j]

            perfil = model(bs)  # warm eval, sin cold_mask

            # Embeddings de candidatos (pos + negs) con combinación completa
            emb_pos = F.normalize(model.combinar(bp.unsqueeze(1)).squeeze(1), dim=-1)
            emb_negs = F.normalize(model.combinar(bn), dim=-1)  # (B, 99, D)

            score_pos = (perfil * emb_pos).sum(dim=-1)
            score_negs = torch.bmm(emb_negs, perfil.unsqueeze(-1)).squeeze(-1)
            all_scores = torch.cat([score_negs, score_pos.unsqueeze(1)], dim=1)
            ranks = (all_scores[:, :-1] >= score_pos.unsqueeze(1)).sum(dim=1) + 1
            ranks = ranks.cpu().numpy()

            within = ranks <= K
            hits += within.sum()
            mrr += np.where(within, 1.0/ranks, 0).sum()
            ndcg += np.where(within, 1.0/np.log2(ranks+1), 0).sum()
            mask_nr = (~br) & within
            hits_nr += mask_nr.sum()

    return {
        "HR@10": float(hits/n_total),
        "MRR@10": float(mrr/n_total),
        "NDCG@10": float(ndcg/n_total),
        "n_usuarios": int(n_total),
        "HR@10_no_recompra": float(hits_nr/max(total_nr,1)),
        "n_no_recompra": int(total_nr)
    }


def entrenar(max_epochs=None):
    if max_epochs is None:
        max_epochs = EPOCHS
    print(f"ENTRENAMIENTO MODELO DUAL (combine_mode={COMBINE_MODE}, p_cold={P_COLD_DROPOUT})")
    set_seed(SEED)

    print("\nCargando datos...")
    historial = pd.read_pickle(HISTORIAL_PATH)
    emb_matrix = np.load(EMBEDDINGS_PATH)
    producto_ids = np.load(PRODUCT_IDS_PATH)
    pid2idx = {int(pid): i+1 for i, pid in enumerate(producto_ids)}
    with open(NEGATIVOS_VAL, 'rb') as f: neg_val = pickle.load(f)
    with open(NEGATIVOS_TEST, 'rb') as f: neg_test = pickle.load(f)
    print(f"  Usuarios: {len(historial):,} | Productos: {len(producto_ids):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    model = DualEmbeddingCNN(
        n_productos=len(producto_ids), emb_dim=EMB_DIM, n_filters=N_FILTERS,
        dropout=DROPOUT, seq_len=SEQ_LEN, combine_mode=COMBINE_MODE
    )
    model.init_frozen_embeddings(emb_matrix)
    model = model.to(device)
    print(f"  Parámetros entrenables: {model.count_parameters():,}")

    dataset = InstacartDataset(historial, pid2idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5
    )

    mejor_hr10 = 0.0
    paciencia = 0
    historia = {"train_loss": [], "val_HR10": [], "val_HR10_nr": []}

    for epoch in range(1, max_epochs+1):
        model.train()
        total_loss = 0
        t0 = time.time()
        for seq, pos, negs in tqdm(dataloader, desc=f"Ep {epoch:3d}/{max_epochs}", leave=False):
            optimizer.zero_grad()
            loss = bpr_loss_dual(model, seq, pos, negs, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        t_train = time.time() - t0

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == max_epochs:
            m = evaluar(model, historial, 'val', neg_val, pid2idx, device)
            historia["train_loss"].append(avg_loss)
            historia["val_HR10"].append(m["HR@10"])
            historia["val_HR10_nr"].append(m["HR@10_no_recompra"])
            print(f"Ep {epoch:3d} | Loss {avg_loss:.4f} | val HR@10 {m['HR@10']:.4f} "
                  f"(nr {m['HR@10_no_recompra']:.4f}) | t {t_train:.0f}s")
            if m["HR@10"] > mejor_hr10:
                mejor_hr10 = m["HR@10"]
                paciencia = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'mejor_val_hr10': mejor_hr10, 'val_metrics': m,
                            'combine_mode': COMBINE_MODE, 'p_cold': P_COLD_DROPOUT},
                           MODELO_BEST)
                print(f"Mejor modelo (val HR@10 = {mejor_hr10:.4f})")
            else:
                paciencia += EVAL_EVERY
                if paciencia >= PATIENCE:
                    print(f"\n⏹ Early stopping en época {epoch}")
                    break

    # Evaluación final en test
    print("EVALUACIÓN FINAL EN TEST")
    ckpt = torch.load(MODELO_BEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    m_test = evaluar(model, historial, 'test', neg_test, pid2idx, device)
    print(f"  HR@10:   {m_test['HR@10']:.4f} (sobre {m_test['n_usuarios']:,})")
    print(f"  HR@10_nr: {m_test['HR@10_no_recompra']:.4f} (sobre {m_test['n_no_recompra']:,})")
    print(f"  MRR@10:  {m_test['MRR@10']:.4f} | NDCG@10: {m_test['NDCG@10']:.4f}")

    resultados = {"combine_mode": COMBINE_MODE, "p_cold": P_COLD_DROPOUT,
                  "validation": ckpt['val_metrics'], "test": m_test, "historia": historia}
    out = RESULTS / f"resultados_dual_{COMBINE_MODE}.json"
    with open(out, 'w') as f:
        json.dump(resultados, f, indent=2, default=float)
    print(f"\n Guardado: {out}")
    return resultados


if __name__ == "__main__":
    entrenar()
