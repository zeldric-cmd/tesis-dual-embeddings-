"""
EXPERIMENTO DE SINONIMIA LEXICA - Modelo dual
Demuestra que la representacion BPE + FastText del modelo dual proyecta
productos morfologicamente relacionados en regiones proximas del espacio
vectorial, abordando la Limitacion 3 (sinonimia lexica).

METODO:
  1. Genera el embedding de cada producto con el modelo dual (rama semantica,
     igual que en cold-start: combinar() con la tabla congelada FastText).
  2. Calcula la similitud coseno entre pares de productos RELACIONADOS
     (misma categoria / variantes) y pares NO RELACIONADOS (categorias
     distintas).
  3. Demuestra que la similitud media de los pares relacionados es mayor
     que la de los no relacionados (capacidad discriminativa).

SALIDA:
  - results/sinonimia.json  (similitudes por par + promedios)
  - Figura de distribucion de similitudes

Uso:
    python sinonimia.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Imports del proyecto (misma ruta que evaluacion.py)
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import (PRODUCT_IDS_PATH, EMBEDDINGS_PATH, DATA_PROC, RESULTS,
                    MODELO_BEST, EMB_DIM, N_FILTERS, DROPOUT, SEQ_LEN, COMBINE_MODE)
from model_dual import DualEmbeddingCNN



# PARES DE PRODUCTOS A COMPARAR
# IMPORTANTE: estos nombres deben EXISTIR en tu products_filtrado.csv.
# El script busca el producto cuyo nombre CONTIENE el texto indicado
# (insensible a mayusculas). Si no encuentra alguno, lo reporta y lo omite.
# Ajusta los nombres a productos reales de tu catalogo Instacart.

PARES_RELACIONADOS = [
    ("Organic Whole Milk",        "Organic Reduced Fat Milk"),
    ("Organic Whole Milk",        "Organic 2% Reduced Fat Milk"),
    ("Greek Nonfat Yogurt",       "Greek Whole Milk Yogurt"),
    ("Organic Baby Spinach",      "Organic Spinach"),
    ("Organic Hass Avocado",      "Organic Avocado"),
    ("Organic Strawberries",      "Organic Blueberries"),
    ("Sparkling Water",           "Sparkling Lemon Water"),
    ("Organic Bananas",           "Banana"),
    ("Sharp Cheddar Cheese",      "Mild Cheddar Cheese"),
    ("Extra Virgin Olive Oil",    "Olive Oil"),
    ("Whole Wheat Bread",         "White Bread"),
    ("Organic Large Brown Eggs",  "Organic Eggs"),
]

PARES_NO_RELACIONADOS = [
    ("Organic Whole Milk",        "Dish Soap"),
    ("Greek Nonfat Yogurt",       "Paper Towels"),
    ("Organic Baby Spinach",      "Laundry Detergent"),
    ("Banana",                    "Aluminum Foil"),
    ("Sharp Cheddar Cheese",      "Toilet Paper"),
    ("Sparkling Water",           "Trash Bags"),
    ("Olive Oil",                 "Hand Soap"),
    ("White Bread",               "Dishwasher Detergent"),
    ("Organic Strawberries",      "Glass Cleaner"),
    ("Organic Eggs",              "Fabric Softener"),
]



def buscar_producto(nombre_buscado, pid2name, name2pid):
    """
    Coincidencia robusta:
    1) nombre EXACTO (insensible a mayusculas), si existe;
    2) si no, el nombre MAS CORTO que contenga el texto (evita variantes
       elaboradas tipo '... French Vanilla Yogurt').
    Devuelve (product_id, nombre_real) o None.
    """
    nb = nombre_buscado.lower().strip()
    for pid, nombre in pid2name.items():
        if str(nombre).lower().strip() == nb:
            return pid, nombre
    candidatos = [(pid, nombre) for pid, nombre in pid2name.items()
                  if nb in str(nombre).lower()]
    if not candidatos:
        return None
    return min(candidatos, key=lambda x: len(str(x[1])))


def embedding_producto(model, pid, pid2idx, device):
    """
    Genera el embedding semantico de un producto con el modelo dual.
    Usa cold_mask=True para obtener SOLO la rama semantica (FastText),
    que es la que captura la similitud lexica/morfologica.
    """
    idx = pid2idx[pid]  # indice 1..n
    idx_t = torch.tensor([[idx]], dtype=torch.long).to(device)  # (1,1)
    cm = torch.ones(1, 1, dtype=torch.bool).to(device)          # cold -> semantica
    with torch.no_grad():
        emb = model.combinar(idx_t, cm).squeeze()               # (emb_dim,)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def similitud_coseno(a, b):
    """Coseno entre dos vectores (ya L2-normalizados -> producto punto)."""
    return float(np.dot(a, b))


def evaluar_pares(pares, model, pid2idx, pid2name, name2pid, device):
    """Calcula la similitud de cada par; omite los que no se encuentren."""
    resultados = []
    for n1, n2 in pares:
        p1 = buscar_producto(n1, pid2name, name2pid)
        p2 = buscar_producto(n2, pid2name, name2pid)
        if p1 is None or p2 is None:
            falta = n1 if p1 is None else n2
            print(f"  [omitido] no se encontro: '{falta}'")
            continue
        e1 = embedding_producto(model, p1[0], pid2idx, device)
        e2 = embedding_producto(model, p2[0], pid2idx, device)
        sim = similitud_coseno(e1, e2)
        resultados.append({
            "producto_1": p1[1],
            "producto_2": p2[1],
            "similitud": sim,
        })
        print(f"  {sim:.4f}  |  {p1[1][:35]:<35} <-> {p2[1][:35]}")
    return resultados


def main():
    print("EXPERIMENTO DE SINONIMIA LEXICA - Modelo dual")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datos
    producto_ids = np.load(PRODUCT_IDS_PATH)
    products = pd.read_csv(DATA_PROC / "products_filtrado.csv")
    pid2idx = {int(pid): i + 1 for i, pid in enumerate(producto_ids)}
    pid2name = dict(zip(products['product_id'], products['product_name']))
    name2pid = {str(v).lower(): k for k, v in pid2name.items()}

    # Modelo 
    model = DualEmbeddingCNN(len(producto_ids), EMB_DIM, N_FILTERS, DROPOUT,
                             SEQ_LEN, COMBINE_MODE)
    model.init_frozen_embeddings(np.load(EMBEDDINGS_PATH))
    ckpt = torch.load(MODELO_BEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    # Evaluar pares
    print("\nPares RELACIONADOS:")
    rel = evaluar_pares(PARES_RELACIONADOS, model, pid2idx, pid2name, name2pid, device)
    print("\nPares NO RELACIONADOS:")
    norel = evaluar_pares(PARES_NO_RELACIONADOS, model, pid2idx, pid2name, name2pid, device)

    sims_rel = [r["similitud"] for r in rel]
    sims_norel = [r["similitud"] for r in norel]
    prom_rel = float(np.mean(sims_rel)) if sims_rel else 0.0
    prom_norel = float(np.mean(sims_norel)) if sims_norel else 0.0
    diferencia = prom_rel - prom_norel

    # Resumen
    print(f"Similitud media pares RELACIONADOS    : {prom_rel:.4f}")
    print(f"Similitud media pares NO RELACIONADOS : {prom_norel:.4f}")
    print(f"Diferencia discriminativa             : {diferencia:+.4f}")

    # Guardar JSON
    out = {
        "pares_relacionados": rel,
        "pares_no_relacionados": norel,
        "promedio_relacionados": prom_rel,
        "promedio_no_relacionados": prom_norel,
        "diferencia_discriminativa": diferencia,
    }
    ruta_json = Path(RESULTS) / "sinonimia.json"
    with open(ruta_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nGuardado: {ruta_json}")

    # Figura
    generar_figura(sims_rel, sims_norel, prom_rel, prom_norel)


def generar_figura(sims_rel, sims_norel, prom_rel, prom_norel):
    """Dos paneles: distribucion (boxplot) y dispersion de similitudes."""
    IMG = Path(__file__).parent.parent / "Imagenes" / "Capitulo5"
    IMG.mkdir(parents=True, exist_ok=True)

    C_REL = "#2c6fbb"
    C_NOREL = "#c44e52"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: boxplot comparativo
    bp = ax1.boxplot([sims_rel, sims_norel], labels=["Relacionados", "No relacionados"],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [C_REL, C_NOREL]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax1.set_ylabel("Similitud coseno", fontsize=11)
    ax1.set_title("Distribución de similitudes por tipo de par", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4); ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: dispersion (cada par como punto)
    x_rel = np.random.normal(1, 0.04, len(sims_rel))
    x_norel = np.random.normal(2, 0.04, len(sims_norel))
    ax2.scatter(x_rel, sims_rel, color=C_REL, alpha=0.7, s=50, label="Relacionados")
    ax2.scatter(x_norel, sims_norel, color=C_NOREL, alpha=0.7, s=50, label="No relacionados")
    ax2.axhline(prom_rel, color=C_REL, linestyle="--", alpha=0.7,
                label=f"Prom. rel. {prom_rel:.3f}")
    ax2.axhline(prom_norel, color=C_NOREL, linestyle="--", alpha=0.7,
                label=f"Prom. no rel. {prom_norel:.3f}")
    ax2.set_xticks([1, 2]); ax2.set_xticklabels(["Relacionados", "No relacionados"])
    ax2.set_ylabel("Similitud coseno", fontsize=11)
    ax2.set_title("Similitud coseno por par de productos", fontsize=12)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(axis="y", linestyle="--", alpha=0.4); ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    ruta = IMG / "sinonimia.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {ruta}")


if __name__ == "__main__":
    main()
