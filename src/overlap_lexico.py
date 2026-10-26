"""
EXPERIMENTO DE CONTROL: Overlap lexico entre productos cold y warm

Mide, para cada producto cold, que fraccion de sus tokens BPE ya aparecia
en algun producto warm. Usa EXACTAMENTE la misma particion 90/10 que
cold_start_dual_completo.py (random.shuffle con SEED).

"""
import sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd

src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import PRODUCT_IDS_PATH, DATA_PROC, RESULTS, TOKENIZER_PATH, SEED
from tokenizers import Tokenizer


def main():
    print("OVERLAP LEXICO: productos cold vs vocabulario warm")

    random.seed(SEED)
    np.random.seed(SEED)

    producto_ids = np.load(PRODUCT_IDS_PATH)
    products = pd.read_csv(DATA_PROC / "products_filtrado.csv")
    pid2name = dict(zip(products['product_id'], products['product_name']))
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # PARTICION IDENTICA: random.shuffle (NO np.random.permutation)
    pl = list(producto_ids)
    random.shuffle(pl)
    n_cold = int(len(pl) * 0.10)
    productos_cold = list(pl[:n_cold])
    productos_warm = set(pl[n_cold:])
    print(f"Cold: {len(productos_cold):,} | Warm: {len(productos_warm):,}")

    print("\nConstruyendo vocabulario BPE de productos warm...")
    tokens_warm = set()
    for pid in productos_warm:
        nombre = str(pid2name.get(pid, ""))
        if not nombre:
            continue
        ids = tokenizer.encode(nombre).ids
        tokens_warm.update(ids)
    print(f"Tokens BPE distintos en warm: {len(tokens_warm):,}")

    print("\nMidiendo overlap de productos cold...")
    overlaps = []
    detalle = []
    for pid in productos_cold:
        nombre = str(pid2name.get(pid, ""))
        if not nombre:
            continue
        ids = tokenizer.encode(nombre).ids
        if not ids:
            continue
        compartidos = sum(1 for t in ids if t in tokens_warm)
        overlap = compartidos / len(ids)
        overlaps.append(overlap)
        detalle.append({
            "producto": nombre,
            "overlap": overlap,
            "n_tokens": len(ids),
            "tokens_compartidos": compartidos,
        })

    overlaps = np.array(overlaps)

    print("RESULTADOS")
    print(f"Productos cold analizados:     {len(overlaps):,}")
    print(f"Overlap promedio:              {overlaps.mean():.4f}")
    print(f"Overlap mediana:               {np.median(overlaps):.4f}")
    print(f"Productos con overlap = 100%:  {(overlaps == 1.0).sum():,} "
          f"({(overlaps == 1.0).mean() * 100:.1f}%)")
    print(f"Productos con overlap >= 75%:  {(overlaps >= 0.75).sum():,} "
          f"({(overlaps >= 0.75).mean() * 100:.1f}%)")
    print(f"Productos con overlap < 50%:   {(overlaps < 0.5).sum():,} "
          f"({(overlaps < 0.5).mean() * 100:.1f}%)")
    print(f"Productos con overlap = 0%:    {(overlaps == 0.0).sum():,} "
          f"({(overlaps == 0.0).mean() * 100:.1f}%)")

    print("\nDistribucion acumulada:")
    for umbral in [0.0, 0.25, 0.50, 0.75, 0.99]:
        frac = (overlaps <= umbral).mean()
        print(f"  overlap <= {umbral:.2f}: {frac * 100:5.1f}% de productos cold")

    print("INTERPRETACION")
    prom = overlaps.mean()
    if prom >= 0.80:
        print("OVERLAP ALTO (>=80%): el cold-start se beneficia del")
        print("solapamiento lexico con productos warm. Reportar como")
        print("LIMITACION con el contraargumento de TF-IDF.")
    elif prom >= 0.50:
        print("OVERLAP MEDIO (50-80%): escenario mixto. Reportar con matiz.")
    else:
        print("OVERLAP BAJO (<50%): cold-start relativamente genuino.")

    out = {
        "n_cold": len(overlaps),
        "overlap_promedio": float(overlaps.mean()),
        "overlap_mediana": float(np.median(overlaps)),
        "frac_overlap_total": float((overlaps == 1.0).mean()),
        "frac_overlap_mayor_75": float((overlaps >= 0.75).mean()),
        "frac_overlap_menor_50": float((overlaps < 0.5).mean()),
        "frac_overlap_cero": float((overlaps == 0.0).mean()),
        "detalle": detalle,
    }
    ruta = Path(RESULTS) / "overlap_lexico.json"
    with open(ruta, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()
