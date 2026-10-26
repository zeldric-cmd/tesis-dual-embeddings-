import sys, pickle
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, DATA_PROC, PRODUCT_IDS_PATH

print("Cargando products.csv")
df = pd.read_csv(DATA_RAW / "products.csv")
product_ids = np.load(str(PRODUCT_IDS_PATH))

nombres = dict(zip(df['product_id'].astype(int), df['product_name'].astype(str)))
filtrado = {int(pid): nombres.get(int(pid), f"Producto {pid}") for pid in product_ids}

out = DATA_PROC / "productos_nombres.pkl"
with open(str(out), "wb") as f:
    pickle.dump(filtrado, f)

print(f"Guardado: {out}")
print(f"Total: {len(filtrado)} productos")
print(f"Ejemplo: {list(filtrado.items())[:3]}")
