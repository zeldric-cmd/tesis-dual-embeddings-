"""
Genera los artefactos del demo (carpeta demo_assets/):
  - token_vectors.npy : vector FastText de cada token BPE (vocab x 300)
  - ids_validos.json  : muestra de 2000 user_id para el botón aleatorio

Requiere que el pipeline ya haya corrido (bpe_tokenizer.json, historial)
y el FastText .bin en data/raw/. Correr desde la raíz del proyecto:
    python generar_demo_assets.py
"""
import json, pickle, random
from pathlib import Path
import numpy as np
import fasttext
from tokenizers import Tokenizer

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
ASSET = ROOT / "demo_assets"
ASSET.mkdir(parents=True, exist_ok=True)

# ── token_vectors.npy : TOKVEC[id] = vector FastText del token BPE con ese id ──
print("Cargando tokenizer BPE y FastText...")
tok = Tokenizer.from_file(str(PROC / "bpe_tokenizer.json"))
ft = fasttext.load_model(str(RAW / "wiki-news-300d-1M-subword.bin"))
vocab = tok.get_vocab()                      # {token: id}
n = max(vocab.values()) + 1
dim = ft.get_dimension()
TOK = np.zeros((n, dim), dtype=np.float32)
for token, idx in vocab.items():
    TOK[idx] = ft.get_word_vector(token)
np.save(ASSET / "token_vectors.npy", TOK)
print(f"  token_vectors.npy: {TOK.shape}")

# ── ids_validos.json : 2000 user_id de muestra (semilla fija) ──
h = pickle.load(open(PROC / "historial_usuarios.pkl", "rb"))
ids = h["user_id"].tolist()
random.seed(42)
muestra = random.sample(ids, min(2000, len(ids)))
json.dump(muestra, open(ASSET / "ids_validos.json", "w"))
print(f"  ids_validos.json: {len(muestra)} IDs de {len(ids)}")
print("Listo: demo_assets/ generado.")
