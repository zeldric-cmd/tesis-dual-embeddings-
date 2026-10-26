"""Configuración central del proyecto de embeddings duales."""
from pathlib import Path

#Rutas base (todo autocontenido dentro del proyecto)
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

# Insumos crudos (se colocan manualmente en data/raw/)
# CSV de Instacart: orders.csv, order_products__prior.csv, products.csv, ...
# Binario FastText preentrenado wiki-news-300d-1M-subword.bin
FASTTEXT_BIN = DATA_RAW / "wiki-news-300d-1M-subword.bin"

# Artefactos derivados (los GENERA el pipeline en data/processed/)
HISTORIAL_PATH = DATA_PROC / "historial_usuarios.pkl"
EMBEDDINGS_PATH = DATA_PROC / "producto_embeddings.npy"   # matriz FastText (tabla congelada)
PRODUCT_IDS_PATH = DATA_PROC / "producto_ids.npy"
NEGATIVOS_VAL = DATA_PROC / "negativos_val.pkl"
NEGATIVOS_TEST = DATA_PROC / "negativos_test.pkl"
TOKENIZER_PATH = DATA_PROC / "bpe_tokenizer.json"

#Checkpoints
MODELO_BEST = EXPERIMENTS / "mejor_modelo_dual.pt"

#Hiperparámetros del modelo (config final)
EMB_DIM = 300            # dimensión de cada tabla de embeddings
N_FILTERS = 256          # filtros por bloque convolucional
DROPOUT = 0.3            # dropout en los bloques convolucionales
SEQ_LEN = 10

# Combinación de embeddings duales
# Opciones: "sum", "concat", "gate"
COMBINE_MODE = "gate"    # config final: compuerta adaptativa

#Entrenamiento 
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
N_NEG_TRAIN = 4
N_NEG_EVAL = 99
EPOCHS = 120
PATIENCE = 16
EVAL_EVERY = 2
TOP_K = 10
SEED = 42

#Preprocesamiento (data_processing.py)
MIN_ORDENES_USUARIO = 5      # mínimo de órdenes por usuario para incluirlo

#Tokenización BPE (tokenization.py)
VOCAB_SIZE_BPE = 8000        # tamaño del vocabulario BPE
MIN_APARICIONES_PROD = 5     # frecuencia mínima de un producto

#Verificación de rutas al importar
def verificar_rutas():
    archivos = {
        "Historial": HISTORIAL_PATH,
        "Embeddings FastText": EMBEDDINGS_PATH,
        "Product IDs": PRODUCT_IDS_PATH,
        "Negativos val": NEGATIVOS_VAL,
        "Negativos test": NEGATIVOS_TEST,
        "FastText .bin": FASTTEXT_BIN,
    }
    print("Verificación de rutas:")
    todo_ok = True
    for nombre, path in archivos.items():
        existe = path.exists()
        estado = "OK" if existe else "FALTA"
        print(f"  [{estado}] {nombre}: {path}")
        if not existe:
            todo_ok = False
    return todo_ok

if __name__ == "__main__":
    verificar_rutas()
