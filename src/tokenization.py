import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import fasttext
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    DATA_RAW, DATA_PROC, RESULTS,
    FASTTEXT_BIN, TOKENIZER_PATH, EMBEDDINGS_PATH, PRODUCT_IDS_PATH,
    VOCAB_SIZE_BPE, EMB_DIM, SEED
)


def cargar_productos_filtrados():
    """Carga productos del CSV filtrado (output de fase 1)."""
    path = DATA_PROC / "products_filtrado.csv"
    products = pd.read_csv(path)
    print(f"Productos cargados: {len(products):,}")
    return products


def entrenar_tokenizer_bpe(products, vocab_size=VOCAB_SIZE_BPE, min_freq=2):
    """
    Entrena tokenizador BPE sobre los nombres de productos.

    Notas:
    - vocab_size=8000 sobre 47,618 nombres con promedio 5.25 tokens/nombre
      resulta en un BPE que preserva mayoritariamente palabras inglesas
      completas. La capacidad sublexical se manifiesta con OOV y erratas.
    """
    print(f"\nENTRENAMIENTO DE TOKENIZADOR BPE")
    print(f"Vocabulario objetivo: {vocab_size:,}")
    print(f"Frecuencia mínima: {min_freq}")

    # Preparar corpus
    corpus = products['product_name'].str.lower().tolist()
    print(f"Corpus: {len(corpus):,} nombres de productos")

    # Guardar corpus en archivo temporal
    corpus_path = DATA_PROC / "corpus_nombres.txt"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for nombre in corpus:
            f.write(nombre + '\n')

    # Inicializar y entrenar
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["[PAD]", "[UNK]"],
        show_progress=False
    )

    print("Entrenando...")
    start = time.time()
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    elapsed = time.time() - start

    vocab_real = tokenizer.get_vocab_size()
    print(f"Entrenado en {elapsed:.1f}s")
    print(f"   Vocabulario final: {vocab_real:,} tokens")

    # Guardar
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"   Guardado: {TOKENIZER_PATH}")

    return tokenizer


def analizar_tokenizer(tokenizer, products, n_ejemplos=8):
    """Analiza el comportamiento del tokenizer sobre el catálogo."""
    print(f"\nANÁLISIS DEL TOKENIZADOR BPE\n")

    # Ejemplos
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(products), n_ejemplos, replace=False)
    print("Ejemplos de tokenización:")
    for idx in sample_idx:
        nombre = products.iloc[idx]['product_name']
        tokens = tokenizer.encode(nombre.lower()).tokens
        print(f"  '{nombre}'")
        print(f"    → {tokens}  ({len(tokens)} tokens)")

    # Estadísticas globales
    print(f"\nEstadísticas globales sobre {len(products):,} productos:")
    longitudes = []
    for nombre in products['product_name'].str.lower():
        tokens = tokenizer.encode(nombre).tokens
        longitudes.append(len(tokens))

    longitudes = np.array(longitudes)
    stats = {
        "tokens_promedio": float(longitudes.mean()),
        "tokens_mediana":  float(np.median(longitudes)),
        "tokens_min":      int(longitudes.min()),
        "tokens_max":      int(longitudes.max()),
        "tokens_p95":      float(np.percentile(longitudes, 95)),
        "tokens_p99":      float(np.percentile(longitudes, 99))
    }

    print(f"  Tokens por producto:")
    print(f"    Promedio: {stats['tokens_promedio']:.2f}")
    print(f"    Mediana:  {stats['tokens_mediana']:.0f}")
    print(f"    Mínimo:   {stats['tokens_min']}")
    print(f"    Máximo:   {stats['tokens_max']}")
    print(f"    P95:      {stats['tokens_p95']:.0f}")
    print(f"    P99:      {stats['tokens_p99']:.0f}")

    return stats


def cargar_fasttext_bin():
    """
    Carga el modelo FastText binario.

    IMPORTANTE: usar .bin (no .vec) para acceder a get_word_vector() que
    compone vectores para tokens OOV usando los n-gramas de caracteres.

    Nota: ft.get_words() puede fallar con UnicodeDecodeError porque el modelo
    fue entrenado sobre Common Crawl multilingüe y contiene bytes no-UTF8.
    Esto NO afecta a get_word_vector() que es la función crítica que usamos.
    """
    print(f"\nCARGANDO FASTTEXT .BIN")
    print(f"Archivo: {FASTTEXT_BIN}")
    print(f"Tamaño: {FASTTEXT_BIN.stat().st_size / 1e9:.2f} GB")
    print(f"Cargando (puede tardar 30-60s)...")

    start = time.time()
    ft = fasttext.load_model(str(FASTTEXT_BIN))
    elapsed = time.time() - start

    print(f"Cargado en {elapsed:.1f}s")
    print(f"   Dimensión: {ft.get_dimension()}")

    # Intentar obtener tamaño del vocabulario (puede fallar por bytes no-UTF8)
    try:
        n_words = len(ft.get_words(on_unicode_error="ignore"))
        print(f"   Vocabulario nominal: {n_words:,}")
    except Exception as e:
        print(f"   Vocabulario nominal: ~1M palabras (get_words() no decodificable)")

    # Test de funcionalidad crítica: get_word_vector con varios casos
    print(f"\n   Tests de get_word_vector():")
    tests = [
        ("organic",      "palabra en vocabulario"),
        ("organi",       "subtoken parcial"),
        ("leche",        "palabra en español (OOV en corpus inglés)"),
        ("xkqwzpt",      "palabra inventada"),
    ]
    for palabra, descripcion in tests:
        vec = ft.get_word_vector(palabra)
        norma = np.linalg.norm(vec)
        ok = "valido" if norma > 0 else "no valido"
        print(f"     {ok} '{palabra}' ({descripcion}): norma = {norma:.4f}")

    return ft


def construir_embeddings_productos(products, tokenizer, ft_model):
    """
    Construye matriz de embeddings de productos usando BPE + FastText .bin.

    Para cada producto:
    1. Tokeniza con BPE
    2. Obtiene vector de cada token con ft.get_word_vector() (siempre devuelve vector)
    3. Promedia los vectores

    NO filtra tokens OOV (corrección del bug crítico).
    """
    print(f"\nCONSTRUCCIÓN DE EMBEDDINGS DE PRODUCTOS")

    n_productos = len(products)
    matriz = np.zeros((n_productos, EMB_DIM), dtype=np.float32)
    producto_ids = np.zeros(n_productos, dtype=np.int64)

    # Estadísticas
    tokens_promedio_por_producto = 0
    productos_sin_tokens = 0

    print(f"Procesando {n_productos:,} productos...")
    for i, (_, row) in enumerate(tqdm(products.iterrows(), total=n_productos)):
        nombre = row['product_name'].lower()
        tokens = tokenizer.encode(nombre).tokens

        if not tokens:
            # Caso extremo: nombre vacío o solo puntuación
            productos_sin_tokens += 1
            # Dejar como vector cero
        else:
            # CLAVE: get_word_vector SIEMPRE devuelve un vector, incluso para OOV
            vecs = np.array([ft_model.get_word_vector(t) for t in tokens])
            matriz[i] = vecs.mean(axis=0)
            tokens_promedio_por_producto += len(tokens)

        producto_ids[i] = row['product_id']

    tokens_promedio_por_producto /= max(n_productos - productos_sin_tokens, 1)

    print(f"\n Matriz construida: {matriz.shape}")
    print(f"   Tokens promedio por producto: {tokens_promedio_por_producto:.2f}")
    print(f"   Productos sin tokens (vector cero): {productos_sin_tokens}")

    # Análisis de calidad
    norms = np.linalg.norm(matriz, axis=1)
    print(f"\n   Distribución de normas de embeddings:")
    print(f"     Min:    {norms.min():.4f}")
    print(f"     Max:    {norms.max():.4f}")
    print(f"     Media:  {norms.mean():.4f}")
    print(f"     Std:    {norms.std():.4f}")

    return matriz, producto_ids


def verificar_calidad_embeddings(matriz, producto_ids, products, ft_model, tokenizer):
    """
    Test de calidad: verificar que productos semánticamente similares
    tengan embeddings cercanos. Esto NO funcionaba en la versión .vec.
    """
    print(f"\nTEST DE CALIDAD SEMÁNTICA")

    # Normalizar para similitud coseno
    matriz_norm = matriz / (np.linalg.norm(matriz, axis=1, keepdims=True) + 1e-8)
    pid2idx = {pid: i for i, pid in enumerate(producto_ids)}
    pid2name = dict(zip(products['product_id'], products['product_name']))

    def encontrar_por_nombre(query):
        """Busca el product_id que mejor coincide con un texto."""
        query_lower = query.lower()
        for pid, name in pid2name.items():
            if name.lower() == query_lower:
                return pid
        return None

    # Pares de prueba: deben dar similitud alta
    pares_test = [
        ("Organic Whole Milk",            "Whole Milk"),
        ("Bag of Organic Bananas",        "Banana"),
        ("Organic Baby Spinach",          "Baby Spinach"),
        ("Greek Nonfat Yogurt",           "Greek Yogurt"),
    ]

    resultados = []
    print("\nPares semánticamente relacionados (deben dar similitud ALTA):")
    for p1, p2 in pares_test:
        pid1 = encontrar_por_nombre(p1)
        pid2 = encontrar_por_nombre(p2)
        if pid1 is None or pid2 is None:
            print(f"  ⚠️  '{p1}' o '{p2}' no encontrado en catálogo")
            continue
        i1, i2 = pid2idx[pid1], pid2idx[pid2]
        sim = float(np.dot(matriz_norm[i1], matriz_norm[i2]))
        resultados.append({"p1": p1, "p2": p2, "tipo": "relacionado", "sim": sim})
        print(f"  {p1:40s} ↔ {p2:30s} sim = {sim:.4f}")

    return resultados


def guardar_resultados(matriz, producto_ids, stats_tokenizer):
    """Guarda embeddings y estadísticas."""
    print(f"\nGUARDANDO ARCHIVOS")

    np.save(EMBEDDINGS_PATH, matriz)
    np.save(PRODUCT_IDS_PATH, producto_ids)
    print(f"Embeddings:  {EMBEDDINGS_PATH}  ({matriz.shape})")
    print(f"Product IDs: {PRODUCT_IDS_PATH}  ({producto_ids.shape})")

    # Guardar estadísticas
    stats_path = RESULTS / "estadisticas_tokenizer.json"
    with open(stats_path, 'w') as f:
        json.dump({
            "vocab_size_bpe": VOCAB_SIZE_BPE,
            "emb_dim": EMB_DIM,
            "fasttext_modelo": "wiki-news-300d-1M-subword.bin",
            "tokenizer_stats": stats_tokenizer,
            "n_productos": int(len(producto_ids)),
            "matriz_shape": list(matriz.shape),
            "norms_promedio": float(np.linalg.norm(matriz, axis=1).mean()),
        }, f, indent=2)
    print(f"Estadísticas: {stats_path}")


def ejecutar_pipeline_completo():
    """Ejecuta toda la fase 2."""
    print("FASE 2: TOKENIZACIÓN BPE + EMBEDDINGS CON FASTTEXT .BIN")

    # 1. Cargar productos filtrados
    products = cargar_productos_filtrados()

    # 2. Entrenar tokenizer BPE
    tokenizer = entrenar_tokenizer_bpe(products)

    # 3. Analizar tokenizer
    stats_tok = analizar_tokenizer(tokenizer, products)

    # 4. Cargar FastText binario
    ft_model = cargar_fasttext_bin()

    # 5. Construir embeddings
    matriz, producto_ids = construir_embeddings_productos(products, tokenizer, ft_model)

    # 6. Verificar calidad semántica
    verificar_calidad_embeddings(matriz, producto_ids, products, ft_model, tokenizer)

    # 7. Guardar
    guardar_resultados(matriz, producto_ids, stats_tok)

    print("FASE 2 COMPLETA")

    return matriz, producto_ids, tokenizer, ft_model


if __name__ == "__main__":
    matriz, producto_ids, tokenizer, ft_model = ejecutar_pipeline_completo()
