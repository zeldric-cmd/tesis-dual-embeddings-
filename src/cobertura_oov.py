"""
Experimento del Objetivo 1: Cobertura de vocabulario abierto (OOV).

Demuestra que la representacion BPE+FastText puede generar vectores para
productos con terminos NO VISTOS durante el entrenamiento (neologismos,
erratas, acentos, terminos compuestos), donde un vocabulario fijo falla.

Comparacion:
  - BPE+FastText: tokeniza con BPE, cada subtoken recibe vector de FastText
    (FastText compone vectores via n-gramas de caracteres -> nunca da cero).
  - Palabras+vocabulario fijo: tokeniza por espacios, cada palabra se busca
    en el vocabulario del catalogo de entrenamiento (da cero si es OOV).

Metrica: % de tokens del nombre que reciben un vector valido (cobertura).

Uso:
    python cobertura_oov.py
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import fasttext
from tokenizers import Tokenizer

src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import (DATA_PROC, FASTTEXT_BIN, TOKENIZER_PATH, RESULTS, SEED)


# Productos OOV de prueba: terminos que NO aparecen en el catalogo Instacart
# (neologismos de marca, erratas, acentos, compuestos sin espacio)
PRODUCTOS_OOV = [
    ("Organique Whole Milk 2%",          "Neologismo con acento no estandar"),
    ("Glutenfrei Sourdough Bread",        "Termino en otro idioma (aleman)"),
    ("Vegano Unsalted Butterblend",       "Termino compuesto nuevo"),
    ("Paleolicious Grain Free Granola",   "Neologismo de marca"),
    ("Keto-Friendly Cauliflower Tortillas","Termino de tendencia dietetica"),
    ("Probiotiqs Greek Yogurt Plain",     "Errata intencional en marca"),
    ("Aquafina Sparklng Water",           "Errata tipografica (letra faltante)"),
    ("Wholesome Organikz Baby Spinach",   "Grafia no convencional de marca"),
    ("Freerange Omega3 Eggs",             "Termino compuesto sin espacio"),
    ("Superfoods Quinoamilk Unsweetened", "Neologismo compuesto (quinoa+milk)"),
]


def main():
    print("OBJETIVO 1 — COBERTURA DE VOCABULARIO ABIERTO (BPE vs PALABRAS)")

    # Cargar tokenizer BPE y FastText
    print("\nCargando tokenizer BPE y FastText...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ft = fasttext.load_model(str(FASTTEXT_BIN))

    # Construir vocabulario fijo del catalogo (palabras vistas en entrenamiento) 
    products = pd.read_csv(DATA_PROC / "products_filtrado.csv")
    vocab_fijo = set()
    for nombre in products['product_name'].str.lower():
        for palabra in nombre.split():
            vocab_fijo.add(palabra)
    print(f"Vocabulario fijo del catalogo: {len(vocab_fijo):,} palabras unicas")

    def cobertura_bpe(nombre):
        """% de subtokens BPE que reciben vector FastText valido (no-cero)."""
        tokens = tokenizer.encode(nombre.lower()).tokens
        if not tokens:
            return 0.0
        validos = 0
        for t in tokens:
            vec = ft.get_word_vector(t)
            if np.linalg.norm(vec) > 1e-8:  # FastText compone via n-gramas
                validos += 1
        return 100.0 * validos / len(tokens)

    def cobertura_palabras(nombre):
        """% de palabras que existen en el vocabulario fijo del catalogo."""
        palabras = nombre.lower().split()
        if not palabras:
            return 0.0
        en_vocab = sum(1 for p in palabras if p in vocab_fijo)
        return 100.0 * en_vocab / len(palabras)

    # Evaluar cada producto OOV
    print(f"{'Producto':<42} {'Tipo':<38} {'Cob.BPE':>8} {'Cob.Word':>9}")

    resultados = []
    cob_bpe_total = []
    cob_word_total = []
    for nombre, tipo in PRODUCTOS_OOV:
        cb = cobertura_bpe(nombre)
        cw = cobertura_palabras(nombre)
        cob_bpe_total.append(cb)
        cob_word_total.append(cw)
        resultados.append({"producto": nombre, "tipo": tipo,
                           "cobertura_bpe": cb, "cobertura_word": cw})
        print(f"{nombre:<42} {tipo:<38} {cb:>7.1f}% {cw:>8.1f}%")

    media_bpe = np.mean(cob_bpe_total)
    media_word = np.mean(cob_word_total)
    print(f"{'PROMEDIO':<80} {media_bpe:>7.1f}% {media_word:>8.1f}%")

    print(f"\n--- INTERPRETACION ---")
    print(f"  Cobertura BPE+FastText:  {media_bpe:.1f}%")
    print(f"  Cobertura palabras+vocab fijo: {media_word:.1f}%")
    print(f"  Ventaja BPE: +{media_bpe - media_word:.1f} puntos porcentuales")
    if media_bpe > media_word:
        print(f"BPE+FastText cubre mas terminos OOV que el vocabulario fijo.")
        print(f"BPE garantiza representacion ante vocabulario abierto.")

    # Guardar
    out = RESULTS / "cobertura_oov.json"
    with open(out, 'w') as f:
        json.dump({"productos": resultados,
                   "promedio_bpe": float(media_bpe),
                   "promedio_word": float(media_word),
                   "ventaja_pp": float(media_bpe - media_word)},
                  f, indent=2, default=float)
    print(f"\n Guardado: {out}")
    del ft
    return resultados


if __name__ == "__main__":
    main()
