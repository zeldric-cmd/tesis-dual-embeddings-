"""
Pre-generación de negativos fijos para evaluación reproducible.

CRÍTICO para evaluación rigurosa:
- Los mismos 99 negativos por usuario se usan en TODOS los modelos
  (tu propuesta, POP, TF-IDF, Item-CF)
- Los negativos EXCLUYEN el historial del usuario (train + val + test)
- Semilla determinística POR usuario (no afectada por orden de iteración)
"""
import pickle
import random
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    DATA_PROC, RESULTS,
    HISTORIAL_PATH, NEGATIVOS_VAL, NEGATIVOS_TEST,
    PRODUCT_IDS_PATH,
    N_NEG_EVAL, SEED
)
import numpy as np


def generar_negativos_fijos(historial, pid2idx, columna_objetivo,
                             n_neg=N_NEG_EVAL, seed_base=42):
    """
    Genera 99 negativos por usuario, fijos y reproducibles.

    Args:
        historial: DataFrame con columnas 'user_id', 'train', 'val', 'test'
        pid2idx: dict mapping product_id → índice
        columna_objetivo: 'val' o 'test' (cuál es el positivo a evitar)
        n_neg: número de negativos por usuario
        seed_base: semilla base; se suma user_id para reproducibilidad

    Returns:
        dict {user_id: [neg_pid_1, neg_pid_2, ..., neg_pid_99]}

    Garantías:
    - Negativos NUNCA están en train, val o test del usuario
    - Mismos negativos cada vez que se ejecuta (con misma semilla)
    - Independiente del orden de iteración del DataFrame
    """
    todos_pids = set(pid2idx.keys())
    negativos = {}

    print(f"Generando {n_neg} negativos por usuario para '{columna_objetivo}'...")

    for _, row in tqdm(historial.iterrows(), total=len(historial)):
        user_id = int(row['user_id'])
        positivo = row[columna_objetivo]

        if positivo not in pid2idx:
            continue

        # Items a excluir: TODO el historial del usuario
        items_usuario = set(row['train']) | {row['val'], row['test']}
        pool = list(todos_pids - items_usuario)

        # Semilla determinística por usuario (no por orden de iteración)
        local_rng = random.Random(seed_base + user_id)

        if len(pool) < n_neg:
            # Caso extremo: usuario con casi todo el catálogo en historial
            # (improbable pero defensivo)
            negativos[user_id] = local_rng.sample(pool, len(pool))
        else:
            negativos[user_id] = local_rng.sample(pool, n_neg)

    return negativos


def validar_negativos(negativos, historial, pid2idx, columna_objetivo):
    """
    Verifica que los negativos cumplan las garantías esperadas.
    """
    print(f"\n=== VALIDACIÓN DE NEGATIVOS ({columna_objetivo}) ===")

    n_users = len(negativos)
    n_neg_promedio = np.mean([len(v) for v in negativos.values()])
    n_neg_min = min(len(v) for v in negativos.values())
    n_neg_max = max(len(v) for v in negativos.values())

    print(f"  Usuarios con negativos: {n_users:,}")
    print(f"  Negativos por usuario:  promedio {n_neg_promedio:.1f}, min {n_neg_min}, max {n_neg_max}")

    # Test de no-contaminación: ningún negativo está en historial
    contaminados = 0
    for _, row in historial.iterrows():
        user_id = int(row['user_id'])
        if user_id not in negativos:
            continue
        items_usuario = set(row['train']) | {row['val'], row['test']}
        if any(neg in items_usuario for neg in negativos[user_id]):
            contaminados += 1

    if contaminados == 0:
        print(f"Ningún negativo está en el historial del usuario")
    else:
        print(f"ALERTA: {contaminados:,} usuarios tienen negativos contaminados")

    # Test de unicidad: ningún negativo se repite en la misma lista
    duplicados = 0
    for user_id, negs in negativos.items():
        if len(negs) != len(set(negs)):
            duplicados += 1

    if duplicados == 0:
        print(f"Sin duplicados dentro de listas")
    else:
        print(f"ALERTA: {duplicados:,} usuarios con negativos duplicados")

    return {
        "n_users": n_users,
        "n_neg_promedio": float(n_neg_promedio),
        "n_neg_min": int(n_neg_min),
        "n_neg_max": int(n_neg_max),
        "contaminados": int(contaminados),
        "duplicados": int(duplicados)
    }


def ejecutar_pipeline_completo():
    """Genera negativos para val y test."""
    print("PRE-GENERACIÓN DE NEGATIVOS FIJOS PARA EVALUACIÓN")

    # Cargar historial y productos
    print("\nCargando datos...")
    historial = pd.read_pickle(HISTORIAL_PATH)
    producto_ids = np.load(PRODUCT_IDS_PATH)
    pid2idx = {int(pid): i+1 for i, pid in enumerate(producto_ids)}

    print(f"  Usuarios: {len(historial):,}")
    print(f"  Productos: {len(producto_ids):,}")

    # Generar negativos para VAL
    print(f"\n--- NEGATIVOS PARA VAL ---")
    neg_val = generar_negativos_fijos(
        historial, pid2idx,
        columna_objetivo='val',
        seed_base=SEED  # semilla A
    )
    stats_val = validar_negativos(neg_val, historial, pid2idx, 'val')

    # Generar negativos para TEST
    print(f"\n--- NEGATIVOS PARA TEST ---")
    neg_test = generar_negativos_fijos(
        historial, pid2idx,
        columna_objetivo='test',
        seed_base=SEED + 10000  # semilla B (distinta para que val y test no compartan)
    )
    stats_test = validar_negativos(neg_test, historial, pid2idx, 'test')

    # Guardar
    print(f"\n=== GUARDANDO ===")
    with open(NEGATIVOS_VAL, 'wb') as f:
        pickle.dump(neg_val, f)
    print(f"{NEGATIVOS_VAL}  ({len(neg_val):,} usuarios)")

    with open(NEGATIVOS_TEST, 'wb') as f:
        pickle.dump(neg_test, f)
    print(f"{NEGATIVOS_TEST}  ({len(neg_test):,} usuarios)")

    # Guardar stats
    stats_path = RESULTS / "estadisticas_negativos.json"
    with open(stats_path, 'w') as f:
        json.dump({
            "n_neg_por_usuario": N_NEG_EVAL,
            "seed_val": SEED,
            "seed_test": SEED + 10000,
            "validacion_val": stats_val,
            "validacion_test": stats_test
        }, f, indent=2)
    print(f"{stats_path}")

    print("NEGATIVOS GENERADOS Y VALIDADOS")

    return neg_val, neg_test


if __name__ == "__main__":
    neg_val, neg_test = ejecutar_pipeline_completo()
