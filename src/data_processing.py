"""
Módulo de preprocesamiento del dataset Instacart Market Basket Analysis 2017.

Incluye:
- Cálculo correcto de esparsidad (pares únicos, no interacciones)
- Filtrado mínimo (≥5 órdenes/usuario, ≥5 apariciones/producto)
- Orden temporal dentro de la canasta (add_to_cart_order)
- Partición leave-one-out con train/val/test
- Análisis de recompras y diversidad
"""
import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    DATA_RAW, DATA_PROC, RESULTS,
    MIN_ORDENES_USUARIO, MIN_APARICIONES_PROD, SEED
)


def cargar_dataset_raw():
    """Carga los 5 CSVs originales del dataset Instacart."""
    print("Cargando dataset raw de Instacart...")
    orders   = pd.read_csv(DATA_RAW / "orders.csv")
    products = pd.read_csv(DATA_RAW / "products.csv")
    prior    = pd.read_csv(DATA_RAW / "order_products__prior.csv")
    aisles   = pd.read_csv(DATA_RAW / "aisles.csv")
    depts    = pd.read_csv(DATA_RAW / "departments.csv")

    print(f"  Órdenes:       {len(orders):,}")
    print(f"  Productos:     {len(products):,}")
    print(f"  Interacciones: {len(prior):,}")
    print(f"  Pasillos:      {len(aisles):,}")
    print(f"  Departamentos: {len(depts):,}")

    return orders, products, prior, aisles, depts


def calcular_esparsidad_real(prior, orders, n_usuarios, n_productos, etiqueta=""):
    """
    Calcula esparsidad como (1 - pares_únicos / (usuarios * productos)).
    Esto es lo CORRECTO. Tu versión anterior usaba len(prior) que cuenta
    cada compra individual, inflando el numerador.
    """
    pares_unicos = (prior
                    .merge(orders[['order_id', 'user_id']], on='order_id')
                    [['user_id', 'product_id']]
                    .drop_duplicates())

    n_pares = len(pares_unicos)
    matriz_posible = n_usuarios * n_productos
    esparsidad = 1 - (n_pares / matriz_posible)

    print(f"\n=== ESPARSIDAD {etiqueta} ===")
    print(f"  Usuarios:                 {n_usuarios:,}")
    print(f"  Productos:                {n_productos:,}")
    print(f"  Matriz posible:           {matriz_posible:,}")
    print(f"  Pares únicos (user,prod): {n_pares:,}")
    print(f"  Interacciones totales:    {len(prior):,}")
    print(f"  Esparsidad REAL:          {esparsidad:.6f}  ({esparsidad*100:.4f}%)")
    print(f"  (Tu cálculo anterior daba {1 - len(prior)/matriz_posible:.4f} — incorrecto)")

    return {
        "n_usuarios": int(n_usuarios),
        "n_productos": int(n_productos),
        "n_pares_unicos": int(n_pares),
        "n_interacciones_totales": int(len(prior)),
        "matriz_posible": int(matriz_posible),
        "esparsidad": float(esparsidad)
    }


def filtrar_dataset(orders, products, prior):
    """
    Aplica filtros estándar:
    - Usuarios con ≥ MIN_ORDENES_USUARIO órdenes
    - Productos con ≥ MIN_APARICIONES_PROD apariciones
    """
    print(f"\n=== FILTROS (≥{MIN_ORDENES_USUARIO} ord/usuario, ≥{MIN_APARICIONES_PROD} apar/producto) ===")

    # Filtro de usuarios
    ord_usr = orders.groupby('user_id')['order_id'].count()
    usuarios_validos = ord_usr[ord_usr >= MIN_ORDENES_USUARIO].index
    print(f"  Usuarios válidos:  {len(usuarios_validos):,} ({len(usuarios_validos)/orders['user_id'].nunique()*100:.1f}%)")

    # Filtro de productos
    apar_prod = prior.groupby('product_id')['order_id'].count()
    productos_validos = apar_prod[apar_prod >= MIN_APARICIONES_PROD].index
    print(f"  Productos válidos: {len(productos_validos):,} ({len(productos_validos)/products['product_id'].nunique()*100:.1f}%)")

    # Aplicar filtros conjuntamente
    orders_filt = orders[orders['user_id'].isin(usuarios_validos)].copy()
    ordenes_validas_ids = set(orders_filt['order_id'])
    prior_filt = prior[
        prior['order_id'].isin(ordenes_validas_ids) &
        prior['product_id'].isin(productos_validos)
    ].copy()

    print(f"  Interacciones restantes: {len(prior_filt):,} ({len(prior_filt)/len(prior)*100:.1f}%)")

    return orders_filt, prior_filt, productos_validos


def construir_historial(orders_filt, prior_filt, productos_validos):
    """
    Construye el historial de compras ordenado cronológicamente.
    CORRECCIÓN CLAVE: usa add_to_cart_order para ordenar dentro de cada canasta.
    Antes el orden dentro de una canasta era arbitrario.
    """
    print("\nConstruyendo historial de compras (con orden temporal correcto)...")

    # Merge con orden por user_id, order_number Y add_to_cart_order
    historial_long = (prior_filt
                      .merge(orders_filt[['order_id', 'user_id', 'order_number']], on='order_id')
                      .sort_values(['user_id', 'order_number', 'add_to_cart_order']))

    # Agrupar por usuario manteniendo el orden
    historial = (historial_long
                 .groupby('user_id', sort=False)['product_id']
                 .apply(list)
                 .reset_index())
    historial.columns = ['user_id', 'producto_ids']

    print(f"  Usuarios con historial: {len(historial):,}")

    # Estadísticas
    long_hist = historial['producto_ids'].apply(len)
    print(f"  Longitud media:    {long_hist.mean():.1f} productos")
    print(f"  Longitud mediana:  {long_hist.median():.0f}")
    print(f"  Longitud mínima:   {long_hist.min()}")
    print(f"  Longitud máxima:   {long_hist.max()}")

    # Verificar pérdida silenciosa de usuarios
    n_usuarios_originales = orders_filt['user_id'].nunique()
    if len(historial) < n_usuarios_originales:
        perdidos = n_usuarios_originales - len(historial)
        print(f"  ⚠️  Usuarios perdidos en el merge: {perdidos:,}")

    return historial


def particion_leave_one_out(historial):
    """
    Aplica leave-one-out:
    - test: última interacción
    - val: penúltima
    - train: resto

    CORRECCIÓN: solo conserva usuarios con AL MENOS 3 productos en historial
    (necesario para tener train, val y test válidos).
    """
    print("\nAplicando partición leave-one-out (train / val / test)...")

    # Filtrar usuarios con al menos 3 productos
    longitudes = historial['producto_ids'].apply(len)
    mask_suficiente = longitudes >= 3
    usuarios_descartados = (~mask_suficiente).sum()

    if usuarios_descartados > 0:
        print(f"  ⚠️  Usuarios descartados por historial < 3: {usuarios_descartados:,}")

    historial = historial[mask_suficiente].reset_index(drop=True)

    # Particionar
    historial['test']  = historial['producto_ids'].apply(lambda x: x[-1])
    historial['val']   = historial['producto_ids'].apply(lambda x: x[-2])
    historial['train'] = historial['producto_ids'].apply(lambda x: x[:-2])

    # Verificaciones
    print(f"  Usuarios finales:        {len(historial):,}")
    print(f"  Longitud media train:    {historial['train'].apply(len).mean():.1f}")
    print(f"  Val: 1 producto/usuario")
    print(f"  Test: 1 producto/usuario")

    return historial


def analizar_recompras(historial):
    """
    Análisis crítico: ¿qué porcentaje de tests son productos que el usuario ya compró?
    Esto importa porque tu HR@10 alto puede deberse a "predicción de recompra"
    en lugar de "recomendación futura genuina".
    """
    print("\n=== ANÁLISIS DE RECOMPRAS ===")

    historial['test_en_train'] = historial.apply(
        lambda r: r['test'] in set(r['train']), axis=1
    )
    historial['val_en_train']  = historial.apply(
        lambda r: r['val']  in set(r['train']), axis=1
    )

    pct_test_recompra = historial['test_en_train'].mean() * 100
    pct_val_recompra  = historial['val_en_train'].mean() * 100

    print(f"  Tests que YA están en train: {historial['test_en_train'].sum():,} "
          f"({pct_test_recompra:.1f}%)")
    print(f"  Vals que YA están en train:  {historial['val_en_train'].sum():,} "
          f"({pct_val_recompra:.1f}%)")

    if pct_test_recompra > 70:
        print(f"{pct_test_recompra:.1f}% de tests son recompras.")
        print(f"Esto significa que un modelo puede ganar HR@10 alto simplemente")
        print(f"memorizando lo que el usuario ya compró.")
        print(f"Reportar HR@10 también filtrando recompras es importante.")

    return {
        "pct_test_recompra": float(pct_test_recompra),
        "pct_val_recompra":  float(pct_val_recompra),
        "n_test_no_recompra": int((~historial['test_en_train']).sum())
    }


def analizar_diversidad(historial):
    """
    Análisis: ¿qué tan diversos son los productos en train?
    Si la diversidad es baja, las secuencias de entrada al CNN son muy repetitivas.
    """
    print("\n=== ANÁLISIS DE DIVERSIDAD ===")

    historial['n_total_train'] = historial['train'].apply(len)
    historial['n_unicos_train'] = historial['train'].apply(lambda x: len(set(x)))
    historial['ratio_diversidad'] = (historial['n_unicos_train'] /
                                     historial['n_total_train'].clip(lower=1))

    print(f"  Productos totales prom/usuario:  {historial['n_total_train'].mean():.1f}")
    print(f"  Productos únicos prom/usuario:   {historial['n_unicos_train'].mean():.1f}")
    print(f"  Ratio de diversidad promedio:    {historial['ratio_diversidad'].mean():.3f}")

    return {
        "n_total_train_promedio":  float(historial['n_total_train'].mean()),
        "n_unicos_train_promedio": float(historial['n_unicos_train'].mean()),
        "ratio_diversidad_promedio": float(historial['ratio_diversidad'].mean()),
    }


def guardar_resultados(historial, products, productos_validos, stats_globales):
    """Guarda historial, productos filtrados y estadísticas."""
    print("\nGuardando archivos procesados...")

    # Limpiar columnas auxiliares que no necesitamos persistir
    cols_a_eliminar = ['test_en_train', 'val_en_train', 'n_total_train',
                       'n_unicos_train', 'ratio_diversidad']
    historial_limpio = historial.drop(columns=[c for c in cols_a_eliminar if c in historial.columns])

    # Guardar historial
    historial_path = DATA_PROC / "historial_usuarios.pkl"
    historial_limpio.to_pickle(historial_path)
    print(f"Historial:  {historial_path}")

    # Guardar productos filtrados
    products_filt = products[products['product_id'].isin(productos_validos)].copy()
    products_path = DATA_PROC / "products_filtrado.csv"
    products_filt.to_csv(products_path, index=False)
    print(f"Productos:  {products_path} ({len(products_filt):,} filas)")

    # Guardar estadísticas
    stats_path = RESULTS / "estadisticas_dataset.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats_globales, f, indent=2)
    print(f"Estadísticas: {stats_path}")


def ejecutar_pipeline_completo():
    """Ejecuta todo el pipeline de preprocesamiento."""

    print("PREPROCESAMIENTO DEL DATASET INSTACART")


    # Asegurar directorio de salida
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. Cargar raw
    orders, products, prior, aisles, depts = cargar_dataset_raw()

    # 2. Esparsidad ANTES del filtrado
    n_usuarios_pre  = orders['user_id'].nunique()
    n_productos_pre = products['product_id'].nunique()
    esparsidad_pre  = calcular_esparsidad_real(
        prior, orders, n_usuarios_pre, n_productos_pre, "ANTES DEL FILTRADO"
    )

    # 3. Filtrado
    orders_filt, prior_filt, productos_validos = filtrar_dataset(orders, products, prior)

    # 4. Esparsidad DESPUÉS del filtrado
    n_usuarios_post  = orders_filt['user_id'].nunique()
    n_productos_post = len(productos_validos)
    esparsidad_post  = calcular_esparsidad_real(
        prior_filt, orders_filt, n_usuarios_post, n_productos_post,
        "DESPUÉS DEL FILTRADO"
    )

    # 5. Historial con orden temporal correcto
    historial = construir_historial(orders_filt, prior_filt, productos_validos)

    # 6. Partición leave-one-out
    historial = particion_leave_one_out(historial)

    # 7. Análisis
    stats_recompras  = analizar_recompras(historial)
    stats_diversidad = analizar_diversidad(historial)

    # 8. Agregar columnas booleanas a las estadísticas guardadas
    stats_globales = {
        "antes_filtro": esparsidad_pre,
        "despues_filtro": esparsidad_post,
        "filtros_aplicados": {
            "min_ordenes_usuario":  MIN_ORDENES_USUARIO,
            "min_apariciones_prod": MIN_APARICIONES_PROD
        },
        "particion_leave_one_out": {
            "usuarios_finales":     len(historial),
            "longitud_media_train": float(historial['train'].apply(len).mean()),
            "longitud_mediana_train": float(historial['train'].apply(len).median()),
            "longitud_min_train":   int(historial['train'].apply(len).min()),
            "longitud_max_train":   int(historial['train'].apply(len).max()),
        },
        "recompras": stats_recompras,
        "diversidad": stats_diversidad,
        "seed": SEED
    }

    # 9. Guardar
    guardar_resultados(historial, products, productos_validos, stats_globales)


    print("PREPROCESAMIENTO COMPLETO")


    return historial, stats_globales


if __name__ == "__main__":
    historial, stats = ejecutar_pipeline_completo()
