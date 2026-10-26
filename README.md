# Sistema de Recomendación Dual con BPE + FastText + CNN-1D

Sistema de recomendación de productos comestibles que integra una **representación dual de embeddings** (semánticos preentrenados + colaborativos aprendidos) sobre el dataset **Instacart Market Basket Analysis 2017**. Desarrollado como tesis de maestría en el IPN-ESCOM.

## Descripción

El sistema aborda cuatro limitaciones de los sistemas de recomendación en el dominio de comestibles: vocabulario abierto, inicio en frío de ítem, sinonimia léxica y escasez de retroalimentación explícita. La arquitectura combina:

- **Tokenización BPE** de vocabulario abierto (8,000 tokens) sobre los nombres de producto.
- **Embeddings FastText** preentrenados de 300 dimensiones (tabla congelada).
- **Embeddings colaborativos** entrenables, integrados con los semánticos mediante un **mecanismo de compuerta (gate)**.
- **Cold-start dropout** que simula la ausencia de señal colaborativa durante el entrenamiento.
- **CNN-1D multiescala** con kernels paralelos k={3,5,7} para modelar el comportamiento secuencial de compra.

## Resultados principales

Evaluación sobre 182,223 usuarios y 47,618 productos, protocolo leave-one-out con 99 negativos fijos por usuario.

| Escenario | Métrica | Dual | Item-CF | TF-IDF | POP |
|-----------|---------|------|---------|--------|-----|
| Warm-start | HR@10 | 0.8019 | 0.8928 | 0.3771 | 0.7617 |
| Cold-start | HR@10 | 0.2949 | 0.0000 | 0.2810 | 0.0000 |

En cold-start de ítem, la representación dual recupera productos sin historial donde el filtrado colaborativo colapsa a 0.0000. La ventaja en HR@10 sobre TF-IDF es estadísticamente significativa (prueba de McNemar, p = 0.0020).

## Estructura del repositorio

    tesis-dual-embeddings/
    ├── src/
    │   ├── config.py              # Configuración e hiperparámetros
    │   ├── data_processing.py     # Preprocesamiento del dataset
    │   ├── tokenization.py        # Tokenizador BPE + embeddings FastText
    │   ├── generar_productos.py   # Mapeo producto-nombre
    │   ├── negativos.py           # Generación de negativos fijos
    │   ├── model_dual.py          # Arquitectura DualEmbeddingCNN
    │   ├── training_dual.py       # Entrenamiento del modelo
    │   ├── evaluacion.py          # Exp 1 y 2 (warm/cold, Tabla 5.2)
    │   ├── ablacion_dual.py       # Exp 3 (ablación)
    │   ├── cobertura_oov.py       # Exp 4 (cobertura OOV)
    │   ├── prueba_estadistica.py  # Exp 5 (McNemar)
    │   ├── sinonimia.py           # Exp 6 (sinonimia léxica)
    │   ├── overlap_lexico.py      # Análisis de validez (overlap cold-warm)
    │   └── verificar_entorno.py   # Verificación del entorno
    ├── notebooks/                 # Notebooks de los experimentos
    ├── results/                   # Resultados en JSON
    ├── Dockerfile
    └── requirements.txt

## Requisitos

- Python 3.11
- PyTorch con soporte CUDA
- GPU NVIDIA (desarrollado sobre RTX 5070 Ti, arquitectura Blackwell)

Instalación:

    conda create -n tesis-dual python=3.11
    conda activate tesis-dual
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python src/verificar_entorno.py

## Datos

El dataset Instacart Market Basket Analysis 2017 se obtiene de [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis); coloca los CSV en `data/raw/`. Los embeddings FastText preentrenados `wiki-news-300d-1M-subword.bin` se descargan de `https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip` y se colocan en `data/raw/`.

Los datos, modelos entrenados y embeddings no se incluyen en el repositorio por su tamaño y la licencia del dataset.

## Reproducción de experimentos

Preprocesamiento (requiere `data/raw/` con los CSV de Instacart y el FastText `.bin`):

    python src/data_processing.py
    python src/tokenization.py
    python src/generar_productos.py
    python src/negativos.py

Entrenamiento:

    python src/training_dual.py

Experimentos:

    python src/evaluacion.py          # Exp 1 y 2: warm + cold (Tabla 5.2)
    python src/ablacion_dual.py       # Exp 3: ablación
    python src/cobertura_oov.py       # Exp 4: cobertura OOV
    python src/prueba_estadistica.py  # Exp 5: McNemar
    python src/sinonimia.py           # Exp 6: sinonimia léxica
    python src/overlap_lexico.py      # Análisis de validez del cold-start

## Autor

Tesis de maestría — IPN-ESCOM.
