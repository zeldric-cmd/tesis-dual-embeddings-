"""
Verificación del entorno de reproducibilidad.

Comprueba la versión de Python, las versiones de las dependencias y que el
código del proyecto (config.py) importe correctamente. Es el CMD por defecto
del contenedor Docker del Apéndice B: documenta el entorno, no entrena.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("VERIFICACIÓN DEL ENTORNO — Sistema de Recomendación Dual")
print(f"Python: {sys.version.split()[0]}")

deps = ["numpy", "pandas", "scipy", "sklearn", "tokenizers",
        "statsmodels", "matplotlib", "tqdm", "torch", "fasttext"]
print("\nDependencias:")
todo_ok = True
for m in deps:
    try:
        mod = __import__(m)
        print(f"  [OK]    {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  [FALTA] {m:<14} {e}")
        todo_ok = False

try:
    import torch
    print(f"\nPyTorch {torch.__version__} | CUDA disponible: {torch.cuda.is_available()}")
    print("  (imagen CPU por diseño; el entrenamiento con GPU se hace fuera del contenedor)")
except Exception:
    pass

try:
    import config
    print(f"\nconfig.py OK — EMB_DIM={config.EMB_DIM}, N_FILTERS={config.N_FILTERS}, "
          f"SEED={config.SEED}, COMBINE_MODE={config.COMBINE_MODE}, "
          f"VOCAB_SIZE_BPE={config.VOCAB_SIZE_BPE}")
except Exception as e:
    print(f"\nconfig.py: ERROR al importar — {e}")
    todo_ok = False

print("ENTORNO OK" if todo_ok else "ENTORNO INCOMPLETO — revisar dependencias")
sys.exit(0 if todo_ok else 1)
