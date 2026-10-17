"""Medir latencia REAL con una foto, desde la raíz del proyecto."""
import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


def main():
    parser = argparse.ArgumentParser(description='Medir inferencia CPU real.')
    parser.add_argument('foto', type=Path, help='Ruta a una fotografía de prueba.')
    parser.add_argument('--repeticiones', type=int, default=3)
    args = parser.parse_args()
    from PIL import Image
    from src.motor_inferencia import inferir_imagen
    from src.camara_vivo import estimar_etiqueta
    with Image.open(args.foto) as im:
        foto = im.convert('RGB')
    for i in range(max(1, min(args.repeticiones, 10))):
        inicio = time.monotonic()
        res = inferir_imagen(foto)
        ms = round(1000 * (time.monotonic() - inicio))
        try:
            import psutil
            memoria = round(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
            print(f'Intento {i+1}: {ms} ms / {memoria} MiB RAM / {estimar_etiqueta(res)}')
        except ImportError:
            print(f'Intento {i+1}: {ms} ms / {estimar_etiqueta(res)}')
    print('Las etiquetas son preliminares; no se mide exactitud sin ejemplos anotados.')


if __name__ == '__main__':
    main()
