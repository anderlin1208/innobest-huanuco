"""Inferencia compartida por fotografías y cámara.

Sin consultas a MongoDB ni llamadas Streamlit en hilos de vídeo. Un bloqueo
protege los objetos Ultralytics frente a múltiples clientes concurrentes.
"""
from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PESOS_MOBILE = BASE / 'modelos' / 'mobilenet' / 'mejor_modelo.pt'
PESOS_YOLO = BASE / 'modelos' / 'yolo_cls' / 'entrenamiento' / 'weights' / 'best.pt'
PESOS_DETECTOR = BASE / 'yolo11n.pt'

# Modelos y recursos se comparten sin recrearlos para cada usuario o fotograma.
BLOQUEO_MODELOS = threading.RLock()


@lru_cache(maxsize=1)
def establecer_hilos_cpu():
    """Evita sobrecargar CPU y saturar el proceso de Render con demasiados hilos."""
    import torch
    try:
        hilos = max(1, min(4, int(os.environ.get('INNOBEST_CPU_THREADS', '2'))))
    except ValueError:
        hilos = 2
    torch.set_num_threads(hilos)


@lru_cache(maxsize=1)
def modelo_mobile():
    from src.modelo import cargar_modelo
    establecer_hilos_cpu()
    return cargar_modelo(PESOS_MOBILE)


@lru_cache(maxsize=1)
def modelo_yolo():
    from ultralytics import YOLO
    establecer_hilos_cpu()
    return YOLO(str(PESOS_YOLO))


@lru_cache(maxsize=1)
def modelo_detector():
    from src.filtro_aves import cargar_detector
    establecer_hilos_cpu()
    if not PESOS_DETECTOR.is_file():
        raise FileNotFoundError('No está incluido el detector general yolo11n.pt.')
    return cargar_detector(PESOS_DETECTOR)


def clasificar_yolo(imagen):
    import numpy as np
    from src.especies import CLASES
    resultado = modelo_yolo().predict(
        source=np.asarray(imagen.convert('RGB')), imgsz=224,
        device='cpu', verbose=False
    )[0]
    if resultado.probs is None:
        raise RuntimeError('YOLO-cls no devolvió probabilidades.')
    ranking = [(str(resultado.names[i]), float(p))
               for i, p in enumerate(resultado.probs.data.cpu().tolist())]
    if len(ranking) != len(CLASES) or set(nombre for nombre, _ in ranking) != set(CLASES):
        raise ValueError('Las clases del checkpoint YOLO no coinciden con las tres especies.')
    return sorted(ranking, key=lambda par: par[1], reverse=True)


def inferir_imagen(imagen, *, imgsz_detector=416):
    """Devuelve decisiones reales y conservadoras, sin persistir fotogramas.

    No obtiene probabilidades para imágenes sin ave detectada; no inventa ceros.
    El tamaño 416 equilibra coste de CPU y precisión (NO es valor validado).
    """
    from src.filtro_aves import verificar_ave
    from src.decision import evaluar_decision
    from src.modelo import predecir
    from src.especies import ESPECIES

    inicio = time.monotonic()
    with BLOQUEO_MODELOS:
        filtro = verificar_ave(modelo_detector(), imagen, imgsz=imgsz_detector)
        mobile, yolo = [], []
        if filtro.estado == 'ave_detectada':
            mobile = predecir(modelo_mobile(), imagen)
            yolo = clasificar_yolo(imagen)
        decision = evaluar_decision(filtro.estado, mobile, yolo)

    especie = ESPECIES.get(decision.especie, {})
    return {
        'estado': decision.estado,
        'especie': decision.especie,
        'nombre': especie.get('comun'),
        'motivo': decision.motivo,
        'filtro': filtro,
        'mobile': mobile,
        'yolo': yolo,
        'duracion_s': time.monotonic() - inicio,
    }
