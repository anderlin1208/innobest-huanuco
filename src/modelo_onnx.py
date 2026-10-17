"""Predicción MobileNetV3 Small (3 especies) sin importar PyTorch.

Preproceso equivalente a MobileNet_V3_Small_Weights.DEFAULT.transforms()
(Resize menor lado=256, CenterCrop=224, normalización ImageNet).
El ONNX incluido deriva del checkpoint entregado mejor_modelo.pt.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.especies import CLASES

RUTA_MODELO = Path(__file__).resolve().parent.parent / 'modelos' / 'mobilenet' / 'mejor_modelo.onnx'
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
DESVIACION = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preparar_imagen(imagen: Image.Image) -> np.ndarray:
    """Convierte la imagen PIL a entrada NCHW float32 igual a torchvision."""
    imagen = ImageOps.exif_transpose(imagen).convert('RGB')
    ancho, alto = imagen.size
    if min(ancho, alto) <= 0:
        raise ValueError('La imagen no tiene dimensiones válidas.')
    # torchvision Resize(256) usa truncamiento entero para el lado mayor.
    if ancho <= alto:
        nuevo_ancho, nuevo_alto = 256, int(256 * alto / ancho)
    else:
        nuevo_ancho, nuevo_alto = int(256 * ancho / alto), 256
    imagen = imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.BILINEAR)
    # torchvision center_crop usa round((dim - crop)/2); Python usa tie-to-even.
    izquierda = round((nuevo_ancho - 224) / 2)
    arriba = round((nuevo_alto - 224) / 2)
    imagen = imagen.crop((izquierda, arriba, izquierda + 224, arriba + 224))
    entrada = np.asarray(imagen, dtype=np.float32) / np.float32(255.0)
    entrada = (entrada - MEDIA) / DESVIACION
    return np.ascontiguousarray(entrada.transpose(2, 0, 1)[None, ...])


def cargar_sesion(ruta: Path = RUTA_MODELO):
    import onnxruntime as ort

    if not ruta.is_file():
        raise FileNotFoundError('No se encontró modelos/mobilenet/mejor_modelo.onnx')
    opciones = ort.SessionOptions()
    # Optimización de 512 MB: limitar hilos y evitar reservar grandes arenas de RAM.
    opciones.intra_op_num_threads = max(1, min(2, int(os.getenv('INNOBEST_CPU_THREADS', '1'))))
    opciones.inter_op_num_threads = 1
    opciones.enable_cpu_mem_arena = False
    opciones.enable_mem_pattern = False
    opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ruta), sess_options=opciones, providers=['CPUExecutionProvider'])


def predecir(sesion, imagen: Image.Image) -> list[tuple[str, float]]:
    entrada = preparar_imagen(imagen)
    entrada_nombre = sesion.get_inputs()[0].name
    logits = np.asarray(sesion.run(None, {entrada_nombre: entrada})[0], dtype=np.float64)
    if logits.shape != (1, len(CLASES)) or not np.isfinite(logits).all():
        raise ValueError('El modelo devolvió una salida inválida.')
    puntuaciones = logits[0] - np.max(logits[0])
    exp = np.exp(puntuaciones)
    prob = exp / exp.sum()
    return sorted(zip(CLASES, (float(p) for p in prob)), key=lambda fila: fila[1], reverse=True)


def interpretar(ranking: list[tuple[str, float]], *, umbral: float = 0.85,
                margen_minimo: float = 0.20) -> dict:
    """Decisión heurística; no es un detector de aves ajenas ni de outliers."""
    if len(ranking) != 3 or {c for c, _ in ranking} != set(CLASES):
        raise ValueError('El modelo debe producir exactamente las tres especies.')
    principal, segunda = ranking[0], ranking[1]
    aceptada = principal[1] >= umbral and principal[1] - segunda[1] >= margen_minimo
    return {
        'estado': 'compatible_sin_verificar' if aceptada else 'no_concluyente',
        'especie': principal[0] if aceptada else None,
        'mayor_puntuacion': principal[1],
        'margen': principal[1] - segunda[1],
        'motivo': ('Coincide con una de las tres especies; requiere verificación humana.'
                   if aceptada else
                   'La puntuación o la separación entre especies no supera los umbrales provisionales.'),
    }
