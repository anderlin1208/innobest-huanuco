"""Filtro genérico antes de clasificar especies: YOLO11n COCO DETECCIÓN.

Este no es el YOLO11-cls de tres especies. Usa pesos generales
``yolo11n.pt`` (descarga automática al iniciar por primera vez, si faltan).
COCO incluye la clase 'bird'; NO garantiza reconocer todas las aves silvestres.
Cuando no hay evidencia suficiente, la clasificación se suspende.
"""
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np


@dataclass(frozen=True)
class ResultadoFiltro:
    estado: str  # ave_detectada, sin_ave, varias_aves, no_verificable
    descripcion: str
    aves: int = 0
    objetos: tuple[str, ...] = ()
    recuadros: tuple[tuple[int, int, int, int], ...] = ()


def cargar_detector(pesos='yolo11n.pt'):
    from ultralytics import YOLO
    return YOLO(str(pesos))


def verificar_ave(detector, imagen: Image.Image, umbral_ave=0.20, umbral_objeto=0.35, imgsz=640):
    arr = np.asarray(ImageOps.exif_transpose(imagen).convert('RGB'))
    resultados = detector.predict(source=arr, imgsz=imgsz, conf=min(umbral_ave, umbral_objeto),
                                  device='cpu', verbose=False)
    result = resultados[0]
    encontrados = []
    num_aves = 0
    recuadros = []
    if result.boxes is not None:
        clases = result.boxes.cls.cpu().tolist()
        puntuaciones = result.boxes.conf.cpu().tolist()
        coordenadas = (result.boxes.xyxy.cpu().tolist()
                       if hasattr(result.boxes, 'xyxy') else [None] * len(clases))
        for clase, puntuacion, xyxy in zip(clases, puntuaciones, coordenadas):
            nombre = str(result.names[int(clase)]).lower()
            if nombre == 'bird' and puntuacion >= umbral_ave:
                num_aves += 1
                if xyxy is not None:
                    recuadros.append(tuple(int(v) for v in xyxy))
            elif puntuacion >= umbral_objeto:
                encontrados.append(nombre)
    if num_aves > 1:
        return ResultadoFiltro('varias_aves', f'Se detectaron {num_aves} aves.', num_aves, tuple(encontrados), tuple(recuadros))
    if num_aves == 1:
        return ResultadoFiltro('ave_detectada', 'Se detectó una posible ave.', num_aves, tuple(encontrados), tuple(recuadros))
    if encontrados:
        return ResultadoFiltro('sin_ave', 'No se detectó un ave; otros objetos visibles.', 0, tuple(encontrados))
    return ResultadoFiltro('no_verificable', 'El detector no encontró evidencia suficiente.', 0, ())
