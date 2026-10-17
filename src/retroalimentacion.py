"""Etiquetas humanas y tratamiento de imágenes para feedback opt-in.

La etiqueta introducida por una persona NO altera automáticamente los pesos de IA,
ni certifica la identidad biológica de una especie.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from PIL import Image, ImageOps

from src.especies import CLASES, ESPECIES

OTRA_AVE = 'otra_ave'
NO_AVE = 'no_ave'
NO_SE = 'no_se_puede_determinar'
ETIQUETAS_FEEDBACK = tuple(CLASES) + (OTRA_AVE, NO_AVE, NO_SE)
ETIQUETAS_ENTRENABLES = frozenset(CLASES + [OTRA_AVE, NO_AVE])
NOMBRES = {k: ESPECIES[k]['comun'] for k in CLASES} | {
    OTRA_AVE: 'Otra especie de ave (fuera de las 3)',
    NO_AVE: 'No es un ave',
    NO_SE: 'No puedo determinarlo',
}
MAX_ARCHIVO = 8 * 1024 * 1024
MAX_PIXELES = 24_000_000


@dataclass(frozen=True)
class ImagenFeedback:
    huella: str
    jpeg_sin_metadatos: bytes
    ancho: int
    alto: int


def preparar_imagen_feedback(contenido: bytes) -> ImagenFeedback:
    """Fingerprint exacto del archivo, JPEG reducido y desprovisto de EXIF/metadatos."""
    if not isinstance(contenido, bytes) or not contenido or len(contenido) > MAX_ARCHIVO:
        raise ValueError('La imagen debe pesar entre 1 byte y 8 MB.')
    try:
        with Image.open(BytesIO(contenido)) as original:
            if original.width * original.height > MAX_PIXELES:
                raise ValueError('La imagen tiene demasiados píxeles.')
            imagen = ImageOps.exif_transpose(original).convert('RGB')
            imagen.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            salida = BytesIO()
            imagen.save(salida, format='JPEG', quality=85, optimize=True)
            return ImagenFeedback(sha256(contenido).hexdigest(), salida.getvalue(),
                                  imagen.width, imagen.height)
    except (OSError, SyntaxError) as exc:
        raise ValueError('Archivo de imagen no válido.') from exc


def validar_etiqueta(etiqueta: str) -> str:
    if etiqueta not in ETIQUETAS_FEEDBACK:
        raise ValueError('La etiqueta humana no corresponde a una opción permitida.')
    return etiqueta


def es_apta_para_reentrenamiento(etiqueta: str, autorizada: bool, revision: str) -> bool:
    return bool(autorizada and etiqueta in ETIQUETAS_ENTRENABLES and revision == 'aprobada')
