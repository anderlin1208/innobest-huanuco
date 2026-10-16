"""Validación de ZIP aportados desde panel web para entrenar cinco clases.

Se exige un conjunto externo de test que nunca se mezcla con feedback.
La revisión de licencias, taxonomía y duplicados visuales sigue siendo humana.
"""
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, BadZipFile

CLASES_CINCO = ('ara_militaris', 'pauxi_koepckeae', 'spizaetus_isidori', 'otra_ave', 'no_ave')
SPLITS = ('train', 'val', 'test')
EXT = {'.jpg', '.jpeg', '.png', '.webp'}
MINIMOS = {'train': 10, 'val': 3, 'test': 3}


def validar_dataset_zip(contenido: bytes):
    if not contenido or len(contenido) > 250 * 1024 * 1024:
        raise ValueError('ZIP vacío o superior a 250 MB.')
    conteos = Counter()
    huellas = set()
    total_expandido = 0
    total_archivos = 0
    try:
        with ZipFile(BytesIO(contenido)) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                ruta = PurePosixPath(info.filename.replace('\\', '/'))
                if '..' in ruta.parts or len(ruta.parts) not in (3, 4):
                    raise ValueError('ZIP inválido: los archivos deben estar en train|val|test/clase/foto.jpg.')
                partes = ruta.parts[-3:]
                split, clase, nombre = partes
                if info.filename.startswith('/') or split not in SPLITS or clase not in CLASES_CINCO or Path(nombre).suffix.lower() not in EXT:
                    raise ValueError('Ruta o clase inesperada en ZIP: ' + info.filename[:100])
                if info.file_size > 8 * 1024 * 1024:
                    raise ValueError('Cada imagen debe pesar como máximo 8 MB.')
                total_expandido += info.file_size
                total_archivos += 1
                if total_expandido > 350 * 1024 * 1024 or total_archivos > 3500:
                    raise ValueError('Demasiados datos en el ZIP.')
                foto = z.read(info)
                huella = sha256(foto).hexdigest()
                if huella in huellas:
                    raise ValueError('Hay duplicados exactos entre train, val o test.')
                huellas.add(huella)
                # No se aceptan falsos .jpg (más adelante PIL los valida).
                from PIL import Image
                with Image.open(BytesIO(foto)) as imagen:
                    if imagen.width * imagen.height > 24_000_000:
                        raise ValueError('Una imagen supera 24 megapíxeles.')
                    imagen.verify()
                conteos[(split, clase)] += 1
    except (BadZipFile, OSError, EOFError) as e:
        raise ValueError('ZIP inválido o con fotografías dañadas.') from e
    for split in SPLITS:
        for clase in CLASES_CINCO:
            if conteos[(split, clase)] < MINIMOS[split]:
                raise ValueError(f'Datos insuficientes en {split}/{clase}: '
                                 f'{conteos[(split, clase)]}; mínimo {MINIMOS[split]}.')
    return {'archivos': total_archivos, 'por_clase': {
        c: {s: conteos[(s, c)] for s in SPLITS} for c in CLASES_CINCO}}


def extraer_validado(contenido: bytes, destino: Path):
    """Solo llamar tras validar. Construye directorios desde clases conocidas."""
    validar_dataset_zip(contenido)
    destino = Path(destino)
    with ZipFile(BytesIO(contenido)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            split, clase, nombre = PurePosixPath(info.filename.replace('\\', '/')).parts[-3:]
            ruta = destino / split / clase / nombre
            if ruta.exists():
                raise ValueError('Nombres de imagen duplicados: ' + str(ruta))
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(z.read(info))
    return destino
