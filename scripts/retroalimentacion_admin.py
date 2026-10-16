"""Administración LOCAL de correcciones. No publicar comandos de aprobación en la web.

Ejemplos (desde la raíz):
py -3.12 scripts/retroalimentacion_admin.py listar
py -3.12 scripts/retroalimentacion_admin.py descargar-pendientes --salida revision_pendientes
py -3.12 scripts/retroalimentacion_admin.py aprobar --id ID --etiqueta otra_ave
py -3.12 scripts/retroalimentacion_admin.py rechazar --id ID
py -3.12 scripts/retroalimentacion_admin.py exportar-aprobadas --salida feedback_aprobado

Debes COMPROBAR VISUALMENTE el contenido y la licencia de la imagen antes de aprobar.
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mongodb import RepositorioPredicciones
from src.repositorio_feedback import RepositorioFeedback
from src.retroalimentacion import ETIQUETAS_FEEDBACK, NO_SE


def guardar_archivos(registros, salida: Path, repositorio, usar_revisada=False):
    salida.mkdir(parents=True, exist_ok=True)
    manifiesto = []
    visitadas = set()
    for doc in registros:
        if not doc.get('imagen_guardada') or not doc.get('imagen_gridfs_id'):
            continue
        etiqueta = doc.get('etiqueta_revisada' if usar_revisada else 'etiqueta_usuario')
        if etiqueta not in ETIQUETAS_FEEDBACK or etiqueta == NO_SE:
            continue
        huella = doc['huella_sha256']
        # Deduplicación por hash de contenido: no entrenar dos veces la misma fotografía.
        if huella in visitadas:
            continue
        visitadas.add(huella)
        foto = repositorio.leer_imagen(doc['imagen_gridfs_id'])
        destino = salida / etiqueta / (huella[:20] + '.jpg')
        destino.parent.mkdir(parents=True, exist_ok=True)
        # Evitar reemplazar archivos existentes sin necesidad.
        if not destino.exists():
            destino.write_bytes(foto)
        manifiesto.append({'id': str(doc['_id']), 'clase': etiqueta,
                           'huella_sha256': huella, 'archivo': str(destino.relative_to(salida)),
                           'estado_revision': doc['estado_revision']})
    with (salida / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'clase', 'huella_sha256', 'archivo', 'estado_revision'])
        writer.writeheader()
        writer.writerows(manifiesto)
    print(f'Exportadas {len(manifiesto)} imágenes a {salida}.')


def main():
    parser = argparse.ArgumentParser(description='Revisar correcciones de INNOBEST')
    parser.add_argument('accion', choices=['listar', 'descargar-pendientes', 'aprobar', 'rechazar',
                                           'exportar-aprobadas'])
    parser.add_argument('--id', default='')
    parser.add_argument('--etiqueta', choices=ETIQUETAS_FEEDBACK)
    parser.add_argument('--salida', default='feedback_revision')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    uri = os.getenv('MONGODB_URI', '')
    if not uri:
        raise SystemExit('Configura MONGODB_URI en .env; nunca compartas esa URI.')
    repo = RepositorioFeedback(RepositorioPredicciones(uri, os.getenv('MONGODB_DB', 'innobest_huanuco')).bd)
    if args.accion == 'listar':
        datos = repo.listar_pendientes(100)
        for doc in datos:
            print(str(doc['_id']), doc['etiqueta_usuario'],
                  'imagen_autorizada=' + str(doc.get('imagen_guardada', False)))
        print('Total pendientes mostrados:', len(datos))
    elif args.accion == 'descargar-pendientes':
        datos = repo.coleccion.find({'estado_revision': 'pendiente', 'imagen_guardada': True})
        guardar_archivos(datos, Path(args.salida), repo)
        print('Revisa fotos y permisos ANTES de aprobar; son etiquetas propuestas por usuarios.')
    elif args.accion in ('aprobar', 'rechazar'):
        if not ObjectId.is_valid(args.id):
            raise SystemExit('Indica --id con el identificador mostrado por listar.')
        if args.accion == 'aprobar' and not args.etiqueta:
            raise SystemExit('Para aprobar debes aportar --etiqueta tras verificar la especie.')
        resultado = repo.revisar(args.id, args.accion == 'aprobar', args.etiqueta)
        print('Revisión guardada:', resultado['estado_revision'],
              'elegible_para_entrenamiento=', resultado['apta_entrenamiento'])
    else:
        datos = repo.aprobadas_exportables()
        guardar_archivos(datos, Path(args.salida), repo, usar_revisada=True)
        print('No mezcles automáticamente con train/val/test originales: revisar duplicados y licencias.')


if __name__ == '__main__':
    main()
