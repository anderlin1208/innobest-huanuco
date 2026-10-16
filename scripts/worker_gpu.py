"""Ejecutor GPU, conectado a la cola creada desde la web (Colab T4 u otra GPU).

Configura MONGODB_URI de forma privada en el entorno. No publiques la URI.
Ejecución: python scripts/worker_gpu.py [--una-vez]
"""
import argparse
import io
import json
import os
import sys
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset_entrenamiento import extraer_validado, validar_dataset_zip, CLASES_CINCO
from src.mongodb import RepositorioPredicciones
from src.repositorio_feedback import RepositorioFeedback
from src.trabajos import RepositorioTrabajos


def incorporar_feedback(datos: Path, repo: RepositorioFeedback):
    """Solo añade aprobados a TRAIN sin alterar VAL/TEST. Sin duplicados de contenido."""
    hashes=set()
    for fichero in datos.rglob('*'):
        if fichero.is_file():
            hashes.add(sha256(fichero.read_bytes()).hexdigest())
    n=0
    for doc in repo.aprobadas_exportables():
        clase = doc.get('etiqueta_revisada')
        if clase not in CLASES_CINCO or not doc.get('imagen_gridfs_id'):
            continue
        imagen=repo.leer_imagen(doc['imagen_gridfs_id'])
        huella=sha256(imagen).hexdigest()
        if huella in hashes: continue
        hashes.add(huella)
        destino=datos/'train'/clase/(huella[:30]+'.jpg')
        if not destino.exists():
            destino.parent.mkdir(parents=True,exist_ok=True)
            destino.write_bytes(imagen)
            n+=1
    return n


def ejecutar_trabajo(job, cola, repo_fb):
    ident=job['_id']
    with tempfile.TemporaryDirectory(prefix='innobest_gpu_') as tmp:
        raiz=Path(tmp)
        contenido=cola.datasets.get(job['dataset_gridfs_id']).read()
        cola.progreso(ident,'Verificando y extrayendo dataset de cinco clases')
        extraer_validado(contenido,raiz/'datos')
        agregadas=incorporar_feedback(raiz/'datos',repo_fb)
        cola.progreso(ident,f'Datos listos. {agregadas} correcciones aprobadas incorporadas al train.')
        from src.entrenamiento_cinco import entrenar_mobilenet, entrenar_yolo
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('No hay GPU CUDA; abre este worker en Colab con T4.')
        anterior=ROOT/'modelos'/'mobilenet'/'mejor_modelo.pt'
        pesos_tres=anterior if anterior.is_file() else None
        salida=raiz/'resultados'
        mobile=entrenar_mobilenet(raiz/'datos',salida/'mobilenet',pesos_tres,
                                on_epoch=lambda msg: cola.progreso(ident,msg))
        cola.progreso(ident,'MobileNetV3 terminado. Iniciando YOLO11n-cls (100 épocas)')
        yolo=entrenar_yolo(raiz/'datos',salida/'yolo_cls',
                          on_epoch=lambda msg: cola.progreso(ident,msg))
        meta={'MobileNetV3':mobile,'YOLO11n-cls':yolo,'feedback_añadido':agregadas}
        salida.joinpath('resultado.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        memoria=io.BytesIO()
        with ZipFile(memoria,'w',ZIP_DEFLATED) as z:
            for archivo in salida.rglob('*'):
                if archivo.is_file() and archivo.stat().st_size < 100*1024*1024:
                    z.write(archivo,archivo.relative_to(salida).as_posix())
        cola.completar(ident,artefacto=memoria.getvalue(),metricas=meta)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--una-vez',action='store_true',help='Finalizar después de ejecutar un trabajo o de encontrar la cola vacía')
    args=ap.parse_args()
    load_dotenv(ROOT/'.env')
    uri=os.getenv('MONGODB_URI')
    if not uri: raise SystemExit('Falta MONGODB_URI en entorno. No ingreses contraseñas en el código.')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('Sin GPU CUDA: este worker requiere Colab T4 o GPU compatible.')
    repo=RepositorioPredicciones(uri,os.getenv('MONGODB_DB','innobest_huanuco'))
    repo.verificar_conexion()
    cola=RepositorioTrabajos(repo.bd)
    feedback=RepositorioFeedback(repo.bd)
    print('Ejecutor GPU conectado. Esperando trabajos desde INNOBEST.')
    while True:
        job=cola.tomar_siguiente()
        if job:
            try:
                ejecutar_trabajo(job,cola,feedback)
                print('Trabajo completado:',str(job['_id']))
            except Exception as exc:
                cola.fallar(job['_id'],f'{type(exc).__name__}: {str(exc)[:180]}')
                print('Error:',type(exc).__name__,str(exc)[:180])
        if args.una_vez: break
        time.sleep(15)

if __name__=='__main__': main()
