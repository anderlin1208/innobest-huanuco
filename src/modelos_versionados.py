"""Carga pesos versionados publicados por un worker GPU confiable desde GridFS.

Se verifica la estructura y la coincidencia exacta de clases antes de habilitar inferencia.
"""
import io
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile
from dataclasses import dataclass

from src.dataset_entrenamiento import CLASES_CINCO


@dataclass
class ModelosVersionados:
    mobile: object
    yolo: object
    clases_mobile: list
    temporal: object


def cargar_bundle_zip(contenido: bytes) -> ModelosVersionados:
    import torch
    from ultralytics import YOLO
    from torch import nn
    from torchvision import models
    nombres = ['mobilenet/mejor_modelo.pt', 'mobilenet/metadatos.json',
               'yolo_cls/weights/best.pt', 'yolo_cls/metadatos.json']
    temporal=tempfile.TemporaryDirectory(prefix='innobest_version_')
    try:
        with ZipFile(io.BytesIO(contenido)) as z:
            for nombre in nombres:
                if nombre not in z.namelist():
                    raise ValueError('El modelo entrenado está incompleto: '+nombre)
            meta=json.loads(z.read('mobilenet/metadatos.json'))
            clases=list(meta.get('clases',[]))
            if set(clases)!=set(CLASES_CINCO) or len(clases)!=5:
                raise ValueError('El orden de las cinco clases no está validado.')
            ruta=Path(temporal.name)/'best.pt'
            ruta.write_bytes(z.read('yolo_cls/weights/best.pt'))
            clasificador=models.mobilenet_v3_small(weights=None)
            dim=clasificador.classifier[-1].in_features
            clasificador.classifier[-1]=nn.Linear(dim,len(clases))
            clasificador.load_state_dict(torch.load(io.BytesIO(z.read('mobilenet/mejor_modelo.pt')),
                                                   map_location='cpu',weights_only=True))
            clasificador.eval()
            yolo=YOLO(str(ruta))
            if set(yolo.names.values())!=set(CLASES_CINCO):
                raise ValueError('Las clases YOLO no corresponden al entrenamiento.')
            return ModelosVersionados(clasificador,yolo,clases,temporal)
    except Exception:
        temporal.cleanup()
        raise


def predecir_mobile_versionado(bundle:ModelosVersionados, imagen):
    import torch
    from PIL import ImageOps
    from src.modelo import transformacion_evaluacion
    x=transformacion_evaluacion()(ImageOps.exif_transpose(imagen).convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        puntajes=torch.softmax(bundle.mobile(x),dim=1)[0].tolist()
    return sorted(zip(bundle.clases_mobile,puntajes),key=lambda v:v[1],reverse=True)


def predecir_yolo_versionado(bundle:ModelosVersionados,imagen):
    import numpy as np
    r=bundle.yolo.predict(np.asarray(imagen.convert('RGB')),imgsz=224,device='cpu',verbose=False)[0]
    return sorted([(r.names[i],float(p)) for i,p in enumerate(r.probs.data.cpu().tolist())],
                  key=lambda v:v[1],reverse=True)
