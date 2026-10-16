"""YOLO-cls, 100 épocas, MISMAS particiones de MobileNetV3.

Usa Ultralytics clasificación (no detección con cuadros).
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from ultralytics import YOLO

from src.especies import CLASES


def ejecutar(args):
    raiz=Path(args.datos).resolve();salida=Path(args.salida).resolve()
    for split in ('train','val','test'):
        for clase in CLASES:
            if not (raiz/split/clase).is_dir(): raise FileNotFoundError(raiz/split/clase)
    salida.mkdir(parents=True,exist_ok=True)
    # Comprobación de seguridad: YOLO necesita entrenar en GPU T4.
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA no disponible. Activa T4 antes de entrenar.')
    print('GPU YOLO:',torch.cuda.get_device_name(0))
    # YOLO11n-cls preentrenado es adecuado para Colab; no es YOLO de detección.
    modelo=YOLO(args.pesos)
    corrida='entrenamiento'
    modelo.train(data=str(raiz),epochs=100,imgsz=224,batch=16,device=0,workers=2,
                 patience=110,seed=2026,deterministic=True,plots=True,
                 project=str(salida),name=corrida,exist_ok=True,
                 save=True,save_period=10)
    dir_corrida=salida/corrida
    pesos_mejor=dir_corrida/'weights'/'best.pt'
    if not pesos_mejor.is_file():raise FileNotFoundError(f'No se creó best.pt: {pesos_mejor}')
    modelo=YOLO(str(pesos_mejor))
    y_true=[];y_pred=[]
    nombres=modelo.names
    for idx,clase in enumerate(CLASES):
        for foto in sorted((raiz/'test'/clase).iterdir()):
            if not foto.is_file():continue
            with Image.open(foto) as im:
                im=im.convert('RGB')
                import numpy as np
                arr=np.array(im)
            r=modelo.predict(source=arr,imgsz=224,device=0,verbose=False)[0]
            clase_predicha=str(nombres[int(r.probs.top1)])
            if clase_predicha not in CLASES:
                raise ValueError(f'Clase inesperada en YOLO: {clase_predicha}')
            y_true.append(idx);y_pred.append(CLASES.index(clase_predicha))
    matriz=confusion_matrix(y_true,y_pred,labels=list(range(len(CLASES))))
    pd.DataFrame(matriz,index=CLASES,columns=CLASES).to_csv(salida/'matriz_confusion.csv')
    historia=dir_corrida/'results.csv'
    if historia.is_file():shutil.copy2(historia,salida/'historial.csv')
    reporte=classification_report(y_true,y_pred,labels=list(range(len(CLASES))),target_names=CLASES,zero_division=0,output_dict=True)
    total=sum(1 for split in ('train','val','test') for c in CLASES for p in (raiz/split/c).iterdir() if p.is_file())
    metricas={'modelo':'YOLO11n-cls','especies':CLASES,'version':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
              'total_imagenes':total,'n_entrenamiento':sum(len(list((raiz/'train'/c).iterdir())) for c in CLASES),
              'n_validacion':sum(len(list((raiz/'val'/c).iterdir())) for c in CLASES),'n_prueba':len(y_true),
              'accuracy_test':round(accuracy_score(y_true,y_pred),5),
              'f1_macro_test':round(f1_score(y_true,y_pred,average='macro',zero_division=0),5),
              'reporte_por_clase':reporte,'gpu':torch.cuda.get_device_name(0),
              'epocas_solicitadas':100,'epocas_ejecutadas':100,'ruta_mejor_modelo':str(pesos_mejor)}
    # Verificar número real de filas del registro por si Ultralytics detuvo o falló antes.
    if (salida/'historial.csv').is_file():
        metricas['epocas_ejecutadas']=len(pd.read_csv(salida/'historial.csv'))
    (salida/'metricas.json').write_text(json.dumps(metricas,ensure_ascii=False,indent=2),encoding='utf-8')
    print('YOLO: ',metricas['epocas_ejecutadas'],'/100 épocas; Accuracy test',metricas['accuracy_test'],'F1 test',metricas['f1_macro_test'])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--datos',required=True)
    parser.add_argument('--salida',required=True)
    parser.add_argument('--pesos',default='yolo11n-cls.pt')
    ejecutar(parser.parse_args())
