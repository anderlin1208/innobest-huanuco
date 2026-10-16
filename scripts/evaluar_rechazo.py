"""Evalúa fotografías etiquetadas; NO modifica los modelos ni fotos.

Uso:
 py -3.12 scripts/evaluar_rechazo.py --objetivo C:/datos/validacion_objetivo --externas C:/datos/validacion_externas

Objetivo debe contener carpetas ara_militaris, pauxi_koepckeae, spizaetus_isidori.
Externas puede contener subcarpetas arbitrarias, por ejemplo otros_animales y otras_aves.
No usar como calibración el conjunto final de prueba de entrenamiento.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from ultralytics import YOLO
from src.especies import CLASES
from src.modelo import cargar_modelo, predecir
from src.filtro_aves import cargar_detector, verificar_ave
from src.decision import evaluar_decision
import numpy as np

EXT = {'.jpg', '.jpeg', '.png', '.webp'}
BASE=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--objetivo', required=True)
    p.add_argument('--externas', required=True)
    p.add_argument('--salida', default='evaluacion_rechazo.csv')
    args=p.parse_args()
    objetivo, externas = Path(args.objetivo), Path(args.externas)
    if not objetivo.is_dir() or not externas.is_dir():
        raise FileNotFoundError('Faltan carpetas de validación objetivo / externas.')
    modelo = cargar_modelo(BASE/'modelos/mobilenet/mejor_modelo.pt')
    yolo = YOLO(str(BASE/'modelos/yolo_cls/entrenamiento/weights/best.pt'))
    detector = cargar_detector(BASE/'modelos/detector_general/yolo11n.pt'
                               if (BASE/'modelos/detector_general/yolo11n.pt').exists() else 'yolo11n.pt')
    archivos=[]
    for clase in CLASES:
        carpeta=objetivo/clase
        if not carpeta.is_dir():
            raise FileNotFoundError(carpeta)
        archivos += [(x, clase) for x in sorted(carpeta.rglob('*')) if x.suffix.lower() in EXT]
    archivos += [(x, 'desconocida') for x in sorted(externas.rglob('*')) if x.is_file() and x.suffix.lower() in EXT]
    resultados=[]
    for ruta, real in archivos:
        try:
            with Image.open(ruta) as im:
                im.load()
                filtro=verificar_ave(detector, im)
                if filtro.estado == 'ave_detectada':
                    mob=predecir(modelo, im)
                    pred=yolo.predict(source=np.asarray(im.convert('RGB')), imgsz=224, device='cpu', verbose=False)[0]
                    yy=sorted([(str(pred.names[i]), float(val)) for i, val in enumerate(pred.probs.data.cpu().tolist())],
                              key=lambda t:t[1], reverse=True)
                else:
                    mob,yy=[],[]
                dec=evaluar_decision(filtro.estado,mob,yy)
                resultados.append({'archivo':str(ruta),'real':real,'filtro':filtro.estado,
                                   'estado':dec.estado, 'predicha':dec.especie or '',
                                   'correcta':bool(dec.especie==real and real!='desconocida'),
                                   'aceptacion_erronea':bool(dec.especie and dec.especie != real)})
        except (OSError, ValueError) as error:
            print(f'Imagen no válida: {ruta} ({type(error).__name__})')
    if not resultados:
        raise ValueError('No se encontraron fotos para evaluar.')
    with open(args.salida,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(resultados[0])); w.writeheader(); w.writerows(resultados)
    conocidas=[x for x in resultados if x['real']!='desconocida']
    desconocidas=[x for x in resultados if x['real']=='desconocida']
    print('Archivos evaluados:',len(resultados))
    print('Conocidas correctas aceptadas:',sum(x['correcta'] for x in conocidas), '/',len(conocidas))
    print('Conocidas rechazadas/no concluyentes:',sum(not x['predicha'] for x in conocidas),'/',len(conocidas))
    print('Desconocidas incorrectamente aceptadas:',sum(x['aceptacion_erronea'] for x in desconocidas), '/',len(desconocidas))
    print('Casos incorrectamente aceptados:',sum(x['aceptacion_erronea'] for x in resultados))
    print('CSV:', args.salida)


if __name__=='__main__':main()
