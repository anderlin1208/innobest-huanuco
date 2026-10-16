"""Entrena MobileNetV3 de 5 clases y YOLO11n-cls en un WORKER con GPU.

Los modelos de 3 clases no se sobrescriben. Dataset externo con train/val/test
para las 5 clases; cada corrida ejecuta 100 épocas (80+20 MobileNet).
"""
from __future__ import annotations
import json
from pathlib import Path

from .dataset_entrenamiento import CLASES_CINCO


def crear_mobilenet_cinco(pesos_tres: Path | None = None, clases=None):
    import torch
    from torch import nn
    from torchvision import models
    from src.especies import CLASES
    clases = list(clases or CLASES_CINCO)
    red = models.mobilenet_v3_small(weights=None if pesos_tres else models.MobileNet_V3_Small_Weights.DEFAULT)
    dim = red.classifier[-1].in_features
    if pesos_tres:
        # Preservar conocimiento de las tres especies en las tres primeras filas.
        red.classifier[-1] = nn.Linear(dim, 3)
        red.load_state_dict(torch.load(str(pesos_tres), map_location='cpu', weights_only=True))
        vieja = red.classifier[-1]
        nueva = nn.Linear(dim, len(clases))
        with torch.no_grad():
            for idx, nombre in enumerate(CLASES):
                destino = clases.index(nombre)
                nueva.weight[destino].copy_(vieja.weight[idx])
                nueva.bias[destino].copy_(vieja.bias[idx])
        red.classifier[-1] = nueva
    else:
        red.classifier[-1] = nn.Linear(dim, len(clases))
    return red


def entrenar_mobilenet(raiz: Path, salida: Path, pesos_tres: Path | None, on_epoch=lambda x: None):
    import torch
    import pandas as pd
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    salida.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clases_ordenadas = sorted(CLASES_CINCO)
    red = crear_mobilenet_cinco(pesos_tres, clases_ordenadas).to(device)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.80,1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(9),
        transforms.ColorJitter(brightness=.12, contrast=.12),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    test_tf = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
    conjuntos = {s: datasets.ImageFolder(raiz/s, transform=train_tf if s=='train' else test_tf)
                 for s in ('train','val','test')}
    if set(conjuntos['train'].classes) != set(CLASES_CINCO):
        raise ValueError('Faltan clases en el entrenamiento.')
    # ImageFolder ordena alfabéticamente: se usa ese mismo orden para métricas e inferencia.
    classes = conjuntos['train'].classes
    cargas = {s: DataLoader(ds, batch_size=16, shuffle=s=='train', num_workers=0)
              for s, ds in conjuntos.items()}
    for param in red.features.parameters(): param.requires_grad = False
    history=[]
    mejor=float('inf')
    pesos=salida/'mejor_modelo.pt'
    for epoca in range(100):
        if epoca==80:
            for param in red.features[-3:].parameters(): param.requires_grad=True
        optim = None
        if epoca in (0,80):
            optim = torch.optim.AdamW((p for p in red.parameters() if p.requires_grad),
                                     lr=1e-3 if epoca==0 else 1e-5)
        if optim is not None: optimizador = optim
        registro={'epoca':epoca+1}
        for fase in ('train','val'):
            red.train(fase=='train')
            if fase=='train':
                for modulo in red.modules():
                    if isinstance(modulo, nn.modules.batchnorm._BatchNorm): modulo.eval()
            perdido=0
            aciertos=0
            total=0
            for imagenes, etiquetas in cargas[fase]:
                imagenes,etiquetas=imagenes.to(device),etiquetas.to(device)
                with torch.set_grad_enabled(fase=='train'):
                    pred=red(imagenes)
                    perdida=nn.functional.cross_entropy(pred,etiquetas)
                    if fase=='train':
                        optimizador.zero_grad(set_to_none=True)
                        perdida.backward()
                        optimizador.step()
                perdido+=perdida.item()*len(etiquetas)
                aciertos+=(pred.argmax(1)==etiquetas).sum().item()
                total+=len(etiquetas)
            registro[f'{fase}_loss']=perdido/total
            registro[f'{fase}_accuracy']=aciertos/total
        history.append(registro)
        if registro['val_loss']<mejor:
            mejor=registro['val_loss']
            torch.save({k:v.detach().cpu() for k,v in red.state_dict().items()},pesos)
        pd.DataFrame(history).to_csv(salida/'historial.csv',index=False)
        on_epoch(f'MobileNetV3 época {epoca+1}/100 · val_acc {registro["val_accuracy"]:.1%}')
    red.load_state_dict(torch.load(pesos,map_location=device,weights_only=True))
    red.eval()
    reales=[]
    predichos=[]
    with torch.inference_mode():
        for x,y in cargas['test']:
            salida_red=red(x.to(device))
            reales.extend(y.tolist()); predichos.extend(salida_red.argmax(1).cpu().tolist())
    metricas={'modelo':'MobileNetV3-5clases', 'epocas_ejecutadas':100,
             'accuracy_test':float(accuracy_score(reales,predichos)),
             'f1_macro_test':float(f1_score(reales,predichos,average='macro',zero_division=0)),
             'n_prueba':len(reales),'clases':classes,
             'matriz_confusion':confusion_matrix(reales,predichos).tolist()}
    (salida/'metadatos.json').write_text(json.dumps(metricas,indent=2),encoding='utf-8')
    return metricas


def entrenar_yolo(raiz: Path, salida: Path, on_epoch=lambda x: None):
    import torch
    from ultralytics import YOLO
    from torchvision import datasets
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from PIL import Image
    import numpy as np
    import json

    if not torch.cuda.is_available():
        raise RuntimeError('YOLO de 100 épocas requiere GPU CUDA en el ejecutor configurado.')
    salida.mkdir(parents=True,exist_ok=True)
    red=YOLO('yolo11n-cls.pt')
    def progreso_yolo(trainer):
        epoca = int(trainer.epoch) + 1
        on_epoch(f'YOLO11n-cls época {epoca}/100')
    red.add_callback('on_fit_epoch_end', progreso_yolo)
    red.train(data=str(raiz), epochs=100, imgsz=224, batch=16,
              device=0, project=str(salida.parent), name=salida.name,
              exist_ok=True, workers=0, patience=0, save_period=10, verbose=False)
    best=salida/'weights'/'best.pt'
    if not best.is_file(): raise RuntimeError('YOLO no produjo best.pt')
    red=YOLO(str(best))
    if set(red.names.values()) != set(CLASES_CINCO):
        raise RuntimeError('Las clases YOLO entrenadas no coinciden con las cinco esperadas.')
    datos=datasets.ImageFolder(raiz/'test')
    reales=[]; predichos=[]
    nombre_por_id={k:v for k,v in red.names.items()}
    for ruta,y in datos.samples:
        with Image.open(ruta) as im: arr=np.asarray(im.convert('RGB'))
        rs=red.predict(arr,device=0,imgsz=224,verbose=False)[0]
        clase=nombre_por_id[int(rs.probs.top1)]
        reales.append(y)
        predichos.append(datos.class_to_idx[clase])
    m={'modelo':'YOLO11n-cls-5clases','epocas_ejecutadas':100,
       'accuracy_test':float(accuracy_score(reales,predichos)),
       'f1_macro_test':float(f1_score(reales,predichos,average='macro',zero_division=0)),
       'n_prueba':len(reales),'clases':datos.classes,
       'matriz_confusion':confusion_matrix(reales,predichos).tolist()}
    (salida/'metadatos.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    on_epoch('YOLO11n-cls completó las 100 épocas.')
    return m
