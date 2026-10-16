"""Entrenamiento de aves amenazadas para Colab GPU T4.

Ejemplo: python entrenar.py --datos /content/innobest_preparado --salida /content/drive/MyDrive/innobest_huanuco_2026/modelos/mobilenet
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.especies import CLASES
from src.modelo import crear_modelo, transformacion_entrenamiento, transformacion_evaluacion

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 2026


class AvesDataset(Dataset):
    def __init__(self, registros, transformar):
        self.registros = registros
        self.transformar = transformar

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, i):
        ruta, clase = self.registros[i]
        with Image.open(ruta) as imagen:
            imagen = imagen.convert("RGB")
            imagen = self.transformar(imagen)
        return imagen, clase


def cargar_particiones(raiz):
    """Lee el MISMO train/val/test usado por YOLO-cls, sin volver a dividir."""
    huellas={}
    particiones={}
    for split in ('train', 'val', 'test'):
        lista=[]
        for idx, clase in enumerate(CLASES):
            carpeta=raiz/split/clase
            if not carpeta.is_dir():
                raise FileNotFoundError(f'Falta {carpeta}. Ejecuta preparar_datos.py primero.')
            archivos=sorted(p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)
            if not archivos: raise ValueError(f'La partición {split}/{clase} está vacía.')
            for archivo in archivos:
                with Image.open(archivo) as im: im.verify()
                huella=hashlib.sha256(archivo.read_bytes()).hexdigest()
                if huella in huellas:
                    raise ValueError(f'Archivo duplicado entre particiones: {archivo} y {huellas[huella]}')
                huellas[huella]=str(archivo)
                lista.append((str(archivo),idx))
        particiones[split]=lista
    return particiones


def paso(modelo, loader, dispositivo, optimizador=None):
    entrenando = optimizador is not None
    modelo.train(entrenando)
    # Evita corromper las medias BatchNorm con conjuntos tan pequeños.
    if entrenando:
        for capa in modelo.modules():
            if isinstance(capa, nn.modules.batchnorm._BatchNorm):
                capa.eval()
    perdida_total = 0.0
    etiquetas, predicciones = [], []
    for imagenes, y in loader:
        imagenes, y = imagenes.to(dispositivo), y.to(dispositivo)
        with torch.set_grad_enabled(entrenando):
            salida = modelo(imagenes)
            perdida = nn.functional.cross_entropy(salida, y)
            if entrenando:
                optimizador.zero_grad(set_to_none=True)
                perdida.backward()
                optimizador.step()
        perdida_total += perdida.item() * len(y)
        etiquetas.extend(y.cpu().tolist())
        predicciones.extend(salida.argmax(dim=1).detach().cpu().tolist())
    return {
        'loss': round(perdida_total / len(loader.dataset), 5),
        'accuracy': round(accuracy_score(etiquetas, predicciones), 5),
    }


def evaluar(modelo, loader, dispositivo):
    modelo.eval()
    verdaderas, predichas = [], []
    with torch.inference_mode():
        for imagenes, y in loader:
            salida = modelo(imagenes.to(dispositivo))
            verdaderas.extend(y.tolist())
            predichas.extend(salida.argmax(dim=1).cpu().tolist())
    return verdaderas, predichas


def entrenar(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Dispositivo:', dispositivo)
    if dispositivo == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    else:
        print('ADVERTENCIA: entrenando sin GPU; comprueba la configuración de Colab.')

    datos, salida = Path(args.datos), Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    particiones = cargar_particiones(datos)
    train, val, test = (particiones[k] for k in ('train', 'val', 'test'))
    for nombre, particion in [('Entrenamiento', train), ('Validación', val), ('Prueba', test)]:
        print(nombre, len(particion), {CLASES[k]: v for k, v in Counter(y for _, y in particion).items()})
    salida.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'archivo': ruta, 'etiqueta': CLASES[y], 'split': split}
                  for split, lista in [('train', train), ('val', val), ('test', test)]
                  for ruta, y in lista]).to_csv(salida / 'particiones.csv', index=False)

    train_loader = DataLoader(AvesDataset(train, transformacion_entrenamiento()), batch_size=args.batch,
                              shuffle=True, num_workers=2, pin_memory=dispositivo == 'cuda')
    val_loader = DataLoader(AvesDataset(val, transformacion_evaluacion()), batch_size=args.batch,
                            num_workers=2, pin_memory=dispositivo == 'cuda')
    test_loader = DataLoader(AvesDataset(test, transformacion_evaluacion()), batch_size=args.batch,
                             num_workers=2, pin_memory=dispositivo == 'cuda')
    modelo = crear_modelo(preentrenado=True).to(dispositivo)
    for p in modelo.features.parameters():
        p.requires_grad = False

    if args.epocas_cabecera < 1 or args.epocas_ajuste < 1:
        raise ValueError('Cada fase debe tener como mínimo una época.')
    if args.epocas_cabecera + args.epocas_ajuste != 100:
        raise ValueError('El requisito del concurso son 100 épocas: la suma de ambas fases debe ser 100.')

    mejor_val = float('inf')
    mejor_epoca_global = None
    historial = []
    ruta_modelo = salida / 'mejor_modelo.pt'
    etapas = [('cabecera', args.epocas_cabecera, 1e-3), ('ajuste_fino', args.epocas_ajuste, 1e-5)]
    print(f'Iniciando {args.epocas_cabecera + args.epocas_ajuste} épocas: '
          f'{args.epocas_cabecera} de cabecera y {args.epocas_ajuste} de ajuste fino.')

    for nombre, max_epocas, tasa in etapas:
        if nombre == 'ajuste_fino':
            # El ajuste fino parte del mejor modelo anterior.
            modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo, weights_only=True))
            for p in modelo.features[-3:].parameters():
                p.requires_grad = True
        optimizador = torch.optim.AdamW((p for p in modelo.parameters() if p.requires_grad), lr=tasa, weight_decay=1e-4)
        for epoca in range(1, max_epocas + 1):
            entrenamiento = paso(modelo, train_loader, dispositivo, optimizador)
            validacion = paso(modelo, val_loader, dispositivo)
            numero_global = len(historial) + 1
            historial.append({'fase': nombre, 'epoca': epoca, 'epoca_global': numero_global,
                              'train_loss': entrenamiento['loss'], 'train_accuracy': entrenamiento['accuracy'],
                              'val_loss': validacion['loss'], 'val_accuracy': validacion['accuracy']})
            print(f"Época {numero_global}/100 [{nombre} {epoca}/{max_epocas}] "
                  f"| val_loss={validacion['loss']:.4f} | val_acc={validacion['accuracy']:.3f}")
            # Guardamos el mejor modelo según la pérdida de validación,
            # pero NO terminamos antes: el requisito es ejecutar 100 épocas completas.
            if validacion['loss'] < mejor_val - 0.0005:
                mejor_val = validacion['loss']
                mejor_epoca_global = numero_global
                torch.save({k: v.detach().cpu() for k, v in modelo.state_dict().items()}, ruta_modelo)
            # Se guarda el historial en Drive tras cada época para reducir pérdidas ante desconexiones.
            pd.DataFrame(historial).to_csv(salida / 'historial.csv', index=False)

    # Pruebas sin utilizar sus resultados para ajustar hiperparámetros.
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo, weights_only=True))
    modelo.eval()
    verdaderas, predichas = evaluar(modelo, test_loader, dispositivo)
    reporte = classification_report(verdaderas, predichas, labels=list(range(len(CLASES))),
                                     target_names=CLASES, output_dict=True, zero_division=0)
    matriz = confusion_matrix(verdaderas, predichas, labels=list(range(len(CLASES))))
    pd.DataFrame(historial).to_csv(salida / 'historial.csv', index=False)
    pd.DataFrame(matriz, index=CLASES, columns=CLASES).to_csv(salida / 'matriz_confusion.csv')
    metricas = {
        'modelo': 'MobileNetV3-Small', 'version': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        'especies': CLASES, 'total_imagenes': len(train)+len(val)+len(test),
        'n_entrenamiento': len(train), 'n_validacion': len(val), 'n_prueba': len(test),
        'accuracy_test': round(accuracy_score(verdaderas, predichas), 5),
        'f1_macro_test': round(f1_score(verdaderas, predichas, average='macro', zero_division=0), 5),
        'reporte_por_clase': reporte, 'gpu': torch.cuda.get_device_name(0) if dispositivo == 'cuda' else None,
        'seed': SEED, 'epocas_solicitadas': 100, 'epocas_ejecutadas': len(historial),
        'epocas_cabecera': args.epocas_cabecera, 'epocas_ajuste_fino': args.epocas_ajuste,
        'mejor_epoca_global': mejor_epoca_global, 'mejor_val_loss': mejor_val,
    }
    (salida / 'metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
    print('ÉPOCAS EJECUTADAS:', len(historial), '/100')
    print('MEJOR ÉPOCA:', mejor_epoca_global)
    print('MODELO GUARDADO:', ruta_modelo)
    print('ACCURACY PRUEBA:', metricas['accuracy_test'], 'F1 MACRO:', metricas['f1_macro_test'])
    print('Nota: un conjunto de prueba pequeño produce estimaciones inciertas.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datos', required=True, help='Raíz preparada: train/val/test con tres subcarpetas por especie')
    parser.add_argument('--salida', required=True, help='Carpeta de resultados y checkpoints persistentes')
    parser.add_argument('--epocas-cabecera', type=int, default=80)
    parser.add_argument('--epocas-ajuste', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    argumentos = parser.parse_args()
    entrenar(argumentos)
