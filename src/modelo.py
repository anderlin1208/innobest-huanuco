"""Arquitectura común para entrenamiento y despliegue."""
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps

from src.especies import CLASES


def crear_modelo(preentrenado=False):
    pesos = models.MobileNet_V3_Small_Weights.DEFAULT if preentrenado else None
    modelo = models.mobilenet_v3_small(weights=pesos)
    entradas = modelo.classifier[-1].in_features
    modelo.classifier[-1] = nn.Linear(entradas, len(CLASES))
    return modelo


def transformacion_evaluacion():
    # Compatible con el preprocesamiento ImageNet de MobileNetV3.
    return models.MobileNet_V3_Small_Weights.DEFAULT.transforms()


def transformacion_entrenamiento():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.78, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def cargar_modelo(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo del modelo: {ruta}")
    modelo = crear_modelo(preentrenado=False)
    # Solo cargar archivos producidos por tu propio entrenamiento.
    pesos = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo


def predecir(modelo, imagen: Image.Image):
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    entrada = transformacion_evaluacion()(imagen).unsqueeze(0)
    with torch.inference_mode():
        probabilidades = torch.softmax(modelo(entrada), dim=1)[0].tolist()
    ranking = sorted(zip(CLASES, probabilidades), key=lambda x: x[1], reverse=True)
    return ranking
