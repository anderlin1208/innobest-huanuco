"""Genera UN dataset train/val/test compartido para MobileNetV3 y YOLO-cls.

No toca las fotos originales. Retira copias byte-a-byte y píxel-a-píxel,
y mantiene posibles casi-duplicados dentro de una misma partición.
No verifica la identificación taxonómica de las aves.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from src.especies import CLASES

EXT={'.jpg','.jpeg','.png','.webp'}
SEED=2026


def imagen_verificada(ruta):
    with Image.open(ruta) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        im.load()
        # fingerprint píxeles después de estandarizar orientación, sin reescalado
        exacto=hashlib.sha256(im.size[0].to_bytes(4,'big') + im.size[1].to_bytes(4,'big') + im.tobytes()).hexdigest()
        gr=im.resize((9,8),Image.Resampling.LANCZOS).convert('L')
        px=list(gr.getdata())
        bits=0
        for y in range(8):
            for x in range(8):
                bits=(bits<<1) | (px[y*9+x] > px[y*9+x+1])
        return exacto, bits


def particionar_grupos(grupos):
    """Asignación determinista por clase que nunca divide un grupo visual."""
    rng=random.Random(SEED)
    grupos=[list(g) for g in grupos]
    rng.shuffle(grupos)
    grupos.sort(key=len,reverse=True)  # orden estable para desempatar
    total=sum(map(len,grupos))
    objetivo={'train':total*0.70,'val':total*0.15,'test':total*0.15}
    resultado={'train':[],'val':[],'test':[]}
    # Garantiza grupos para validación y prueba.
    if len(grupos)<3:
        raise ValueError('Hay menos de 3 grupos visuales independientes en una especie; no es posible evaluar.')
    for split,group in zip(['train','val','test'],grupos[:3]): resultado[split].extend(group)
    for group in grupos[3:]:
        split=max(['train','val','test'],key=lambda s:(objetivo[s]-len(resultado[s]),{'train':2,'val':1,'test':0}[s]))
        resultado[split].extend(group)
    return resultado


def ejecutar(args):
    fuente=Path(args.datos).resolve()
    salida=Path(args.destino).resolve()
    reporte=Path(args.reporte).resolve()
    if not fuente.is_dir():raise FileNotFoundError(f'No existe el dataset: {fuente}')
    if salida == fuente or fuente in salida.parents or salida in fuente.parents:
        raise ValueError('La salida debe estar fuera de la carpeta original para evitar mezclar imágenes.')
    if salida.exists():
        if not args.recrear: raise FileExistsError(f'Ya existe {salida}; especifica --recrear solo para la copia local.')
        if not str(salida).startswith('/content/') and not args.permitir_fuera_de_content:
            raise ValueError('Por seguridad, --recrear solo está habilitado dentro de /content/.')
        shutil.rmtree(salida)
    global_sha={}; global_pix={}; manifest=[]; rechazadas=[]
    for clase in CLASES:
        carpeta=fuente/clase
        if not carpeta.is_dir():raise FileNotFoundError(f'Falta carpeta: {carpeta}')
        archivos=sorted(p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXT)
        if not archivos:raise ValueError(f'{clase}: carpeta vacía')
        unicas=[]; hashes=[]
        for ruta in archivos:
            try:
                sh=hashlib.sha256(ruta.read_bytes()).hexdigest()
                pix,dhash=imagen_verificada(ruta)
            except (OSError,UnidentifiedImageError,ValueError) as exc:
                rechazadas.append({'archivo':str(ruta),'clase':clase,'motivo':f'imagen_no_valida: {type(exc).__name__}'})
                continue
            if sh in global_sha:
                original, otra=global_sha[sh]
                if otra!=clase:raise ValueError(f'Imagen exacta en clases distintas: {original} y {ruta}')
                rechazadas.append({'archivo':str(ruta),'clase':clase,'motivo':f'duplicado_exacto: {original}'})
                continue
            if pix in global_pix:
                original, otra=global_pix[pix]
                if otra!=clase:raise ValueError(f'Misma imagen en clases distintas: {original} y {ruta}')
                rechazadas.append({'archivo':str(ruta),'clase':clase,'motivo':f'duplicado_pixeles: {original}'})
                continue
            global_sha[sh]=(str(ruta),clase);global_pix[pix]=(str(ruta),clase)
            unicas.append(ruta);hashes.append(dhash)
        if len(unicas)<6:raise ValueError(f'{clase} tiene menos de 6 fotografías legibles independientes.')
        # Unir solamente vecinos extremadamente similares (dHash <= 2 bits)
        padres=list(range(len(unicas)))
        def raiz(k):
            while padres[k]!=k:
                padres[k]=padres[padres[k]]
                k=padres[k]
            return k
        for i in range(len(unicas)):
            for j in range(i+1,len(unicas)):
                if (hashes[i]^hashes[j]).bit_count()<=2:
                    a,b=raiz(i),raiz(j)
                    if a!=b:padres[b]=a
        grupos={}
        for i,ruta in enumerate(unicas):grupos.setdefault(raiz(i),[]).append(ruta)
        cortes=particionar_grupos(grupos.values())
        for split,paths in cortes.items():
            for ruta in paths:
                dest=salida/split/clase/ruta.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                # prefijo SHA evita sobrescritura por nombres repetidos
                archivo_hash=hashlib.sha256(ruta.read_bytes()).hexdigest()[:10]
                dest=dest.with_name(archivo_hash+'_'+ruta.name)
                shutil.copy2(ruta,dest)
                manifest.append({'archivo_fuente':str(ruta),'clase':clase,'particion':split,'archivo_preparado':str(dest)})
        resumen=Counter(x['particion'] for x in manifest if x['clase']==clase)
        print(f'{clase}: {len(archivos)} originales, {len(unicas)} imágenes únicas por archivo/píxeles, '
              f'{len(grupos)} grupos visuales, train={resumen["train"]}, val={resumen["val"]}, test={resumen["test"]}')
        if len(unicas)<50:
            print(f'ADVERTENCIA: {clase} tiene menos de 50 imágenes únicas; no declarar 50 únicas en el informe.')
    reporte.mkdir(parents=True,exist_ok=True)
    with (reporte/'particiones_compartidas.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['archivo_fuente','clase','particion','archivo_preparado']);w.writeheader();w.writerows(manifest)
    with (reporte/'descartes_duplicados.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['archivo','clase','motivo']);w.writeheader();w.writerows(rechazadas)
    conteos={clase:{sp:sum(x['clase']==clase and x['particion']==sp for x in manifest) for sp in ('train','val','test')} for clase in CLASES}
    (reporte/'resumen_datos.json').write_text(json.dumps({'total_utilizadas':len(manifest),'descartadas':len(rechazadas), 'distribucion':conteos, 'advertencia':'Las etiquetas biológicas y licencias requieren revisión externa.'},indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'PREPARACIÓN FINALIZADA. Muestras únicas utilizadas: {len(manifest)}; duplicados/ilegibles apartados: {len(rechazadas)}')
    print('Los originales NO fueron modificados.');print('Dataset común:',salida)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--datos',required=True)
    p.add_argument('--destino',required=True)
    p.add_argument('--reporte',required=True)
    p.add_argument('--recrear',action='store_true')
    p.add_argument('--permitir-fuera-de-content',action='store_true')
    ejecutar(p.parse_args())
