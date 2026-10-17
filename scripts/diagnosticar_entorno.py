"""Diagnóstico seguro: no imprime credenciales ni requiere conexión a Atlas."""
from pathlib import Path
import importlib.util
import platform

BASE = Path(__file__).resolve().parent.parent


def main():
    print('INNOBEST - verificación de despliegue')
    print('Python:', platform.python_version())
    paquetes = ['streamlit', 'streamlit_webrtc', 'av', 'cv2', 'torch',
                'torchvision', 'ultralytics', 'pymongo', 'plotly']
    ausentes = []
    for nombre in paquetes:
        presente = importlib.util.find_spec(nombre) is not None
        print(f'{nombre}: {"OK" if presente else "FALTA"}')
        if not presente:
            ausentes.append(nombre)
    for ruta in [
        'modelos/mobilenet/mejor_modelo.pt',
        'modelos/yolo_cls/entrenamiento/weights/best.pt',
        'yolo11n.pt',
    ]:
        archivo = BASE / ruta
        ok = archivo.is_file() and archivo.stat().st_size > 10000
        print(f'{ruta}: {"OK" if ok else "FALTA"}')
        if not ok:
            ausentes.append(ruta)
    if ausentes:
        print('RESULTADO: Faltan dependencias o modelos: ' + ', '.join(ausentes))
        return 1
    print('RESULTADO: Archivos y módulos detectados. Esto NO verifica cámara, MongoDB ni latencia.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
