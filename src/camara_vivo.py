"""Webcam WebRTC: vídeo fluido y análisis asíncrono de fotogramas espaciados.

No modifica Streamlit/session_state desde hilos multimedia; no accede a MongoDB.
Los resultados se superponen directamente en la señal de vídeo.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


TITULOS_ASCII = {
    'ara_militaris': 'Guacamayo militar',
    'pauxi_koepckeae': 'Paujil del Sira',
    'spizaetus_isidori': 'Aguila andina',
}


def estimar_etiqueta(resultado):
    """Estado explícito en vez de clasificar forzosamente otras imágenes."""
    if resultado.get('especie'):
        return 'PRELIMINAR: ' + TITULOS_ASCII.get(resultado['especie'], resultado['especie'])
    estados = {
        'rechazado_no_ave': 'SIN AVE DETECTADA',
        'no_verificable': 'SIN EVIDENCIA SUFICIENTE',
        'modelos_discrepan': 'MODELOS NO COINCIDEN',
        'no_concluyente': 'PREDICCION NO CONCLUYENTE',
        'rechazado_no_objetivo': 'FUERA DE LAS ESPECIES',
    }
    return estados.get(resultado.get('estado'), 'SIN PREDICCION VALIDA')


def analizar_rgb(rgb):
    """Ejecutado exclusivamente en un hilo secundario; no llama a st.*."""
    from src.motor_inferencia import inferir_imagen
    return inferir_imagen(Image.fromarray(rgb), imgsz_detector=416)


class ProcesadorCamara:
    """Mantiene vídeo activo mientras la CPU analiza como máximo un frame a la vez."""

    def __init__(self, intervalo=2.5):
        self.intervalo = max(1.5, float(intervalo))
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='innobest-inferencia')
        self._pending = None
        self._last_started = 0.0
        self._resultado = None
        self._fecha_resultado = 0.0
        self._error_tipo = None
        self._cerrado = False

    def actualizar_intervalo(self, segundos):
        with self._lock:
            self.intervalo = max(1.5, float(segundos))

    def cerrar(self):
        with self._lock:
            if self._cerrado:
                return
            self._cerrado = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        try:
            self.cerrar()
        except Exception:
            pass

    def procesar_muestra(self, rgb, ahora=None):
        """Gestiona el ciclo de trabajo sin esperar nunca al modelo."""
        ahora = time.monotonic() if ahora is None else ahora
        with self._lock:
            if self._pending is not None and self._pending.done():
                try:
                    self._resultado = self._pending.result()
                    self._fecha_resultado = ahora
                    self._error_tipo = None
                except Exception as error:
                    self._resultado = None
                    self._fecha_resultado = 0.0
                    self._error_tipo = type(error).__name__
                self._pending = None

            if (not self._cerrado and self._pending is None
                    and ahora - self._last_started >= self.intervalo):
                self._last_started = ahora
                # El frame se copia antes de pasar al trabajador multimedia.
                self._pending = self._executor.submit(analizar_rgb, rgb.copy())
            # No dejar una especie anticuada sobre el vídeo si se bloqueó un modelo.
            visible = self._resultado
            if visible is not None and ahora - self._fecha_resultado > max(12.0, self.intervalo * 4):
                visible = None
            return visible, self._error_tipo, self._pending is not None

    def __call__(self, frame):
        """Callback de streamlit-webrtc; retorna un frame real anotado."""
        import av
        import cv2
        import numpy as np

        bgr = frame.to_ndarray(format='bgr24')
        alto, ancho = bgr.shape[:2]
        if ancho > 640:
            nuevo = (640, max(1, round(alto * 640 / ancho)))
            bgr = cv2.resize(bgr, nuevo, interpolation=cv2.INTER_AREA)
            alto, ancho = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resultado, error, procesando = self.procesar_muestra(rgb)

        if error:
            titulo = 'ERROR DE INFERENCIA: ' + error[:36]
            detalle = 'Revisar logs. El video sigue activo.'
            color = (50, 50, 230)
        elif resultado:
            titulo = estimar_etiqueta(resultado)
            if resultado.get('especie'):
                m = resultado.get('mobile') or []
                y = resultado.get('yolo') or []
                pm = m[0][1] if m else 0.0
                py = y[0][1] if y else 0.0
                detalle = f'MobileNetV3: {pm:.0%}  |  YOLO-cls: {py:.0%} (softmax)'
                color = (40, 180, 65)
            else:
                detalle = resultado.get('motivo', 'Resultado no concluyente')
                # cv2.putText emplea letras ASCII para evitar glifos sin soporte.
                detalle = detalle.encode('ascii', errors='ignore').decode('ascii')[:80]
                color = (30, 170, 255)
            # Coordenadas corresponden al frame reducido analizado.
            for recuadro in resultado['filtro'].recuadros:
                x1, y1, x2, y2 = recuadro
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        else:
            titulo = 'CAMARA ACTIVA - ANALIZANDO...'
            detalle = 'No se requiere capturar una fotografia.'
            color = (180, 210, 210)

        # Encabezado opaco para mantener legibilidad en fondos variados.
        cv2.rectangle(bgr, (0, 0), (ancho, min(alto, 96)), (13, 30, 23), -1)
        tam = 0.50 if ancho < 470 else 0.65
        cv2.putText(bgr, titulo[:60], (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    tam, color, 2, cv2.LINE_AA)
        cv2.putText(bgr, detalle[:82], (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38 if ancho < 470 else 0.47, (225, 236, 231), 1, cv2.LINE_AA)
        estado = 'ANALIZANDO' if procesando else 'VIDEO EN VIVO'
        cv2.putText(bgr, estado, (12, 78), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (155, 221, 178), 1, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format='bgr24')


def configuracion_rtc():
    """Primero TURN definido por usuario; luego credenciales Cloudflare; luego STUN.

    Cloudflare TURN es gestionado automáticamente por streamlit-webrtc si se
    configuran sus claves de servidor; jamás se exponen esas claves al cliente.
    """
    turn_url = os.getenv('INNOBEST_TURN_URL', '').strip()
    turn_user = os.getenv('INNOBEST_TURN_USERNAME', '').strip()
    turn_pass = os.getenv('INNOBEST_TURN_PASSWORD', '').strip()
    stun = {'urls': ['stun:stun.l.google.com:19302']}
    if turn_url and turn_user and turn_pass:
        return {'iceServers': [stun, {'urls': [turn_url],
                                       'username': turn_user, 'credential': turn_pass}]}
    if os.getenv('CLOUDFLARE_TURN_KEY_ID') and os.getenv('CLOUDFLARE_TURN_KEY_API_TOKEN'):
        return None  # El paquete genera credenciales temporales en el servidor.
    return {'iceServers': [stun]}


def mostrar_camara_vivo():
    import streamlit as st
    st.subheader('Reconocimiento de aves en tiempo real')
    st.caption('Vídeo continuo de tu cámara y predicción automática. No guardamos ni enviamos '
               'fotogramas a MongoDB. La inferencia es en CPU, con modelos reales y sin etiquetas ficticias.')
    st.info('Pulsa START, concede permiso a la cámara y muestra un ave. El vídeo permanece '
            'activo mientras el análisis ocurre cada pocos segundos. '
            'La primera inferencia tarda más porque carga los modelos.')
    intervalo = st.slider('Intervalo mínimo entre análisis (segundos)',
                          min_value=1.5, max_value=6.0, value=2.5, step=0.5)
    try:
        import streamlit_webrtc
    except ImportError:
        st.error('Falta streamlit-webrtc. Instala las dependencias de requirements.txt.')
        return

    procesador = st.session_state.get('innobest_procesador_camara')
    if procesador is None or getattr(procesador, '_cerrado', False):
        procesador = ProcesadorCamara(intervalo)
        st.session_state['innobest_procesador_camara'] = procesador
    else:
        procesador.actualizar_intervalo(intervalo)

    kwargs = {}
    rtc = configuracion_rtc()
    if rtc is not None:
        kwargs['rtc_configuration'] = rtc
    streamlit_webrtc.webrtc_streamer(
        key='innobest-video-vivo',
        video_frame_callback=procesador,
        on_video_ended=procesador.cerrar,
        async_processing=True,
        media_stream_constraints={
            'video': {'width': {'ideal': 640}, 'height': {'ideal': 480},
                      'frameRate': {'ideal': 15, 'max': 20}},
            'audio': False,
        },
        **kwargs,
    )
    st.caption('Las puntuaciones son softmax de cada modelo, no una probabilidad '
               'conjunta ni verificación científica. Para registrar correcciones, '
               'usa «Identificar una fotografía» en el menú.')
    st.caption('En Render la transmisión WebRTC puede requerir un servidor TURN. '
               'Si la cámara no conecta, consulta la guía de despliegue y prueba '
               'el modo de fotografías mientras configuras TURN.')
