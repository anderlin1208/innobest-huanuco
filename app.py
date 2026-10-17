"""INNOBEST Huánuco — Render Free (512 MB), solo MobileNetV3 ONNX.

Interfaz para fotografías/captura puntual. Sin torch, YOLO ni video WebRTC.
No sustituye la identificación de un especialista ni detecta 'no es ave'.
"""
from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from src.especies import CLASES, ESPECIES
from src.modelo_onnx import RUTA_MODELO, cargar_sesion, interpretar, predecir
from src.retroalimentacion import ETIQUETAS_FEEDBACK, NOMBRES, NO_SE, preparar_imagen_feedback

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')
st.set_page_config(page_title='INNOBEST | Aves de Huánuco', page_icon='🦜',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container{padding-top:1.5rem;max-width:1250px}
.hero{border:1px solid #2a624e;border-radius:18px;padding:24px 26px;
background:linear-gradient(115deg,#174334,#102522)}
.hero h1{color:#f0fff5;margin:0 0 8px 0;font-size:2.1rem}
.hero p{color:#bde3d1;margin:0;font-size:1rem}
</style>''', unsafe_allow_html=True)


def configuracion(nombre: str, predeterminado: str = '') -> str:
    valor = os.getenv(nombre)
    if valor:
        return valor
    try:
        return str(st.secrets.get(nombre, predeterminado))
    except Exception:
        return predeterminado


@st.cache_resource(show_spinner='Preparando MobileNetV3 optimizado...')
def obtener_modelo():
    return cargar_sesion()


@st.cache_resource(show_spinner=False)
def obtener_repositorio(uri: str, bd: str):
    from src.mongodb import RepositorioPredicciones
    return RepositorioPredicciones(uri, bd)


def repositorio():
    uri = configuracion('MONGODB_URI').strip()
    if not uri:
        return None
    return obtener_repositorio(uri, configuracion('MONGODB_DB', 'innobest_huanuco'))


def abrir_imagen(contenido: bytes) -> Image.Image:
    if not contenido or len(contenido) > 8 * 1024 * 1024:
        raise ValueError('El archivo debe ser una fotografía de hasta 8 MB.')
    try:
        with Image.open(io.BytesIO(contenido)) as origen:
            if origen.width * origen.height > 12_000_000:
                raise ValueError('La imagen supera 12 megapíxeles. Reduce su resolución.')
            imagen = ImageOps.exif_transpose(origen).convert('RGB')
            imagen.thumbnail((960, 960), Image.Resampling.LANCZOS)
            return imagen
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError('Archivo de imagen inválido.') from exc


def mostrar_puntuaciones(ranking):
    st.subheader('Puntuaciones del clasificador')
    for clave, prob in ranking:
        st.write(f'**{ESPECIES[clave]["comun"]}** · {prob:.1%}')
        st.progress(min(1.0, max(0.0, float(prob))))
    st.caption('Las puntuaciones softmax no son probabilidades biológicas calibradas. '
               'El modelo solo distingue entre las tres especies entrenadas.')


st.markdown('''<div class="hero"><h1>INNOBEST · Aves amenazadas de Huánuco</h1>
<p>Reconocimiento de fotografías con MobileNetV3 optimizado · Conservación de biodiversidad</p>
</div>''', unsafe_allow_html=True)
st.caption('Prototipo educativo. Cada identificación requiere verificación humana.')

st.sidebar.header('INNOBEST Huánuco')
pagina = st.sidebar.radio('Explorar', [
    'Identificar un ave', 'Especies protegidas', 'Actividad del sistema', 'Acerca del modelo'
])
st.sidebar.divider()
st.sidebar.success('Modo ligero: 1 modelo ONNX')
st.sidebar.caption('Sin YOLO, sin PyTorch y sin transmisión de video WebRTC. '
                   'Se permite tomar una fotografía con la cámara.')

if pagina == 'Identificar un ave':
    st.subheader('Identifica una de las tres especies')
    st.info('El modelo clasifica entre tres especies. No verifica que la imagen contenga un ave: '
            'usa una fotografía clara de un solo ejemplar y revisa el resultado.')
    izquierda, derecha = st.columns([1, 1], gap='large')
    with izquierda:
        modo = st.radio('Origen de la fotografía', ['Subir una imagen', 'Tomar foto con la cámara'],
                        horizontal=True)
        if modo == 'Subir una imagen':
            archivo = st.file_uploader('Selecciona JPG, JPEG, PNG o WEBP',
                                       type=['jpg', 'jpeg', 'png', 'webp'], key='foto_subida')
        else:
            archivo = st.camera_input('Toma una foto para analizar', key='foto_capturada')
        st.caption('Tamaño máximo: 8 MB y 12 megapíxeles. La captura analiza una foto, no video en vivo.')
        foto, contenido, huella = None, None, None
        if archivo is not None:
            try:
                contenido = archivo.getvalue()
                foto = abrir_imagen(contenido)
                huella = hashlib.sha256(contenido).hexdigest()
                vista_previa = foto.copy()
                vista_previa.thumbnail((600, 600))
                st.image(vista_previa, caption='Fotografía seleccionada', use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))
        analizar = st.button('Analizar fotografía', type='primary',
                             disabled=foto is None, use_container_width=True)

    with derecha:
        prediccion = st.session_state.get('resultado_fotografia')
        if prediccion and prediccion['huella'] != huella:
            prediccion = None
        if analizar and foto is not None and huella is not None:
            try:
                if not RUTA_MODELO.is_file():
                    raise FileNotFoundError('Falta el archivo mejor_modelo.onnx en el despliegue.')
                with st.spinner('Analizando la fotografía con MobileNetV3...'):
                    inicio = time.perf_counter()
                    ranking = predecir(obtener_modelo(), foto)
                    tiempo = time.perf_counter() - inicio
                    decision = interpretar(ranking)
                prediccion = {'huella': huella, 'ranking': ranking,
                              'decision': decision, 'tiempo': tiempo}
                st.session_state['resultado_fotografia'] = prediccion
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
                st.error(f'No se pudo ejecutar el modelo ({type(exc).__name__}). '
                         'Consulta los registros de Render para más detalles.')
                prediccion = None
        if prediccion:
            ranking = prediccion['ranking']
            decision = prediccion['decision']
            principal, puntuacion = ranking[0]
            st.write('**Especie sugerida por MobileNetV3:**')
            st.markdown(f'### {ESPECIES[principal]["comun"]}')
            st.caption(f'*{ESPECIES[principal]["cientifico"]}*')
            st.metric('Puntuación softmax', f'{puntuacion:.1%}')
            st.caption(f'Inferencia: {prediccion["tiempo"]:.2f} segundos')
            if decision['especie']:
                st.success('Compatibilidad preliminar: supera los umbrales heurísticos.')
            else:
                st.warning('Identificación no concluyente según los umbrales provisionales.')
            st.write(decision['motivo'])
            mostrar_puntuaciones(ranking)
            st.warning('Una puntuación elevada NO demuestra que sea un ave ni una de estas especies. '
                       'No utilices el resultado como confirmación de un avistamiento.')
            if st.button('Guardar predicción en MongoDB', key='guardar_prediccion'):
                try:
                    rep = repositorio()
                    if rep is None:
                        st.warning('MongoDB no está configurado. La predicción sigue funcionando sin base de datos.')
                    else:
                        rep.registrar(especie=decision['especie'] or 'no_concluyente',
                                     confianza=puntuacion, concluyente=bool(decision['especie']),
                                     probabilidades=dict(ranking))
                        st.success('Predicción anónima guardada en MongoDB.')
                except Exception as exc:
                    st.warning(f'No se pudo guardar en MongoDB ({type(exc).__name__}). '
                               'La predicción sigue disponible.')
        elif foto is not None:
            st.info('Presiona «Analizar fotografía» para obtener la clasificación.')
        else:
            st.info('Sube una fotografía para comenzar.')

    if prediccion and contenido is not None and huella is not None:
        with st.expander('Enviar una corrección humana (opcional)'):
            st.caption('Tu corrección no reentrena el modelo automáticamente. '
                       'Solo se guarda una copia de la imagen si lo autorizas.')
            with st.form('retroalimentacion'):
                etiqueta = st.selectbox('¿Qué muestra realmente esta fotografía?',
                                        ETIQUETAS_FEEDBACK,
                                        index=ETIQUETAS_FEEDBACK.index(NO_SE),
                                        format_func=lambda clave: NOMBRES[clave])
                autorizacion = st.checkbox('Autorizo guardar una copia reducida de esta '
                                           'imagen sin metadatos para revisión y futuro entrenamiento.')
                enviar = st.form_submit_button('Enviar corrección')
            if enviar:
                try:
                    rep = repositorio()
                    if rep is None:
                        st.warning('Configura MongoDB para almacenar correcciones.')
                    else:
                        from src.repositorio_feedback import RepositorioFeedback
                        preparada = preparar_imagen_feedback(contenido)
                        feedback = RepositorioFeedback(rep.bd)
                        feedback.registrar(imagen=preparada, etiqueta=etiqueta,
                                          autorizacion=bool(autorizacion), filtro=None,
                                          mobile=prediccion['ranking'], yolo=[],
                                          decision=prediccion['decision']['estado'])
                        st.success('Corrección registrada para revisión humana.')
                except Exception as exc:
                    st.warning(f'No se pudo guardar la corrección ({type(exc).__name__}).')

elif pagina == 'Especies protegidas':
    st.subheader('Especies incluidas en el reconocimiento')
    for clave in CLASES:
        especie = ESPECIES[clave]
        with st.expander(f'{especie["comun"]} · {especie["estado"]}', expanded=True):
            st.write(f'**Nombre científico:** *{especie["cientifico"]}*')
            st.write(f'**Área de referencia:** {especie["lugar"]}')
            st.write(f'**Amenazas:** {especie["amenazas"]}')
            st.link_button('Fuente informativa', especie['fuente'])
    st.caption('La distribución geográfica no se confirma mediante imágenes subidas al sistema.')

elif pagina == 'Actividad del sistema':
    st.subheader('Historial anónimo (MongoDB Atlas)')
    st.caption('Las predicciones son resultados del programa; no equivalen a avistamientos confirmados.')
    if not configuracion('MONGODB_URI').strip():
        st.info('MongoDB es opcional. Para habilitar el historial, configura MONGODB_URI '
                'y MONGODB_DB en Environment de Render.')
    elif st.button('Consultar historial de predicciones'):
        try:
            rep = repositorio()
            rep.verificar_conexion()
            conteos, recientes = rep.obtener_estadisticas()
            total = sum(int(x['total']) for x in conteos)
            st.metric('Predicciones guardadas', total)
            if conteos:
                st.dataframe([{
                    'Modelo': x['_id'].get('modelo', 'Desconocido'),
                    'Especie': ESPECIES.get(x['_id'].get('especie'), {}).get('comun',
                                                   x['_id'].get('especie', 'No concluyente')),
                    'Cantidad': x['total']
                } for x in conteos], hide_index=True, use_container_width=True)
            if recientes:
                st.subheader('Registros recientes')
                st.dataframe(recientes, hide_index=True, use_container_width=True)
            else:
                st.info('Todavía no existen predicciones guardadas.')
        except Exception as exc:
            st.error(f'MongoDB no respondió ({type(exc).__name__}). '
                     'Puedes seguir clasificando fotografías sin conexión a MongoDB.')

else:
    st.subheader('Motor de inferencia')
    st.write('**Modelo:** MobileNetV3 Small, entrenado para tres especies de aves de Huánuco.')
    st.write('**Formato:** ONNX, ejecutado en CPU mediante ONNX Runtime.')
    st.write('**Preprocesamiento:** Resize 256, recorte central 224 × 224 y normalización ImageNet.')
    st.write('**Salida:** puntuaciones softmax para Guacamayo militar, Paujil del Sira y Águila andina.')
    st.write('**Decisión orientativa:** puntuación ≥ 85 % y diferencia frente a la segunda especie ≥ 20 puntos.')
    st.warning('Los umbrales no están calibrados. No hay detector de aves ni reconocimiento de '
               'especies externas al conjunto de entrenamiento.')
    st.caption('La cámara de video continuo, la comparación con YOLO y el entrenamiento desde '
               'la interfaz se han desactivado para utilizar menos RAM en Render Free.')

st.divider()
st.caption('INNOBEST Huánuco · Versión ligera de clasificación visual · '
           'No compartir ubicaciones precisas de especies amenazadas.')
