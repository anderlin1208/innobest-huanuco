"""Dashboard INNOBEST Huánuco 2026: inferencia real y estadísticas verificables."""
import io
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from pymongo.errors import PyMongoError
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.especies import CLASES, ESPECIES
from src.modelo import cargar_modelo, predecir
from src.decision import evaluar_decision, evaluar_decision_cinco
from src.filtro_aves import cargar_detector, verificar_ave
from src.mongodb import RepositorioPredicciones
from src.repositorio_feedback import RepositorioFeedback
from src.auth_admin import verificar_clave
from src.trabajos import RepositorioTrabajos
from src.dataset_entrenamiento import validar_dataset_zip
from src.modelos_versionados import cargar_bundle_zip, predecir_mobile_versionado, predecir_yolo_versionado
from src.retroalimentacion import (
    preparar_imagen_feedback, NOMBRES, ETIQUETAS_FEEDBACK, NO_SE
)

st.set_page_config(page_title='INNOBEST | Aves de Huánuco', page_icon='🦜', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container{padding-top:1.7rem;max-width:1300px}
.hero{border:1px solid #275447;border-radius:20px;padding:24px 28px;background:linear-gradient(105deg,#16382d,#0c2522)}
.hero h1{font-size:2.3rem;margin:0 0 10px 0;color:#f0fff5}
.hero p{font-size:1.04rem;color:#b4d7c5;margin:0}
.small{color:#a8cbbd;font-size:.87rem}
</style>''', unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
MODELO_PATH = BASE / 'modelos' / 'mobilenet' / 'mejor_modelo.pt'
METRICAS_PATH = BASE / 'modelos' / 'mobilenet' / 'metricas.json'
YOLO_PATH = BASE / 'modelos' / 'yolo_cls' / 'entrenamiento' / 'weights' / 'best.pt'
YOLO_METRICAS_PATH = BASE / 'modelos' / 'yolo_cls' / 'metricas.json'
DETECTOR_PATH = BASE / 'modelos' / 'detector_general' / 'yolo11n.pt'


load_dotenv(BASE / '.env')


def configuracion(llave: str, predeterminado: str = '') -> str:
    # Las variables del entorno tienen prioridad sobre st.secrets.
    valor = os.environ.get(llave)
    if valor:
        return valor
    try:
        return str(st.secrets.get(llave, predeterminado))
    except Exception:
        return predeterminado


@st.cache_resource(show_spinner=False)
def obtener_repositorio(uri: str, nombre: str):
    return RepositorioPredicciones(uri, nombre)


def repositorio_configurado():
    uri = configuracion('MONGODB_URI').strip()
    if not uri:
        return None
    return obtener_repositorio(uri, configuracion('MONGODB_DB', 'innobest_huanuco'))


def guardar_prediccion(etiqueta, probabilidad, version, concluyente, ranking):
    repo = repositorio_configurado()
    if repo is None:
        return False, 'MongoDB no configurado. Configura MONGODB_URI en .env.'
    try:
        repo.registrar(
            especie=etiqueta,
            confianza=probabilidad,
            modelo=version,
            concluyente=concluyente,
            probabilidades=dict(ranking),
        )
        return True, 'Predicción anónima guardada en MongoDB.'
    except (PyMongoError, ValueError) as error:
        # La excepción de Mongo puede incluir la URI y las credenciales.
        return False, f'Error al guardar ({type(error).__name__}). Revisa la conexión de MongoDB.'


@st.cache_resource(show_spinner='Cargando modelo entrenado...')
def obtener_modelo():
    return cargar_modelo(MODELO_PATH)


@st.cache_resource(show_spinner='Cargando detector general de aves (primera vez requiere internet)...')
def obtener_detector():
    # Descarga automática de pesos COCO por Ultralytics cuando aún no están presentes.
    # No confundir yolo11n.pt de detección con best.pt de clasificación propia.
    ruta = DETECTOR_PATH if DETECTOR_PATH.is_file() else 'yolo11n.pt'
    return cargar_detector(ruta)


@st.cache_resource(show_spinner='Cargando YOLO clasificación...')
def obtener_yolo():
    from ultralytics import YOLO
    return YOLO(str(YOLO_PATH))


def predecir_yolo(imagen):
    import numpy as np
    im = np.asarray(imagen.convert('RGB'))
    resultado = obtener_yolo().predict(source=im, imgsz=224, device='cpu', verbose=False)[0]
    ranking = [(str(resultado.names[i]), float(p)) for i, p in enumerate(resultado.probs.data.cpu().tolist())]
    ranking = [(nombre, prob) for nombre, prob in ranking if nombre in CLASES]
    if len(ranking) != len(CLASES):
        raise ValueError('Las clases de YOLO no corresponden a las tres especies configuradas.')
    return sorted(ranking, key=lambda x: x[1], reverse=True)


@st.cache_resource(show_spinner='Descargando y cargando nuevos modelos desde MongoDB...')
def obtener_bundle_activo(id_trabajo: str, uri: str, nombre_bd: str):
    # La única fuente autorizada es el ZIP de un trabajo generado por nuestro worker GPU.
    rep = RepositorioPredicciones(uri, nombre_bd)
    cola = RepositorioTrabajos(rep.bd)
    job = cola.obtener(id_trabajo)
    if not job or job.get('estado') != 'completado_pendiente_validacion':
        raise ValueError('La versión activada no está disponible.')
    contenido = cola.resultados.get(job['resultado_gridfs_id']).read()
    return cargar_bundle_zip(contenido)


def detectar_version_activa():
    uri=configuracion('MONGODB_URI').strip()
    if not uri: return None
    bd=configuracion('MONGODB_DB','innobest_huanuco')
    rep=RepositorioTrabajos(obtener_repositorio(uri,bd).bd)
    actual=rep.activa()
    if not actual: return None
    identificador=str(actual['id_trabajo'])
    return identificador, obtener_bundle_activo(identificador,uri,bd)


def leer_metricas(ruta):
    if ruta.exists():
        try:
            return json.loads(ruta.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            st.warning(f'No se pudieron leer métricas de {ruta.name}')
    return None


st.markdown('''<div class="hero"><h1>INNOBEST · Aves amenazadas de Huánuco</h1>
<p>Identificación asistida por IA · Monitoreo responsable · Conservación de biodiversidad</p></div>''', unsafe_allow_html=True)
st.caption('Prototipo educativo. Una predicción no equivale a confirmar la presencia de una especie ni su estado de conservación.')

metricas = leer_metricas(METRICAS_PATH)
metricas_yolo = leer_metricas(YOLO_METRICAS_PATH)
modelo_disponible = MODELO_PATH.exists()
yolo_disponible = YOLO_PATH.exists()

st.sidebar.header('Explorar el sistema')
vista = st.sidebar.radio('Sección', ['Identificar un ave', 'Resultados del entrenamiento',
                                    'Especies protegidas', 'Actividad del sistema',
                                    'Administración de IA'])
st.sidebar.divider()
st.sidebar.info('Cámara y correcciones en la web. Entrenamiento solicitado desde la web y ejecutado por GPU Colab T4. Datos: MongoDB.')
version_activa = None
if vista == 'Identificar un ave':
    try:
        version_activa = detectar_version_activa()
        if version_activa:
            st.sidebar.success('Modelos de 5 clases activos: versión ' + version_activa[0][-8:])
        else:
            st.sidebar.caption('Modelos originales de 3 clases activos.')
    except Exception as error:
        st.sidebar.warning('Versión actual no disponible ('+type(error).__name__+'); se probará la original.')
        version_activa = None

if vista == 'Identificar un ave':
    st.subheader('Reconocimiento con control de imágenes ajenas')
    st.caption('La verificación de especie utiliza ambos modelos y un detector general de aves. '
               'Los resultados son provisionales y pueden fallar; no se declara certeza biológica.')
    algoritmo = st.selectbox('Modelo principal que deseas inspeccionar', ['MobileNetV3', 'YOLO-cls'])
    izquierda, derecha = st.columns([1, 1], gap='large')
    with izquierda:
        origen_imagen = st.radio('Entrada de imagen', ['Subir archivo', 'Tomar foto con cámara'], horizontal=True)
        if origen_imagen == 'Tomar foto con cámara':
            archivo = st.camera_input('Activa tu cámara y captura una fotografía', key='camara_aves')
            st.caption('Debes autorizar la cámara en el navegador. Funciona en localhost o HTTPS; clasifica tras capturar la foto.')
        else:
            archivo = st.file_uploader('Sube una fotografía', type=['jpg', 'jpeg', 'png', 'webp'])
        st.caption('Máximo 8 MB. Solo se conservará una copia reducida si autorizas su uso en retroalimentación.')
        foto = None
        if archivo is not None:
            try:
                if archivo.size > 8 * 1024 * 1024:
                    raise ValueError('Para identificar un ave, la fotografía debe pesar como máximo 8 MB.')
                foto = Image.open(io.BytesIO(archivo.getvalue()))
                foto.load()
                st.image(foto, caption='Imagen seleccionada', use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                st.error('El archivo no contiene una imagen válida.')

    with derecha:
        # Inicializar por cada ejecución; nunca reutilizar predicciones de una foto anterior.
        filtro = None
        decision = None
        ranking_m, ranking_y = [], []
        if not (modelo_disponible and yolo_disponible) and not version_activa:
            st.error('Para validar por consenso faltan pesos de uno o ambos clasificadores.')
            if not modelo_disponible:
                st.code(str(MODELO_PATH.relative_to(BASE)))
            if not yolo_disponible:
                st.code(str(YOLO_PATH.relative_to(BASE)))
        elif foto is not None:
            try:
                # Primero se comprueba la presencia de un ave: no invocar clasificadores
                # de tres clases sobre fotos de gatos, tigres u otros animales.
                filtro = verificar_ave(obtener_detector(), foto)
                st.caption('Filtro general de aves: ' + filtro.descripcion)
                if filtro.estado != 'ave_detectada':
                    decision = evaluar_decision(filtro.estado, [], [])
                    st.warning('SIN PREDICCIÓN VÁLIDA — ' + decision.motivo)
                    st.info('No se asigna 0 % a ninguna especie: 0 % sería una probabilidad ficticia.')
                    ranking_m, ranking_y = [], []
                else:
                    with st.spinner('Comparando MobileNetV3 y YOLO11-cls...'):
                        if version_activa:
                            ranking_m = predecir_mobile_versionado(version_activa[1], foto)
                            ranking_y = predecir_yolo_versionado(version_activa[1], foto)
                        else:
                            ranking_m = predecir(obtener_modelo(), foto)
                            ranking_y = predecir_yolo(foto)
                    # Umbrales provisionales hasta calibrar con validación conocida y OOD.
                    decision = (evaluar_decision_cinco(filtro.estado, ranking_m, ranking_y)
                                if version_activa else evaluar_decision(filtro.estado, ranking_m, ranking_y))
                    orden = [('MobileNetV3', ranking_m), ('YOLO-cls', ranking_y)]
                    orden.sort(key=lambda item: item[0] != algoritmo)
                    for nombre, ranking in orden:
                        especie_clave, valor = ranking[0]
                        st.write(f'**{nombre}**: {NOMBRES.get(especie_clave, especie_clave)} — puntuación softmax {valor:.1%}')
                    if decision.especie:
                        especie = ESPECIES[decision.especie]
                        st.success(f'IDENTIFICACIÓN PRELIMINAR: {especie["comun"]}')
                        st.write(f'*{especie["cientifico"]}* · Pendiente de verificación humana')
                    else:
                        st.warning('SIN IDENTIFICACIÓN VÁLIDA — ' + decision.motivo)
                    with st.expander('Ver puntuaciones individuales de los dos modelos'):
                        for nombre, ranking in [('MobileNetV3', ranking_m), ('YOLO-cls', ranking_y)]:
                            st.markdown('**' + nombre + '**')
                            df = pd.DataFrame([{'Especie': NOMBRES.get(c, c), 'Puntuación': v}
                                               for c, v in ranking])
                            st.plotly_chart(px.bar(df, x='Puntuación', y='Especie',
                                                   orientation='h', range_x=[0, 1], text_auto='.1%'),
                                            use_container_width=True, key='chart_' + nombre)
                    st.caption('Los umbrales (85 % y margen 20 puntos) son heurísticos y NO están '
                               'calibrados. Dos modelos pueden coincidir y equivocarse.')
                if st.button('Guardar evaluación en MongoDB', type='primary'):
                    repo = repositorio_configurado()
                    if repo is None:
                        st.warning('MongoDB no está configurado. Revisa MONGODB_URI en .env.')
                    else:
                        try:
                            repo.registrar_evaluacion(decision=decision, mobile=ranking_m,
                                                     yolo=ranking_y, resultado_filtro=filtro)
                            st.success('Evaluación almacenada (sin imagen y sin identificación confirmada).')
                        except (PyMongoError, ValueError):
                            st.error('No se pudo guardar la evaluación; verifica MongoDB.')
            except Exception as exc:
                # Mostramos categoría de error, sin exponer URIs ni datos internos.
                st.error(f'No se pudo verificar la fotografía ({type(exc).__name__}). '
                         'El filtro necesita los pesos yolo11n.pt de detección general; '
                         'se descargan una vez con acceso a internet. No se asignó una especie.')
        else:
            st.info('Sube una foto. Primero verificaremos si hay un ave; luego compararemos ambos modelos.')
    if foto is not None:
        st.divider()
        st.subheader('Enseñar mediante una corrección humana')
        st.caption('**Tiempo real:** la etiqueta se registra en MongoDB y puede consultarse al volver a subir el MISMO archivo. '
                   '**Entrenamiento:** ninguno de los dos modelos cambia sus pesos por un clic; las imágenes '
                   'autorizadas deben revisarse y reentrenarse posteriormente con GPU.')
        try:
            preparada = preparar_imagen_feedback(archivo.getvalue())
            repo_feedback = None
            repo_general = repositorio_configurado()
            if repo_general:
                repo_feedback = RepositorioFeedback(repo_general.bd)
                registro_previo = repo_feedback.reciente(preparada.huella)
                if registro_previo:
                    etiqueta_anterior = NOMBRES.get(registro_previo.get('etiqueta_revisada') or registro_previo['etiqueta_usuario'],
                                                   registro_previo.get('etiqueta_revisada') or registro_previo['etiqueta_usuario'])
                    estado = registro_previo.get('estado_revision', 'pendiente')
                    st.info(f'Etiqueta humana guardada antes para este archivo: **{etiqueta_anterior}** '
                            f'(revisión: **{estado}**). Esta información NO es una predicción nueva de IA.')
            else:
                st.info('Para recordar correcciones configura MongoDB en el archivo .env.')
            with st.form('formulario_correccion_' + preparada.huella[:16]):
                etiqueta_corregida = st.selectbox(
                    '¿Qué aparece realmente en la imagen?',
                    ETIQUETAS_FEEDBACK, format_func=lambda x: NOMBRES[x],
                    index=ETIQUETAS_FEEDBACK.index(NO_SE),
                )
                st.caption('Marca «Otra especie de ave» para loros u otras aves distintas de las tres '
                           'especies del concurso. Marca «No es un ave» para tigres, perros, etc.')
                autoriza_guardar_foto = st.checkbox(
                    'Autorizo conservar una copia reducida y sin metadatos de esta fotografía '
                    'para revisión y futuro entrenamiento; tengo permiso para utilizarla.',
                    value=False,
                )
                envio = st.form_submit_button('Enviar corrección al sistema', type='primary')
            if envio:
                if not repo_feedback:
                    st.error('MongoDB no está configurado; no se guardó ninguna corrección.')
                else:
                    try:
                        identificador = repo_feedback.registrar(
                            imagen=preparada,
                            etiqueta=etiqueta_corregida,
                            autorizacion=autoriza_guardar_foto,
                            filtro=filtro.estado if filtro is not None else None,
                            mobile=ranking_m,
                            yolo=ranking_y,
                            decision=decision.estado if decision is not None else None,
                        )
                        st.success('Corrección guardada en MongoDB, pendiente de revisión. '
                                   'No se han modificado todavía los pesos de MobileNetV3 ni YOLO-cls.')
                        if not autoriza_guardar_foto:
                            st.caption('No se guardó la fotografía: esta corrección no se podrá usar para reentrenar.')
                    except (PyMongoError, ValueError, OSError) as exc:
                        st.error('No se pudo registrar la corrección (' + type(exc).__name__ + ').')
        except ValueError as exc:
            st.error('No se puede preparar la imagen para retroalimentación: ' + str(exc))
    st.warning('El detector general puede no reconocer aves raras. Tampoco demuestra que una ave '
               'pertenezca a una de las tres especies: hace falta un conjunto de pruebas de especies externas.')
    st.warning('No publiques coordenadas precisas de nidos, refugios o poblaciones sensibles.')

elif vista == 'Administración de IA':
    from src.panel_administrador import mostrar_panel_administrador
    mostrar_panel_administrador(repositorio_configurado, configuracion)

elif vista == 'Resultados del entrenamiento':
    st.subheader('Comparación sobre el mismo conjunto de prueba')
    filas=[]
    for nombre, m in [('MobileNetV3', metricas), ('YOLO-cls', metricas_yolo)]:
        if m:
            filas.append({'Modelo':nombre,'Accuracy prueba':m['accuracy_test'],
                          'F1 macro prueba':m['f1_macro_test'],
                          'Épocas':m.get('epocas_ejecutadas', '?'),
                          'Fotos utilizadas':m['total_imagenes'],'Fotos prueba':m['n_prueba']})
    if not filas:
        st.info('No hay métricas reales todavía. Primero entrena ambos modelos en Colab.')
    else:
        st.dataframe(pd.DataFrame(filas),hide_index=True,use_container_width=True)
        st.caption('Mismo conjunto de imágenes de prueba para ambos modelos. Una muestra pequeña NO da un resultado concluyente.')
    seleccion = st.selectbox('Ver detalles', ['MobileNetV3', 'YOLO-cls'])
    if seleccion == 'MobileNetV3':
        m=metricas
        carpeta=BASE/'modelos'/'mobilenet'
    else:
        m=metricas_yolo
        carpeta=BASE/'modelos'/'yolo_cls'
    if m:
        st.info(f"Entrenamiento: {m.get('epocas_ejecutadas','?')}/100 épocas; exactitud prueba {m['accuracy_test']:.1%}; F1 macro {m['f1_macro_test']:.3f}")
        hist=carpeta/'historial.csv'
        if hist.exists():
            df=pd.read_csv(hist)
            if seleccion == 'MobileNetV3':
                st.plotly_chart(px.line(df,x='epoca_global',y=['train_loss','val_loss'],title='Curvas MobileNetV3'),use_container_width=True)
            else:
                st.dataframe(df.tail(12),hide_index=True,use_container_width=True)
        matriz=carpeta/'matriz_confusion.csv'
        if matriz.exists():
            cm=pd.read_csv(matriz,index_col=0)
            st.plotly_chart(px.imshow(cm,text_auto=True,title=f'Matriz de confusión {seleccion}',
                                      labels={'x':'Predicción','y':'Real'}),use_container_width=True)
        st.caption('Revise las imágenes y las etiquetas originales antes de extraer conclusiones científicas.')

elif vista == 'Especies protegidas':
    st.subheader('Especies incluidas en el clasificador')
    for clave in CLASES:
        ave = ESPECIES[clave]
        with st.expander(f"{ave['comun']} — {ave['estado']}", expanded=True):
            st.write(f"**Nombre científico:** *{ave['cientifico']}*")
            st.write(f"**Área de referencia:** {ave['lugar']}")
            st.write(f"**Amenazas:** {ave['amenazas']}")
            st.link_button('Consultar fuente', ave['fuente'])
    st.caption('Los rangos de distribución son orientativos, no observaciones confirmadas por la aplicación.')

else:
    st.subheader('Predicciones guardadas en MongoDB')
    st.caption('Son registros de uso del clasificador; NO son avistamientos ni censos de aves.')
    repo = repositorio_configurado()
    if repo is None:
        st.warning('MongoDB no está configurado. Agrega MONGODB_URI al archivo .env.')
        st.code('MONGODB_DB=innobest_huanuco', language='bash')
    else:
        if st.button('Comprobar conexión MongoDB'):
            try:
                repo.verificar_conexion()
                repo.preparar_indices()
                st.success('MongoDB conectado. Lectura y escritura disponibles si el usuario tiene permisos.')
            except PyMongoError as error:
                st.error(f'No se pudo conectar: {type(error).__name__}. Revisa usuario, contraseña y IP autorizada.')
        try:
            conteos, recientes = repo.obtener_estadisticas()
            filas = [
                {'Modelo': fila['_id'].get('modelo', 'Desconocido'),
                 'Especie': ESPECIES.get(fila['_id'].get('especie'), {}).get('comun', 'No reconocida / sin validación'),
                 'Total': int(fila['total'])}
                for fila in conteos
            ]
            if filas:
                tabla = pd.DataFrame(filas)
                st.metric('Predicciones registradas', int(tabla['Total'].sum()))
                st.plotly_chart(px.bar(tabla, x='Especie', y='Total', color='Modelo', barmode='group'),
                                use_container_width=True)
                st.dataframe(tabla, hide_index=True, use_container_width=True)
                st.subheader('Últimas predicciones')
                historial = pd.DataFrame(recientes)
                if not historial.empty:
                    historial = historial.rename(columns={
                        'fecha_utc': 'Fecha UTC', 'modelo': 'Modelo',
                        'especie_predicha': 'Especie predicha',
                        'confianza': 'Puntuación modelo anterior', 'concluyente': 'Indicador antiguo (no es validación)',
                        'estado_validacion': 'Estado evaluación', 'filtro_ave': 'Filtro',
                    })
                    st.dataframe(historial, hide_index=True, use_container_width=True)
            else:
                st.info('MongoDB funciona, pero todavía no hay predicciones registradas.')
        except PyMongoError as error:
            st.error(f'No fue posible consultar MongoDB ({type(error).__name__}). Revisa la conexión.')
    if repo is not None:
        st.subheader('Aprendizaje con retroalimentación humana')
        try:
            fb = RepositorioFeedback(repo.bd)
            filas_fb = []
            for fila in fb.estadisticas():
                filas_fb.append({'Etiqueta registrada': NOMBRES.get(fila['_id']['etiqueta'], fila['_id']['etiqueta']),
                                 'Estado de revisión': fila['_id']['revision'], 'Cantidad': fila['total']})
            if filas_fb:
                st.dataframe(pd.DataFrame(filas_fb), hide_index=True, use_container_width=True)
                st.caption('Las correcciones guardadas NO equivalen a modelos reentrenados ni a verificaciones biológicas.')
            else:
                st.info('No hay correcciones humanas registradas todavía.')
        except PyMongoError as exc:
            st.warning('No se pudo consultar la retroalimentación (' + type(exc).__name__ + ').')
    st.caption('Las evaluaciones anteriores no guardaron fotografías. Las nuevas correcciones '
               'solo guardan imágenes cuando el usuario lo autoriza expresamente. '
               'No se guardan coordenadas ni identidades personales.')

st.divider()
st.caption('INNOBEST Huánuco 2026 | Proyecto académico de clasificación visual. Verifica taxonomía, licencias y resultados con especialistas.')
