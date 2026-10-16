"""Panel de revisión y cola de entrenamiento. Nunca llamar desde una vista sin autenticación."""
from __future__ import annotations
import time
import io
from zipfile import ZipFile

import pandas as pd
import streamlit as st
from pymongo.errors import PyMongoError

from src.auth_admin import verificar_clave
from src.dataset_entrenamiento import validar_dataset_zip
from src.repositorio_feedback import RepositorioFeedback
from src.retroalimentacion import ETIQUETAS_FEEDBACK, NOMBRES, NO_SE
from src.trabajos import RepositorioTrabajos


def mostrar_panel_administrador(repositorio_configurado, configuracion):
    st.subheader('Administración de IA · Revisión y entrenamiento')
    st.caption('Este panel permite aprobar etiquetas, preparar entrenamiento y consultar resultados. '
               'La ejecución de 100 épocas por modelo requiere que el worker de GPU esté conectado.')
    clave_hash = configuracion('INNOBEST_ADMIN_PASSWORD_HASH')
    if not clave_hash:
        st.error('Panel bloqueado: no se configuró contraseña de administrador.')
        st.code('py -3.12 scripts/configurar_admin.py', language='powershell')
        st.caption('Solo se ejecuta una vez en el equipo de desarrollo. Reinicia Streamlit después.')
        return
    if not st.session_state.get('innobest_autorizado_admin', False):
        fallos = st.session_state.get('innobest_fallos_admin', 0)
        hasta = st.session_state.get('innobest_bloqueo_admin', 0.0)
        if time.monotonic() < hasta:
            st.error('Demasiados intentos. Espera un minuto.'); return
        with st.form('inicio_admin'):
            clave=st.text_input('Contraseña de administrador', type='password')
            entrar=st.form_submit_button('Acceder al panel')
        if entrar:
            if verificar_clave(clave,clave_hash):
                st.session_state['innobest_autorizado_admin']=True
                st.session_state['innobest_fallos_admin']=0
                st.rerun()
            else:
                fallos+=1
                st.session_state['innobest_fallos_admin']=fallos
                if fallos>=5: st.session_state['innobest_bloqueo_admin']=time.monotonic()+60
                st.error('Contraseña incorrecta.')
        return

    if st.button('Cerrar sesión de administrador'):
        st.session_state['innobest_autorizado_admin']=False
        st.rerun()
    conexion=repositorio_configurado()
    if not conexion:
        st.warning('Primero configura MONGODB_URI en .env y reinicia la aplicación.');return
    fb=RepositorioFeedback(conexion.bd)
    cola=RepositorioTrabajos(conexion.bd)
    tab_revision, tab_entrenamiento=st.tabs(['Revisar fotografías','Entrenamiento y versiones'])

    with tab_revision:
        st.write('Las etiquetas propuestas por usuarios no son verdaderas hasta su revisión. '
                 'Aprobar requiere imagen autorizada y comprobar su origen y especie.')
        try:
            pendientes=fb.listar_pendientes(30)
            st.metric('Revisiones pendientes',len(pendientes))
            for doc in pendientes:
                doc_id=str(doc['_id'])
                etiqueta=doc.get('etiqueta_usuario',NO_SE)
                with st.expander(f'{doc_id[-8:]} · {NOMBRES.get(etiqueta,etiqueta)}'):
                    detalle=fb.coleccion.find_one({'_id':doc['_id']})
                    tiene_foto=bool(detalle and detalle.get('imagen_gridfs_id') and detalle.get('imagen_guardada'))
                    if tiene_foto:
                        try:
                            st.image(fb.leer_imagen(detalle['imagen_gridfs_id']),width=500)
                        except Exception:
                            st.warning('No se pudo obtener la fotografía de GridFS.')
                            tiene_foto=False
                    else:
                        st.warning('No existe una imagen autorizada. No debe aprobarse para entrenamiento.')
                    with st.form('revision_'+doc_id):
                        clase=st.selectbox('Etiqueta verificada',ETIQUETAS_FEEDBACK,
                                           index=ETIQUETAS_FEEDBACK.index(etiqueta),
                                           format_func=lambda e:NOMBRES[e])
                        confirmacion=st.checkbox('He verificado la fotografía, la especie y el permiso de uso.')
                        aprobar=st.form_submit_button('Aprobar',disabled=not tiene_foto)
                        rechazar=st.form_submit_button('Rechazar')
                    if aprobar or rechazar:
                        if aprobar and not confirmacion:
                            st.error('Primero confirma la revisión visual y los permisos.')
                        else:
                            try:
                                r=fb.revisar(doc_id,aprobar,clase if aprobar else None)
                                st.success('Registro actualizado: '+r['estado_revision'])
                                st.rerun()
                            except (PyMongoError,ValueError,RuntimeError) as exc:
                                st.error('No se pudo revisar ('+type(exc).__name__+').')
            if not pendientes: st.success('No hay revisiones pendientes.')
        except PyMongoError as exc:
            st.error('No se pudieron leer las revisiones ('+type(exc).__name__+').')

    with tab_entrenamiento:
        st.markdown('**Preparar reentrenamiento desde la web**')
        st.info('Las correcciones aprobadas se incorporarán SOLO al entrenamiento. '
                'Necesitas además un ZIP con imágenes verificadas de las CINCO clases '
                'en train/, val/ y test/. No se fabrican datos ni se toca el test anterior.')
        with st.expander('Ver estructura del dataset ZIP'):
            st.code('''dataset.zip\n  train/\n    ara_militaris/\n    pauxi_koepckeae/\n    spizaetus_isidori/\n    otra_ave/\n    no_ave/\n  val/\n    (las mismas cinco carpetas)\n  test/\n    (las mismas cinco carpetas)''')
            st.caption('Mínimos técnicos: 10 train + 3 val + 3 test por clase; '
                       'para uso fiable hacen falta más fotografías, validación experta y test externo.')
        def plantilla_dataset():
            from src.dataset_entrenamiento import CLASES_CINCO
            memoria=io.BytesIO()
            with ZipFile(memoria,'w') as z:
                for split in ('train','val','test'):
                    for clase in CLASES_CINCO:
                        z.writestr(f'{split}/{clase}/','')
            return memoria.getvalue()
        st.download_button('Descargar plantilla de carpetas (ZIP)',plantilla_dataset(),
                           file_name='plantilla_5_clases.zip',mime='application/zip')
        zip_usuario=st.file_uploader('Subir dataset supervisado (ZIP, hasta 250 MB)',type=['zip'],key='zip_entrenamiento')
        autoriza=st.checkbox('He comprobado las cinco clases, las licencias y que el test no se usó para entrenar.')
        if st.button('Solicitar 100 épocas de MobileNetV3 + 100 de YOLO11-cls',type='primary'):
            if not zip_usuario or not autoriza:
                st.error('Debes seleccionar el ZIP y confirmar la revisión del dataset.')
            else:
                try:
                    contenido=zip_usuario.getvalue()
                    resumen=validar_dataset_zip(contenido)
                    identificador=cola.crear(zip_bytes=contenido,resumen=resumen,epocas=100)
                    st.success('Trabajo en cola: '+identificador)
                    st.info('Conecta el worker de Colab T4 para que comience. '
                            'El trabajo no se ejecuta en Render ni modifica modelos activos automáticamente.')
                except (ValueError,PyMongoError) as exc:
                    st.error('No se pudo solicitar el entrenamiento: '+str(exc)[:280])
        st.caption('El administrador solicita el trabajo desde esta web. El cálculo se realiza '
                   'en un worker Python con GPU T4 mientras Colab permanezca conectado.')
        st.divider()
        st.markdown('**Trabajos y resultados**')
        if st.button('Actualizar estados de entrenamiento'):
            st.rerun()
        try:
            for job in cola.recientes():
                st.write(f"**{str(job['_id'])[-8:]}** — {job['estado']} · {job.get('progreso','')}")
                if job.get('metricas'):
                    filas=[]
                    for nombre,m in job['metricas'].items():
                        if isinstance(m,dict) and 'accuracy_test' in m:
                            filas.append({'Modelo':nombre,'Exactitud de prueba':m['accuracy_test'],
                                          'F1 macro':m['f1_macro_test'],'Épocas':m['epocas_ejecutadas']})
                    if filas:st.dataframe(pd.DataFrame(filas),hide_index=True)
                if job.get('estado')=='completado_pendiente_validacion' and job.get('resultado_gridfs_id'):
                    try:
                        resultado=cola.resultados.get(job['resultado_gridfs_id']).read()
                        st.download_button('Descargar modelos entrenados (pendientes de validación)',
                                           resultado, file_name=f"innobest_modelos_{str(job['_id'])[-8:]}.zip",
                                           mime='application/zip',key='res_'+str(job['_id']))
                        with st.expander('Activar esta versión de cinco clases'):
                            st.warning('Antes de activar prueba las tres especies y las dos clases de rechazo con fotos externas. '
                                       'El test del ZIP debe permanecer totalmente independiente del entrenamiento.')
                            confirmar=st.checkbox('He validado la taxonomía, ejemplos externos, cinco clases y métricas.',
                                                 key='conf_'+str(job['_id']))
                            if st.button('Activar versión',disabled=not confirmar,key='activar_'+str(job['_id'])):
                                # Validar que ambos checkpoints realmente pueden cargarse antes de cambiar la referencia.
                                from src.modelos_versionados import cargar_bundle_zip
                                try:
                                    bundle=cargar_bundle_zip(resultado)
                                    bundle.temporal.cleanup()
                                    cola.activar(str(job['_id']))
                                    st.success('Versión activada. La próxima predicción utilizará cinco clases.')
                                    st.rerun()
                                except Exception as error:
                                    st.error('No se pudo activar el modelo: '+type(error).__name__)
                    except Exception:st.warning('No se pudo recuperar el resultado de MongoDB.')
        except PyMongoError as exc:
            st.error('No se pudieron consultar los trabajos ('+type(exc).__name__+').')
        activo=cola.activa()
        if activo:
            st.info('Versión activa: '+str(activo['id_trabajo'])+' (5 clases).')
            if st.button('Volver a los modelos originales de tres especies'):
                cola.volver_originales()
                st.success('Se restauraron los modelos originales.')
                st.rerun()
        else:
            st.caption('Versión activa: modelos originales (3 especies).')
        st.caption('No hay aprendizaje automático por un clic: solo tras revisar ejemplos y '
                   'completar 100 épocas. Los nuevos pesos no se activan hasta ser validados.')
