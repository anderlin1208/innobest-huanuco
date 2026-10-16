"""Colección de etiquetas humanas y archivos opt-in en MongoDB GridFS.

El historial es append-only: nunca transforma una corrección en verdad biológica.
Una fotografía sin consentimiento no se conserva, por lo que no se exporta para entrenamiento.
"""
from __future__ import annotations

from datetime import datetime, timezone
from bson import ObjectId
from gridfs import GridFS
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from src.retroalimentacion import (ImagenFeedback, validar_etiqueta,
                                   es_apta_para_reentrenamiento, NO_SE)


class RepositorioFeedback:
    def __init__(self, bd):
        self.bd = bd
        self.coleccion = bd['retroalimentaciones']
        self.archivos = GridFS(bd, collection='fotografias_feedback')

    def preparar_indices(self):
        self.coleccion.create_index([('huella_sha256', 1), ('fecha_utc', DESCENDING)])
        self.coleccion.create_index([('estado_revision', 1), ('fecha_utc', DESCENDING)])

    def reciente(self, huella: str):
        """Solo la última corrección al mismo archivo; no afecta los pesos de IA."""
        campos = {'_id': 0, 'etiqueta_usuario': 1, 'etiqueta_revisada': 1,
                  'estado_revision': 1, 'fecha_utc': 1, 'autorizacion_entrenamiento': 1}
        # Una etiqueta aprobada tiene prioridad sobre una pendiente más reciente.
        aprobada = self.coleccion.find_one(
            {'huella_sha256': huella, 'estado_revision': 'aprobada'},
            campos, sort=[('fecha_revision_utc', DESCENDING)])
        return aprobada or self.coleccion.find_one(
            {'huella_sha256': huella}, campos, sort=[('fecha_utc', DESCENDING)])

    def registrar(self, *, imagen: ImagenFeedback, etiqueta: str,
                  autorizacion: bool, filtro: str | None,
                  mobile: list, yolo: list, decision: str | None) -> str:
        validar_etiqueta(etiqueta)
        if not isinstance(autorizacion, bool):
            raise ValueError('La autorización debe ser explícita.')
        if len(imagen.huella) != 64:
            raise ValueError('Huella digital inválida.')
        doc = {
            'fecha_utc': datetime.now(timezone.utc),
            'huella_sha256': imagen.huella,
            'etiqueta_usuario': etiqueta,
            'estado_revision': 'pendiente',
            'autorizacion_entrenamiento': autorizacion and etiqueta != NO_SE,
            'imagen_guardada': False,
            'filtro_ave': filtro,
            'decision_previa': decision,
            'predicciones_previas': {},
            'origen': 'correccion_usuario_web',
        }
        for nombre, ranking in [('MobileNetV3', mobile), ('YOLO11-cls', yolo)]:
            if ranking:
                doc['predicciones_previas'][nombre] = {
                    'especie': str(ranking[0][0]), 'puntuacion': round(float(ranking[0][1]), 5)
                }
        id_gridfs = None
        if doc['autorizacion_entrenamiento']:
            id_gridfs = self.archivos.put(imagen.jpeg_sin_metadatos, content_type='image/jpeg',
                                         filename='feedback.jpg')
            doc['imagen_gridfs_id'] = id_gridfs
            doc['imagen_guardada'] = True
        try:
            resultado = self.coleccion.insert_one(doc)
        except Exception:
            if id_gridfs is not None:
                self.archivos.delete(id_gridfs)
            raise
        return str(resultado.inserted_id)

    def estadisticas(self):
        return list(self.coleccion.aggregate([
            {'$group': {'_id': {'etiqueta': '$etiqueta_usuario', 'revision': '$estado_revision'},
                        'total': {'$sum': 1}}},
            {'$sort': {'_id.etiqueta': 1}},
        ]))

    def listar_pendientes(self, limite=50):
        return list(self.coleccion.find(
            {'estado_revision': 'pendiente'},
            {'_id': 1, 'huella_sha256': 1, 'etiqueta_usuario': 1,
             'autorizacion_entrenamiento': 1, 'imagen_guardada': 1, 'fecha_utc': 1}
        ).sort('fecha_utc', DESCENDING).limit(limite))

    def revisar(self, id_doc: str, aprobar: bool, etiqueta_correcta: str | None = None):
        """Solo invocar desde CLI confiable o panel autenticado de administrador."""
        if not ObjectId.is_valid(id_doc):
            raise ValueError('ID de retroalimentación inválido.')
        documento = self.coleccion.find_one({'_id': ObjectId(id_doc)})
        if not documento:
            raise ValueError('No existe esa retroalimentación.')
        if documento['estado_revision'] != 'pendiente':
            raise ValueError('Esta retroalimentación ya fue revisada.')
        etiqueta = validar_etiqueta(etiqueta_correcta or documento['etiqueta_usuario'])
        if aprobar and not documento.get('imagen_guardada'):
            raise ValueError('No se puede aprobar para entrenamiento sin imagen autorizada.')
        cambios = {'estado_revision': 'aprobada' if aprobar else 'rechazada',
                   'fecha_revision_utc': datetime.now(timezone.utc),
                   'etiqueta_revisada': etiqueta if aprobar else None,
                   'apta_entrenamiento': es_apta_para_reentrenamiento(
                       etiqueta, documento.get('autorizacion_entrenamiento', False),
                       'aprobada' if aprobar else 'rechazada')}
        resultado = self.coleccion.update_one({'_id': documento['_id'],
                                             'estado_revision': 'pendiente'}, {'$set': cambios})
        if resultado.modified_count != 1:
            raise RuntimeError('Conflicto al revisar la etiqueta.')
        return cambios

    def aprobadas_exportables(self):
        return self.coleccion.find({'estado_revision': 'aprobada',
                                    'apta_entrenamiento': True,
                                    'imagen_guardada': True})

    def leer_imagen(self, id_gridfs):
        return self.archivos.get(id_gridfs).read()
