"""Trabajos de entrenamiento solicitados desde Streamlit y ejecutados en una GPU externa.

El servidor web NO afirma entrenar sin un worker disponible. Se usa una cola persistente
MongoDB; el estado puede leerse después de reiniciar Streamlit.
"""
from datetime import datetime, timezone
from bson import ObjectId
from gridfs import GridFS
from pymongo import DESCENDING, ReturnDocument


class RepositorioTrabajos:
    def __init__(self, bd):
        self.bd = bd
        self.cola = bd['trabajos_entrenamiento']
        self.datasets = GridFS(bd, collection='datasets_entrenamiento')
        self.resultados = GridFS(bd, collection='modelos_versionados')

    def crear(self, *, zip_bytes: bytes, resumen: dict, epocas: int = 100):
        if epocas != 100:
            raise ValueError('Las 100 épocas son obligatorias para cada uno de los dos modelos.')
        if not zip_bytes or len(zip_bytes) > 250 * 1024 * 1024:
            raise ValueError('El ZIP debe pesar menos de 250 MB.')
        if self.cola.count_documents({'estado': {'$in': ['en_cola', 'entrenando']}}, limit=1):
            raise ValueError('Hay otro entrenamiento pendiente o activo.')
        contenido_id = self.datasets.put(zip_bytes, filename='dataset_aprobado.zip', content_type='application/zip')
        doc = {
            'estado': 'en_cola', 'fecha_utc': datetime.now(timezone.utc),
            'epocas_por_modelo': epocas, 'dataset_gridfs_id': contenido_id,
            'resumen': resumen, 'progreso': 'A la espera de un ejecutor GPU',
            'modelo_activo_actualizado': False,
        }
        try:
            return str(self.cola.insert_one(doc).inserted_id)
        except Exception:
            self.datasets.delete(contenido_id)
            raise

    def recientes(self, limite=12):
        return list(self.cola.find({}, {'dataset_gridfs_id': 0, 'error_privado': 0})
                    .sort('fecha_utc', DESCENDING).limit(limite))

    def obtener(self, id_trabajo):
        if not ObjectId.is_valid(id_trabajo):
            raise ValueError('ID inválido')
        return self.cola.find_one({'_id': ObjectId(id_trabajo)})

    def tomar_siguiente(self):
        return self.cola.find_one_and_update(
            {'estado': 'en_cola'},
            {'$set': {'estado': 'entrenando', 'inicio_utc': datetime.now(timezone.utc),
                      'progreso': 'Worker GPU conectado. Preparando datos.'}},
            sort=[('fecha_utc', 1)], return_document=ReturnDocument.AFTER)

    def progreso(self, id_trabajo, mensaje):
        self.cola.update_one({'_id': id_trabajo, 'estado': 'entrenando'},
                             {'$set': {'progreso': str(mensaje)[:400], 'actualizado_utc': datetime.now(timezone.utc)}})

    def completar(self, id_trabajo, *, artefacto: bytes, metricas: dict):
        archivo_id = self.resultados.put(artefacto, filename='modelos_reentrenados.zip', content_type='application/zip')
        res = self.cola.update_one({'_id': id_trabajo, 'estado': 'entrenando'}, {'$set': {
            'estado': 'completado_pendiente_validacion', 'resultado_gridfs_id': archivo_id,
            'metricas': metricas, 'progreso': 'Modelos entrenados. Revisión necesaria antes de activarlos.',
            'fin_utc': datetime.now(timezone.utc)}})
        if res.modified_count != 1:
            self.resultados.delete(archivo_id)
            raise RuntimeError('Estado cambió durante el entrenamiento.')

    def fallar(self, id_trabajo, detalle):
        self.cola.update_one({'_id': id_trabajo, 'estado': 'entrenando'}, {'$set': {
            'estado': 'fallido', 'progreso': str(detalle)[:350],
            'fin_utc': datetime.now(timezone.utc)}})

    def activa(self):
        return self.bd['configuracion_modelos'].find_one({'_id': 'activa'})

    def activar(self, id_trabajo):
        documento = self.obtener(id_trabajo)
        if not documento or documento['estado'] != 'completado_pendiente_validacion':
            raise ValueError('Solo se pueden activar trabajos completos y revisados.')
        if not documento.get('resultado_gridfs_id') or not documento.get('metricas'):
            raise ValueError('Los pesos o las métricas no están disponibles.')
        self.bd['configuracion_modelos'].update_one({'_id': 'activa'}, {'$set': {
            'id_trabajo': documento['_id'], 'actualizado_utc': datetime.now(timezone.utc),
            'origen': 'aprobacion_administrador_web'}}, upsert=True)
        return documento

    def volver_originales(self):
        self.bd['configuracion_modelos'].delete_one({'_id':'activa'})
