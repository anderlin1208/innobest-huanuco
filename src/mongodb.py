"""Historial anónimo compatible con los registros previos de INNOBEST."""
from datetime import datetime, timezone
from math import isfinite

from pymongo import ASCENDING, DESCENDING, MongoClient
from src.especies import CLASES

MODELO = 'MobileNetV3-ONNX'


class RepositorioPredicciones:
    def __init__(self, uri: str, nombre_bd: str = 'innobest_huanuco'):
        if not uri.startswith(('mongodb://', 'mongodb+srv://')):
            raise ValueError('MONGODB_URI debe ser una URI válida de MongoDB.')
        if not nombre_bd or any(ch in nombre_bd for ch in '/\\.\x00"$ '):
            raise ValueError('Nombre de base de datos inválido.')
        self.cliente = MongoClient(uri, serverSelectionTimeoutMS=3000,
                                   connectTimeoutMS=3000, socketTimeoutMS=4000,
                                   appname='INNOBEST-Huanuco-Ligero')
        self.bd = self.cliente[nombre_bd]
        self.predicciones = self.bd['predicciones']

    def verificar_conexion(self):
        self.cliente.admin.command('ping')

    def preparar_indices(self):
        self.predicciones.create_index([('fecha_utc', DESCENDING)])
        self.predicciones.create_index([('modelo', ASCENDING), ('especie_predicha', ASCENDING)])

    def registrar(self, *, especie, confianza, concluyente, probabilidades):
        if especie not in CLASES and especie != 'no_concluyente':
            raise ValueError('Especie inválida.')
        if set(probabilidades) != set(CLASES):
            raise ValueError('Probabilidades incorrectas.')
        valores = [float(confianza)] + [float(v) for v in probabilidades.values()]
        if not all(isfinite(x) and 0 <= x <= 1 for x in valores):
            raise ValueError('Puntuación fuera de rango.')
        documento = {
            'fecha_utc': datetime.now(timezone.utc),
            'modelo': MODELO,
            'especie_predicha': especie,
            'confianza': round(float(confianza), 5),
            'concluyente': bool(concluyente),
            'estado_validacion': 'compatible_sin_verificar' if concluyente else 'no_concluyente',
            'probabilidades': {k: float(v) for k, v in probabilidades.items()},
            'origen': 'aplicativo_web',
        }
        return str(self.predicciones.insert_one(documento).inserted_id)

    def obtener_estadisticas(self, limite=20):
        conteos = list(self.predicciones.aggregate([
            {'$group': {'_id': {'modelo': '$modelo', 'especie': '$especie_predicha'}, 'total': {'$sum': 1}}},
            {'$sort': {'_id.modelo': 1, '_id.especie': 1}},
        ]))
        recientes = list(self.predicciones.find({}, {
            '_id': 0, 'fecha_utc': 1, 'modelo': 1, 'especie_predicha': 1,
            'confianza': 1, 'concluyente': 1, 'estado_validacion': 1,
        }).sort('fecha_utc', DESCENDING).limit(min(max(int(limite), 1), 100)))
        return conteos, recientes
