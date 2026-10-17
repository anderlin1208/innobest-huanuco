"""Almacenamiento de predicciones anónimas en MongoDB Atlas o MongoDB local.

No almacena imágenes, direcciones IP, coordenadas ni identificadores personales.
"""
from datetime import datetime, timezone
from math import isfinite

from pymongo import ASCENDING, DESCENDING, MongoClient

from src.especies import CLASES
from src.dataset_entrenamiento import CLASES_CINCO

MODELOS = {'MobileNetV3', 'YOLO-cls'}


class RepositorioPredicciones:
    def __init__(self, uri: str, nombre_bd: str = 'innobest_huanuco'):
        if not uri or not uri.startswith(('mongodb://', 'mongodb+srv://')):
            raise ValueError('MONGODB_URI debe ser una URI válida de MongoDB.')
        if not nombre_bd or any(ch in nombre_bd for ch in '/\\.\x00"$ '):
            raise ValueError('El nombre de la base de datos no es válido.')
        self.cliente = MongoClient(
            uri,
            serverSelectionTimeoutMS=4500,
            connectTimeoutMS=4500,
            socketTimeoutMS=6000,
            appname='INNOBEST-Huanuco',
        )
        self.bd = self.cliente[nombre_bd]
        self.predicciones = self.bd['predicciones']

    def verificar_conexion(self) -> None:
        self.cliente.admin.command('ping')

    def preparar_indices(self) -> None:
        self.predicciones.create_index([('fecha_utc', DESCENDING)])
        self.predicciones.create_index([('modelo', ASCENDING), ('especie_predicha', ASCENDING)])

    def registrar(self, *, especie: str, confianza: float, modelo: str,
                 concluyente: bool, probabilidades: dict[str, float]):
        if modelo not in MODELOS:
            raise ValueError('Modelo no admitido.')
        if especie not in CLASES and especie != 'no_concluyente':
            raise ValueError('Especie no admitida.')
        confianza = float(confianza)
        if not isfinite(confianza) or not 0.0 <= confianza <= 1.0:
            raise ValueError('La confianza debe estar entre 0 y 1.')
        if set(probabilidades) != set(CLASES):
            raise ValueError('Las probabilidades deben contener las tres especies.')
        probabilidades = {k: float(v) for k, v in probabilidades.items()}
        if any(not isfinite(v) or not 0 <= v <= 1 for v in probabilidades.values()):
            raise ValueError('Probabilidades inválidas.')
        documento = {
            'fecha_utc': datetime.now(timezone.utc),
            'modelo': modelo,
            'especie_predicha': especie,
            'confianza': round(confianza, 5),
            'concluyente': bool(concluyente),
            'probabilidades': probabilidades,
            'origen': 'aplicativo_web',
        }
        resultado = self.predicciones.insert_one(documento)
        return str(resultado.inserted_id)

    def registrar_evaluacion(self, *, decision, mobile, yolo, resultado_filtro):
        """Almacena una evaluación de ambos modelos; nunca la declara identificación confirmada.

        No se guardan fotos, coordenadas ni datos personales.
        """
        if decision.especie is not None and decision.especie not in CLASES:
            raise ValueError('Especie no admitida.')
        autorizados = {'compatible_sin_verificar', 'rechazado_no_ave',
                       'no_verificable', 'no_concluyente', 'modelos_discrepan',
                       'rechazado_no_objetivo'}
        if decision.estado not in autorizados:
            raise ValueError('Estado de validación inválido.')
        por_modelo = {}
        for nombre, ranking in [('MobileNetV3', mobile), ('YOLO-cls', yolo)]:
            if not ranking:
                continue
            probabilidades = dict(ranking)
            if set(probabilidades) not in (set(CLASES), set(CLASES_CINCO)):
                raise ValueError('Clases inválidas.')
            if any(not isfinite(float(v)) or not 0 <= float(v) <= 1 for v in probabilidades.values()):
                raise ValueError('Probabilidades fuera de rango.')
            por_modelo[nombre] = {
                'especie_predicha': ranking[0][0],
                'confianza': round(float(ranking[0][1]), 5),
                'probabilidades': {k: round(float(v), 5) for k, v in probabilidades.items()},
            }
        doc = {
            'fecha_utc': datetime.now(timezone.utc),
            'modelo': 'Comparación MobileNetV3 / YOLO-cls',
            'especie_predicha': decision.especie or 'no_reconocida',
            'confianza': None,  # No existe una probabilidad conjunta calibrada.
            'concluyente': False,  # Nunca constituye verificación biológica.
            'estado_validacion': decision.estado,
            'motivo': decision.motivo,
            'filtro_ave': resultado_filtro.estado,
            'predicciones_por_modelo': por_modelo,
            'origen': 'aplicativo_web',
        }
        return str(self.predicciones.insert_one(doc).inserted_id)

    def obtener_estadisticas(self, limite: int = 20):
        conteos = list(self.predicciones.aggregate([
            {'$group': {'_id': {'modelo': '$modelo', 'especie': '$especie_predicha'},
                         'total': {'$sum': 1}}},
            {'$sort': {'_id.modelo': 1, '_id.especie': 1}},
        ]))
        recientes = list(self.predicciones.find(
            {}, {'_id': 0, 'fecha_utc': 1, 'modelo': 1, 'especie_predicha': 1,
                 'confianza': 1, 'concluyente': 1, 'estado_validacion': 1,
                 'filtro_ave': 1}
        ).sort('fecha_utc', DESCENDING).limit(min(max(int(limite), 1), 100)))
        return conteos, recientes
