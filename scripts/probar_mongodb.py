"""Prueba no destructiva: ping, escritura, lectura y eliminación de un documento temporal."""
import os
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / '.env')


def main():
    uri = os.environ.get('MONGODB_URI', '').strip()
    nombre = os.environ.get('MONGODB_DB', 'innobest_huanuco').strip()
    if not uri:
        print('ERROR: configura MONGODB_URI en el archivo .env (no publiques la contraseña).')
        return 1
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=6000,
                             connectTimeoutMS=6000, appname='INNOBEST-Prueba')
        client.admin.command('ping')
        print('1/4 Conexión MongoDB: OK')
        coleccion = client[nombre]['_pruebas_conexion']
        identificador = str(uuid4())
        resultado = coleccion.insert_one({'id_prueba': identificador, 'temporal': True})
        print('2/4 Escritura: OK')
        try:
            assert coleccion.find_one({'_id': resultado.inserted_id}) is not None
            print('3/4 Lectura: OK')
        finally:
            coleccion.delete_one({'_id': resultado.inserted_id})
        print('4/4 Eliminación del documento temporal: OK')
        print(f'LISTO: MongoDB funciona. Base de datos: {nombre}')
        return 0
    except (PyMongoError, AssertionError, ValueError) as error:
        # Evitamos imprimir la URI: puede contener usuario y contraseña.
        print(f'ERROR en MongoDB: {type(error).__name__}')
        print('Revisa credenciales, usuario de BD, IP autorizada y conexión a Internet.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
