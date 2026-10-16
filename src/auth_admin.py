"""Autenticación de administrador con PBKDF2; sin acceso anónimo a revisiones."""
import base64
import hashlib
import hmac
import os

ITERACIONES = 310_000


def generar_hash(clave: str, salt: bytes | None = None) -> str:
    if len(clave) < 12:
        raise ValueError('Utiliza al menos 12 caracteres para el administrador.')
    salt = salt or os.urandom(16)
    resultado = hashlib.pbkdf2_hmac('sha256', clave.encode(), salt, ITERACIONES)
    return 'pbkdf2_sha256$%d$%s$%s' % (
        ITERACIONES,
        base64.urlsafe_b64encode(salt).decode('ascii'),
        base64.urlsafe_b64encode(resultado).decode('ascii'),
    )


def verificar_clave(clave: str, representacion: str) -> bool:
    try:
        metodo, n, salt_b64, esperado_b64 = representacion.split('$')
        n = int(n)
        if metodo != 'pbkdf2_sha256' or not 200_000 <= n <= 600_000:
            return False
        salt = base64.urlsafe_b64decode(salt_b64)
        esperado = base64.urlsafe_b64decode(esperado_b64)
        if len(salt) != 16 or len(esperado) != 32:
            return False
        actual = hashlib.pbkdf2_hmac('sha256', clave.encode(), salt, n)
        return hmac.compare_digest(actual, esperado)
    except (ValueError, TypeError, UnicodeError):
        return False
