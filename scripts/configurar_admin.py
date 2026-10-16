"""Configuración inicial (única vez) del administrador; nunca imprime contraseñas."""
import getpass
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.auth_admin import generar_hash


def main():
    primera = getpass.getpass('Contraseña del administrador (mínimo 12 caracteres): ')
    segunda = getpass.getpass('Repítela: ')
    if primera != segunda:
        raise SystemExit('Las contraseñas no coinciden.')
    huella = generar_hash(primera)
    env = BASE / '.env'
    if not env.exists():
        raise SystemExit('Crea .env y configura MongoDB antes de continuar.')
    lineas = [s for s in env.read_text(encoding='utf-8').splitlines()
              if not s.startswith('INNOBEST_ADMIN_PASSWORD_HASH=')]
    lineas.append('INNOBEST_ADMIN_PASSWORD_HASH=' + huella)
    env.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    print('Administrador configurado. Reinicia Streamlit. No subas .env a Git.')


if __name__ == '__main__':
    main()
