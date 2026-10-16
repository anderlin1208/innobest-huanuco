# INNOBEST Huánuco 2026 — Aplicación Python + MongoDB, versión de validación conservadora

**Empieza por [README_MEJORA.md](README_MEJORA.md)** para instalar, configurar MongoDB y probar el control de fotografías fuera de clase.

Esta entrega preserva dos checkpoints de clasificación entrenados anteriormente, pero **no los reentrena ni garantiza que reconozcan todas las fotos**. Añade un detector de aves genérico y consenso entre clasificadores para reducir predicciones forzadas; el comportamiento debe evaluarse con fotografías nuevas etiquetadas.

Para Windows con Python 3.12, **sin entorno virtual**:

```powershell
py -3.12 -m pip install --user -r requirements.txt
Copy-Item .env.example .env
# Edita .env y configura una cuenta MongoDB de permisos mínimos.
py -3.12 -m unittest discover -s tests -v
py -3.12 -m streamlit run app.py
```

El ZIP corregido **no contiene `.env`**. Si enviaste un archivo `.env` dentro de una copia anterior, rota esas credenciales en MongoDB Atlas. La aplicación funciona sin MongoDB para predecir, pero el registro de evaluaciones requiere credenciales nuevas.
