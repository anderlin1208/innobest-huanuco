# INNOBEST Huánuco — Versión ligera para Render Free (512 MB)

Esta es una versión **separada y simplificada** de INNOBEST. Incluye el MobileNetV3 Small entrenado en el proyecto original, convertido a ONNX. **No instala PyTorch, Torchvision, Ultralytics, OpenCV, Plotly ni WebRTC** en Render.

## Funciones incluidas

- Subir fotografías JPG/JPEG/PNG/WEBP y clasificar una de las **tres especies**: Guacamayo militar, Paujil del Sira y Águila andina.
- Tomar **una fotografía puntual** con `st.camera_input` (no video en vivo).
- Ver puntuaciones individuales, sugerencia principal y advertencia cuando el umbral no se supera.
- Información de conservación sobre las especies.
- Guardar resultados anónimos e ingresar correcciones humanas **si se configura MongoDB Atlas**. La base de datos es opcional para clasificar.

## Limitaciones científicas

- Ya no incorpora el detector YOLO11n, el consenso MobileNetV3 / YOLO-cls, el entrenamiento desde la página ni la cámara en vivo WebRTC.
- Al ser un clasificador cerrado de tres clases, **no determina si una fotografía contiene un ave**. Una foto de un perro puede recibir una puntuación alta para una de las tres especies. El umbral 85 % y margen de 20 puntos son heurísticos, no calibrados. La sugerencia es provisional.
- Los datos de evaluación de la versión original no se incluyen: no se inventan métricas de exactitud ni F1.
- ONNX y la retirada de modelos reducen el uso de memoria, pero solo la monitorización de Render puede confirmar que permanece bajo 512 MB en sus condiciones reales.

## Prueba local (Windows PowerShell)

En una **carpeta nueva** (no sobrescribas el proyecto original):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Visita `http://localhost:8501`, selecciona una foto y presiona **Analizar fotografía**. No se necesita MongoDB para clasificar.

## Publicar en Render Free

1. Descomprime este ZIP y sube **el contenido de su carpeta** a la raíz de un repositorio GitHub nuevo, o a una rama de prueba. **No elimines el proyecto original hasta comprobar el funcionamiento.** Verifica que `modelos/mobilenet/mejor_modelo.onnx` esté presente en GitHub.
2. En tu servicio Render Free, configura el repositorio y rama que contienen esta versión.
3. Confirma que el entorno sea **Python** y `Python 3.12.10` mediante `.python-version`.
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
6. **Environment**, opcional: `MONGODB_URI`, `MONGODB_DB=innobest_huanuco`, `INNOBEST_CPU_THREADS=1`. Las credenciales nunca deben añadirse a GitHub.
7. Ejecuta el despliegue, espera el mensaje **Live** y prueba fotografías de las tres especies.
8. Si hay reinicios, examina **Render → Events / Logs**. No afirmes que una app es estable sin estas pruebas. El plan Free puede entrar en suspensión por inactividad y presentar inicio lento.

**Aviso sobre repositorios existentes:** el ZIP es una distribución mínima. Si deseas reemplazar los archivos de la rama `main` de tu repositorio actual, asegúrate de eliminar de esa rama las dependencias pesadas anteriores y configurar Build/Start según los comandos anteriores. Preferimos una rama nueva para poder volver atrás.

## Archivos desplegados

```text
app.py
requirements.txt
.python-version
.streamlit/config.toml
modelos/mobilenet/mejor_modelo.onnx
src/__init__.py
src/especies.py
src/modelo_onnx.py
src/mongodb.py
src/retroalimentacion.py
src/repositorio_feedback.py
```

Los pesos originales `.pt` y los notebooks se conservan en tu proyecto original para entrenar y comparar, **pero no se suben al despliegue ligero**.

## Comprobación de memoria

La página solo carga ONNX Runtime al pulsar **Analizar fotografía**. MongoDB solo se conecta al consultar/guardar información. La primera inferencia incluye la preparación de la sesión; las siguientes reutilizan el modelo mediante `st.cache_resource`.
