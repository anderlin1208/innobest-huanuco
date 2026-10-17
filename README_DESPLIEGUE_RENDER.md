# INNOBEST Huánuco — instalación, cámara en vivo y despliegue en Render

**Versión entregada:** 2026-10-16 · **Aplicación:** `app.py` · **Python:** 3.12.10

El proyecto contiene **los checkpoints originales reales** de MobileNetV3 (`mejor_modelo.pt`),
YOLO11-cls (`best.pt`) y YOLO11n COCO (`yolo11n.pt`). No incluye claves de Atlas ni contraseñas.

> Importante: esta versión está preparada para el despliegue, pero NO se ha validado
> con una webcam conectada ni en una instancia real de Render. **No se puede prometer**
> cero latencia, precisión perfecta ni compatibilidad WebRTC sin probar el entorno.

## 1. Qué cambia respecto del ZIP original

| Antes | Ahora |
|---|---|
| `st.camera_input()` exige tomar una foto | Se añadió **Cámara en vivo** con WebRTC y rotulación sobre el vídeo |
| Inferencia por foto se repite al cambiar casillas | Se reutiliza el resultado por huella SHA-256 + versión del modelo |
| Se consulta MongoDB al entrar a Identificar | Consultar versión remota y correcciones previas es **opcional** |
| Error TLS interrumpe formulario | La lectura opcional de correcciones captura errores de PyMongo |
| Modelos podrían crearse de nuevo por modalidad | Originales compartidos mediante caché y bloqueo de inferencia |
| Los tres modelos podrían analizar cada frame | Solo un fotograma cada 1.5–6 s; vídeo sigue activo entre análisis |
| `.gitignore` podía omitir pesos requeridos | Checkpoints y detector permitidos expresamente al publicar en Git |

**Cámara en vivo** emplea **siempre los modelos originales de 3 clases**.
La **clasificación de fotos** permite consultar manualmente la versión de 5 clases
activada desde MongoDB. No se promete que la última versión esté calibrada.

## 2. Abrir en Windows con Visual Studio Code (primero)

1. Descomprime este ZIP y abre la carpeta `innobest_web_mongodb_tiempo_real` en VS Code.
2. Abre **Terminal → New Terminal** (PowerShell). Comprueba Python 3.12:

   ```powershell
   py -3.12 --version
   ```

   Si no está instalado, instala **Python 3.12 x64** y reinicia VS Code.

3. Crea un entorno virtual para no mezclar Python 3.13 con Python 3.12:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   La instalación de PyTorch, PyAV y Ultralytics puede descargar bastantes MB.

4. Ejecuta las comprobaciones (desde la raíz del proyecto):

   ```powershell
   .\.venv\Scripts\python.exe -m pip check
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   .\.venv\Scripts\python.exe scripts/diagnosticar_entorno.py
   ```

5. Ejecuta:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
   ```

6. Abre `http://localhost:8501`. Selecciona **Cámara en vivo**, pulsa **START**
   y acepta el permiso de cámara. Debes ver vídeo continuo y un rótulo que se
   actualiza automáticamente; no hay botón de captura.
7. En **Identificar un ave** puedes subir imágenes para análisis detallado y enviar
   correcciones. Ahí continúa disponible la captura de foto tradicional.

**Si Chrome o Edge muestran `removeChild`**, desactiva la traducción automática
para el sitio y recarga con `Ctrl+Shift+R`. Comprueba también permisos de cámara
(del navegador y Configuración de Windows > Privacidad > Cámara).

## 3. Configurar MongoDB Atlas (opcional para clasificar)

**Puedes probar la webcam y las predicciones SIN MongoDB.** Solo registro,
retroalimentación, estadísticas y entrenamiento requieren Atlas.

1. Copia `.env.example` como `.env` y reemplaza los valores con la URI de Atlas
   en tu computadora. NO adjuntes este archivo ni lo subas a GitHub.
2. En Atlas comprueba el clúster activo, usuario de base de datos y **Network Access**.
   Agrega la IP pública de tu computadora (y la salida de Render cuando corresponda).
3. Ejecuta:

   ```powershell
   .\.venv\Scripts\python.exe scripts/probar_mongodb.py
   ```

4. Si aparece `ServerSelectionTimeoutError` con `TLSV1_ALERT_INTERNAL_ERROR`,
   verifica IP autorizada, red/firewall y TLS del sistema; prueba otra red.
   En PowerShell:

   ```powershell
   Test-NetConnection ac-hdi4zkj-shard-00-00.h7kclta.mongodb.net -Port 27017
   ```

   El servidor del comando es el que aparecía en tu error: puede cambiar si
   Atlas recrea el clúster. **No uses `tlsAllowInvalidCertificates=true`, `tlsInsecure=true`
   ni desactives TLS**. No uses `0.0.0.0/0` permanentemente sin controles adicionales.
5. Para configurar el administrador, utiliza `scripts/configurar_admin.py` y guarda
   el hash que genera en un entorno privado. No subas las credenciales al repositorio.

## 4. Medir rendimiento de verdad

Ejecuta con una foto real de un ave:

```powershell
.\.venv\Scripts\python.exe scripts/medir_inferencia.py "C:\ruta\foto_ave.jpg" --repeticiones 3
```

Se imprimen latencia total en ms, memoria residente (MiB) y resultado. La primera
pasada incluye carga de pesos; las siguientes miden el coste con modelos en memoria.
**Esta prueba no mide exactitud científica**, que requiere un conjunto etiquetado
independiente de entrenamiento y validación.

Si la CPU no puede procesar en el intervalo elegido, la webcam seguirá mostrando
el resultado anterior hasta terminar el siguiente análisis. Se evalúa como máximo
un frame a la vez; el sistema no afirma procesar 30 inferencias por segundo.

## 5. Subir cambios al repositorio GitHub existente

Repositorio del proyecto: `https://github.com/anderlin1208/innobest-huanuco`

1. Conserva una copia de seguridad de tu carpeta anterior.
2. Copia los archivos de este ZIP **dentro de tu carpeta de proyecto que ya contiene
   el directorio `.git`**. No borres `.git` ni sobrescribas tu `.env` local.
3. En VS Code abre la terminal en esa carpeta y ejecuta:

   ```powershell
   git status
   git add -A
   git add -f modelos/mobilenet/mejor_modelo.pt modelos/yolo_cls/entrenamiento/weights/best.pt yolo11n.pt
   git status
   git commit -m "Implementar camara WebRTC y optimizar inferencia de INNOBEST"
   git push origin main
   ```

4. En GitHub comprueba que `app.py`, `src/camara_vivo.py`, `src/motor_inferencia.py`,
   `requirements.txt`, `.python-version` y los tres `.pt` aparecen en `main`.
   **Revisa que `.env` y `.streamlit/secrets.toml` NO se publiquen.**

Si creaste un repositorio nuevo sin `.git`, no copies comandos a ciegas; primero
conecta ese repositorio con el remoto original.

## 6. Configurar el servicio Render existente

Entra en **Dashboard → innobest-huanuco** (no crees otro servicio si ya existe).

- **Runtime:** Python 3.
- **Branch:** `main`.
- **Root Directory:** vacío.
- **Build Command:** `pip install -r requirements.txt`.
- **Start Command:**

  ```bash
  python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
  ```

- **Variables de entorno** (Render → Environment):

  | Key | Value | Obligatoria |
  |---|---|---|
  | `PYTHON_VERSION` | `3.12.10` | Sí, coincide con `.python-version` |
  | `INNOBEST_CPU_THREADS` | `1` para 1 CPU, o `2` para 2+ CPU | Recomendada |
  | `MONGODB_URI` | URI privada de Atlas | Solo para guardar/administrar |
  | `MONGODB_DB` | `innobest_huanuco` | Solo con Atlas |
  | `INNOBEST_ADMIN_PASSWORD_HASH` | Hash generado, nunca contraseña en texto claro | Solo para panel admin |
  | `CLOUDFLARE_TURN_KEY_ID` | Key ID privado de Cloudflare Realtime TURN | Recomendado para WebRTC remoto |
  | `CLOUDFLARE_TURN_KEY_API_TOKEN` | Token privado de Cloudflare Realtime TURN | Recomendado para WebRTC remoto |

**Memoria/CPU:** Free (512 MB / 0.1 CPU) y el plan de pago de 512 MB NO
son apropiados para garantizar inferencia estable de tres modelos PyTorch.
En Render, **1c-2g (1 CPU, 2 GB RAM)** es un punto de partida para hacer pruebas,
no una garantía de fluidez; ese plan **es de pago**. Verifica precio en tu cuenta
antes de cambiarlo. Prueba latencia, memoria y cámara una vez desplegado.

El nivel gratuito se suspende tras periodos de inactividad; al reanudarse recarga
Python, librerías y modelos. Evitar una espera inicial requiere un servicio sin
suspensión y recursos suficientes. No podemos eliminar toda demora solo cambiando
código.

5. Usa **Manual Deploy → Deploy latest commit** o espera el autodeploy tras el push.
6. Inspecciona **Logs** por `ModuleNotFoundError`, `Out of memory`, errores de
   carga `.pt`, `ServerSelectionTimeoutError` y errores de conexión WebRTC.
7. Abre tu sitio HTTPS, concede permiso a la cámara y prueba desde **dos redes
   distintas** (por ejemplo laptop WiFi y teléfono con datos móviles).
8. En la página de fotos verifica imágenes de especies objetivo, no aves y aves
   de otras especies. Si el filtro rechaza indebidamente una ave rara, toma nota:
   **no elimines el filtro sin validación OOD**.

## 7. Webcam remota y TURN: punto crítico

`streamlit-webrtc` usa WebRTC. Para acceder a la webcam desde Render necesitas
**HTTPS**, negociación ICE y conectividad STUN/TURN; Render solo expone el puerto
web HTTP y no garantiza el acceso UDP directo de WebRTC desde cualquier red.

La configuración incluida usa STUN público cuando no hay credenciales TURN.
Si en Render aparece START pero el vídeo nunca se conecta, habilita un servicio
TURN legítimo en Cloudflare y agrega **las dos variables CLOUDFLARE_TURN_***
en Render. La biblioteca genera credenciales TURN de corta duración. No
expongas las claves en `st.secrets` públicos ni en GitHub. También se admite
`INNOBEST_TURN_URL`, `INNOBEST_TURN_USERNAME` y `INNOBEST_TURN_PASSWORD` para
un servidor TURN propio (usar URL `turns:` sobre TLS cuando el proveedor la soporte).

No podemos certificar conectividad desde todas las redes sin esas pruebas reales.
Si aun con TURN no funciona en Render, habrá que utilizar alojamiento que soporte
correctamente WebRTC o replantear el transporte de vídeo, en vez de prometer que
cambiar una opción de Streamlit resolverá toda la infraestructura.

**Documentación:** https://whitphx.github.io/streamlit-webrtc/deployment/

## 8. Si se producen errores

| Síntoma | Comprobación |
|---|---|
| `No module named streamlit_webrtc` | Reinstalar `requirements.txt` en el entorno 3.12 |
| `ModuleNotFoundError: plotly` | Ejecutar pip de `.venv` y verificar `pip check` |
| `removeChild` (Streamlit) | Desactivar traductor automático del navegador, recargar |
| Vídeo en vivo sin imagen | HTTPS, permiso, cámara ocupada, WebRTC ICE/TURN, logs |
| `ServerSelectionTimeoutError` | Atlas, IP autorizada, firewall, DNS/TLS, ping de prueba |
| Web inicia muy lento | Cold start de Render, PyTorch, RAM, métricas de `medir_inferencia.py` |
| `Out of memory` / reinicio | RAM insuficiente; no es un error de Plotly ni WebRTC |
| No reconoce una ave rara | Detector general COCO limitado; validar con imágenes externas |
| Aparece otra especie con softmax alta | Conjunto cerrado y OOD: usar consenso y abstención |
| No guarda corrección | Atlas debe funcionar y el usuario debe recibir confirmación |

## 9. Entrenamiento / correcciones (se conserva el flujo anterior)

La revisión de correcciones y los trabajos de 100 épocas de MobileNetV3/YOLO-cls
continúan en **Administración de IA**. Los pesos no cambian instantáneamente con
un clic ni al enviar una etiqueta. Se necesita un worker GPU (por ejemplo Colab T4)
para producir y evaluar nuevos checkpoints. Los resultados versionados deben
validarse y activarse por un administrador. Revisa `GUIA_APRENDIZAJE_CONTINUO.md`.

## 10. Checklist antes de decir que está listo

- [ ] Pruebas unitarias pasan y Python 3.12 en el equipo.
- [ ] Los tres pesos aparecen en GitHub y se pueden cargar en Render.
- [ ] Webcam muestra video continuo en localhost.
- [ ] Rotulación cambia cuando aparece/desaparece un ave de prueba.
- [ ] WebRTC funciona desde otra red en Render (TURN, si es necesario).
- [ ] Latencia y memoria medidas en el plan de Render contratado.
- [ ] Clasificación no se bloquea cuando Atlas está desconectado.
- [ ] Escrituras de retroalimentación informan éxito/error reales.
- [ ] Predicciones no se confunden con confirmación biológica.
