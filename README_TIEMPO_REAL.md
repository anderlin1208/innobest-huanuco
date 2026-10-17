# INNOBEST — Cámara, correcciones y entrenamiento desde la web

**Actualización de esta entrega:** la cámara en vídeo continuo se documenta en
[README_DESPLIEGUE_RENDER.md](README_DESPLIEGUE_RENDER.md). Los puntos que siguen
explican la captura de fotografía tradicional, que se conserva como alternativa.

## Capacidades implementadas
- Cámara del navegador (`st.camera_input`) para **tomar una foto y clasificarla** inmediatamente después de capturarla. No es vídeo continuo; el navegador necesita permiso de cámara (localhost o HTTPS).
- Correcciones instantáneas guardadas en MongoDB (con consentimiento optativo para almacenar foto). Volver a enviar **los mismos bytes** permite consultar la corrección previa, pero **no modifica los pesos**.
- Panel Administración de IA con contraseña, revisión visual de correcciones, aprobación/rechazo, carga de dataset ZIP, creación de trabajos y consulta de sus estados desde la web.
- El botón solicita un entrenamiento real de **100 épocas por modelo**. La inferencia web no consume la GPU Colab: un **worker T4 conectado** toma el trabajo de MongoDB, entrena MobileNetV3 y YOLO11n-cls de cinco clases, deposita checkpoints/metricas en GridFS y los deja para descarga y validación humana.
- Los nuevos checkpoints **no se ponen automáticamente en producción**. El administrador puede validarlos y pulsar **Activar versión** desde la web; también puede restaurar la versión original. La activación descarga desde GridFS los modelos de cinco clases en cada instancia web.

## Instalación inicial en Windows sin entorno virtual
1. Descomprime el ZIP y abre la carpeta en Visual Studio Code. Copia manualmente tu `.env` local sin publicarlo y actualiza allí las credenciales de MongoDB.
2. Configura por única vez la contraseña del administrador: `py -3.12 scripts/configurar_admin.py`. **No pegues esa contraseña en el chat.**
3. Instala dependencias (si hace falta): `py -3.12 -m pip install --user -r requirements.txt`.
4. `py -3.12 -m unittest discover -s tests -v`.
5. `py -3.12 -m streamlit run app.py`. Abre `http://localhost:8501`.
6. En **Identificar un ave**, elige **Tomar foto con cámara**, concede permiso, fotografía y obtén resultado. Corrige en la misma pantalla.
7. En **Administración de IA**, inicia sesión y revisa/autoriza fotos antes de pulsar Aprobar. Esto sustituye comandos repetidos en PowerShell.

## Entrenamiento con GPU desde la web
1. Organiza un ZIP con carpetas `train/`, `val/`, `test/` y cinco subcarpetas por split: `ara_militaris`, `pauxi_koepckeae`, `spizaetus_isidori`, `otra_ave`, `no_ave`. Se requieren al menos **10/3/3 fotografías** por clase y split solo para una prueba técnica, no para confiabilidad científica. Es preferible muchas más y test externo.
2. En **Administración de IA → Entrenamiento y versiones**, sube ZIP validado y selecciona **Solicitar 100 épocas**. Se genera un trabajo persistente MongoDB. Fotos de feedback aprobadas se incorporan a `train`, evitando duplicados exactos; no afectan test.
3. En Google Colab, abre `notebooks/worker_colab_t4.ipynb` (súbelo al Drive) y conecta una GPU T4. Sube la misma carpeta ZIP del proyecto actual, sin `.env`, y sigue el notebook. La celda final escucha trabajos hasta desconectar. **Debes permitir la IP pública de Colab en Atlas cuando sea necesario.**
4. La web muestra `en_cola`, `entrenando`, `completado_pendiente_validacion` o `fallido`. Tras completar, descarga los pesos y métricas. El administrador, después de una evaluación humana independiente, puede pulsar **Activar versión**. El sitio cargará desde MongoDB los modelos de cinco clases. También hay opción de **Volver a los modelos originales**.

## Limitaciones importantes
- En **Identificar un ave**, la cámara tradicional captura una foto. En **Cámara en vivo** ahora sí existe vídeo continuo con inferencia de fotogramas espaciados; no se analiza cada fotograma.
- No hay entrenamiento por clic sin GPU/dataset supervisado. Las 100 épocas de dos modelos pueden tardar y la sesión T4 de Colab puede interrumpirse. Si se interrumpe, un trabajo `entrenando` requiere revisión manual.
- En Render normalmente se ejecuta la inferencia CPU y el servidor web; el entrenamiento GPU queda a cargo de Colab u otro worker con GPU.
- Las clases nuevas `no_ave` y `otra_ave` requieren múltiples ejemplos variados. Aprobar fotos aisladas **no garantiza** generalización.
- No se incluyen `.env`, fotos privadas ni credenciales en el ZIP entregable. Usa un usuario MongoDB limitado a la base `innobest_huanuco`.
- No se activa automáticamente la versión nueva: el administrador debe validarla y activarla expresamente desde la web. Los umbrales permanecen provisionales y se requiere evaluación OOD real para afirmar que generaliza bien.
