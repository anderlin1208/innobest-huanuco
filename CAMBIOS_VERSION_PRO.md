# Registro de modificaciones — INNOBEST profesional (2026-10-16)

## Cambios realizados

- Se añadió `src/camara_vivo.py` con transmisión WebRTC y procesamiento asíncrono controlado (sin `st.*` en hilo de vídeo).
- Se añadió `src/motor_inferencia.py` con pesos originales compartidos en memoria, bloqueo de inferencia y control de hilos CPU.
- El reconocimiento en vivo muestra una especie **solo cuando YOLO COCO detecta ave y ambos clasificadores concuerdan con umbrales provisionales**; ante desacuerdos se abstiene.
- Se mantienen `st.file_uploader` y la captura de imagen tradicional para revisión y correcciones.
- El modo de fotografía reutiliza inferencias de la misma imagen y versión, sin volver a ejecutar modelos al interactuar con el formulario.
- Se hicieron opcionales las lecturas de versión activa y de corrección previa desde MongoDB para impedir esperas en navegación y clasificación.
- `src/mongodb.py` ahora admite los estados de cinco clases al guardar evaluaciones; no altera pesos.
- Se fijó Python 3.12.10, añadieron dependencias de WebRTC y se corrigió `.gitignore` para publicar los checkpoints esenciales.
- Se añadieron diagnóstico de entorno, medición real de latencia y memoria, y tests nuevos de cámara sin hardware.

## Validación efectivamente realizada en este entorno

- Los **33 tests de Python** pasan (`unittest discover`).
- `python -m py_compile` pasa para `app.py`, scripts y módulos.
- El checkpoint original **MobileNetV3** cargó con `torch.load(weights_only=True)` y produjo salida real con una imagen de prueba sintética.
- Los archivos de pesos son idénticos a los recibidos; ver `SHA256_PESOS.txt`.

## Validaciones pendientes (no pueden garantizarse desde esta sesión)

- Instalación completa de `streamlit-webrtc`, PyAV, Ultralytics y PyMongo en Windows/Render (no disponibles en este contenedor sin acceso a PyPI).
- Inicio y prueba manual de la interfaz Streamlit con cámara física y red WebRTC.
- Calidad de predicciones frente a aves reales y especies ajenas (sin dataset de evaluación etiquetado).
- Rendimiento, memoria y compatibilidad TURN con una instancia real de Render.
- Conectividad y escritura real a MongoDB Atlas (no tenemos las credenciales privadas y tu TLS había fallado).

El objetivo es una base más robusta y documentada para verificar en despliegue, NO una garantía de cero demora o precisión biológica.
