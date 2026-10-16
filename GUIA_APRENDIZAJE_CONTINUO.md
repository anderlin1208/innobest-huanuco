# INNOBEST: retroalimentación humana con mejora progresiva

## Qué sí funciona ahora

- El dashboard ofrece etiquetas humanas `ara_militaris`, `pauxi_koepckeae`, `spizaetus_isidori`, `otra_ave`, `no_ave`, `no_se_puede_determinar`.
- Cada corrección se almacena inmediatamente en MongoDB Atlas, colección **`retroalimentaciones`**, sin modificar las predicciones antiguas.
- Se utiliza SHA-256 para identificar el **mismo archivo exacto**, y en una carga posterior se puede mostrar la corrección previa con su estado `pendiente / aprobada / rechazada`. **NO** significa que los modelos hayan aprendido de forma automática.
- La foto NO se conserva salvo que la persona autorice el uso para entrenamiento y tenga permiso de utilización. En ese caso se guarda en **MongoDB GridFS** como JPEG reducido a 1280 px máximo y sin metadatos EXIF.
- El panel `Actividad del sistema` muestra el número de correcciones, separado del histórico de predicciones.
- Una revisión local permite aprobar o rechazar correcciones y exportar únicamente las imágenes autorizadas y aprobadas.

## Por qué no se reentrena inmediatamente con cada clic

- Una etiqueta equivocada podría contaminar el entrenamiento. Las etiquetas de usuarios NO equivalen a validación biológica.
- Con una sola fotografía de loro no se aprende a identificar todas las fotos de otras especies de aves.
- Cambiar de 3 clases a 5 (`otras_aves`, `no_ave`) obliga a crear y evaluar pesos nuevos para ambas arquitecturas. No se pueden interpretar las probabilidades antiguas como probabilidades de cinco clases.
- Colab T4 no es un servicio de inferencia o entrenamiento **siempre activo**: puede desconectarse. Render web tampoco es el entorno adecuado para entrenamientos pesados en cada solicitud.
- Para mejorar con datos progresivamente se requiere trabajo supervisado: **recolección → revisión de calidad/licencias → reentrenamiento por lotes en GPU → evaluación con conjunto de prueba independiente → versión nueva → rollback si empeora**.

## Primer uso en Windows (sin venv)

1. Descomprimir en **carpeta nueva**. Mantener como respaldo el ZIP anterior.
2. Abrir la carpeta en VS Code. Instalar dependencias si no están instaladas: `py -3.12 -m pip install --user -r requirements.txt`.
3. Copiar `.env.example` como `.env`, y configurar únicamente tus credenciales **nuevas** y restringidas de MongoDB. El ZIP no incluye `.env`. MongoDB necesita permisos `readWrite` sobre la base INNOBEST, para poder escribir GridFS.
4. Ejecutar `py -3.12 -m unittest discover -s tests -v`.
5. Ejecutar `py -3.12 -m streamlit run app.py`.
6. Cargar la imagen del tigre: esperar «sin predicción válida». En «Enseñar mediante una corrección humana» marcar **No es un ave**, dar autorización solo si tienes derechos para usar la foto y enviar. Repetir con un loro de otra especie marcando **Otra especie de ave**. Luego cargar exactamente el mismo archivo: se muestra la corrección humana anterior como referencia, **sin sustituir engañosamente la salida del modelo**.
7. En `Actividad del sistema` verificar la tabla de correcciones. En Atlas verificar colección `retroalimentaciones` y, si diste autorización, colecciones GridFS `fotografias_feedback.files` y `fotografias_feedback.chunks`.

## Revisión del dataset desde la terminal local (SOLO para administrador)

> Ejecutar desde la raíz de tu proyecto. **No expongas** las opciones de aprobación en el dashboard público.

```powershell
py -3.12 scripts/retroalimentacion_admin.py listar
py -3.12 scripts/retroalimentacion_admin.py descargar-pendientes --salida revision_pendientes
```

Abre y comprueba manualmente las imágenes exportadas en `revision_pendientes`. **Solo después** de comprobar identidad, cantidad de ejemplares, duplicados y licencias de uso, aprueba con ID y etiqueta:

```powershell
py -3.12 scripts/retroalimentacion_admin.py aprobar --id ID_REAL --etiqueta otra_ave
```

Para rechazar una etiqueta incorrecta:

```powershell
py -3.12 scripts/retroalimentacion_admin.py rechazar --id ID_REAL
```

Exporta únicamente fotografías aprobadas y con consentimiento:

```powershell
py -3.12 scripts/retroalimentacion_admin.py exportar-aprobadas --salida feedback_aprobado
```

El directorio `feedback_aprobado` contiene subcarpetas por etiqueta y un `manifest.csv`. No incluyas fotos privadas/no licenciadas en repositorios públicos.

## Próxima etapa necesaria: mejorar realmente los dos pesos de IA

1. Recuperar el **dataset original** y conservar el conjunto `test` separado (intocable). La versión de este ZIP solo incluye los `.pt`, no las 128 fotos depuradas.
2. Reunir un conjunto suficiente y **equilibrado** de `otras_aves` (loros, garzas, palomas, etc.), animales no aviares y las tres especies objetivo; no mezclar imágenes del mismo ejemplar en train y test.
3. Decidir arquitectura del rechazo: primera etapa ave/no ave, segunda etapa 3 especies/otra ave (preferible a entrenar sin validación cinco clases mezcladas).
4. Afinar **MobileNetV3 y YOLO11-cls** en GPU, con subconjunto de nuevas fotografías verificadas, reteniendo ejemplos antiguos para disminuir olvido catastrófico.
5. Medir tanto accuracy/F1 por especie como recall de especies objetivo, falsa aceptación de OOD y tasa de rechazos de aves válidas. No elegir umbrales sobre `test`.
6. Guardar checkpoints versionados `v2`, métricas y fecha; sustituir modelos solo si pasan las pruebas. Mantener rollback al `v1`.

Esta primera entrega implementa la **captura, almacenamiento, consulta de feedback y cola de revisión**. NO añade todavía entrenamiento automático ni reconocimiento generalizado de imágenes desconocidas. Es una distinción técnica fundamental para explicárselo al profesor.
