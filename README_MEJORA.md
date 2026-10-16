# INNOBEST — Validación conservadora de imágenes (etapa 1)

## Qué corrige esta versión

Se conserva `mejor_modelo.pt` (MobileNetV3) y `best.pt` (YOLO11-cls) **sin volver a entrenar**.
El sistema ya **no fuerza** una de las tres especies ante cualquier fotografía.

1. Detector YOLO11n de **objetos** preentrenado en COCO verifica presencia de un ave.
2. Si detecta otros objetos sin ave (por ejemplo un tigre reconocido como felino), rechaza la foto.
3. Si no detecta nada confiable, responde **No se pudo verificar**, no asigna especie.
4. Si ve un ave, ejecuta ambos clasificadores. Solo muestra una **identificación preliminar** si coinciden y cumplen los umbrales provisionales. Si discrepan, no asigna especie.
5. La evaluación puede guardarse en MongoDB como **no verificada**, incluso cuando hay rechazo. No se guardan fotos.

**Limitaciones:** el detector puede no detectar un ave rara; otras aves pueden atravesar el filtro y ambos clasificadores pueden equivocarse coincidiendo. No existe garantía de que todas las tres especies se identifiquen correctamente ni de que se rechace siempre una cuarta especie. Los umbrales 85 % y margen 20 % son heurísticos, no calibrados, y deben evaluarse con muestras conocidas y externas. La salida "sin predicción válida" NO equivale a una probabilidad numérica 0 %.

## Seguridad: importante

El ZIP original recibido incluía `.env` con potenciales credenciales. **Este ZIP corregido NO incluye `.env`.** Cambia en Atlas la contraseña del usuario incluido en el archivo original; es preferible crear un usuario `readWrite` restringido a `innobest_huanuco`, y usarlo en tu `.env` local. NO compartas el `.env` ni publiques permisos `atlasAdmin` en Render.

## Instalar y ejecutar sin venv (Windows, Python 3.12)

Descomprime en carpeta **nueva** y abre esa carpeta en VS Code. Se incluyen los dos checkpoints de los modelos anteriores, pero no el dataset ni las métricas del entrenamiento.

```powershell
py -3.12 -m pip install --user -r requirements.txt
Copy-Item .env.example .env
```

Abre `.env` y configura tu cadena de Atlas con las credenciales **nuevas**. No pegues la URI en mensajes de chat.

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 -m streamlit run app.py
```

URL local: http://localhost:8501. La **primera inferencia** intentará descargar `yolo11n.pt` de detección general desde Ultralytics si no existe en el proyecto. Necesitas internet durante esa primera descarga. Este archivo es distinto de `best.pt` del YOLO11-cls entrenado. Si no se descarga, la aplicación no debe realizar una clasificación sin filtro.

Para descargar el detector por adelantado:

```powershell
py -3.12 -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('Detector disponible')"
```

**Pruebas de aceptación:** una foto de cada especie verdadera, un tigre, un perro y **otras especies de aves**. Revisa el resultado de los dos modelos y el estado final. No confundas una prueba unitaria con una evaluación de eficacia biológica.

## Evaluación cuantitativa (etapa 2)

Aún falta el dataset original completo. Prepara dos conjuntos nuevos, verificados, sin duplicados, distintos de las imágenes que se usaron para entrenar y del conjunto final de prueba:

```
validacion_objetivo/
  ara_militaris/
  pauxi_koepckeae/
  spizaetus_isidori/
validacion_externas/
  otras_aves/
  otros_animales/
  objetos/
```

Ejecuta:

```powershell
py -3.12 scripts/evaluar_rechazo.py --objetivo "C:\datos\validacion_objetivo" --externas "C:\datos\validacion_externas" --salida evaluacion_rechazo.csv
```

El CSV y el resumen mostrarán conocidas correctamente aceptadas, conocidas rechazadas y **desconocidas incorrectamente aceptadas**. Para mejorar la identificación real y el rechazo de otras aves hará falta ampliar/corregir el dataset y entrenar un enfoque con ejemplos negativos (p. ej., una clase adicional *otras aves*), usando particiones independientes y evaluación honesta. Si cambias la cantidad de clases, los checkpoints existentes no serán directamente compatibles: conserva backups y entrena nuevos pesos en Colab.

## Qué falta para Render

- Una política de despliegue para los 3 conjuntos de pesos (MobileNetV3, YOLO-cls y detector general), que no están versionados en Git.
- Usuario MongoDB de privilegios mínimos y estrategia de conexión de red desde Render.
- Probar memoria y tiempos de arranque en Render. `app.py` requiere Torch y Ultralytics, que pueden superar límites de un servicio gratuito.
- Medición validada de rechazo de especies ajenas y errores del clasificador antes de afirmaciones científicas.
