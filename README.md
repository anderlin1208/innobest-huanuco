# INNOBEST Huánuco 2026 — Clasificación de aves y cámara en vivo

**Comienza aquí:** [README_DESPLIEGUE_RENDER.md](README_DESPLIEGUE_RENDER.md).

Contiene los checkpoints reales entregados por el proyecto: MobileNetV3 Small,
YOLO11-cls (tres especies) y YOLO11n COCO para verificar presencia de aves.
La cámara en vivo funciona a través de WebRTC y envía al servidor únicamente
fotogramas destinados a inferencia, sin guardar vídeo en la base de datos.

Incluye reconocimiento en vivo, clasificación de fotografías, rechazo de imágenes
no verificadas, correcciones humanas optativas, estadísticas MongoDB y panel
administrativo de entrenamiento externo.

**Limitaciones:** la detección de aves raras no está garantizada; la puntuación
softmax no es una certeza biológica. El servidor Render requiere recursos CPU/RAM
adecuados y TURN para redes que bloquean conexiones WebRTC directas. La formación
de modelos se ejecuta con worker GPU aparte; ningún clic cambia los pesos solo.

**Inicio rápido (PowerShell con Python 3.12):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No guardes contraseñas ni URI privadas en el repositorio Git.
