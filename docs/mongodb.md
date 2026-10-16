# Esquema de MongoDB

Base: `innobest_huanuco` · Colección: `predicciones`.

Documento de ejemplo ficticio (ilustrativo, no una predicción real):

```json
{
  "fecha_utc": "ISODate(...) — almacenado como fecha BSON",
  "modelo": "MobileNetV3",
  "especie_predicha": "ara_militaris",
  "confianza": 0.91,
  "concluyente": true,
  "probabilidades": {
    "ara_militaris": 0.91,
    "pauxi_koepckeae": 0.05,
    "spizaetus_isidori": 0.04
  },
  "origen": "aplicativo_web"
}
```

Los documentos no contienen imágenes, identidad del usuario, dirección IP ni geolocalización. La base/colección se crea automáticamente al insertar el primer registro. Se recomiendan índices en `fecha_utc` y `modelo + especie_predicha`.
