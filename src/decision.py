"""Reglas conservadoras para inferencia de conjunto cerrado.

No representa un certificado biológico ni un detector OOD validado.
Los umbrales son provisionales: ajustarlos con validación etiquetada,
NUNCA con el conjunto final de prueba.
"""
from dataclasses import dataclass
from math import isfinite
from src.especies import CLASES


@dataclass(frozen=True)
class Decision:
    estado: str
    especie: str | None
    motivo: str


def evaluar_decision(estado_detector: str, mobile: list, yolo: list,
                     umbral: float = 0.85, margen_minimo: float = 0.20) -> Decision:
    """Acepta coincidencia SOLO provisional si ambas redes y filtro concuerdan."""
    if estado_detector == 'sin_ave':
        return Decision('rechazado_no_ave', None, 'No se detectó un ave; se detectaron otros objetos.')
    if estado_detector == 'varias_aves':
        return Decision('no_concluyente', None, 'Hay varias aves; utiliza una foto de un solo ejemplar.')
    if estado_detector != 'ave_detectada':
        return Decision('no_verificable', None, 'No se pudo verificar visualmente la presencia de un ave.')

    def verificar(ranking):
        if len(ranking) != len(CLASES) or {x[0] for x in ranking} != set(CLASES):
            raise ValueError('Las predicciones deben contener las tres especies.')
        if any(not isfinite(float(v)) or not 0 <= float(v) <= 1 for _, v in ranking):
            raise ValueError('Predicciones fuera del rango 0..1.')
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    mobile, yolo = verificar(mobile), verificar(yolo)
    if mobile[0][0] != yolo[0][0]:
        return Decision('modelos_discrepan', None, 'MobileNetV3 y YOLO11-cls discrepan: revisión necesaria.')
    if any(r[0][1] < umbral or (r[0][1] - r[1][1]) < margen_minimo for r in (mobile, yolo)):
        return Decision('no_concluyente', None, 'La coincidencia no supera los umbrales provisionales.')
    return Decision('compatible_sin_verificar', mobile[0][0], 'Los modelos coinciden; identificación preliminar no verificada.')



def evaluar_decision_cinco(estado_detector: str, mobile: list, yolo: list,
                           umbral: float = 0.85, margen_minimo: float = 0.20) -> Decision:
    """Consenso para cinco categorías, incluyendo rechazos supervisados.

    Los umbrales no sustituyen un estudio de calibración/OOD independiente.
    """
    from src.dataset_entrenamiento import CLASES_CINCO
    if estado_detector == 'sin_ave':
        return Decision('rechazado_no_ave', None, 'El detector no encontró aves y encontró otros objetos.')
    if estado_detector == 'varias_aves':
        return Decision('no_concluyente', None, 'Se detectaron varias aves.')
    if estado_detector != 'ave_detectada':
        return Decision('no_verificable', None, 'El detector no pudo verificar un ave.')
    for ranking in (mobile,yolo):
        if len(ranking)!=5 or set(c for c,_ in ranking)!=set(CLASES_CINCO):
            raise ValueError('El modelo versionado debe tener las cinco clases.')
        if any(not isfinite(float(p)) or not 0 <= float(p) <= 1 for _,p in ranking):
            raise ValueError('Las puntuaciones deben estar entre 0 y 1.')
    mobile=sorted(mobile,key=lambda x:x[1],reverse=True)
    yolo=sorted(yolo,key=lambda x:x[1],reverse=True)
    if mobile[0][0] != yolo[0][0]:
        return Decision('modelos_discrepan',None,'Los clasificadores discrepan; requiere revisión.')
    primero=mobile[0][0]
    if primero in ('no_ave','otra_ave'):
        return Decision('rechazado_no_objetivo',None,'Ambos modelos coinciden en que no es una de las tres especies objetivo.')
    if any(r[0][1]<umbral or r[0][1]-r[1][1]<margen_minimo for r in (mobile,yolo)):
        return Decision('no_concluyente',None,'Confianza o margen insuficiente.')
    return Decision('compatible_sin_verificar',primero,'Identificación preliminar, sin confirmación biológica.')
