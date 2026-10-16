import unittest
from src.decision import evaluar_decision
from src.especies import CLASES


def ranking(clase, p=.95):
    resto=(1-p)/2
    return [(c, p if c == clase else resto) for c in CLASES]


class DecisionTest(unittest.TestCase):
    def test_tigre_con_objetos_no_ave_no_se_clasifica(self):
        r=evaluar_decision('sin_ave', ranking(CLASES[0]), ranking(CLASES[0]))
        self.assertEqual(r.estado, 'rechazado_no_ave')
        self.assertIsNone(r.especie)

    def test_detector_sin_evidencia_suspende_prediccion(self):
        r=evaluar_decision('no_verificable', ranking(CLASES[0]), ranking(CLASES[0]))
        self.assertEqual(r.estado, 'no_verificable')

    def test_discrepancia_suspende_prediccion(self):
        r=evaluar_decision('ave_detectada', ranking(CLASES[0]), ranking(CLASES[2]))
        self.assertEqual(r.estado, 'modelos_discrepan')

    def test_coincidencia_alta_es_solo_preliminar(self):
        r=evaluar_decision('ave_detectada', ranking(CLASES[1]), ranking(CLASES[1]))
        self.assertEqual(r.estado, 'compatible_sin_verificar')
        self.assertEqual(r.especie, CLASES[1])

    def test_coincidencia_confianza_baja_no_se_acepta(self):
        r=evaluar_decision('ave_detectada', ranking(CLASES[1], .62), ranking(CLASES[1], .99))
        self.assertEqual(r.estado, 'no_concluyente')

    def test_varias_aves_no_clasifica(self):
        r=evaluar_decision('varias_aves', [], [])
        self.assertEqual(r.estado, 'no_concluyente')

if __name__ == '__main__': unittest.main()

class FiltroTest(unittest.TestCase):
    def _fake(self, clases, nombres, puntuaciones):
        from types import SimpleNamespace
        import numpy as np
        class Caja:
            def __init__(self, c, s):
                self.cls = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: c))
                self.conf = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: s))
        class Detector:
            def predict(self, **kwargs):
                return [SimpleNamespace(boxes=Caja(clases, puntuaciones), names=nombres)]
        return Detector()

    def test_detector_descarta_objeto_sin_ave(self):
        from PIL import Image
        from src.filtro_aves import verificar_ave
        resultado=verificar_ave(self._fake([15],{15:'cat'},[.93]), Image.new('RGB',(32,32)))
        self.assertEqual(resultado.estado,'sin_ave')

    def test_detector_encuentra_ave(self):
        from PIL import Image
        from src.filtro_aves import verificar_ave
        resultado=verificar_ave(self._fake([14],{14:'bird'},[.91]), Image.new('RGB',(32,32)))
        self.assertEqual(resultado.estado,'ave_detectada')

    def test_detector_sin_objetos(self):
        from PIL import Image
        from src.filtro_aves import verificar_ave
        resultado=verificar_ave(self._fake([],{14:'bird'},[]), Image.new('RGB',(32,32)))
        self.assertEqual(resultado.estado,'no_verificable')
