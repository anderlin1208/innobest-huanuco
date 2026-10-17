"""Pruebas que no requieren torch, ni Streamlit, ni conexión de red."""
import unittest

import numpy as np
from PIL import Image

from src.especies import CLASES
from src.modelo_onnx import RUTA_MODELO, interpretar, preparar_imagen, predecir


class SesionSimulada:
    def get_inputs(self):
        class Entrada:
            name = 'input'
        return [Entrada()]

    def run(self, salidas, entradas):
        valor = entradas['input']
        assert valor.shape == (1, 3, 224, 224)
        assert valor.dtype == np.float32
        return [np.array([[1.0, 0.0, -1.0]], dtype=np.float32)]


class TestModeloLigero(unittest.TestCase):
    def test_archivo_onnx_incluido(self):
        self.assertTrue(RUTA_MODELO.is_file())
        self.assertGreater(RUTA_MODELO.stat().st_size, 1_000_000)

    def test_tamano_y_tipo_para_imagen_rectangular(self):
        foto = Image.new('RGB', (300, 600), (60, 120, 180))
        entrada = preparar_imagen(foto)
        self.assertEqual(entrada.shape, (1, 3, 224, 224))
        self.assertEqual(entrada.dtype, np.float32)
        self.assertTrue(np.isfinite(entrada).all())

    def test_ranking_de_tres_clases(self):
        foto = Image.new('RGB', (400, 300), (255, 255, 255))
        ranking = predecir(SesionSimulada(), foto)
        self.assertEqual({c for c, _ in ranking}, set(CLASES))
        self.assertAlmostEqual(sum(x for _, x in ranking), 1.0)
        self.assertEqual(ranking[0][0], 'ara_militaris')

    def test_rechazo_heuristico(self):
        ranking = [('ara_militaris', .45), ('pauxi_koepckeae', .35), ('spizaetus_isidori', .20)]
        self.assertIsNone(interpretar(ranking)['especie'])

    def test_aceptacion_heuristica(self):
        ranking = [('ara_militaris', .90), ('pauxi_koepckeae', .08), ('spizaetus_isidori', .02)]
        self.assertEqual(interpretar(ranking)['especie'], 'ara_militaris')


if __name__ == '__main__':
    unittest.main()
