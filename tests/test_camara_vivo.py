"""Pruebas sin webcam ni dependencias opcionales de WebRTC."""
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.camara_vivo import ProcesadorCamara, estimar_etiqueta, configuracion_rtc


class TestCamaraEnVivo(unittest.TestCase):
    def test_etiqueta_real_se_muestra_como_preliminar(self):
        r = {'especie': 'ara_militaris', 'estado': 'compatible_sin_verificar'}
        self.assertEqual(estimar_etiqueta(r), 'PRELIMINAR: Guacamayo militar')

    def test_sin_ave_no_se_inventa_especie(self):
        self.assertEqual(estimar_etiqueta({'estado': 'rechazado_no_ave'}),
                         'SIN AVE DETECTADA')

    def test_condiciones_stun_sin_credenciales(self):
        with patch.dict('os.environ', {}, clear=True):
            rtc = configuracion_rtc()
            self.assertIn('stun:', rtc['iceServers'][0]['urls'][0])

    def test_turn_cloudflare_no_exhibe_claves(self):
        with patch.dict('os.environ', {'CLOUDFLARE_TURN_KEY_ID': 'x',
                                        'CLOUDFLARE_TURN_KEY_API_TOKEN': 'secreto'}, clear=True):
            self.assertIsNone(configuracion_rtc())

    def test_inferencia_lenta_no_bloquea_callback(self):
        import numpy as np
        imagen = np.zeros((30, 30, 3), dtype='uint8')
        def lenta(rgb):
            time.sleep(.07)
            return {'especie': None, 'estado': 'rechazado_no_ave'}
        with patch('src.camara_vivo.analizar_rgb', side_effect=lenta):
            procesador = ProcesadorCamara(intervalo=2)
            try:
                inicio = time.monotonic()
                r, error, pendiente = procesador.procesar_muestra(imagen)
                self.assertIsNone(r)
                self.assertTrue(pendiente)
                self.assertLess(time.monotonic() - inicio, .05)
                time.sleep(.12)
                r, error, pendiente = procesador.procesar_muestra(imagen)
                self.assertEqual(r['estado'], 'rechazado_no_ave')
                self.assertFalse(pendiente)
                self.assertIsNone(error)
            finally:
                procesador.cerrar()

    def test_video_anotado_se_devuelve_sin_webcam_fisica(self):
        import numpy as np
        fake_av = SimpleNamespace(VideoFrame=SimpleNamespace(from_ndarray=lambda a, format: a))
        clase_frame = SimpleNamespace(to_ndarray=lambda format: np.zeros((120, 240, 3), dtype='uint8'))
        with patch.dict(sys.modules, {'av': fake_av}), \
                patch('src.camara_vivo.analizar_rgb', return_value={'estado': 'rechazado_no_ave', 'especie': None}):
            proc = ProcesadorCamara()
            try:
                salida = proc(clase_frame)
                self.assertEqual(salida.shape, (120, 240, 3))
                self.assertTrue(np.any(salida != 0))
            finally:
                proc.cerrar()

    def test_no_procesa_tras_cerrar(self):
        import numpy as np
        procesador = ProcesadorCamara()
        procesador.cerrar()
        _, _, pendiente = procesador.procesar_muestra(np.zeros((10,10,3),dtype='uint8'))
        self.assertFalse(pendiente)


if __name__ == '__main__':
    unittest.main()
