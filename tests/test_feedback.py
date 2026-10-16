import unittest
from io import BytesIO
from PIL import Image
from src.retroalimentacion import (
    preparar_imagen_feedback, validar_etiqueta,
    es_apta_para_reentrenamiento, NO_AVE, OTRA_AVE, NO_SE,
)


class TestImagenYFeedback(unittest.TestCase):
    def setUp(self):
        i = Image.new('RGB', (33, 25), (150, 60, 30))
        buf = BytesIO()
        i.save(buf, format='PNG')
        self.b = buf.getvalue()

    def test_fingerprint_estable(self):
        uno, dos = preparar_imagen_feedback(self.b), preparar_imagen_feedback(self.b)
        self.assertEqual(uno.huella, dos.huella)
        self.assertEqual((uno.ancho, uno.alto), (33, 25))

    def test_imagen_reducida_es_jpeg_sin_exif(self):
        dato = preparar_imagen_feedback(self.b)
        with Image.open(BytesIO(dato.jpeg_sin_metadatos)) as img:
            self.assertEqual(img.format, 'JPEG')
            self.assertFalse(img.getexif())

    def test_archivo_malformado(self):
        with self.assertRaises(ValueError):
            preparar_imagen_feedback(b'no es imagen')

    def test_limite_de_peso(self):
        with self.assertRaises(ValueError):
            preparar_imagen_feedback(b'0' * (8 * 1024 * 1024 + 1))

    def test_etiquetas_validas(self):
        self.assertEqual(validar_etiqueta(OTRA_AVE), OTRA_AVE)
        self.assertEqual(validar_etiqueta(NO_AVE), NO_AVE)
        with self.assertRaises(ValueError):
            validar_etiqueta('tigre')

    def test_no_autorizada_no_entrenar(self):
        self.assertFalse(es_apta_para_reentrenamiento(OTRA_AVE, False, 'aprobada'))

    def test_pendiente_no_entrenar(self):
        self.assertFalse(es_apta_para_reentrenamiento(NO_AVE, True, 'pendiente'))

    def test_confirmada_apta(self):
        self.assertTrue(es_apta_para_reentrenamiento(OTRA_AVE, True, 'aprobada'))

    def test_no_determinada_no_entrenar(self):
        self.assertFalse(es_apta_para_reentrenamiento(NO_SE, True, 'aprobada'))


if __name__ == '__main__':
    unittest.main()
