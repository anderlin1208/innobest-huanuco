"""Pruebas puras de seguridad y datasets; no consumen GPU ni se conectan a MongoDB."""
import io
import unittest
import zipfile
from PIL import Image

from src.auth_admin import generar_hash, verificar_clave
from src.dataset_entrenamiento import CLASES_CINCO, validar_dataset_zip
from src.decision import evaluar_decision_cinco


def imagen_bytes(x):
    im=Image.new('RGB',(8,8),(x%255,(x*2)%255,(x*3)%255))
    buff=io.BytesIO();im.save(buff,format='PNG')
    return buff.getvalue()


def construir_zip(faltante=None):
    data=io.BytesIO();counter=0
    with zipfile.ZipFile(data,'w') as z:
        for split,n in [('train',10),('val',3),('test',3)]:
            for clase in CLASES_CINCO:
                for i in range(n):
                    if faltante == (split,clase) and i==n-1: continue
                    z.writestr(f'{split}/{clase}/{i}.png',imagen_bytes(counter))
                    counter+=1
    return data.getvalue()


class AdminTest(unittest.TestCase):
    def test_password_hash_verification(self):
        stored=generar_hash('UnaFraseLarga-Segura-2026')
        self.assertTrue(verificar_clave('UnaFraseLarga-Segura-2026',stored))
        self.assertFalse(verificar_clave('cualquier-otra',stored))
        self.assertFalse(verificar_clave('', 'malformado'))

    def test_password_short_rejected(self):
        with self.assertRaises(ValueError): generar_hash('corta')

    def test_dataset_valido(self):
        resumen=validar_dataset_zip(construir_zip())
        self.assertEqual(resumen['archivos'],80)
        self.assertEqual(resumen['por_clase']['no_ave']['test'],3)

    def test_dataset_incompleto_rechazado(self):
        with self.assertRaisesRegex(ValueError,'insuficientes'):
            validar_dataset_zip(construir_zip(('test','otra_ave')))

    def test_dataset_duplicado_rechazado(self):
        # Los 2 archivos con nombres distintos contienen la misma imagen.
        dat=io.BytesIO()
        with zipfile.ZipFile(dat,'w') as z:
            z.writestr('train/no_ave/1.png',imagen_bytes(9))
            z.writestr('train/no_ave/2.png',imagen_bytes(9))
        with self.assertRaisesRegex(ValueError,'duplicados'):
            validar_dataset_zip(dat.getvalue())

    def test_modelos_rechazan_otra_ave(self):
        m=[('otra_ave',.96),('ara_militaris',.02),('no_ave',.01),
           ('pauxi_koepckeae',.005),('spizaetus_isidori',.005)]
        res=evaluar_decision_cinco('ave_detectada',m,m)
        self.assertEqual(res.estado,'rechazado_no_objetivo')
        self.assertIsNone(res.especie)

    def test_modelos_aceptan_provisional(self):
        m=[('spizaetus_isidori',.96),('ara_militaris',.02),('no_ave',.01),
           ('pauxi_koepckeae',.005),('otra_ave',.005)]
        res=evaluar_decision_cinco('ave_detectada',m,m)
        self.assertEqual(res.especie,'spizaetus_isidori')

    def test_discrepan_rechaza(self):
        m=[('otra_ave',.96),('ara_militaris',.02),('no_ave',.01),
           ('pauxi_koepckeae',.005),('spizaetus_isidori',.005)]
        y=[('spizaetus_isidori',.96),('ara_militaris',.02),('no_ave',.01),
           ('pauxi_koepckeae',.005),('otra_ave',.005)]
        self.assertEqual(evaluar_decision_cinco('ave_detectada',m,y).estado,'modelos_discrepan')


if __name__=='__main__':unittest.main()
