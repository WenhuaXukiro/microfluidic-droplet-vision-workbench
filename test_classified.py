import unittest,tempfile,zipfile,io,json
from pathlib import Path
import server
class Classified(unittest.TestCase):
    def test_class_folders_and_exclusions(self):
        with tempfile.TemporaryDirectory() as t:
            old=server.DATA,server.DB
            try:
                server.DATA=Path(t);server.DB=Path(t)/'project.json';(Path(t)/'images').mkdir()
                rows=[]
                for i in range(8):
                    file=f'{i}.png';(Path(t)/'images'/file).write_bytes(b'unchanged-image')
                    rows.append(dict(id=str(i),projectId='p',parentId='original',file=file,name='same.png',reviewed=True,boxes=[dict(cls=i%5)]))
                rows[5]['reviewed']=False;rows[6]['boxes']=[dict(cls=0),dict(cls=1)];rows[7]['deletedAt']=1
                server.write(dict(projects=[dict(id='p',name='test')],images=rows))
                with zipfile.ZipFile(io.BytesIO(server.export_classified('p'))) as z:
                    manifest=json.loads(z.read('manifest.json'));self.assertEqual(len(manifest['exported']),5);self.assertEqual(len(manifest['skipped']),2)
                    for r in manifest['exported']:self.assertEqual(z.read(r['path']),b'unchanged-image')
                    for name in server.CLASS_NAMES:self.assertIn(name+'/',z.namelist())
                with self.assertRaises(ValueError):server.export_classified('other')
            finally:server.DATA,server.DB=old
if __name__=='__main__':unittest.main()
