"""Use temporary data to verify export and leakage protection without touching user data."""
import tempfile,json,unittest
from pathlib import Path
import server
class Workflow(unittest.TestCase):
    def test_export_and_batch_isolation(self):
        with tempfile.TemporaryDirectory() as t:
            old_data,old_db=server.DATA,server.DB
            try:
                server.DATA=Path(t);server.DB=Path(t)/'project.json';(Path(t)/'images').mkdir()
                rows=[]
                for i,split in enumerate(('train','val')):
                    file=f'{i}.png';(Path(t)/'images'/file).write_bytes(b'test')
                    rows.append(dict(file=file,group=str(i),split=split,reviewed=True,boxes=[dict(cls=1,x=.1,y=.2,w=.2,h=.4)]))
                server.write({'images':rows});dest=server.export_dataset()
                self.assertEqual((dest/'labels/train/0.txt').read_text(),'1 0.20000000 0.40000000 0.20000000 0.40000000')
                self.assertTrue((dest/'images/val/1.png').exists())
                yaml=(dest/'data.yaml').read_text()
                for name in ['1cell 1bead','only cell','only beads','multiplet','empty']:
                    self.assertIn(name,yaml)
                rows[0]['boxes']=[dict(cls=i,x=.1,y=.2,w=.2,h=.4) for i in range(5)]
                server.write({'images':rows}); five=server.export_dataset()
                self.assertEqual([line.split()[0] for line in (five/'labels/train/0.txt').read_text().splitlines()],['0','1','2','3','4'])
                rows[1]['group']='0';server.write({'images':rows})
                with self.assertRaisesRegex(ValueError,'同一实验批次'):server.export_dataset()
                rows[1]['reviewed']=False;server.write({'images':rows})
                with self.assertRaisesRegex(ValueError,'验证集'):server.export_dataset()
            finally:server.DATA,server.DB=old_data,old_db
if __name__=='__main__':unittest.main()
