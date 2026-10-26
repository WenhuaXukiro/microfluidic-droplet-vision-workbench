import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import server

class Recycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.old=server.DATA,server.DB
        server.DATA=Path(self.temp.name);server.DB=server.DATA/'project.json';(server.DATA/'images').mkdir()
        self.p={'projects':[{'id':'p','datasetSplit':{'mode':'source'}}], 'images':[
            dict(id='original',projectId='p',file='original.png',name='original',split='train',group='batch',reviewed=False,boxes=[],deletedAt=1),
            dict(id='child',parentId='original',projectId='p',file='child.png',name='child',split='train',group='batch',reviewed=True,boxes=[dict(cls=0,x=0,y=0,w=1,h=1)]),
            dict(id='val',parentId='other-original',projectId='p',file='val.png',name='val',split='val',group='other',reviewed=True,boxes=[dict(cls=0,x=0,y=0,w=1,h=1)]),
            dict(id='foreign',projectId='other',file='foreign.png',name='foreign',split='train',deletedAt=1)]}
        for r in self.p['images']:(server.DATA/'images'/r['file']).write_bytes(b'test')
        server.write(self.p)
    def tearDown(self):server.DATA,server.DB=self.old;self.temp.cleanup()
    def test_purge_preserves_active_child_and_training_export(self):
        result=server.purge_images(self.p,'p',['original']);self.assertEqual(result['count'],1)
        self.assertFalse((server.DATA/'images/original.png').exists());self.assertTrue((server.DATA/'images/child.png').exists())
        self.assertEqual(len(result['project']['images']),3);self.assertTrue(server.export_dataset('p').is_dir())
    def test_active_cross_project_and_invalid_path_are_rejected(self):
        for ids in (['child'],['foreign'],['original','child'],[],None):
            with self.assertRaises(ValueError):server.purge_images(self.p,'p',ids)
        self.p['images'][0]['file']='../project.json'
        with self.assertRaises(ValueError):server.purge_images(self.p,'p',['original'])
        self.assertTrue((server.DATA/'images/original.png').exists())
    def test_failed_commit_rolls_back_files(self):
        with patch.object(server,'write',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):server.purge_images(self.p,'p',['original'])
        self.assertTrue((server.DATA/'images/original.png').exists())
        self.assertEqual(len(json.loads(server.DB.read_text(encoding='utf-8'))['images']),4)
    def test_source_split_keeps_siblings_after_original_purged(self):
        self.p['images'].append({**self.p['images'][1],'id':'sibling','file':'sibling.png'})
        updated=server.purge_images(self.p,'p',['original'])['project']
        server.auto_split(updated,'p',mode='source')
        rows={r['id']:r for r in updated['images']};self.assertEqual(rows['child']['split'],rows['sibling']['split'])
    def test_running_job_blocks_purge(self):
        with patch.dict(server.JOB,{'status':'running'}):
            with self.assertRaises(ValueError):server.purge_images(self.p,'p',['original'])

if __name__=='__main__':unittest.main()
