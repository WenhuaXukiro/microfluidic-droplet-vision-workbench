import copy
import unittest
import server


def image(i, group, **extra):
    return dict(id=i, projectId='p', group=group, split='train', reviewed=True,
                boxes=[dict(cls=0,x=0,y=0,w=1,h=1)], **extra)


class AutoSplit(unittest.TestCase):
    def project(self, rows):
        return dict(projects=[dict(id='p')], images=rows)

    def test_batch_ratio_and_family(self):
        rows=[image(str(i),str(i//2)) for i in range(20)]
        child=image('child','0',parentId='0',deletedAt=1)
        child['reviewed']=False
        rows.append(child)
        other=image('other','0');other['projectId']='other';rows.append(other)
        p=self.project(rows)
        result=server.auto_split(p,'p')
        self.assertEqual((result['train'],result['val']),(16,4))
        self.assertEqual(rows[0]['split'],child['split'])
        self.assertEqual(other['split'],'train')
        for i in range(0,20,2): self.assertEqual(rows[i]['split'],rows[i+1]['split'])
        first=[r['split'] for r in rows]
        server.auto_split(p,'p')
        self.assertEqual(first,[r['split'] for r in rows])

    def test_one_batch_and_source_opt_in(self):
        p=self.project([image(str(i),'video') for i in range(10)])
        before=copy.deepcopy(p)
        with self.assertRaisesRegex(ValueError,'批次'): server.auto_split(p,'p')
        self.assertEqual(before,p)
        result=server.auto_split(p,'p',mode='source')
        self.assertEqual((result['train'],result['val']),(8,2))
        self.assertTrue(all(r['group']=='video' for r in p['images']))

    def test_test_set_preserved_and_validation(self):
        rows=[image(str(i),str(i)) for i in range(10)]
        rows[-1]['split']='test'
        p=self.project(rows)
        self.assertEqual(server.auto_split(p,'p')['test'],1)
        self.assertEqual(rows[-1]['split'],'test')
        for ratio in (0,1,True,'0.2',float('nan')):
            with self.assertRaises(ValueError): server.auto_split(p,'p',ratio)
        with self.assertRaises(ValueError): server.auto_split(p,'missing')

    def test_source_keeps_family_together(self):
        rows=[image(str(i),'same') for i in range(10)]
        rows.append(image('child','different',parentId='0'))
        p=self.project(rows)
        server.auto_split(p,'p',mode='source')
        self.assertEqual(rows[0]['split'],rows[-1]['split'])
        self.assertEqual(sum(r['split']=='val' for r in rows),2)


if __name__=='__main__': unittest.main()
