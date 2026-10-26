"""Regression check: project-scoped exports audit only their own reviewed rows."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import evaluate_model

class ProjectAudit(unittest.TestCase):
    def test_other_project_not_required_but_missing_same_project_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'export';(root/'data').mkdir()
            rows=[]
            for project,uid,split in [('p','a','train'),('p','b','val'),('other','c','train')]:
                rows.append(dict(id=uid,projectId=project,name=uid+'.png',file=uid+'.png',split=split,group=uid,reviewed=True,boxes=[dict(cls=0,x=0,y=0,w=1,h=1)]))
            for uid,split in [('a','train'),('b','val')]:
                images=dataset/'images'/split;labels=dataset/'labels'/split;images.mkdir(parents=True);labels.mkdir(parents=True)
                (images/(uid+'.png')).write_bytes(uid.encode());(labels/(uid+'.txt')).write_text('0 0.5 0.5 1 1')
            snapshot={'projects':[{'id':'p'},{'id':'other'}],'images':rows}
            def save():
                data=json.dumps(snapshot);(dataset/'annotations.json').write_text(data);(root/'data/project.json').write_text(data)
            save()
            with patch.object(evaluate_model,'ROOT',root):
                result,_=evaluate_model.audit(dataset);self.assertFalse(result['errors'])
                rows.append({**rows[0],'id':'missing','name':'missing.png'})
                save()
                with self.assertRaisesRegex(ValueError,'exactly cover'):evaluate_model.audit(dataset)

if __name__=='__main__':unittest.main()
