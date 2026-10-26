"""Check label ordering and edge preservation across the new inference path."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw
import roi_classifier
import worker


class ROISafety(unittest.TestCase):
    def test_edge_particles_are_preserved(self):
        image=Image.new('RGB',(288,512),'black')
        ImageDraw.Draw(image).rectangle((0,0,10,10),fill='red')
        tensor=roi_classifier.image_tensor(image,[512,288])
        restored=tensor*torch.tensor(roi_classifier.STD)[:,None,None]+torch.tensor(roi_classifier.MEAN)[:,None,None]
        self.assertEqual(tuple(tensor.shape),(3,512,288))
        self.assertGreater(float(restored[0,0,0]),.99)
        self.assertLess(float(restored[0,-1,-1]),.01)

    def test_probability_index_matches_workbench_category(self):
        class PredictMultiplet(torch.nn.Module):
            def forward(self,x):
                return torch.tensor([[0.,0.,0.,5.,0.]])
        with tempfile.TemporaryDirectory() as t:
            image=Path(t)/'roi.png';Image.new('RGB',(263,495),'gray').save(image)
            with patch.object(roi_classifier,'load_classifier',return_value=(PredictMultiplet(),{'image_size':[512,288]})):
                result=roi_classifier.predict_roi(image,'classifier_best.pt')
        self.assertEqual(result['boxes'][0]['cls'],3)
        self.assertGreater(result['probabilities']['multiplet'],.95)
        self.assertEqual(result['boxes'][0]['w'],1)
        self.assertEqual(result['task'],'roi_classification')

    def test_multi_droplet_images_do_not_route_to_classification(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            row={'id':'one','projectId':'p','parentId':'source','split':'train','reviewed':True,
                 'boxes':[{'x':0,'y':0,'w':1,'h':1,'cls':0}]}
            snapshot={'images':[row]}
            (root/'annotations.json').write_text(json.dumps(snapshot))
            self.assertTrue(worker.classification_dataset(root,'p'))
            row['boxes'].append(dict(row['boxes'][0]))
            (root/'annotations.json').write_text(json.dumps(snapshot))
            self.assertFalse(worker.classification_dataset(root,'p'))


if __name__=='__main__':unittest.main()
