"""Check preservation of geometry and agreement among preview, validation and inference."""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from image_preprocessing import PRESETS, enhance_image, preview_payload
from roi_classifier import image_tensor, predict_roi


class EnhancementConsistency(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.tile(np.arange(64, 192, dtype=np.uint8), (96, 1))).convert("RGB")
        ImageDraw.Draw(self.image).ellipse((0, 0, 8, 8), fill=(25, 25, 25))

    def test_none_preserves_every_pixel(self):
        np.testing.assert_array_equal(np.asarray(self.image), np.asarray(enhance_image(self.image)))

    def test_preview_matches_model_enhancement_and_preserves_dimensions(self):
        buffer = io.BytesIO();self.image.save(buffer, format="PNG")
        for preset in PRESETS:
            with self.subTest(preset=preset):
                output = preview_payload({"data": base64.b64encode(buffer.getvalue()).decode(), "preprocessing": preset})
                with Image.open(io.BytesIO(base64.b64decode(output["data"]))) as actual:
                    self.assertEqual(actual.size, self.image.size)
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(enhance_image(self.image, preset)))
                    # Dark particle at the image corner must not be cropped or moved.
                    self.assertLess(float(np.asarray(actual)[4, 4].mean()), float(np.asarray(actual)[4, 20].mean()))

    def test_inference_reads_the_saved_enhancement(self):
        seen = []
        class Model(torch.nn.Module):
            def forward(self, x):
                seen.append(x.detach().clone())
                return torch.tensor([[0., 4., 0., 0., 0.]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"roi.png";self.image.save(path)
            with patch("roi_classifier.load_classifier", return_value=(Model(), {"image_size": [128, 128], "preprocessing": "clahe_soft"})):
                result = predict_roi(path, "model.pt")
        torch.testing.assert_close(seen[0][0], image_tensor(self.image, [128, 128], "clahe_soft"), rtol=0, atol=0)
        self.assertEqual(result["preprocessing"], "clahe_soft")
        self.assertEqual(result["boxes"][0]["cls"], 1)

    def test_training_and_validation_use_the_same_image_pipeline(self):
        from train_roi_classifier import ROIDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"annotations.json").write_text(json.dumps({"images": [
                {"id": "sample", "name": "roi.png", "parentId": "original", "boxes": [{"cls": 1}]}]}))
            for split in ("train", "val"):
                (root/"images"/split).mkdir(parents=True)
                self.image.save(root/"images"/split/"sample.png")
                dataset = ROIDataset(root, split, [128, 128], preprocessing="illumination_soft")
                tensor, _, _ = dataset[0]
                torch.testing.assert_close(tensor, image_tensor(self.image, [128, 128], "illumination_soft"), rtol=0, atol=0)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "增强方案"):
            enhance_image(self.image, "typo")


if __name__ == "__main__":
    unittest.main()
