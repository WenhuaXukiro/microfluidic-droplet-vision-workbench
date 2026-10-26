"""Whole-ROI classification shared by training, validation and workbench inference."""
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision import models
from torchvision.transforms import functional as TF
from image_preprocessing import PREPROCESS_VERSION, enhance_image, validate_preset

NAMES = ["1cell 1bead", "only cell", "only beads", "multiplet", "empty"]
FORMAT = "droplet-roi-resnet18-v1"
FORMATS = {"resnet18": FORMAT, "convnext_tiny": "droplet-roi-convnext-tiny-v1"}
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def build_classifier(architecture, num_classes=5):
    if architecture == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    if architecture == "convnext_tiny":
        return models.convnext_tiny(weights=None, num_classes=num_classes)
    raise ValueError(f"不支持的分类网络：{architecture}")


def image_tensor(image, image_size, preprocessing="none"):
    """Preserve the full ROI and its aspect ratio; never center-crop particles."""
    height, width = image_size
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = enhance_image(image, preprocessing)
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return TF.normalize(TF.to_tensor(canvas), MEAN, STD)


def load_classifier(checkpoint, device="cpu"):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = ckpt.get("architecture", ckpt.get("training_config", {}).get("architecture", "resnet18"))
    if ckpt.get("format") != FORMATS.get(architecture) or ckpt.get("names") != NAMES:
        raise ValueError("分类模型格式或类别不匹配")
    validate_preset(ckpt.get("preprocessing", ckpt.get("training_config", {}).get("preprocessing", "none")))
    if ckpt.get("preprocessing_version", PREPROCESS_VERSION) != PREPROCESS_VERSION:
        raise ValueError("分类模型的识别增强版本不受支持")
    model = build_classifier(architecture, num_classes=len(NAMES))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def predict_roi(image_path, checkpoint, device="cpu"):
    model, ckpt = load_classifier(checkpoint, device)
    preprocessing = ckpt.get("preprocessing", ckpt.get("training_config", {}).get("preprocessing", "none"))
    with Image.open(image_path) as image:
        x = image_tensor(image, ckpt["image_size"], preprocessing).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(x).softmax(dim=1)[0].cpu()
    cls = int(probabilities.argmax())
    return {"boxes": [{"x": 0., "y": 0., "w": 1., "h": 1., "cls": cls,
                       "confidence": float(probabilities[cls])}],
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(NAMES)},
            "task": "roi_classification", "model": Path(checkpoint).parent.name,
            "architecture": ckpt.get("architecture", ckpt.get("training_config", {}).get("architecture", "resnet18")),
            "preprocessing": preprocessing,
            "message": f"单液滴分类：{NAMES[cls]} · {float(probabilities[cls]):.1%}（需人工审核）"}
