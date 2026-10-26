"""Read-only GPU training smoke check for the existing ConvNeXt model."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torchvision
from PIL import Image
from roi_classifier import load_classifier, image_tensor

root = Path(__file__).resolve().parent
assert torch.cuda.is_available(), "CUDA is unavailable"
torch.set_num_threads(8)
manifest = json.loads(next((root / "models").glob("*/active_model.json")).read_text(encoding="utf-8"))
model, checkpoint = load_classifier(root / manifest["checkpoint"], "cuda")
database = json.loads((root / "data/project.json").read_text(encoding="utf-8"))
row = next(r for r in database["images"] if r.get("projectId") == manifest["project_id"] and r.get("reviewed") and r.get("parentId") and not r.get("deletedAt"))
with Image.open(root / "data/images" / row["file"]) as image:
    batch = image_tensor(image, checkpoint["image_size"], checkpoint.get("preprocessing", "none")).unsqueeze(0).cuda()
with torch.inference_mode():
    prediction = model(batch).softmax(1)
assert torch.isfinite(prediction).all()
for name, parameter in model.named_parameters():
    parameter.requires_grad_(not name.startswith(tuple(f"features.{i}." for i in range(4))))
model.train()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0001)
loss = torch.nn.functional.cross_entropy(model(batch), torch.tensor([row["boxes"][0]["cls"]], device="cuda"))
loss.backward()
assert torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
torch.cuda.synchronize()
print(json.dumps({"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "architecture": checkpoint.get("architecture"), "image_size": checkpoint["image_size"], "loss": loss.item(), "peak_gpu_mb": round(torch.cuda.max_memory_allocated()/1024**2), "forward_backward_optimizer": "passed", "checkpoint_modified": False}, ensure_ascii=False, indent=2))
