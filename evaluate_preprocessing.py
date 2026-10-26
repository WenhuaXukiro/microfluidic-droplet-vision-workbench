"""Compare three fixed enhancement presets against unchanged ROI classifier weights."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader

from image_preprocessing import PRESETS, PREPROCESS_VERSION
from roi_classifier import load_classifier
from train_roi_classifier import ROIDataset, evaluate, save_json, write_predictions

ROOT = Path(__file__).resolve().parent


def compare(checkpoint, dataset, device="mps", batch_size=16):
    checkpoint, dataset = Path(checkpoint).resolve(), Path(dataset).resolve()
    torch.set_num_threads(4)
    model, ckpt = load_classifier(checkpoint, device)
    if ckpt.get("preprocessing", ckpt.get("training_config", {}).get("preprocessing", "none")) != "none":
        raise ValueError("本对照实验要求输入未绑定增强的模型，防止重复处理")
    output = ROOT / "data/preprocessing_evaluations" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    config = {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "dataset": str(dataset), "device": device, "batch_size": batch_size,
              "preprocessing_version": PREPROCESS_VERSION, "presets": PRESETS,
              "selection": "accuracy and macro F1 strictly improve; balanced accuracy does not decrease; all classes have correct predictions",
              "weights_changed": False, "validation_used_for_selection": True}
    save_json(output / "config.json", config)
    results = {}
    print(json.dumps({"output": str(output), **config}, ensure_ascii=False), flush=True)
    for preset in PRESETS:
        data = ROIDataset(dataset, "val", ckpt["image_size"], preprocessing=preset)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics, probabilities, indices = evaluate(model, loader, device)
        directory = output / preset
        directory.mkdir()
        save_json(directory / "metrics.json", metrics)
        write_predictions(directory, data, probabilities, indices)
        results[preset] = metrics
        save_json(output / "progress.json", {"status": "running", "completed": list(results), "results": results})
        print(json.dumps({"preset": preset, "label": PRESETS[preset]["label"], "accuracy": metrics["accuracy"],
                          "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"]}, ensure_ascii=False), flush=True)
    baseline = results["none"]
    eligible = [p for p, m in results.items() if p != "none" and m["accuracy"] > baseline["accuracy"]
                and m["macro_f1"] > baseline["macro_f1"] and m["balanced_accuracy"] >= baseline["balanced_accuracy"]
                and all(c["correct"] > 0 for c in m["per_class"])]
    selected = max(eligible, key=lambda p: (results[p]["macro_f1"], results[p]["accuracy"])) if eligible else "none"
    summary = {"config": config, "results": results, "selected": selected,
               "activated": False, "cpu_verification_required_before_activation": selected != "none"}
    save_json(output / "summary.json", summary)
    save_json(output / "progress.json", {"status": "completed", "selected": selected})
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["preset", "label", "correct", "images", "accuracy", "macro_f1", "balanced_accuracy"])
        writer.writeheader()
        for p, m in results.items():
            writer.writerow({"preset": p, "label": PRESETS[p]["label"], **{k: m[k] for k in ("correct", "images", "accuracy", "macro_f1", "balanced_accuracy")}})
    print(json.dumps({"completed": str(output), "selected": selected}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    compare(args.checkpoint, args.dataset, args.device, args.batch)
