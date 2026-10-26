"""Re-evaluate an existing checkpoint without training or changing annotations."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/mpl"))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("MPLBACKEND", "Agg")
NAMES = ["1cell 1bead", "only cell", "only beads", "multiplet", "empty"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def audit(dataset):
    snapshot = json.loads((dataset / "annotations.json").read_text())
    rows = {r["id"]: r for r in snapshot["images"]}
    active = [r for r in rows.values() if r.get("reviewed") and not r.get("deletedAt")]
    selected = {}
    hashes = defaultdict(list)
    counts = {}
    errors = []
    full_boxes = 0
    for split in ("train", "val"):
        images = sorted((dataset / "images" / split).glob("*"))
        labels = sorted((dataset / "labels" / split).glob("*.txt"))
        class_counts = Counter()
        if {p.stem for p in images} != {p.stem for p in labels}:
            errors.append(f"{split}: image/label stems differ")
        selected[split] = []
        for image in images:
            row = rows.get(image.stem)
            if not row or not row.get("reviewed") or row.get("deletedAt") or row["split"] != split:
                errors.append(f"Invalid snapshot row: {image.name}")
                continue
            selected[split].append(row)
            label = dataset / "labels" / split / (image.stem + ".txt")
            parsed = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            if len(parsed) != len(row["boxes"]):
                errors.append(f"Label count mismatch: {image.name}")
            for values, box in zip(parsed, row["boxes"]):
                expected = [box["cls"], box["x"] + box["w"] / 2, box["y"] + box["h"] / 2, box["w"], box["h"]]
                if len(values) != 5 or any(abs(a - b) > 1e-7 for a, b in zip(values, expected)):
                    errors.append(f"Label values mismatch: {image.name}")
                if not all(0 <= v <= 1 for v in values[1:]) or int(values[0]) not in range(5):
                    errors.append(f"Out-of-range label: {image.name}")
                class_counts[int(values[0])] += 1
                full_boxes += values[1:] == [.5, .5, 1., 1.]
            hashes[digest(image)].append((split, image.name))
        counts[split] = {"images": len(images), "labels": len(labels), "classes": {NAMES[i]: class_counts[i] for i in range(5)}}
    actual_ids = {r["id"] for values in selected.values() for r in values}
    exported_projects = {r["projectId"] for values in selected.values() for r in values}
    expected_ids = {r["id"] for r in active if r["split"] in ("train", "val") and r.get("projectId") in exported_projects}
    if actual_ids != expected_ids:
        errors.append("Exported images do not exactly cover reviewed active snapshot rows")
    sources = {s: {r.get("parentId") or r["id"] for r in selected[s]} for s in selected}
    shared = [v for v in hashes.values() if len({s for s, _ in v}) > 1]
    groups = {s: sorted({r.get("group", "") for r in selected[s]}) for s in selected}
    distances = []
    for group in set(groups["train"]) & set(groups["val"]):
        frames = {}
        for split in selected:
            frames[split] = sorted(int(m.group(1)) for r in selected[split] if r.get("group") == group
                                   if (m := re.search(r"frame_(\d+)", r["name"])))
        train = frames["train"]
        for frame in frames["val"]:
            pos = bisect_left(train, frame)
            neighbors = train[max(0, pos - 1):pos + 1]
            if neighbors:
                distances.append(min(abs(frame - t) for t in neighbors))
    result = {
        "dataset": str(dataset), "snapshot_sha256": digest(dataset / "annotations.json"),
        "current_annotations_match_snapshot": snapshot == json.loads((ROOT / "data/project.json").read_text()),
        "splits": counts, "full_image_boxes": full_boxes, "errors": errors,
        "shared_original_ids": sorted(sources["train"] & sources["val"]),
        "cross_split_exact_duplicate_groups": shared, "groups": groups,
        "video_val_frames_checked": len(distances),
        "video_val_frames_within_5_of_train": sum(d <= 5 for d in distances),
        "median_nearest_train_frame_distance": sorted(distances)[len(distances) // 2] if distances else None,
    }
    if errors or result["shared_original_ids"] or shared:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result, selected["val"]


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union else 0.


def workbench_evaluation(model, dataset, annotations, output, device, size, conf):
    import numpy as np
    ground_truth = {r["id"]: r for r in annotations}
    if not all(len(r["boxes"]) == 1 for r in annotations):
        raise ValueError("Per-image top-1 accuracy is only defined here for one annotated droplet per image")
    images = sorted((dataset / "images/val").glob("*"))
    matrix = np.zeros((5, 6), dtype=int)
    totals = Counter()
    records, raw = [], []
    results = model.predict(source=[str(p) for p in images], stream=True, batch=1, imgsz=size,
                            conf=conf, iou=.7, max_det=300, device=device, quantize=32,
                            rect=True, augment=False, verbose=False, save=False)
    for index, result in enumerate(results, 1):
        row = ground_truth[Path(result.path).stem]
        g = row["boxes"][0]
        actual = int(g["cls"])
        target = [g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]]
        detections = sorted([
            {"class": int(box.cls.item()), "confidence": float(box.conf.item()), "xyxyn": box.xyxyn[0].tolist()}
            for box in result.boxes
        ], key=lambda d: d["confidence"], reverse=True)
        top = detections[0] if detections else None
        predicted = top["class"] if top else 5
        overlap = iou(top["xyxyn"], target) if top else 0.
        class_correct = predicted == actual
        localized_correct = class_correct and overlap >= .5
        exact = localized_correct and len(detections) == 1
        matrix[actual, predicted] += 1
        # At a fixed threshold, each ground-truth box can match at most one prediction.
        tp = int(any(d["class"] == actual and iou(d["xyxyn"], target) >= .5 for d in detections))
        fp, fn = len(detections) - tp, 1 - tp
        totals.update(tp=tp, fp=fp, fn=fn, class_correct=int(class_correct),
                      localized_correct=int(localized_correct), exact_correct=int(exact),
                      no_detection=int(not detections), multiple_detections=int(len(detections) > 1))
        records.append({"id": row["id"], "source_name": row["name"], "image": result.path,
                        "true_class": NAMES[actual], "top1_class": NAMES[predicted] if predicted < 5 else "no detection",
                        "top1_confidence": top["confidence"] if top else 0., "top1_iou": overlap,
                        "class_correct": class_correct, "top1_class_and_iou_correct": localized_correct,
                        "exactly_one_correct_detection": exact, "detections": len(detections),
                        "tp": tp, "fp": fp, "fn": fn})
        raw.append({"id": row["id"], "ground_truth": g, "detections": detections})
        if index % 50 == 0 or index == len(images):
            print(f"Workbench policy evaluated: {index}/{len(images)}", flush=True)
    if len(records) != len(images) or {r["id"] for r in records} != set(ground_truth):
        raise ValueError("Prediction coverage differs from validation annotations")
    with (output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (output / "errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader()
        writer.writerows(r for r in records if not r["exactly_one_correct_detection"])
    write_json(output / "predictions.json", raw)
    per_class = []
    for c in range(5):
        correct = int(matrix[c, c]); count = int(matrix[c].sum()); predicted_count = int(matrix[:, c].sum())
        p = correct / predicted_count if predicted_count else 0.
        recall = correct / count if count else 0.
        per_class.append({"class": NAMES[c], "images": count, "correct": correct,
                          "top1_precision": p, "top1_recall": recall,
                          "top1_f1": 2*p*recall/(p+recall) if p+recall else 0.})
    n = len(images); tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    return {"images": n, "settings": {"imgsz": size, "conf": conf, "nms_iou": .7, "matching_iou": .5,
                                      "device": device, "batch": 1, "rect": True},
            "counts": dict(totals), "top1_accuracy": totals["class_correct"]/n,
            "top1_class_and_iou_accuracy": totals["localized_correct"]/n,
            "exact_image_accuracy": totals["exact_correct"]/n,
            "detection_precision": tp/(tp+fp) if tp+fp else 0., "detection_recall": tp/(tp+fn) if tp+fn else 0.,
            "detection_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "balanced_top1_accuracy": sum(c["top1_recall"] for c in per_class)/5,
            "macro_top1_f1": sum(c["top1_f1"] for c in per_class)/5,
            "majority_class_baseline": max(c["images"] for c in per_class)/n,
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist(), "per_class": per_class}, records


def plots(output, policy, records):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    matrix = np.array(policy["confusion_matrix_true_rows_predicted_columns"])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(6), xticklabels=NAMES+["no detection"], yticks=range(5), yticklabels=NAMES,
           xlabel="Predicted top-1 class", ylabel="Annotated class",
           title=f"Workbench top-1 accuracy: {policy['top1_accuracy']:.1%} ({policy['images']} images)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for y in range(5):
        for x in range(6):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center",
                    color="white" if matrix[y, x] > matrix.max()*.55 else "black")
    fig.tight_layout(); fig.savefig(output / "top1_confusion_matrix.png", dpi=180); plt.close(fig)
    failures = sorted((r for r in records if not r["exactly_one_correct_detection"]),
                      key=lambda r: r["top1_confidence"], reverse=True)[:12]
    if failures:
        fig, axes = plt.subplots(3, 4, figsize=(13, 12))
        for ax, row in zip(axes.flat, failures):
            with Image.open(row["image"]) as image:
                ax.imshow(image)
            ax.set_title(f"GT: {row['true_class']}\nTop: {row['top1_class']} ({row['top1_confidence']:.2f})\n{row['source_name']}", fontsize=9)
            ax.axis("off")
        for ax in list(axes.flat)[len(failures):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(output / "error_examples.png", dpi=140); plt.close(fig)


def report(output, summary):
    p, a, d = summary["workbench"], summary["audit"], summary["standard_detection"]
    pct = lambda x: f"{100*x:.2f}%"
    lines = ["# 已训练模型验证报告", "", f"评估时间：{summary['evaluated_at']}", "",
             f"模型：`{summary['model']['source']}`；提取文件：`{summary['model']['extracted']}`。",
             f"SHA-256：`{summary['model']['sha256']}`。", "",
             "## 实测结果", "", f"验证集共 {p['images']} 张图片，每张均有一个人工类别标注。",
             f"按工作台实际识别设置（1024、置信度0.25、CPU、每次1张）逐图预测：", "",
             f"- 最高置信度类别准确率：**{pct(p['top1_accuracy'])}**，正确 {p['counts']['class_correct']}/{p['images']} 张；漏检计错。",
             f"- 最高置信框同时类别正确且 IoU≥0.5：{pct(p['top1_class_and_iou_accuracy'])}。",
             f"- 整张预测完全正确（恰好一个预测框，类别正确且 IoU≥0.5）：{pct(p['exact_image_accuracy'])}。",
             f"- 固定阈值检测精确率/召回率/F1：{pct(p['detection_precision'])} / {pct(p['detection_recall'])} / {pct(p['detection_f1'])}。",
             f"- 漏检图片 {p['counts'].get('no_detection',0)} 张，多框图片 {p['counts'].get('multiple_detections',0)} 张。",
             f"- 类别平衡准确率：{pct(p['balanced_top1_accuracy'])}；宏平均F1：{pct(p['macro_top1_f1'])}。",
             f"- 总是预测验证集最多类别的基线：{pct(p['majority_class_baseline'])}。", ""]
    with (output / "per_image.csv").open(encoding="utf-8-sig") as f:
        predicted_counts = Counter(r["top1_class"] for r in csv.DictReader(f))
    if len(predicted_counts) == 1:
        only_class = next(iter(predicted_counts))
        lines += [f"**本次全部 {p['images']} 张图片均预测为 `{only_class}`，没有体现出有效的五类区分能力。**", ""]
        if abs(p['top1_accuracy'] - p['majority_class_baseline']) < 1e-12:
            lines += ["模型出现类别塌缩；这批验证数据上的准确率等于恒定预测多数类别的基线。", ""]
    lines += [
             "| 人工类别 | 图片数 | 最高置信类别正确数 | 类别命中率 | Top-1精确率 | Top-1 F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in p["per_class"]:
        lines.append(f"| {c['class']} | {c['images']} | {c['correct']} | {pct(c['top1_recall'])} | {pct(c['top1_precision'])} | {pct(c['top1_f1'])} |")
    lines += ["", "## 标准目标检测验证", "",
              "另用 Ultralytics val、原训练分辨率1024、batch=4、rect=True、conf=0.001、NMS IoU=0.7 验证全部385张图片。mAP是检测指标，不能当作逐图分类准确率；以下精确率/召回率取官方最大F1工作点。", "",
              f"- mAP@0.5：{pct(d['map50'])}；mAP@0.5:0.95：{pct(d['map50_95'])}。",
              f"- 宏平均精确率/召回率：{pct(d['precision'])} / {pct(d['recall'])}。", "",
              "标准验证采用低置信度阈值和多标签候选，当前每张图都保留了五个类别的候选框，因此这里的100%召回率不能解读为工作台五类都能识别；实际工作台使用单类别预测，结果以逐图评估为准。", "",
              "| 类别 | Precision | Recall | AP@0.5 | AP@0.5:0.95 |", "|---|---:|---:|---:|---:|"]
    for c in d["per_class"]:
        lines.append(f"| {c['class']} | {pct(c['precision'])} | {pct(c['recall'])} | {pct(c['ap50'])} | {pct(c['ap50_95'])} |")
    lines += ["", "## 数据核对与适用范围", "",
              f"训练集 {a['splits']['train']['images']} 张，验证集 {a['splits']['val']['images']} 张；图片/标注一一对应，YOLO标签与快照一致。当前工作台标注与训练快照一致：{a['current_annotations_match_snapshot']}。",
              "未发现跨集合完全相同的图片或同一原图的处理副本。",
              f"但两集合来自同一视频批次；{a['video_val_frames_within_5_of_train']}/{a['video_val_frames_checked']} 张验证帧距训练帧不超过5帧，中位距离 {a['median_nearest_train_frame_distance']} 帧。这会带来时间相关性与数据泄漏风险，结果不能视作新实验或新视频上的独立测试准确率。",
              f"训练与验证共 {a['full_image_boxes']} 个框覆盖整张处理图，此任务实际侧重单液滴分类；框的IoU只衡量与这些整图框的一致程度，不能证明精确的液滴定位能力。",
              "本次没有重新训练、重新划分数据或修改人工标注。若要测新实验泛化能力，须另留整段视频/独立实验为测试集；用当前旧模型重新划分已参与训练的数据仍不构成独立测试。", "",
              "## 输出文件", "", "- `summary.json`：结果、版本、参数及模型校验值。",
              "- `per_image.csv`：全部逐图预测与人工类别对照；`errors.csv`：整张预测未完全正确的图片。",
              "- `predictions.json`：工作台阈值下全部预测框及置信度。",
              "- `top1_confusion_matrix.png`：人工类别为行，最高置信预测为列。",
              "- `error_examples.png`：部分未完全正确图片及最高置信预测（若存在）。",
              "- `standard/`：官方验证的混淆矩阵、PR曲线及标注/预测样例。", "",
              "## 安装与工作台检查", "",
              f"项目独立训练环境：`.venv`；Ultralytics {summary['environment']['ultralytics']}，PyTorch {summary['environment']['torch']}。验证采用CPU以对应工作台实际识别设置。",
              "已安装版本保存在项目目录的 `training-environment.lock.txt`。"]
    check_path = output / "environment_check.json"
    if check_path.exists():
        check = json.loads(check_path.read_text())
        lines += [f"工作台已识别训练环境：{'是' if check['workbench_training_environment_installed'] else '否'}；已有模型的工作台预测调用成功。",
                  f"在本机正常权限下，Apple GPU/MPS可用：{'是' if check['mps_available'] else '否'}；GPU前向与反向计算检查：{'通过' if check['mps_forward_backward_success'] else '未通过'}。"]
    lines += ["",
              "复现：在项目目录执行 `.venv/bin/python evaluate_model.py`。", "",
              "指标参考：[Ultralytics官方验证文档](https://docs.ultralytics.com/modes/val/)。"]
    (output / "验证报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    import yaml
    settings.update({"sync": False})
    weights = args.weights or max((ROOT / "data/runs").glob("droplet_five_*/weights/best.pt"), key=lambda p:p.stat().st_mtime)
    weights = weights.resolve()
    if args.dataset:
        dataset = args.dataset.resolve()
    else:
        train_args = yaml.safe_load((weights.parent.parent / "args.yaml").read_text())
        snapshot_id = re.split(r"[/\\]", train_args["data"])[-2]
        dataset = ROOT / "data/exports" / snapshot_id
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    output = args.output or ROOT / "data/evaluations" / now.strftime("best_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    audit_result, val_rows = audit(dataset)
    write_json(output / "audit.json", audit_result)
    model = YOLO(str(weights))
    if list(model.names.values()) != NAMES or model.task != "detect":
        raise ValueError(f"Unexpected model classes/task: {model.names}, {model.task}")
    project_ids = {r["projectId"] for r in val_rows}
    if len(project_ids) != 1:
        raise ValueError("Validation snapshot includes multiple projects")
    package = ROOT / "models" / next(iter(project_ids))
    package.mkdir(parents=True, exist_ok=True)
    extracted = package / "best.pt"
    if extracted.exists() and digest(extracted) != digest(weights):
        raise FileExistsError(f"Different extracted model already exists: {extracted}")
    if not extracted.exists():
        shutil.copy2(weights, extracted)
    model_info = {"source": str(weights), "extracted": str(extracted), "sha256": digest(weights),
                  "classes": NAMES, "task": model.task, "checkpoint_version": model.ckpt.get("version"),
                  "checkpoint_date": model.ckpt.get("date"), "historical_metrics": model.ckpt.get("train_metrics")}
    write_json(package / "model_info.json", model_info)
    metrics = model.val(data=str(write_local_yaml(dataset, output, yaml)), imgsz=1024, batch=4,
                        device=args.device, workers=0, conf=.001, iou=.7, rect=True, quantize=32,
                        split="val", plots=True, save_json=False, save_txt=True, save_conf=True,
                        project=str(output), name="standard", exist_ok=False, verbose=True)
    standard = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class": []}
    for index, cls in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        standard["per_class"].append({"class": NAMES[int(cls)], "precision": float(p), "recall": float(r),
                                      "ap50": float(ap50), "ap50_95": float(ap)})
    write_json(output / "standard_metrics.json", standard)
    # Fresh model avoids validation overrides changing workbench prediction defaults.
    policy, records = workbench_evaluation(YOLO(str(extracted)), dataset, val_rows, output, args.device, 1024, .25)
    summary = {"evaluated_at": now.isoformat(), "model": model_info, "audit": audit_result,
               "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                               "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                               "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available()},
               "standard_detection": standard, "workbench": policy}
    write_json(output / "summary.json", summary)
    plots(output, policy, records)
    report(output, summary)
    (package / "使用说明.md").write_text(
        f"# 已提取的最佳模型\n\n来源：`{weights}`\n\nSHA-256：`{model_info['sha256']}`\n\n"
        "适用五类单液滴ROI；输入应使用训练时相同的裁剪与前处理。使用项目的 .venv 环境加载此 best.pt。\n\n"
        "```python\nfrom ultralytics import YOLO\nmodel = YOLO('best.pt')\n"
        "results = model.predict('processed_roi.png', imgsz=1024, conf=0.25, device='cpu')\n```\n\n"
        f"最近验证报告：`{output / '验证报告.md'}`\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "extracted_model": str(extracted),
                      "top1_accuracy": policy["top1_accuracy"], "exact_image_accuracy": policy["exact_image_accuracy"],
                      "map50": standard["map50"], "map50_95": standard["map50_95"]}, ensure_ascii=False, indent=2), flush=True)


def write_local_yaml(dataset, output, yaml):
    config = yaml.safe_load((dataset / "data.yaml").read_text())
    config["path"] = str(dataset.resolve())
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
