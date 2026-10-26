"""Fine-tune a full-ROI classifier and compare it with the existing workbench model."""
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from roi_classifier import NAMES, FORMATS, build_classifier, image_tensor, load_classifier
from image_preprocessing import PRESETS, PREPROCESS_VERSION, validate_preset


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ROIDataset(Dataset):
    def __init__(self, dataset, split, image_size, augment=False, preprocessing="none"):
        snapshot = json.loads((dataset / "annotations.json").read_text())
        rows = {r["id"]: r for r in snapshot["images"]}
        self.items = []
        self.image_size = image_size
        self.preprocessing = validate_preset(preprocessing)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(.5), transforms.RandomVerticalFlip(.5),
            transforms.ColorJitter(brightness=.10, contrast=.15),
        ]) if augment else None
        for image in sorted((dataset / "images" / split).glob("*")):
            row = rows[image.stem]
            if len(row["boxes"]) != 1 or not row.get("parentId"):
                raise ValueError("单液滴分类训练只接受每张一个已审核液滴的处理图")
            self.items.append((image, int(row["boxes"][0]["cls"]), row["name"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, cls, _ = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.augment:
                image = self.augment(image)
            x = image_tensor(image, self.image_size, self.preprocessing)
        return x, cls, index


def classification_metrics(targets, probabilities):
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((5, 5), dtype=np.int64)
    for actual, predicted in zip(targets, predictions):
        matrix[int(actual), int(predicted)] += 1
    tp = matrix.diagonal().astype(float)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(5), where=matrix.sum(1) != 0)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(5), where=matrix.sum(0) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=precision + recall != 0)
    return {"images": len(targets), "correct": int(tp.sum()), "accuracy": float(tp.sum()/len(targets)),
            "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
            "predicted_counts": {NAMES[i]: int(matrix.sum(0)[i]) for i in range(5)},
            "per_class": [{"class": NAMES[i], "images": int(matrix.sum(1)[i]), "correct": int(tp[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])} for i in range(5)]}


def evaluate(model, loader, device):
    model.eval()
    targets, probabilities, indices = [], [], []
    loss_sum = 0.
    with torch.inference_mode():
        for x, y, index in loader:
            logits = model(x.to(device))
            loss_sum += float(nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item())
            targets.extend(y.tolist()); probabilities.extend(logits.softmax(1).cpu().tolist()); indices.extend(index.tolist())
    probabilities = np.asarray(probabilities)
    result = classification_metrics(targets, probabilities)
    result["cross_entropy"] = loss_sum/len(targets)
    return result, probabilities, indices


def find_baseline(val_rows):
    expected={r['id']:NAMES[r['boxes'][0]['cls']] for r in val_rows}
    candidates=sorted((ROOT/'data/evaluations').glob('*/summary.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    project_ids={r['projectId'] for r in val_rows}
    if len(project_ids)==1:
        manifest=ROOT/'models'/next(iter(project_ids))/'active_model.json'
        if manifest.exists():
            active=json.loads(manifest.read_text());previous=ROOT/active['run']/'summary.json'
            if previous.exists():candidates.insert(0,previous)
    for path in candidates:
        csv_path=path.parent/'per_image.csv'
        if not csv_path.exists():continue
        baseline=json.loads(path.read_text())
        with csv_path.open(encoding='utf-8-sig') as f:
            labels={r['id']:r['true_class'] for r in csv.DictReader(f)}
        if labels!=expected:continue
        if 'workbench' in baseline:return baseline,str(path)
        if 'validation' in baseline:
            metrics=baseline['validation']
            workbench={'images':metrics['images'],'counts':{'class_correct':metrics['correct']},'top1_accuracy':metrics['accuracy'],
                       'balanced_top1_accuracy':metrics['balanced_accuracy'],'macro_top1_f1':metrics['macro_f1'],
                       'per_class':[{'class':c['class'],'top1_recall':c['recall']} for c in metrics['per_class']]}
            return {'workbench':workbench,'kind':'prior_classification','source':str(path),
                    'architecture':baseline.get('training_config',{}).get('architecture')},str(path)
    # New projects need a useful baseline even before a prior detector evaluation exists.
    targets=[r['boxes'][0]['cls'] for r in val_rows]
    majority=Counter(targets).most_common(1)[0][0]
    probabilities=np.zeros((len(targets),5));probabilities[:,majority]=1
    metrics=classification_metrics(targets,probabilities)
    workbench={'images':len(targets),'counts':{'class_correct':metrics['correct']},'top1_accuracy':metrics['accuracy'],
               'balanced_top1_accuracy':metrics['balanced_accuracy'],'macro_top1_f1':metrics['macro_f1'],
               'per_class':[{'class':c['class'],'top1_recall':c['recall']} for c in metrics['per_class']]}
    return {'workbench':workbench,'kind':'majority_class'},'validation majority-class baseline'


def write_predictions(output, dataset, probabilities, indices):
    rows = []
    for probability, index in zip(probabilities, indices):
        path, true_class, name = dataset.items[index]
        predicted = int(probability.argmax())
        rows.append({"id": path.stem, "source_name": name, "true_class": NAMES[true_class],
                     "predicted_class": NAMES[predicted], "confidence": float(probability[predicted]),
                     "correct": predicted == true_class,
                     **{f"probability_{i}": float(p) for i,p in enumerate(probability)}})
    for filename, selected in [("per_image.csv", rows), ("errors.csv", [r for r in rows if not r["correct"]])]:
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(selected)


def create_report(output, summary, history):
    import matplotlib.pyplot as plt
    new, old = summary["validation"], summary["baseline"]["workbench"]
    pct = lambda value: f"{100*value:.2f}%"
    baseline_label='多数类基线' if summary['baseline'].get('kind')=='majority_class' else '旧模型'
    architecture=summary['training_config'].get('architecture','resnet18')
    model_label={'resnet18':'ResNet18','convnext_tiny':'ConvNeXt-Tiny'}[architecture]
    if summary['baseline'].get('architecture'):
        baseline_label={'resnet18':'ResNet18（现用基线）','convnext_tiny':'ConvNeXt-Tiny（现用基线）'}.get(summary['baseline']['architecture'],'旧模型')
    lines = ["# 五类单液滴重新训练结果", "", f"时间：{summary['completed_at']}", "",
             f"本次使用 {model_label} 预训练网络微调，输入整张单液滴ROI、等比例缩放并补边，输入尺寸 {summary['image_size'][1]}×{summary['image_size'][0]}。训练时保留全部图像内容；使用轻微亮度/对比度变化、水平/垂直翻转及按训练类别频数平方根倒数设置的损失权重。", "",
             f"识别增强：{PRESETS[summary['training_config'].get('preprocessing','none')]['label']}；方案随权重保存，验证和工作台预测自动使用同一实现，处理图不重复生成。", "",
             f"训练集 {summary['audit']['splits']['train']['images']} 张；验证集 {new['images']} 张。对比采用完全相同的验证图片及类别，未改变人工标注和训练/验证划分。", "",
             f"| 指标 | {baseline_label} | {model_label} 候选 |", "|---|---:|---:|",
             f"| 类别准确率 | {pct(old['top1_accuracy'])} | **{pct(new['accuracy'])}** |",
             f"| 正确图片 | {old['counts']['class_correct']}/{old['images']} | {new['correct']}/{new['images']} |",
             f"| 类别平衡准确率 | {pct(old['balanced_top1_accuracy'])} | {pct(new['balanced_accuracy'])} |",
             f"| 宏平均F1 | {pct(old['macro_top1_f1'])} | {pct(new['macro_f1'])} |", "",
             f"准确率变化 {(new['accuracy']-old['top1_accuracy'])*100:+.2f} 个百分点。", "",
             "| 人工类别 | 验证图片 | 旧模型命中率 | 新模型正确数 | 新精确率 | 新召回率 | 新F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for i,c in enumerate(new["per_class"]):
        lines.append(f"| {c['class']} | {c['images']} | {pct(old['per_class'][i]['top1_recall'])} | {c['correct']} | {pct(c['precision'])} | {pct(c['recall'])} | {pct(c['f1'])} |")
    if architecture=='convnext_tiny':
        config=summary['training_config']
        lines += ["", "ConvNeXt训练配置：先训练分类头2轮，再固定前两个特征阶段、微调后两个阶段和分类头；保留官方预训练权重。",
                  f"物理批量{config['batch_size']}，梯度累积{config['gradient_accumulation_steps']}次，有效批量{config['effective_batch_size']}。随机种子、输入图像范围、训练/验证划分与基线一致。"]
    if summary["training_config"].get("initial_checkpoint"):
        lines += ["", f"本次基于当前项目已有分类权重继续训练，未重新初始化分类头。初始权重SHA-256：`{summary['training_config']['initial_checkpoint_sha256']}`。",
                  f"学习率：特征网络{summary['training_config']['backbone_lr']}，分类头{summary['training_config']['head_lr']}。旧模型也在完全相同的当前验证图上重新评估。"]
    lines += ["", "## 如何使用", "",
              f"已{'启用' if summary['activated'] else '保留为候选，未启用'}：`{summary['model']}`。在工作台选择当前项目的一张**单液滴处理图**，点击“使用已训练模型识别”。分类模型返回整张处理图的一个类别；预测草稿仍需人工审核。多液滴原图应使用目标检测模型。", "",
              "## 评估限制", "",
              f"这{new['images']}张图是训练过程中用于选择最佳轮次的验证集，不是独立测试集。训练与验证来自同一视频；审计中有{summary['audit']['video_val_frames_within_5_of_train']}张验证帧与训练帧间隔不超过5帧。分数不能直接代表新视频/新实验的识别率，应另采集独立视频或实验作测试。验证图片不用于梯度更新。",
              "新模型针对单液滴ROI分类，未训练目标定位；不报告或对比检测mAP。新的整图框只是分类结果在工作台中的呈现形式。", "",
              "## 可复现记录", "", f"最佳轮次：{summary['best_epoch']}；模型 SHA-256：`{summary['sha256']}`。",
              "`training_config.json`、`history.csv`、`summary.json` 记录数据、参数、每轮结果及环境；`per_image.csv`、`errors.csv` 记录逐图结果；`confusion_matrix.png` 和 `learning_curves.png` 提供图示。",
              "最终结果用CPU重新加载落盘权重评估，输入预处理与工作台预测共用实现。",
              "参考：[PyTorch官方迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)。"]
    (output / "重新训练报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    matrix = np.array(new["confusion_matrix"])
    fig,ax=plt.subplots(figsize=(8,7));ax.imshow(matrix,cmap="Blues")
    ax.set(xticks=range(5),xticklabels=NAMES,yticks=range(5),yticklabels=NAMES,
           xlabel="Predicted class",ylabel="Annotated class",title=f"Retrained ROI classifier: {new['accuracy']:.1%} ({new['correct']}/{new['images']})")
    plt.setp(ax.get_xticklabels(),rotation=25,ha="right")
    for y in range(5):
        for x in range(5):ax.text(x,y,str(matrix[y,x]),ha="center",va="center",color="white" if matrix[y,x]>matrix.max()*.55 else "black")
    fig.tight_layout();fig.savefig(output/"confusion_matrix.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key in ("accuracy","balanced_accuracy","macro_f1"):
        axes[0].plot([h["epoch"] for h in history],[h[key] for h in history],label=key)
    axes[0].axhline(old["top1_accuracy"],color="gray",linestyle="--",label="old accuracy")
    axes[0].set(xlabel="Epoch",ylabel="Validation score",ylim=(0,1));axes[0].legend()
    axes[1].plot([h["epoch"] for h in history],[h["train_loss"] for h in history],label="weighted training loss")
    axes[1].plot([h["epoch"] for h in history],[h["val_loss"] for h in history],label="validation loss")
    axes[1].set(xlabel="Epoch",ylabel="Cross entropy");axes[1].legend()
    fig.tight_layout();fig.savefig(output/"learning_curves.png",dpi=160);plt.close(fig)


def train(dataset, project_id, epochs=30, long_edge=512, device="auto", batch_size=None, patience=None, architecture="resnet18", preprocessing="none", initial_checkpoint=None):
    from evaluate_model import audit
    dataset = Path(dataset).resolve()
    preprocessing = validate_preset(preprocessing)
    if architecture not in FORMATS:raise ValueError('不支持的分类网络')
    batch_size=batch_size or 16
    patience=patience or (6 if architecture=='convnext_tiny' else 8)
    accumulation=max(1,16//batch_size)
    head_prefix='classifier.' if architecture=='convnext_tiny' else 'fc.'
    frozen_prefixes=tuple(f'features.{i}.' for i in range(4)) if architecture=='convnext_tiny' else ()
    if device == "auto":device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "0":device = "cuda:0"
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(8)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    tag='_convnext_tiny' if architecture=='convnext_tiny' else ''
    output = ROOT / "data/classifier_runs" / (project_id + tag + now.strftime("_%Y%m%d_%H%M%S"))
    output.mkdir(parents=True,exist_ok=False)
    audit_result, val_rows = audit(dataset)
    if {r["projectId"] for r in val_rows} != {project_id}:raise ValueError("训练快照项目不匹配")
    baseline,baseline_path = find_baseline(val_rows)
    image_size = [int(long_edge),int(math.ceil(long_edge*.5625/32)*32)]
    train_data=ROIDataset(dataset,"train",image_size,True,preprocessing);val_data=ROIDataset(dataset,"val",image_size,preprocessing=preprocessing)
    loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
    val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=False,num_workers=0)
    counts=np.bincount([c for _,c,_ in train_data.items],minlength=5)
    if not counts.all():raise ValueError("训练集必须覆盖全部五类")
    weights=counts.astype(float)**(-.5);weights/=weights.mean()
    initial_checkpoint=Path(initial_checkpoint).resolve() if initial_checkpoint else None
    backbone_lr,head_lr=(.00005,.0002) if initial_checkpoint else (.0001,.001)
    pretrained_name='ConvNeXt_Tiny_Weights.IMAGENET1K_V1' if architecture=='convnext_tiny' else 'ResNet18_Weights.IMAGENET1K_V1'
    config={"architecture":architecture,"pretrained":pretrained_name,"seed":42,"dataset":str(dataset),
            "preprocessing":preprocessing,"preprocessing_version":PREPROCESS_VERSION,
            "project_id":project_id,"image_size":image_size,"epochs":epochs,"batch_size":batch_size,"device":device,
            "patience":patience,"selection":"macro_f1, tie-break accuracy","warmup_head_epochs":2,
            "gradient_accumulation_steps":accumulation,"effective_batch_size":batch_size*accumulation,
            "frozen_feature_stages":list(range(4)) if frozen_prefixes else [],
            "class_counts":counts.tolist(),"class_weights":weights.tolist(),"label_smoothing":.02,
            "backbone_lr":backbone_lr,"head_lr":head_lr,"weight_decay":.0001,
            "initial_checkpoint":str(initial_checkpoint) if initial_checkpoint else None,
            "initial_checkpoint_sha256":sha256(initial_checkpoint) if initial_checkpoint else None,
            "augmentation":{"horizontal_flip":.5,"vertical_flip":.5,"brightness":.10,"contrast":.15,"crop":False},
            "torch":str(torch.__version__),"torchvision":__import__('torchvision').__version__,"baseline":str(baseline_path)}
    save_json(output/"training_config.json",config);save_json(output/"audit.json",audit_result)
    if initial_checkpoint:
        model,initial=load_classifier(initial_checkpoint,'cpu')
        if initial.get('project_id')!=project_id or initial.get('architecture')!=architecture:
            raise ValueError('继续训练的模型必须属于当前项目且使用相同网络')
        if initial.get('image_size')!=image_size or initial.get('preprocessing','none')!=preprocessing:
            raise ValueError('继续训练须保持原模型输入尺寸与识别增强方案一致')
    else:
        filename='convnext_tiny-983f1562.pth' if architecture=='convnext_tiny' else 'resnet18-f37072fd.pth'
        pretrained=ROOT/'weights'/filename
        if not sha256(pretrained).startswith(filename.split('-')[-1].split('.')[0]):raise ValueError("预训练权重校验失败")
        model=build_classifier(architecture,num_classes=1000)
        model.load_state_dict(torch.load(pretrained,map_location="cpu",weights_only=True))
        if architecture=='convnext_tiny':
            model.classifier[2]=nn.Linear(model.classifier[2].in_features,5)
        else:
            model.fc=nn.Linear(model.fc.in_features,5)
    for name,param in model.named_parameters():
        if frozen_prefixes and name.startswith(frozen_prefixes):param.requires_grad_(False)
    model.to(device)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=.02)
    optimizer=torch.optim.AdamW([{"params":[p for name,p in model.named_parameters() if not name.startswith(head_prefix) and p.requires_grad],"lr":backbone_lr},
                                {"params":[p for name,p in model.named_parameters() if name.startswith(head_prefix)],"lr":head_lr}],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,epochs),eta_min=.000001)
    best_score=(-1.,-1.);best_epoch=0;history=[]
    print(json.dumps({"run":str(output),"device":device,"training_images":len(train_data),"validation_images":len(val_data),"class_weights":weights.tolist()},ensure_ascii=False),flush=True)
    for epoch in range(1,epochs+1):
        started=time.monotonic();model.train();head_only=epoch<=2
        for name,param in model.named_parameters():
            param.requires_grad_(name.startswith(head_prefix) or (not head_only and not (frozen_prefixes and name.startswith(frozen_prefixes))))
        if head_only:
            model.eval();(model.classifier if architecture=='convnext_tiny' else model.fc).train()
        elif frozen_prefixes:
            for block in list(model.features)[:4]:block.eval()
        train_loss=0.;seen=0;train_correct=0
        optimizer.zero_grad(set_to_none=True)
        for step,(x,y,_) in enumerate(loader,1):
            x=x.to(device);y=y.to(device)
            logits=model(x);loss=criterion(logits,y)
            group_start=((step-1)//accumulation)*accumulation
            group_size=min(accumulation,len(loader)-group_start)
            (loss/group_size).backward()
            if step%accumulation==0 or step==len(loader):
                nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            train_loss+=float(loss.item())*len(y);train_correct+=int((logits.argmax(1)==y).sum().item());seen+=len(y)
            if step%40==0:print(f"Epoch {epoch}/{epochs}: batch {step}/{len(loader)}, train loss {train_loss/seen:.4f}",flush=True)
        scheduler.step();metrics,probabilities,indices=evaluate(model,val_loader,device)
        row={"epoch":epoch,"seconds":round(time.monotonic()-started,2),"train_loss":train_loss/seen,"train_accuracy":train_correct/seen,
             "val_loss":metrics["cross_entropy"],"accuracy":metrics["accuracy"],"balanced_accuracy":metrics["balanced_accuracy"],"macro_f1":metrics["macro_f1"]}
        history.append(row)
        with (output/"history.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        checkpoint={"format":FORMATS[architecture],"architecture":architecture,"preprocessing":preprocessing,"preprocessing_version":PREPROCESS_VERSION,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "names":NAMES,"image_size":image_size,"project_id":project_id,"epoch":epoch,"validation":metrics,"training_config":config}
        score=(metrics["macro_f1"],metrics["accuracy"])
        if score>best_score:
            best_score=score;best_epoch=epoch;torch.save(checkpoint,output/"best.tmp");(output/"best.tmp").replace(output/"best.pt")
        torch.save(checkpoint,output/"last.tmp");(output/"last.tmp").replace(output/"last.pt")
        save_json(output/"progress.json",{"status":"running","epoch":epoch,"best_epoch":best_epoch,"latest":row,"best_macro_f1":best_score[0]})
        print(f"Epoch {epoch}/{epochs} ({row['seconds']:.1f}s): accuracy={metrics['accuracy']:.2%}, balanced={metrics['balanced_accuracy']:.2%}, macro F1={metrics['macro_f1']:.2%}; best epoch={best_epoch}",flush=True)
        if epoch>=10 and epoch-best_epoch>=patience:
            print(f"Early stopping after {patience} epochs without macro-F1 improvement",flush=True);break
    del model,optimizer
    if device=="mps":torch.mps.empty_cache()
    print('训练轮次结束，正在CPU重新加载最佳权重并复核完整验证集。',flush=True)
    best_model,ckpt=load_classifier(output/"best.pt","cpu")
    metrics,probabilities,indices=evaluate(best_model,val_loader,"cpu")
    write_predictions(output,val_data,probabilities,indices)
    improved=(metrics["accuracy"]>baseline["workbench"]["top1_accuracy"] and
              metrics["macro_f1"]>baseline["workbench"]["macro_top1_f1"] and all(c["correct"]>0 for c in metrics["per_class"]))
    package=ROOT/"models"/project_id;package.mkdir(parents=True,exist_ok=True)
    exported=package/"classifier_best.pt"
    if improved:
        shutil.copy2(output/"best.pt",exported.with_suffix(".tmp"));exported.with_suffix(".tmp").replace(exported)
        manifest={"task":"roi_classification","architecture":architecture,"preprocessing":preprocessing,"preprocessing_version":PREPROCESS_VERSION,"checkpoint":(output/'best.pt').relative_to(ROOT).as_posix(),"project_id":project_id,
                  "names":NAMES,"sha256":sha256(exported),"run":output.relative_to(ROOT).as_posix(),"validation":metrics}
        existing=package/"active_model.json"
        if existing.exists():shutil.copy2(existing,package/"previous_active_model.json")
        save_json(existing,manifest)
    summary={"completed_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"validation":metrics,"best_epoch":best_epoch,
             "epochs_run":len(history),"image_size":image_size,"training_config":config,"audit":audit_result,"baseline":baseline,
             "activated":improved,"model":str(exported if improved else output/"best.pt"),"sha256":sha256(output/"best.pt")}
    save_json(output/"summary.json",summary);create_report(output,summary,history)
    save_json(output/"progress.json",{"status":"completed","epoch":len(history),"best_epoch":best_epoch,"validation":metrics,"activated":improved})
    print(json.dumps({"completed":str(output),"accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"],"activated":improved},ensure_ascii=False),flush=True)
    return summary


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True);parser.add_argument("--project-id",required=True)
    parser.add_argument("--epochs",type=int,default=30);parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--device",default="auto");parser.add_argument("--batch",type=int)
    parser.add_argument("--architecture",choices=list(FORMATS),default='resnet18');parser.add_argument('--patience',type=int)
    parser.add_argument('--preprocessing',choices=list(PRESETS),default='none')
    parser.add_argument('--initial-checkpoint',type=Path)
    a=parser.parse_args();train(a.dataset,a.project_id,a.epochs,a.size,a.device,a.batch,a.patience,a.architecture,a.preprocessing,a.initial_checkpoint)
