"""Optionally re-estimate BatchNorm from training images, then validate and publish."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import update_bn
from roi_classifier import load_classifier
from train_roi_classifier import ROOT, ROIDataset, evaluate, write_predictions, create_report, save_json, sha256


def recalibrate(output, device):
    output=Path(output).resolve()
    summary=json.loads((output/'summary.json').read_text())
    if not summary['activated']:
        raise ValueError('训练候选未超过原模型，暂不启用进一步校准')
    config=summary['training_config'];project_id=config['project_id'];dataset=Path(config['dataset'])
    train_data=ROIDataset(dataset,'train',summary['image_size'],False)
    val_data=ROIDataset(dataset,'val',summary['image_size'],False)
    train_loader=DataLoader(train_data,batch_size=16,shuffle=False,num_workers=0)
    val_loader=DataLoader(val_data,batch_size=16,shuffle=False,num_workers=0)
    model,checkpoint=load_classifier(output/'best.pt',device)
    print(f'Re-estimating BatchNorm from {len(train_data)} training images only',flush=True)
    update_bn(train_loader,model,device=device)
    model.to('cpu').eval()
    if device=='mps':torch.mps.empty_cache()
    metrics,probabilities,indices=evaluate(model,val_loader,'cpu')
    previous=summary['validation']
    accepted=(metrics['accuracy']>=previous['accuracy'] and metrics['balanced_accuracy']>=previous['balanced_accuracy'] and
              metrics['macro_f1']>=previous['macro_f1'] and all(c['correct']>0 for c in metrics['per_class']) and
              (metrics['accuracy']>previous['accuracy'] or metrics['macro_f1']>previous['macro_f1']))
    comparison={'method':'BatchNorm statistics recomputed using unaugmented training images only; no gradient updates',
                'training_images':len(train_data),'before':previous,'after':metrics,'accepted':accepted}
    save_json(output/'batchnorm_comparison.json',comparison)
    chosen=output/'best.pt'
    if accepted:
        checkpoint['state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        checkpoint['validation']=metrics;checkpoint['batchnorm_recalibration']=comparison['method']
        chosen=output/'best_calibrated.pt'
        torch.save(checkpoint,chosen.with_suffix('.tmp'));chosen.with_suffix('.tmp').replace(chosen)
        summary['validation']=metrics;summary['sha256']=sha256(chosen)
        summary['post_training']=comparison
        exported=Path(summary['model'])
        shutil.copy2(chosen,exported.with_suffix('.tmp'));exported.with_suffix('.tmp').replace(exported)
        write_predictions(output,val_data,probabilities,indices)
    else:
        summary['post_training']={'method':comparison['method'],'accepted':False}
    save_json(output/'summary.json',summary)
    manifest_path=ROOT/'models'/project_id/'active_model.json'
    manifest=json.loads(manifest_path.read_text())
    manifest.update(checkpoint=str(chosen.relative_to(ROOT)),sha256=sha256(chosen),validation=summary['validation'])
    save_json(manifest_path,manifest)
    progress=json.loads((output/'progress.json').read_text())
    progress['validation']=summary['validation'];progress['batchnorm_recalibration_accepted']=accepted
    save_json(output/'progress.json',progress)
    with (output/'history.csv').open() as f:
        history=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    create_report(output,summary,history)
    with (output/'重新训练报告.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## 训练后统计检查\n\n使用未经增强的{len(train_data)}张训练图片重新估计BatchNorm统计，不使用验证图进行参数或统计更新。')
        f.write('重新估计后的模型在准确率、类别平衡准确率和宏平均F1上均不下降，已采用。\n' if accepted else '重估版本未同时改善准确率和各类别综合表现，最终采用原最佳轮次权重。\n')
    print(json.dumps({'accepted':accepted,'before_accuracy':previous['accuracy'],'after_accuracy':metrics['accuracy'],
                      'final_accuracy':summary['validation']['accuracy'],'checkpoint':str(chosen)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--device',default='mps')
    a=parser.parse_args();recalibrate(a.run,a.device)
