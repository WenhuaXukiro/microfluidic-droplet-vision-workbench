from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent

def active_model(project_id):
    manifest=ROOT/'models'/project_id/'active_model.json'
    if not manifest.exists(): return None
    config=json.loads(manifest.read_text())
    if config.get('project_id')!=project_id: raise RuntimeError('启用模型与当前项目不匹配')
    return config

def classification_dataset(dest,project_id):
    p=json.loads((dest/'annotations.json').read_text())
    rows=[r for r in p['images'] if r.get('projectId')==project_id and r.get('reviewed') and not r.get('deletedAt') and r.get('split') in ('train','val')]
    return bool(rows) and all(r.get('parentId') and len(r['boxes'])==1 for r in rows) and sum(
        r['boxes'][0]['w']*r['boxes'][0]['h']>=.95 for r in rows)/len(rows)>=.95

if __name__=='__main__':
    if sys.argv[1]=='enhance':
        from image_preprocessing import preview_payload
        try:
            print(json.dumps(preview_payload(json.load(sys.stdin))))
        except Exception as error:
            print(json.dumps({'error':str(error)}));sys.exit(1)
        sys.exit(0)
    import torch
    def device_for(value):
        if value!='auto': return value
        return '0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if sys.argv[1]=='train':
        dest=Path(sys.argv[2]);task=sys.argv[7] if len(sys.argv)>7 else 'auto'
        if task=='roi_classify' or (task=='auto' and classification_dataset(dest,sys.argv[6])):
            from train_roi_classifier import train
            architecture=sys.argv[8] if len(sys.argv)>8 else 'resnet18'
            preprocessing=sys.argv[9] if len(sys.argv)>9 else 'none'
            batch=int(sys.argv[10]) if len(sys.argv)>10 else None
            initial_checkpoint=sys.argv[11] if len(sys.argv)>11 and sys.argv[11] else None
            train(dest,sys.argv[6],batch_size=batch,epochs=int(sys.argv[3]),long_edge=int(sys.argv[4]),device=sys.argv[5],architecture=architecture,preprocessing=preprocessing,initial_checkpoint=initial_checkpoint)
        else:
            if len(sys.argv)>9 and sys.argv[9]!='none':raise ValueError('识别增强目前用于单液滴分类，请选择单液滴分类或保持原样')
            from ultralytics import YOLO
            config=dest/'local.yaml'
            config.write_text((dest/'data.yaml').read_text().replace('path: .','path: '+json.dumps(str(dest))))
            model=YOLO('yolo11n.pt')
            model.train(data=str(config),epochs=int(sys.argv[3]),imgsz=int(sys.argv[4]),device=device_for(sys.argv[5]),batch=2,workers=0,project=str(ROOT/'data'/'runs'),name='droplet_five_'+sys.argv[6],exist_ok=False,mosaic=0.0,fliplr=0.5,flipud=0.5,cls_pw=.5)
            checkpoint=Path(model.trainer.best).resolve()
            package=ROOT/'models'/sys.argv[6];package.mkdir(parents=True,exist_ok=True)
            manifest=package/'active_model.json'
            if manifest.exists():
                import shutil
                shutil.copy2(manifest,package/'previous_active_model.json')
            temporary=manifest.with_suffix('.tmp')
            temporary.write_text(json.dumps({'task':'detect','checkpoint':checkpoint.relative_to(ROOT).as_posix(),
                                             'project_id':sys.argv[6],'run':checkpoint.parent.parent.relative_to(ROOT).as_posix()}))
            temporary.replace(manifest)
    else:
        active=active_model(sys.argv[4])
        if active and active.get('task')=='roi_classification':
            from roi_classifier import predict_roi
            result=predict_roi(sys.argv[2],ROOT/active['checkpoint'])
            Path(sys.argv[3]).write_text(json.dumps(result))
            sys.exit(0)
        from ultralytics import YOLO
        if active and active.get('task')=='detect':
            best=ROOT/active['checkpoint']
        else:
            weights=list((ROOT/'data'/'runs').glob('droplet_five_'+sys.argv[4]+'*/weights/best.pt'))
            if not weights: raise RuntimeError('没有已训练模型，请先完成训练。')
            best=max(weights,key=lambda p:p.stat().st_mtime)
        model=YOLO(str(best))
        expected=['1cell 1bead','only cell','only beads','multiplet','empty']
        if list(model.names.values()) != expected: raise RuntimeError('模型类别与当前五类不匹配，请重新训练。')
        result=model.predict(sys.argv[2],imgsz=1024,conf=0.25,device='cpu',verbose=False)[0]
        boxes=[]
        for b in result.boxes:
            x,y,w,h=b.xywhn[0].tolist(); cls=int(b.cls.item())
            if cls not in (0,1,2,3,4): continue
            boxes.append(dict(x=x-w/2,y=y-h/2,w=w,h=h,cls=cls,confidence=float(b.conf.item())))
        Path(sys.argv[3]).write_text(json.dumps({'boxes':boxes,'model':best.parent.parent.name}))
