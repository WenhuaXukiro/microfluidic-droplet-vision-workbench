from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import time, random, mimetypes
import json, base64, uuid, shutil, zipfile, io, subprocess, sys, threading, importlib.util, math
from image_preprocessing import PRESETS, validate_preset
ROOT=Path(__file__).resolve().parent
import os
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"data"/"config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"data"/"mpl"))
(ROOT/"data"/"config").mkdir(parents=True,exist_ok=True)
(ROOT/"data"/"mpl").mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data'; DB=DATA/'project.json'
VENV_PYTHON=ROOT/('.venv/Scripts/python.exe' if os.name=='nt' else '.venv/bin/python')
TRAIN_PYTHON=str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
os.environ.setdefault('PYTHONUTF8','1')
def installed():
    return (ROOT/'.venv/Lib/site-packages/ultralytics/__init__.py').exists() or any((ROOT/'.venv/lib').glob('python*/site-packages/ultralytics/__init__.py')) or importlib.util.find_spec('ultralytics') is not None
def active_models():
    models={}
    for manifest in (ROOT/'models').glob('*/active_model.json'):
        try:
            config=json.loads(manifest.read_text())
            preset=config.get('preprocessing','none')
            models[config['project_id']]={'task':config['task'],'architecture':config.get('architecture','resnet18'),'accuracy':config.get('validation',{}).get('accuracy'),
                                          'preprocessing':preset,'preprocessing_label':PRESETS.get(preset,{}).get('label',preset)}
        except (ValueError,KeyError,OSError):
            continue
    return models
LOCK=threading.RLock(); JOB={'status':'idle'}
def read():
    p=json.loads(DB.read_text()) if DB.exists() else {'images':[]}
    p.setdefault('projects',[])
    mapping={}
    for r in p['images']:
        if not r.get('projectId'):
            name=r.get('group') or '之前导入的图片'
            if name not in mapping:
                existing=next((q for q in p['projects'] if q['name']==name),None)
                if not existing:
                    existing={'id':uuid.uuid4().hex,'name':name};p['projects'].append(existing)
                mapping[name]=existing['id']
            r['projectId']=mapping[name]
    if p['images'] and any('projectId' not in r for r in json.loads(DB.read_text())['images']): write(p)
    return p
def write(p):
    t=DB.with_suffix('.tmp'); t.write_text(json.dumps(p,ensure_ascii=False)); t.replace(DB)
def auto_split(p, project_id, ratio=0.2, mode='batch'):
    if JOB['status']=='running': raise ValueError('请等待训练结束后划分集合')
    config=next((q for q in p['projects'] if q['id']==project_id),None)
    if config is None: raise ValueError('请先选择项目')
    if isinstance(ratio,bool) or not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or not 0.05<=ratio<=0.5:
        raise ValueError('验证集比例应为 5%–50%')
    if mode not in ('batch','source'): raise ValueError('划分方式无效')
    rows=[r for r in p['images'] if r.get('projectId')==project_id]
    parents={r['id']:r['id'] for r in rows}
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(a,b): parents[root(a)]=root(b)
    batches={}; sources={}
    for r in rows:
        if r.get('parentId') in parents: union(r['id'],r['parentId'])
        if r.get('parentId'):
            source=r['parentId']
            if source in sources: union(r['id'],sources[source])
            else: sources[source]=r['id']
        g=r.get('group','').strip()
        if mode=='batch' and g:
            if g in batches: union(r['id'],batches[g])
            else: batches[g]=r['id']
    blocks={}
    for r in rows: blocks.setdefault(root(r['id']),[]).append(r)
    candidates=[];test_count=0
    for key,block in blocks.items():
        reviewed=[r for r in block if r.get('reviewed') and not r.get('deletedAt')]
        if not reviewed: continue
        if any(not r.get('group','').strip() for r in reviewed): raise ValueError('请先为已审核图片填写实验批次')
        if any(r.get('split')=='test' for r in block):
            if any(r.get('split')!='test' for r in block): raise ValueError('测试集与其他集合存在同批次或同源图片，请先统一其集合')
            test_count+=len(reviewed);continue
        candidates.append((key,len(reviewed)))
    if len(candidates)<2:
        raise ValueError('至少需要两个含已审核图片的独立'+('实验批次；当前批次不足。独立图片可选择按原图随机划分，同一视频帧请勿随机拆分。' if mode=='batch' else '原图，测试集不参与划分。'))
    candidates.sort()
    random.Random(42).shuffle(candidates)
    total=sum(n for _,n in candidates);target=total*ratio
    # Choose the closest nonempty prefix, always leaving a training block.
    count=0;best=None
    for i,(_,n) in enumerate(candidates[:-1],1):
        count+=n
        score=abs(count-target)
        if best is None or score<best[0]: best=(score,i,count)
    val_keys={key for key,_ in candidates[:best[1]]}
    for key,_ in candidates:
        for r in blocks[key]: r['split']='val' if key in val_keys else 'train'
    config['datasetSplit']={'mode':mode,'validationRatio':ratio,'seed':42}
    return {'train':total-best[2],'val':best[2],'test':test_count,'mode':mode}

def export_dataset(project_id=None):
    p=read(); rows=[r for r in p['images'] if r.get('reviewed') and not r.get('deletedAt') and (project_id is None or r.get('projectId')==project_id)]
    splits={r['id']:r['split'] for r in p['images'] if 'id' in r}
    source_splits={}
    for r in rows:
        if r.get('parentId') in splits and splits[r['parentId']]!=r['split']:
            raise ValueError('处理图必须与原图使用相同的数据集合：'+r['name'])
        if r.get('parentId'):
            source=(r.get('projectId'),r['parentId'])
            if source in source_splits and source_splits[source]!=r['split']:
                raise ValueError('同源处理图必须使用相同的数据集合：'+r['name'])
            source_splits[source]=r['split']
    if not any(r['split']=='train' for r in rows) or not any(r['split']=='val' for r in rows):
        raise ValueError('训练集和验证集各需至少一张已审核图片。请按实验批次分组。')
    groups={}
    for r in rows:
        g=r.get('group','').strip()
        if not g: raise ValueError('请为每张已审核图片填写实验批次。')
        config=next((q for q in p['projects'] if q['id']==r.get('projectId')), {})
        key=(r.get('projectId'),g)
        if config.get('datasetSplit',{}).get('mode')!='source':
            if key in groups and groups[key]!=r['split']: raise ValueError('同一实验批次不能分到不同集合：'+g)
            groups[key]=r['split']
    dest=DATA/'exports'/uuid.uuid4().hex; dest.mkdir(parents=True)
    for r in rows:
        split=r['split']; (dest/'images'/split).mkdir(parents=True,exist_ok=True); (dest/'labels'/split).mkdir(parents=True,exist_ok=True)
        shutil.copy2(DATA/'images'/r['file'],dest/'images'/split/r['file'])
        lines=[]
        for b in r['boxes']:
            x,y,w,h=b['x'],b['y'],b['w'],b['h']
            lines.append(f"{b['cls']} {x+w/2:.8f} {y+h/2:.8f} {w:.8f} {h:.8f}")
        (dest/'labels'/split/(Path(r['file']).stem+'.txt')).write_text('\n'.join(lines))
    (dest/'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: 1cell 1bead\n  1: only cell\n  2: only beads\n  3: multiplet\n  4: empty\n')
    (dest/'annotations.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
    return dest
def remove_processed_originals(p, project_id):
    children={r.get('parentId') for r in p['images'] if r.get('projectId')==project_id and r.get('parentId') and not r.get('deletedAt') and (DATA/'images'/r['file']).is_file()}
    count=0
    for r in p['images']:
        if r.get('projectId')==project_id and not r.get('parentId') and not r.get('deletedAt') and r['id'] in children:
            r['deletedAt']=time.time();count+=1
    return count
def purge_images(p, project_id, ids):
    """Permanently remove only explicitly selected recycled images."""
    if JOB['status']=='running': raise ValueError('请等待训练完成后永久删除')
    if not isinstance(ids,list) or not ids or any(not isinstance(i,str) for i in ids):
        raise ValueError('请选择回收区中的图片')
    ids=set(ids)
    allowed={r['id'] for r in p['images'] if r.get('projectId')==project_id and r.get('deletedAt')}
    if not ids<=allowed: raise ValueError('只能永久删除当前项目回收区中的图片')
    image_root=(DATA/'images').resolve()
    targets=[]
    remaining=[r for r in p['images'] if r['id'] not in ids]
    retained_files={r['file'] for r in remaining}
    for r in p['images']:
        if r['id'] not in ids or r['file'] in retained_files: continue
        target=(image_root/r['file']).resolve()
        if not target.is_relative_to(image_root): raise ValueError('图片路径无效')
        if target.exists() and not target.is_file(): raise ValueError('图片路径不是文件')
        targets.append(target)
    # Stage files before committing metadata so failures can be rolled back.
    staging=DATA/('purge-'+uuid.uuid4().hex)
    staging.mkdir()
    moved=[]
    try:
        for index,target in enumerate(dict.fromkeys(targets)):
            if target.exists():
                staged=staging/str(index);target.replace(staged);moved.append((target,staged))
        updated={**p,'images':remaining}
        write(updated)
    except Exception:
        for target,staged in reversed(moved): staged.replace(target)
        staging.rmdir()
        raise
    shutil.rmtree(staging)
    return {'project':updated,'count':len(ids)}

def clear_all_projects():
    if JOB['status']=='running': raise ValueError('请等待训练完成后清空')
    for name in ('images','exports'):
        target=DATA/name
        if target.exists(): shutil.rmtree(target)
    (DATA/'images').mkdir(parents=True,exist_ok=True)
    for backup in DATA.glob('project-before-five-*.json'): backup.unlink()
    prediction=DATA/'prediction.json'
    if prediction.exists(): prediction.unlink()
    p={'images':[],'projects':[],'schema':'droplet-five-v1'};write(p);return p
CLASS_NAMES=['1cell 1bead','only cell','only beads','multiplet','empty']
def export_classified(project_id):
    p=read()
    if not any(q['id']==project_id for q in p['projects']): raise ValueError('请选择项目')
    rows=[r for r in p['images'] if r.get('projectId')==project_id and r.get('parentId') and not r.get('deletedAt')]
    included=[]; skipped=[]; buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in CLASS_NAMES: z.writestr(name+'/',b'')
        for r in rows:
            categories={b['cls'] for b in r['boxes']}
            if not r.get('reviewed') or len(categories)!=1:
                skipped.append({'id':r['id'],'name':r['name'],'reason':'未审核或无标注' if not r.get('reviewed') or not categories else '同图包含多个类别，无法归入单一文件夹'});continue
            cls=next(iter(categories))
            if cls not in range(5): raise ValueError('图片类别无效')
            original=Path(r['name']).name
            filename=Path(original).stem+'_'+r['id'][:12]+Path(r['file']).suffix
            dest=CLASS_NAMES[cls]+'/'+filename
            z.write(DATA/'images'/r['file'],dest)
            included.append({'id':r['id'],'sourceName':r['name'],'path':dest,'category':CLASS_NAMES[cls],'parentId':r['parentId']})
        if not included: raise ValueError('当前项目还没有可导出的已审核处理图。请先标注并审核；同一图片须只有一种类别。')
        z.writestr('manifest.json',json.dumps({'projectId':project_id,'exported':included,'skipped':skipped},ensure_ascii=False,indent=2))
    return buf.getvalue()
class LocalServer(ThreadingHTTPServer):
    def server_bind(self):
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name="localhost"
        self.server_port=self.server_address[1]
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send(self,obj,status=200):
        raw=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/api/project': return self.send(read())
        if urlparse(self.path).path=='/api/export-classified':
            try:
                project_id=parse_qs(urlparse(self.path).query).get('projectId',[None])[0]
                with LOCK: raw=export_classified(project_id)
                self.send_response(200); self.send_header('Content-Type','application/zip'); self.send_header('Content-Disposition','attachment; filename=classified-images.zip'); self.end_headers(); self.wfile.write(raw)
            except Exception as e: self.send({'error':str(e)},400)
            return
        if self.path=='/api/status':
            log=DATA/'job.log'; return self.send({**JOB,'installed':installed(),'models':active_models(),'log':log.read_text(errors='replace')[-16000:] if log.exists() else ''})
        if urlparse(self.path).path=='/api/export':
            try:
                with LOCK: dest=export_dataset(parse_qs(urlparse(self.path).query).get('projectId',[None])[0])
                buf=io.BytesIO()
                with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                    for f in dest.rglob('*'):
                        if f.is_file(): z.write(f,f.relative_to(dest))
                self.send_response(200); self.send_header('Content-Type','application/zip'); self.send_header('Content-Disposition','attachment; filename=cell-bead-dataset.zip'); self.end_headers(); self.wfile.write(buf.getvalue())
            except Exception as e: self.send({'error':str(e)},400)
            return
        target=(ROOT/'static'/'index.html') if self.path=='/' else (DATA/self.path.lstrip('/')) if self.path.startswith('/images/') else ROOT/'static'/self.path.lstrip('/')
        target=target.resolve()
        if not target.is_relative_to(ROOT) or not target.is_file(): return self.send({'error':'文件不存在'},404)
        content_type=mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        if target.suffix in ('.html','.js','.css','.json'):content_type+='; charset=utf-8'
        self.send_response(200); self.send_header('Content-Type',content_type); self.end_headers(); self.wfile.write(target.read_bytes())
    def do_POST(self):
        try:
            if self.headers.get('Origin') and self.headers['Origin'] not in ('http://127.0.0.1:8765','http://localhost:8765'): raise ValueError('不允许跨站请求')
            n=int(self.headers.get('Content-Length',0))
            if n>40*1024*1024: raise ValueError('文件过大，单次上传限制为 40 MB')
            body=json.loads(self.rfile.read(n))
            with LOCK:
                p=read()
                if self.path=='/api/preprocessing-preview':
                    validate_preset(body.get('preprocessing','none'))
                    result=subprocess.run([TRAIN_PYTHON,str(ROOT/'worker.py'),'enhance'],input=json.dumps(body),capture_output=True,text=True,timeout=60,cwd=ROOT)
                    response=json.loads(result.stdout) if result.stdout.strip() else {'error':'增强预览失败，请确认训练环境已安装'}
                    if result.returncode or 'error' in response:raise ValueError(response.get('error','增强预览失败'))
                    return self.send(response)
                if self.path=='/api/preprocessing-config':
                    if JOB['status']=='running':raise ValueError('请等待训练结束后修改识别增强方案')
                    preset=validate_preset(body.get('preprocessing','none'))
                    q=next((q for q in p['projects'] if q['id']==body.get('projectId')),None)
                    if q is None:raise ValueError('请选择项目')
                    q['classifierPreprocessing']=preset;write(p);return self.send(p)
                if self.path=='/api/auto-split':
                    counts=auto_split(p,body.get('projectId'),body.get('validationRatio',0.2),body.get('mode','batch'))
                    write(p);return self.send({'project':p,'counts':counts})
                if self.path=='/api/remove-processed-originals':
                    if JOB['status']=='running': raise ValueError('请等待训练结束后清理原图')
                    count=remove_processed_originals(p,body.get('projectId'));write(p);return self.send({'project':p,'count':count})
                if self.path=='/api/clear-all-projects':
                    if body.get('confirmation')!='清空所有项目': raise ValueError('请输入：清空所有项目')
                    return self.send(clear_all_projects())
                if self.path=='/api/project-rename':
                    name=str(body.get('name','')).strip()
                    if not name or len(name)>200: raise ValueError('项目名不能为空且不能超过200字')
                    if any(q['name']==name and q['id']!=body.get('projectId') for q in p['projects']): raise ValueError('项目名已存在')
                    q=next(q for q in p['projects'] if q['id']==body.get('projectId'));q['name']=name;write(p);return self.send(p)
                if self.path=='/api/project-create':
                    name=str(body.get('name','')).strip()
                    if not name or len(name)>200: raise ValueError('项目名不能为空且不能超过200字')
                    q=next((q for q in p['projects'] if q['name']==name),None)
                    if not q: q={'id':uuid.uuid4().hex,'name':name};p['projects'].append(q)
                    write(p); return self.send({'project':p,'id':q['id']})
                if self.path=='/api/purge':
                    if body.get('confirmation')!='永久删除': raise ValueError('请输入“永久删除”确认')
                    return self.send(purge_images(p,body.get('projectId'),body.get('ids')))
                if self.path in ('/api/remove','/api/restore'):
                    if JOB['status']=='running': raise ValueError('训练期间请等待任务完成后再移除或恢复')
                    ids=set(body.get('ids',[])); project_id=body.get('projectId')
                    allowed={r['id'] for r in p['images'] if r.get('projectId')==project_id}
                    if not ids or not ids<=allowed: raise ValueError('请选择当前项目中的图片')
                    if self.path=='/api/remove':
                        affected=ids|{r['id'] for r in p['images'] if r.get('parentId') in ids}
                        for r in p['images']:
                            if r['id'] in affected: r['deletedAt']=time.time()
                    else:
                        affected=ids|{r['parentId'] for r in p['images'] if r['id'] in ids and r.get('parentId')}
                        for r in p['images']:
                            if r['id'] in affected: r.pop('deletedAt',None)
                    write(p); return self.send(p)
                if self.path=='/api/rule':
                    rule=body['rule']
                    if rule.get('version')!=1 or rule.get('units') not in ('relative','pixels'): raise ValueError('前处理规则无效')
                    q=next(q for q in p['projects'] if q['id']==body.get('projectId'));q['processingRule']=rule; write(p); return self.send(p)
                if self.path=='/api/upload':
                    name=str(body['name']); ext=Path(name).suffix.lower()
                    if ext not in ('.png','.jpg','.jpeg'): raise ValueError('请导入 PNG 或 JPG 图片')
                    w,h=int(body['width']),int(body['height'])
                    if w<=0 or h<=0: raise ValueError('图像尺寸无效')
                    uid=uuid.uuid4().hex; file=uid+ext
                    (DATA/'images'/file).write_bytes(base64.b64decode(body['data'],validate=True))
                    parent=next((r for r in p['images'] if r['id']==body.get('parentId')),None)
                    if body.get('parentId') and parent is None: raise ValueError('找不到原始图像')
                    project_id=parent['projectId'] if parent else body.get('projectId')
                    if not any(q['id']==project_id for q in p['projects']): raise ValueError('请选择导入项目')
                    p['images'].append({'projectId':project_id,'parentId':parent['id'] if parent else None,'processingRule':body.get('processingRule') if parent else None,'cropPixels':body.get('cropPixels') if parent else None,'id':uid,'file':file,'name':name,'width':w,'height':h,'group':parent['group'] if parent else body.get('group',''),'split':parent['split'] if parent else 'train','reviewed':False,'boxes':[]})
                    if parent and body.get('removeOriginal'):
                        parent['deletedAt']=time.time()
                    write(p); return self.send(p)
                if self.path=='/api/save':
                    r=next(r for r in p['images'] if r['id']==body['id'])
                    boxes=body['boxes']
                    for b in boxes:
                        if b['cls'] not in (0,1,2,3,4) or any(not isinstance(b[k],(float,int)) or not math.isfinite(b[k]) for k in ('x','y','w','h')): raise ValueError('标注格式无效')
                        if min(b['x'],b['y'])<0 or min(b['w'],b['h'])<=0 or b['x']+b['w']>1.000001 or b['y']+b['h']>1.000001: raise ValueError('标注框超出图片')
                    if body['split'] not in ('train','val','test'): raise ValueError('集合无效')
                    root_id=r.get('parentId') or r['id']
                    for related in p['images']:
                        if related['id']==root_id or related.get('parentId')==root_id:
                            related['group']=body['group']; related['split']=body['split']
                    r.update({k:body[k] for k in ('boxes','group','split','reviewed')}); write(p); return self.send(p)
                if self.path=='/api/train':
                    if JOB['status']=='running': raise ValueError('已有训练任务在运行')
                    if not installed(): raise ValueError('训练环境尚未安装，请运行安装训练环境.command 后重新启动软件')
                    epochs=int(body.get('epochs',50)); size=int(body.get('size',1024))
                    if not 1<=epochs<=500 or size not in (320,512,640,1024,1280): raise ValueError('训练参数无效')
                    task=body.get('task','auto')
                    if task not in ('auto','roi_classify','detect'): raise ValueError('训练方式无效')
                    architecture=body.get('architecture','resnet18')
                    if architecture not in ('resnet18','convnext_tiny'): raise ValueError('分类网络无效')
                    preprocessing=validate_preset(body.get('preprocessing','none'))
                    if task=='detect' and preprocessing!='none':raise ValueError('识别增强目前用于单液滴分类，目标检测请选择保持原样')
                    project_id=body.get('projectId')
                    if not any(q['id']==project_id for q in p['projects']): raise ValueError('请选择项目')
                    initial_checkpoint=''
                    if body.get('continueFromActive',False):
                        current=ROOT/'models'/project_id/'active_model.json'
                        if not current.is_file():raise ValueError('当前项目没有可继续训练的模型')
                        active=json.loads(current.read_text())
                        if active.get('task')!='roi_classification' or active.get('architecture')!=architecture or task=='detect':
                            raise ValueError('继续训练须选择现用分类模型的网络和任务')
                        source=(ROOT/active['checkpoint']).resolve()
                        if not source.is_relative_to(ROOT) or not source.is_file():raise ValueError('现用模型文件无效')
                        initial_checkpoint=str(source)
                    dest=export_dataset(project_id); device=body.get('device','mps')
                    if device not in ('auto','mps','cpu','0'): raise ValueError('设备无效')
                    batch=int(body.get('batch',4))
                    if not 1<=batch<=64: raise ValueError('批量大小范围为1至64')
                    args=[TRAIN_PYTHON,str(ROOT/'worker.py'),'train',str(dest),str(epochs),str(size),device,project_id,task,architecture,preprocessing,str(batch),initial_checkpoint]
                    log=open(DATA/'job.log','w'); proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                    JOB.update(status='running',pid=proc.pid)
                    def wait():
                        code=proc.wait(); log.close(); JOB.update(status='completed' if code==0 else 'failed')
                    threading.Thread(target=wait,daemon=True).start(); return self.send(JOB)
                if self.path=='/api/predict':
                    if JOB['status']=='running': raise ValueError('请等待训练完成')
                    r=next(r for r in p['images'] if r['id']==body['id']); out=DATA/'prediction.json'
                    manifest=ROOT/'models'/r['projectId']/'active_model.json'
                    if manifest.exists() and json.loads(manifest.read_text()).get('task')=='roi_classification' and not r.get('parentId'):
                        raise ValueError('当前模型用于单液滴处理图，请先按项目规则生成裁剪处理图，再进行识别')
                    result=subprocess.run([TRAIN_PYTHON,str(ROOT/'worker.py'),'predict',str(DATA/'images'/r['file']),str(out),r['projectId']],capture_output=True,text=True,timeout=180,cwd=ROOT)
                    if result.returncode: raise ValueError(result.stderr[-2000:] or result.stdout[-2000:])
                    return self.send(json.loads(out.read_text()))
                raise ValueError('未知操作')
        except Exception as e: self.send({'error':str(e)},400)
if __name__=='__main__':
    DATA.mkdir(exist_ok=True); (DATA/'images').mkdir(exist_ok=True)
    print('软件已启动：http://127.0.0.1:8765',flush=True)
    LocalServer(('127.0.0.1',8765),Handler).serve_forever()
