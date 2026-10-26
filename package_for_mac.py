"""Build and verify a complete Mac/Windows source-and-data ZIP."""
import hashlib,json,os,stat,zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parent
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
archive=ROOT.parent/f'DropletLab_完整项目_{stamp}.zip'
temporary_archive=archive.with_suffix('.zip.partial')
assert archive.resolve().is_relative_to(ROOT.parent.resolve())
excluded_dirs={'.venv','.install-cache','__pycache__','.git','.agents','.codex','.pytest_cache','node_modules','__MACOSX'}
excluded_prefixes={'data/config','data/mpl','data/pip-cache'}
files=[]
for folder,dirs,names in os.walk(ROOT):
    dirs[:]=sorted(d for d in dirs if d not in excluded_dirs and (Path(folder)/d).relative_to(ROOT).as_posix() not in excluded_prefixes)
    for name in sorted(names):
        path=Path(folder)/name;relative=path.relative_to(ROOT).as_posix()
        if name=='package-manifest.json' or name=='.DS_Store' or name.endswith(('.pyc','.pyo','.tmp')) or name.startswith(('server-windows.','startup.')) or relative in {'data/prediction.json'}:continue
        if path.is_symlink():raise ValueError('Unexpected symlink: '+relative)
        files.append((path,relative))
source_stats={relative:(path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ino) for path,relative in files}
database=(ROOT/'data/project.json').read_bytes();project=json.loads(database)
records=[];models=[]
for path in sorted((ROOT/'models').glob('*/active_model.json')):
    m=json.loads(path.read_text(encoding='utf-8'));checkpoint=(ROOT/m['checkpoint']).resolve()
    assert not Path(m['checkpoint']).is_absolute() and checkpoint.is_relative_to(ROOT) and checkpoint.is_file(),checkpoint
    assert '\\' not in m['checkpoint']
    actual=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert actual==m['sha256']
    models.append({'manifest':path.relative_to(ROOT).as_posix(),'project_id':m['project_id'],'checkpoint':m['checkpoint'],'architecture':m['architecture'],'validation_images':m.get('validation',{}).get('images'),'validation_accuracy':m.get('validation',{}).get('accuracy'),'sha256':actual})
for row in project['images']:
    assert (ROOT/'data/images'/row['file']).is_file(),row['file']
print(f'Packing {len(files)} files; {len(project["images"])} image records; {len(project["projects"])} projects',flush=True)
with zipfile.ZipFile(temporary_archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
    for index,(path,relative) in enumerate(files,1):
        info=zipfile.ZipInfo('DropletLab/'+relative,date_time=datetime.now().timetuple()[:6]);info.create_system=3;info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=((stat.S_IFREG | (0o755 if path.suffix=='.command' else 0o644))<<16)
        h=hashlib.sha256();size=0
        with z.open(info,'w',force_zip64=True) as target:
            if relative=='data/project.json':target.write(database);h.update(database);size=len(database)
            else:
                with path.open('rb') as source:
                    for chunk in iter(lambda:source.read(1024*1024),b''):
                        target.write(chunk);h.update(chunk);size+=len(chunk)
        records.append({'path':relative,'size':size,'sha256':h.hexdigest()})
        if index%3000==0:print(f'Packed {index}/{len(files)}',flush=True)
    manifest={'format':'dropletlab-project-package-v3','created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'target':'Mac/Windows source and data; Apple Silicon macOS 14+, Python 3.12/3.13 for Mac','root_directory':'DropletLab','projects':project['projects'],'image_records':len(project['images']),'active_models':models,'file_count_excluding_manifest':len(records),'uncompressed_bytes_excluding_manifest':sum(r['size'] for r in records),'exclusions':sorted(excluded_dirs|excluded_prefixes)+['.DS_Store','*.pyc','*.pyo','*.tmp','server-windows.*','startup.*','data/prediction.json','previous package-manifest.json'],'files':records}
    raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    info=zipfile.ZipInfo('DropletLab/package-manifest.json');info.create_system=3;info.external_attr=(stat.S_IFREG|0o644)<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
print('ZIP created. Verifying every archived file...',flush=True)
with zipfile.ZipFile(temporary_archive) as z:
    assert len(z.infolist())==len(records)+1
    assert len(set(z.namelist()))==len(z.infolist())
    for index,row in enumerate(records,1):
        h=hashlib.sha256();size=0
        with z.open('DropletLab/'+row['path']) as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk);size+=len(chunk)
        assert h.hexdigest()==row['sha256'] and size==row['size'],row['path']
        if index%5000==0:print(f'Verified {index}/{len(records)}',flush=True)
    for name in ['启动软件.command','安装训练环境.command','检查训练环境.command']:
        assert z.getinfo('DropletLab/'+name).external_attr>>16 & 0o111
for path,relative in files:
    current=path.stat()
    assert (current.st_size,current.st_mtime_ns,current.st_ino)==source_stats[relative], 'Source changed during packaging: '+relative
assert (ROOT/'data/project.json').read_bytes()==database, 'Database changed during packaging; retry when editing has finished'
(ROOT/'package-manifest.json').write_bytes(raw)
temporary_archive.replace(archive)
h=hashlib.sha256()
with archive.open('rb') as stream:
    for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
archive.with_suffix('.zip.sha256').write_text(h.hexdigest()+'  '+archive.name+'\n',encoding='utf-8')
result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest(),'files':len(records)+1,'image_records':len(project['images']),'projects':[q['name'] for q in project['projects']],'active_models':models,'verified':True}
archive.with_suffix('.zip.校验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
