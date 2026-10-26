"""Verify the extracted migration package against its SHA-256 manifest."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'package-manifest.json').read_text(encoding='utf-8'))
errors=[]
for row in manifest['files']:
    path=(root/row['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        errors.append(row['path']+' missing or invalid');continue
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    if path.stat().st_size!=row['size'] or h.hexdigest()!=row['sha256']:errors.append(row['path']+' changed')
print('Files checked:',len(manifest['files']))
if errors:
    print('\n'.join(errors));sys.exit(1)
print('All packaged files match the manifest.')
