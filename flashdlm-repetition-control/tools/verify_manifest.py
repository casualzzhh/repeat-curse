"""Verify the released source files without loading models or third-party packages."""
from pathlib import Path
import hashlib,json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for name,entry in manifest['files'].items():
    path=(root/name).resolve()
    assert path.is_relative_to(root),f'Invalid manifest path: {name}'
    data=path.read_bytes()
    assert len(data)==entry['bytes'],f'Size mismatch: {name}'
    assert hashlib.sha256(data).hexdigest()==entry['sha256'],f'Hash mismatch: {name}'
print('Verified',len(manifest['files']),'source files')
