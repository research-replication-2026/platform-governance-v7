"""Restore exact archived CSV bytes; refuse a changed existing file."""
import gzip,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def materialize():
    m=json.loads((B/'PUBLIC_ECR_MANIFEST.json').read_text())
    for name,digest in m['files'].items():
        if name.endswith('.py') or name in ['V8_PROTOCOL.md','BUDGET_MANIFEST.json']:
            assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
    for name,meta in m['compressed'].items():
        dest=B/name
        if dest.exists():
            assert hashlib.sha256(dest.read_bytes()).hexdigest()==meta['sha256'],name
            continue
        data=gzip.decompress((B/(name+'.gz')).read_bytes())
        assert hashlib.sha256(data).hexdigest()==meta['sha256'],name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    print('Restored/verified exact ECR simulation data.')
if __name__=='__main__':materialize()
