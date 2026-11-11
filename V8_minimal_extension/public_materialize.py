"""Restore byte-identical CSV files from the lossless public gzip archives."""
import sys
sys.dont_write_bytecode=True
import gzip,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def materialize():
    manifest=json.loads((BASE/'PUBLIC_DATA_MANIFEST.json').read_text())
    restored=0
    for item in manifest['compressed_csvs']:
        archive=BASE/item['archive'];target=BASE/item['uncompressed']
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['compressed_sha256'],archive
        raw=gzip.decompress(archive.read_bytes())
        assert hashlib.sha256(raw).hexdigest()==item['uncompressed_sha256'],archive
        if target.exists():
            # Analysis-derived tables may be recomputed; never silently replace.
            if hashlib.sha256(target.read_bytes()).hexdigest()!=item['uncompressed_sha256']:
                if item['role']=='derived_analysis': continue
                raise RuntimeError(f'Existing input differs from archive: {target}')
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw);restored+=1
    print(f'Validated {len(manifest["compressed_csvs"])} lossless CSV archives; restored {restored} missing files.')

if __name__=='__main__': materialize()
