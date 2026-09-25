"""Verify bundled dataset fingerprints without loading pickle objects."""
import hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'data/MANIFEST.json').read_text())
    for record in manifest['files']:
        path=root/record['path']
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(2**20),b''):h.update(block)
        if h.hexdigest()!=record['sha256']:raise ValueError(f'Checksum mismatch: {record["path"]}')
        print('OK',record['path'])

if __name__=='__main__':main()
