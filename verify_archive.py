"""Read-only archive integrity and syntax checks. Does not import model code."""
from pathlib import Path
import ast
import hashlib
import json

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'docs' / 'SOURCE_MANIFEST.json').read_text())
    errors = []
    for entry in manifest:
        path = root / entry['archive_file']
        if not path.is_file():
            errors.append(f'Missing: {entry["archive_file"]}')
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['archive_sha256']:
            errors.append(f'Changed: {entry["archive_file"]}')
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            except SyntaxError as exc:
                errors.append(str(exc))
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(manifest)} archived source/data/model files match their recorded hashes; Python syntax parses.')
    print('This does not validate dependencies, serialized models, databases, APIs, or end-to-end execution.')

if __name__ == '__main__':
    main()
