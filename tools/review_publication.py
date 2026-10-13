"""Review candidate publication files without reading ignored local artifacts."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.pytest_cache', 'workspace', '.venv', 'build', 'dist'}
SECRET = re.compile(r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
FORBIDDEN = {'.pdf', '.docx', '.pptx', '.zip', '.index', '.safetensors', '.bin', '.pt', '.pth', '.pkl', '.pickle', '.ipynb'}


def main():
    inventory = ROOT / 'docs/PUBLICATION_FILES.csv'
    files = [p for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(part in IGNORED or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts)]
    issues = []
    for p in files:
        relative = p.relative_to(ROOT).as_posix()
        if p.suffix in FORBIDDEN or p.name.startswith('.env') and p.name != '.env.example':
            issues.append(f'{relative}: excluded file type')
        if p.stat().st_size > 10 * 1024 * 1024:
            issues.append(f'{relative}: exceeds 10 MiB publication budget')
        text = p.read_text(encoding='utf8')
        if SECRET.search(text):
            issues.append(f'{relative}: credential pattern (value redacted)')
        if p.suffix == '.py':
            ast.parse(text, filename=relative)
        if p.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', text):
                target = p.parent / link.split('#')[0]
                if '://' not in link and target.resolve() != inventory.resolve() and not target.exists():
                    issues.append(f'{relative}: missing local link {link}')
    with inventory.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'bytes', 'sha256'])
        writer.writeheader()
        for p in files:
            if p == inventory:
                continue  # self-hashing would be circular
            writer.writerow({'path':p.relative_to(ROOT).as_posix(), 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    print(json.dumps({'candidate_files_excluding_inventory':len([p for p in files if p != inventory]), 'issues':issues}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
