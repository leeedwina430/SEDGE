"""Check bundle contents against SHA256SUMS."""
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
rows=(root/'SHA256SUMS').read_text().splitlines()
for line in rows:
    expected,name=line.split('  ',1)
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print(f'Verified {len(rows)} files')
