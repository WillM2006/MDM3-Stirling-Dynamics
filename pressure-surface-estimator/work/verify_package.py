"""Verify every original packaged evidence file and frozen estimator source."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((ROOT/'PACKAGING_MANIFEST.json').read_text())
for name,expected in m['files'].items():
    assert sha(ROOT/name)==expected, 'Packaged file changed: '+name
for name,expected in m['frozen_estimator_sha256'].items():
    assert sha(ROOT/'src'/name)==expected, 'Frozen estimator changed: '+name
for case in m['cases']:
    assert sha(ROOT/'outputs/independent-wave-package/cases'/case['case']/'pressures.csv')==case['pressure_sha256']
print('Verified',len(m['files']),'manifest files, frozen estimator sources and all four unchanged pressure records.')
