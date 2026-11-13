"""A failed fresh mean prediction must remain a failed comparison."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import check_processed_prediction as study


class ProcessedPredictionTests(unittest.TestCase):
    def test_bad_candidate_is_reported_without_refitting(self):
        f=np.arange(10,63)*.005;observed=np.ones((200,53,81))
        predictions=np.stack([np.full((53,81),v) for v in [3.,3.,2.,4.]])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);payload=dict(f=f)
            for group in study.GROUPS:
                payload[f'{group}_standard']=np.ones((53,81))
                for headings in [12,36]:payload[f'{group}_{headings}']=predictions
            np.savez_compressed(root/'prediction.npz',**payload)
            (root/'protocol.json').write_text(json.dumps(dict(headings=[12,36])))
            manifest=dict(prediction_sha256=study.digest(root/'prediction.npz'),protocol_sha256=study.digest(root/'protocol.json'))
            (root/'prediction_manifest.json').write_text(json.dumps(manifest))
            with patch.object(study,'load',return_value=(f,observed,np.ones((53,9,9)))):
                study.analyse(root,root)
            rows=list(csv.DictReader((root/'processed_prediction_changes.csv').open()))
            failed=[r for r in rows if r['method']=='full_processing']
            self.assertEqual(len(failed),18)
            for r in failed:
                self.assertAlmostEqual(float(r['improvement_pct']),-50.)
                self.assertAlmostEqual(float(r['ci95_low']),-50.)
                self.assertAlmostEqual(float(r['ci95_high']),-50.)
            self.assertEqual(study.digest(root/'prediction.npz'),manifest['prediction_sha256'])


if __name__=='__main__':unittest.main()
