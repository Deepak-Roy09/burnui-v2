import unittest
import pandas as pd

from app.preparation.splitting import split_by_engine
from ml_fusion_weight_diagnostic import training_partitions


class FusionWeightDiagnosticTests(unittest.TestCase):
    def test_rebuilds_validation_partition_from_metadata(self):
        frame = pd.DataFrame({"engine_id": [engine for engine in range(1, 21) for _ in range(30)], "cycle": list(range(1, 31)) * 20})
        original = split_by_engine(frame, seed=42)
        rebuilt = training_partitions(frame, original.metadata)
        self.assertEqual(rebuilt.metadata, original.metadata)
        self.assertFalse(rebuilt.validation.empty)
        self.assertTrue(set(rebuilt.validation.engine_id).isdisjoint(set(rebuilt.test.engine_id)))
