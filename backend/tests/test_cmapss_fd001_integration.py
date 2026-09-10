import os
import unittest
from zipfile import ZipFile

from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss
from app.preparation.early_features import extract_early_features
from app.preparation.splitting import split_by_engine


class CMAPSSFD001IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CMAPSS_ZIP"), "Set CMAPSS_ZIP to run the read-only FD001 integration test.")
    def test_fd001_train_loads_and_generates_early_features(self) -> None:
        with ZipFile(os.environ["CMAPSS_ZIP"]) as archive:
            with archive.open("train_FD001.txt") as source:
                frame = load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)

        self.assertEqual(frame["engine_id"].nunique(), 100)
        self.assertEqual(frame.columns.tolist()[:5], ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"])
        self.assertEqual([column for column in frame.columns if column.startswith("sensor_")], [f"sensor_{index}" for index in range(1, 22)])
        self.assertTrue((frame.groupby("engine_id")["cycle"].diff().dropna() > 0).all())
        features = extract_early_features(frame)
        self.assertEqual(len(features.skipped_engines), 0)
        self.assertEqual(features.features.shape, (100, 169))
        split = split_by_engine(frame)
        self.assertEqual((len(split.metadata.train_engine_ids), len(split.metadata.validation_engine_ids), len(split.metadata.test_engine_ids)), (70, 15, 15))
