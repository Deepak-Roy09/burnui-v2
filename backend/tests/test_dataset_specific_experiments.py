import unittest
import numpy as np
import pandas as pd
from app.evaluation.dataset_specific_experiments import official_test_truth, train_dataset_specific
from app.evaluation.real_only_evaluation import FutureDegradationTarget

class DatasetSpecificExperimentTests(unittest.TestCase):
    def _train_frame(self):
        rows = []
        for engine_id in range(1, 21):
            for cycle in range(1, 41):
                row = {"engine_id": engine_id, "cycle": cycle}
                row.update({f"sensor_{number}": engine_id * .1 + cycle * number * .001 for number in range(1, 22)})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_official_rul_alignment_uses_sorted_engine_ids(self):
        frame = pd.DataFrame({"engine_id": [2, 1, 2, 1], "cycle": [40, 35, 45, 42]})
        target = FutureDegradationTarget(30, .2, 20.0, (9,))
        truth = official_test_truth(frame, [3, 7], target)
        self.assertEqual(truth.loc[1, "rul"], 3)
        self.assertEqual(truth.loc[2, "rul"], 7)
        self.assertEqual(truth.loc[1, "remaining_cycles"], 15)
        self.assertEqual(truth.loc[2, "remaining_cycles"], 22)

    def test_training_is_deterministic_and_engine_disjoint(self):
        first = train_dataset_specific(self._train_frame())
        second = train_dataset_specific(self._train_frame())
        first_ids = first.split_metadata
        self.assertEqual(first_ids, second.split_metadata)
        train, validation, internal = map(set, (first_ids.train_engine_ids, first_ids.validation_engine_ids, first_ids.test_engine_ids))
        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(internal))
        self.assertTrue(validation.isdisjoint(internal))
        self.assertEqual(set(first.pipeline.preprocessor.fit_engine_ids), train)
        self.assertEqual(set(first.pipeline.validation_engine_ids), validation)

    def test_rul_is_not_an_argument_to_training(self):
        # The public training helper deliberately accepts trajectories only.
        with self.assertRaises(TypeError):
            train_dataset_specific(self._train_frame(), [1] * 20)  # type: ignore[call-arg]
