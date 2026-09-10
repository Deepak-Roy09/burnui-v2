import unittest

from app.datasets.validation import validate_csv_upload


VALID_CSV = b"""engine_id,cycle,setting_1,setting_2,setting_3,sensor_1,sensor_2
1,1,0.1,0.2,100,518.67,641.82
1,2,0.1,0.2,100,518.68,642.10
2,1,0.3,0.4,90,519.00,640.00
"""


class DatasetValidationTests(unittest.TestCase):
    def test_accepts_sequential_configurable_sensor_columns(self) -> None:
        report = validate_csv_upload("normalized.csv", VALID_CSV)

        self.assertTrue(report["valid"])
        self.assertEqual(report["summary"]["sensor_count"], 2)
        self.assertEqual(report["summary"]["engine_count"], 2)

    def test_rejects_future_contract_shape_errors(self) -> None:
        invalid = b"""engine_id,cycle,setting_1,setting_2,setting_3,sensor_1,sensor_3
1,2,0.1,0.2,100,1.0,2.0
1,1,0.1,0.2,100,1.1,2.1
"""
        report = validate_csv_upload("normalized.csv", invalid)

        self.assertFalse(report["valid"])
        codes = {issue["code"] for issue in report["errors"]}
        self.assertIn("non_sequential_sensor_columns", codes)
        self.assertIn("non_monotonic_cycles", codes)

    def test_rejects_non_csv_upload(self) -> None:
        report = validate_csv_upload("source.txt", VALID_CSV)

        self.assertFalse(report["valid"])
        self.assertEqual(report["errors"][0]["code"], "invalid_file_type")
