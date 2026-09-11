import unittest
from io import BytesIO

from app.preparation.cmapss_adapter import (
    CMAPSSFileKind,
    CMAPSSSubset,
    CMAPSSVariant,
    detect_cmapss_filename,
    load_cmapss,
    load_cmapss_rul,
    resolve_cmapss_source,
)


def raw_cmapss_text(engine_count: int = 2, cycles: int = 30) -> bytes:
    rows = []
    for engine in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            values = [engine, cycle, 0.0, 0.0, 100.0, *(engine + cycle * sensor * 0.001 for sensor in range(1, 22))]
            rows.append(" ".join(str(value) for value in values))
    return ("\n".join(rows) + "\n").encode("utf-8")


class CMAPSSUploadTests(unittest.TestCase):
    def test_rul_values_preserve_file_order(self) -> None:
        self.assertEqual(load_cmapss_rul(BytesIO(b"12\n34\n")), [12.0, 34.0])

    def test_official_train_and_test_names_identify_all_variants(self) -> None:
        for variant in CMAPSSVariant:
            for kind in (CMAPSSFileKind.TRAIN, CMAPSSFileKind.TEST):
                descriptor = detect_cmapss_filename(f"{kind.value}_{variant.value}.txt")
                self.assertIsNotNone(descriptor)
                assert descriptor is not None
                self.assertEqual(descriptor.variant, variant)
                self.assertEqual(descriptor.kind, kind)
                self.assertEqual(descriptor.subset.value, kind.value)

    def test_raw_fd001_to_fd004_parse_to_normalized_columns_without_value_changes(self) -> None:
        from io import BytesIO

        for variant in CMAPSSVariant:
            for subset in CMAPSSSubset:
                frame = load_cmapss(BytesIO(raw_cmapss_text()), variant=variant, subset=subset)
                self.assertEqual(frame.shape, (60, 26))
                self.assertEqual(frame.columns.tolist(), ["engine_id", "cycle", "setting_1", "setting_2", "setting_3", *[f"sensor_{i}" for i in range(1, 22)]])
                self.assertEqual(frame.iloc[0]["sensor_1"], 1.001)
                self.assertEqual(frame.iloc[-1]["engine_id"], 2)

    def test_ambiguous_name_requires_explicit_variant_and_file_type(self) -> None:
        with self.assertRaises(ValueError):
            resolve_cmapss_source("sample.txt")
        variant, subset, descriptor = resolve_cmapss_source("sample.txt", selected_variant="FD004", selected_subset="TEST")
        self.assertEqual((variant, subset, descriptor), (CMAPSSVariant.FD004, CMAPSSSubset.TEST, None))

    def test_filename_selection_cannot_silently_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            resolve_cmapss_source("test_FD002.txt", selected_variant="FD001")
        with self.assertRaises(ValueError):
            resolve_cmapss_source("RUL_FD003.txt")

    def test_unknown_or_normalized_csv_names_are_not_raw_cmapss_matches(self) -> None:
        self.assertIsNone(detect_cmapss_filename("normalized.csv"))
        self.assertIsNone(detect_cmapss_filename("test_FD005.txt"))
