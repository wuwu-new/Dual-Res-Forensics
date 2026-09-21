import unittest

from drf.config_overrides import apply_data_path_overrides, parse_named_paths


class ConfigOverridesTest(unittest.TestCase):
    def test_parse_named_paths(self) -> None:
        self.assertEqual(
            parse_named_paths(["cdfv2=/data/cdf.json", "dfdc=/data/dfdc.json"], "--test-json"),
            {"cdfv2": "/data/cdf.json", "dfdc": "/data/dfdc.json"},
        )

    def test_parse_rejects_invalid_and_duplicate_names(self) -> None:
        with self.assertRaises(ValueError):
            parse_named_paths(["missing-separator"], "--test-json")
        with self.assertRaises(ValueError):
            parse_named_paths(["dfdc=a", "dfdc=b"], "--test-json")

    def test_apply_replaces_only_requested_paths(self) -> None:
        cfg = {
            "data": {
                "train_json": "old-train.json",
                "val_jsons": {"old": "old-val.json"},
                "test_jsons": {"old": "old-test.json"},
            }
        }
        apply_data_path_overrides(
            cfg,
            train_json="new-train.json",
            val_jsons={"ffpp_val": "new-val.json"},
        )
        self.assertEqual(cfg["data"]["train_json"], "new-train.json")
        self.assertEqual(cfg["data"]["val_jsons"], {"ffpp_val": "new-val.json"})
        self.assertEqual(cfg["data"]["test_jsons"], {"old": "old-test.json"})


if __name__ == "__main__":
    unittest.main()
