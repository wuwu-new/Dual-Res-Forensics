import json
import tempfile
import unittest
from pathlib import Path

from tools.build_ffpp_index import build_indexes


class BuildFfppIndexTest(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        splits = {
            "train": [["000", "001"]],
            "val": [["002", "003"]],
            "test": [["004", "005"]],
        }
        for split, pairs in splits.items():
            (root / f"{split}.json").write_text(json.dumps(pairs), encoding="utf-8")
            for first, second in pairs:
                for source_id in (first, second):
                    frame = root / "original_sequences/youtube/c23/frames" / source_id / "0.png"
                    frame.parent.mkdir(parents=True, exist_ok=True)
                    frame.write_bytes(b"frame")
                for video_name in (f"{first}_{second}", f"{second}_{first}"):
                    frame = root / "manipulated_sequences/Deepfakes/c23/frames" / video_name / "0.png"
                    frame.parent.mkdir(parents=True, exist_ok=True)
                    frame.write_bytes(b"frame")

    def test_builds_disjoint_portable_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ffpp"
            root.mkdir()
            self._make_dataset(root)

            summaries = build_indexes(root, root, methods=("Deepfakes",), strict=True)

            self.assertEqual([summary.images for summary in summaries], [4, 4, 4])
            video_sets = []
            for split in ("train", "val", "test"):
                index_path = root / f"ffpp_c23_{split}.json"
                records = json.loads(index_path.read_text(encoding="utf-8"))
                self.assertEqual({record["label"] for record in records}, {0, 1})
                self.assertTrue(all(not Path(record["image_path"]).is_absolute() for record in records))
                self.assertTrue(
                    all((index_path.parent / record["image_path"]).is_file() for record in records)
                )
                video_sets.append({record["video_id"] for record in records})

            self.assertFalse(video_sets[0] & video_sets[1])
            self.assertFalse(video_sets[0] & video_sets[2])
            self.assertFalse(video_sets[1] & video_sets[2])

    def test_strict_mode_rejects_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ffpp"
            root.mkdir()
            self._make_dataset(root)
            missing = root / "original_sequences/youtube/c23/frames/000/0.png"
            missing.unlink()

            with self.assertRaises(FileNotFoundError):
                build_indexes(root, root, methods=("Deepfakes",), strict=True)


if __name__ == "__main__":
    unittest.main()
