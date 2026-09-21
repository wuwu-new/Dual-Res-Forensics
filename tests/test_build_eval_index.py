import json
import tempfile
import unittest
from pathlib import Path

from tools.build_eval_index import build_celebdf_v2, build_dfdc


class BuildEvalIndexTest(unittest.TestCase):
    @staticmethod
    def _frame(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frame")

    def test_celebdf_labels_and_video_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "celebdf"
            root.mkdir()
            (root / "List_of_testing_videos.txt").write_text(
                "1 YouTube-real/001.mp4\n0 Celeb-synthesis/id1_id2_0001.mp4\n",
                encoding="utf-8",
            )
            self._frame(root / "YouTube-real/frames/001/000.png")
            self._frame(root / "Celeb-synthesis/frames/id1_id2_0001/000.png")
            output = root / "cdfv2_test.json"

            summary = build_celebdf_v2(root, output, strict=True)

            records = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary.videos, 2)
            self.assertEqual([item["label"] for item in records], [0, 1])
            self.assertEqual(
                [item["video_id"] for item in records],
                ["celebdfv2:YouTube-real/001", "celebdfv2:Celeb-synthesis/id1_id2_0001"],
            )
            self.assertTrue(all((output.parent / item["image_path"]).is_file() for item in records))

    def test_dfdc_reports_missing_face_extractions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dfdc"
            root.mkdir()
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "real.mp4": {"is_fake": 0},
                        "fake.mp4": {"is_fake": 1},
                        "missing.mp4": {"is_fake": 1},
                    }
                ),
                encoding="utf-8",
            )
            self._frame(root / "frames/real/000.png")
            self._frame(root / "frames/fake/000.png")
            output = root / "dfdc_test.json"

            summary = build_dfdc(root, output)

            self.assertEqual(summary.images, 2)
            self.assertEqual(summary.videos, 2)
            self.assertEqual(summary.missing_videos, ("missing.mp4",))
            with self.assertRaises(FileNotFoundError):
                build_dfdc(root, output, strict=True)


if __name__ == "__main__":
    unittest.main()
