import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.package_indexed_data import collect_files, create_archive


class PackageIndexedDataTest(unittest.TestCase):
    def test_packages_deduplicated_relative_files_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "dataset"
            frame = root / "frames/video/000.png"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"frame")
            index = root / "test.json"
            index.write_text(
                json.dumps(
                    [
                        {"image_path": "frames/video/000.png", "label": 0, "video_id": "v"},
                        {"image_path": "frames/video/000.png", "label": 0, "video_id": "v"},
                    ]
                ),
                encoding="utf-8",
            )
            output = base / "dataset.tar"

            files, manifest = collect_files(root, [index])
            self.assertEqual(len(files), 2)
            self.assertEqual(manifest["indexes"][0]["records"], 2)
            create_archive(root, [index], output)

            with tarfile.open(output) as archive:
                self.assertEqual(
                    set(archive.getnames()),
                    {"frames/video/000.png", "test.json", "PACKAGE_MANIFEST.json"},
                )

    def test_rejects_absolute_file_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "dataset"
            root.mkdir()
            outside = base / "outside.png"
            outside.write_bytes(b"frame")
            index = root / "test.json"
            index.write_text(
                json.dumps([{"image_path": str(outside), "label": 0, "video_id": "v"}]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                collect_files(root, [index])


if __name__ == "__main__":
    unittest.main()
