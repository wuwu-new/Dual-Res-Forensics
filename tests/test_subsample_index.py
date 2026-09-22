import unittest

from tools.subsample_index import evenly_spaced


class SubsampleIndexTest(unittest.TestCase):
    def test_even_sampling_preserves_video_order_and_endpoints(self) -> None:
        records = [
            {"image_path": f"v1/{index}.png", "label": 0, "video_id": "v1"}
            for index in range(10)
        ] + [
            {"image_path": f"v2/{index}.png", "label": 1, "video_id": "v2"}
            for index in range(3)
        ]
        sampled = evenly_spaced(records, 4)
        self.assertEqual(
            [item["image_path"] for item in sampled],
            ["v1/0.png", "v1/3.png", "v1/6.png", "v1/9.png", "v2/0.png", "v2/1.png", "v2/2.png"],
        )

    def test_single_frame_uses_middle(self) -> None:
        records = [
            {"image_path": f"{index}.png", "label": 0, "video_id": "v"}
            for index in range(5)
        ]
        self.assertEqual(evenly_spaced(records, 1)[0]["image_path"], "2.png")

    def test_rejects_missing_video_id(self) -> None:
        with self.assertRaises(ValueError):
            evenly_spaced([{"image_path": "0.png", "label": 0}], 1)


if __name__ == "__main__":
    unittest.main()
