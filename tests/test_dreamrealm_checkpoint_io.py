import json
from pathlib import Path
import tempfile
import unittest
import torch
from dreamrealm.checkpoint_io import save_parts, load_parts


class CheckpointTest(unittest.TestCase):
    def test_exact_roundtrip_across_small_parts(self):
        state = {
            "weights": torch.randn(7000),
            "nested": {"step": 3, "values": [torch.randn(37, 29), None, 1.5]},
            "names": ["a", "b"],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "parts"
            save_parts(state, root, part_bytes=997)
            self.assertTrue(all(x.stat().st_size <= 997 for x in root.glob("*.bin")))
            got = load_parts(root)
            torch.testing.assert_close(got["weights"], state["weights"], atol=0, rtol=0)
            torch.testing.assert_close(
                got["nested"]["values"][0], state["nested"]["values"][0], atol=0, rtol=0
            )
            self.assertEqual(got["nested"]["step"], 3)
            self.assertEqual(got["names"], state["names"])

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "parts"
            save_parts({"tensor": torch.ones(900)}, root, part_bytes=1000)
            path = next(root.glob("*.bin"))
            data = bytearray(path.read_bytes())
            data[20] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(IOError, "checksum"):
                load_parts(root)


if __name__ == "__main__":
    unittest.main()
