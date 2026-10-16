import os
import sys
import re
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import export_weights as E


class TestExport(unittest.TestCase):
    def test_export_roundtrip(self):
        d = {
            "conv1.W": np.arange(8 * 3 * 3 * 3, dtype=np.float32).reshape(8, 3, 3, 3) / 7.0,
            "conv1.b": np.arange(8, dtype=np.float32),
            "conv2.W": np.arange(16 * 8 * 3 * 3, dtype=np.float32).reshape(16, 8, 3, 3) / 13.0,
            "conv2.b": np.arange(16, dtype=np.float32),
            "fc.W": np.arange(4 * 4096, dtype=np.float32).reshape(4, 4096) / 100.0,
            "fc.b": np.arange(4, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as td:
            npz = os.path.join(td, "w.npz")
            np.savez(npz, **d)
            out = os.path.join(td, "weights.h")
            E.write_header(npz, np.array([0.5, 0.5, 0.5], np.float32),
                           np.array([0.25, 0.25, 0.25], np.float32), out)
            with open(out) as f:
                text = f.read()
        self.assertIn("NORM_MEAN", text)
        self.assertIn("W1", text)
        self.assertIn("B3", text)

        def parse(name):
            m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{(.*?)\}};", text, re.S)
            return np.array([float(v) for v in re.findall(r"[-+0-9.eE]+", m.group(1))], dtype=np.float32)

        self.assertTrue(np.allclose(parse("B1"), d["conv1.b"], atol=1e-4))
        self.assertTrue(np.allclose(parse("NORM_STD"), [0.25, 0.25, 0.25], atol=1e-6))

    def test_cfloat_emits_valid_c_literals(self):
        self.assertEqual(E._cfloat(0.0), "0.0f")
        self.assertEqual(E._cfloat(1.0), "1.0f")
        self.assertEqual(E._cfloat(-3.0), "-3.0f")
        self.assertEqual(E._cfloat(0.25), "0.25f")


if __name__ == "__main__":
    unittest.main()
