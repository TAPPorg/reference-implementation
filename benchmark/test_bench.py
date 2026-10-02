import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import bench
import bench_data
import bench_file
import tapp_bindings


class TestParseContractions(unittest.TestCase):
    def write_file(self, text):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        path = Path(tmp_dir.name) / "contractions.txt"
        path.write_text(text)
        return path

    def test_required_fields_and_defaults(self):
        path = self.write_file("""
[matmul]
indices = ab-bc-ac
extents = a:4 b:5 c:6
""")
        specs = bench_file.parse_contractions(path)
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec["name"], "matmul")
        self.assertEqual(spec["indices"], "ab-bc-ac")
        self.assertEqual(spec["extents"], {"a": 4, "b": 5, "c": 6})
        for t in ("a", "b", "c", "d"):
            self.assertEqual(spec[f"datatype_{t}"], "f32")
            self.assertEqual(spec[f"op_{t}"], "identity")
        self.assertEqual(spec["precision"], "default")
        self.assertEqual(spec["alpha"], 1.0)
        self.assertEqual(spec["beta"], 0.0)
        self.assertEqual(spec["repeats"], 10)

    def test_datatype_default_applies_only_to_unset_tensors(self):
        path = self.write_file("""
[mixed]
indices = ab-bc-ac
extents = a:2 b:2 c:2
datatype = f32
datatype_d = f64
""")
        spec = bench_file.parse_contractions(path)[0]
        self.assertEqual(spec["datatype_a"], "f32")
        self.assertEqual(spec["datatype_b"], "f32")
        self.assertEqual(spec["datatype_c"], "f32")
        self.assertEqual(spec["datatype_d"], "f64")

    def test_op_override_leaves_others_at_default(self):
        path = self.write_file("""
[conj]
indices = ab-bc-ac
extents = a:2 b:2 c:2
op_a = conjugate
""")
        spec = bench_file.parse_contractions(path)[0]
        self.assertEqual(spec["op_a"], "conjugate")
        self.assertEqual(spec["op_b"], "identity")
        self.assertEqual(spec["op_c"], "identity")
        self.assertEqual(spec["op_d"], "identity")

    def test_multiple_sections_preserve_order(self):
        path = self.write_file("""
[first]
indices = ab-bc-ac
extents = a:1 b:1 c:1

[second]
indices = ij-jk-ik
extents = i:2 j:2 k:2
""")
        specs = bench_file.parse_contractions(path)
        self.assertEqual([s["name"] for s in specs], ["first", "second"])

    def test_alpha_beta_repeats_are_typed(self):
        path = self.write_file("""
[typed]
indices = ab-bc-ac
extents = a:2 b:2 c:2
alpha = 2.5
beta = 1.0
repeats = 7
""")
        spec = bench_file.parse_contractions(path)[0]
        self.assertIsInstance(spec["alpha"], complex)
        self.assertIsInstance(spec["beta"], complex)
        self.assertIsInstance(spec["repeats"], int)
        self.assertEqual(spec["alpha"], 2.5)
        self.assertEqual(spec["repeats"], 7)

    def test_alpha_beta_accept_complex_literals(self):
        path = self.write_file("""
[complex_scalars]
indices = ab-bc-ac
extents = a:2 b:2 c:2
datatype_d = c64
alpha = 1.0+2.0j
beta = -1.0j
""")
        spec = bench_file.parse_contractions(path)[0]
        self.assertEqual(spec["alpha"], complex(1.0, 2.0))
        self.assertEqual(spec["beta"], complex(0.0, -1.0))

    def test_build_scalar_rejects_imaginary_for_real_datatype(self):
        with self.assertRaises(ValueError):
            bench.build_scalar(complex(1.0, 2.0), "f32")

    def test_build_scalar_accepts_complex_for_complex_datatype(self):
        values = bench.build_scalar(complex(1.0, 2.0), "c64")
        self.assertEqual(list(values), [1.0, 2.0])

    def test_comments_and_blank_lines_ignored(self):
        path = self.write_file("""
# a leading comment
[commented]
# another comment
indices = ab-bc-ac
extents = a:2 b:2 c:2

""")
        specs = bench_file.parse_contractions(path)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["name"], "commented")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            bench_file.parse_contractions(Path("does_not_exist.txt"))


class TestFLOPs(unittest.TestCase):
    def test_complex_matmul_is_four_times_real(self):
        args = ("ab", "bc", "ac", [2, 2], [2, 2], [2, 2])
        self.assertEqual(bench_data.FLOPs(*args), 24)
        self.assertEqual(bench_data.FLOPs(*args, is_complex=True), 96)

    def test_complex_reduction_is_two_times_real(self):
        # e is isolated in A: (3 - 1) * 2 * 2 = 8 reduction additions
        args = ("abe", "bc", "ac", [2, 2, 3], [2, 2], [2, 2])
        self.assertEqual(bench_data.FLOPs(*args), 8 + 24)
        self.assertEqual(bench_data.FLOPs(*args, is_complex=True), 2 * 8 + 4 * 24)

    def test_complex_intensity_is_four_times_real_for_matmul(self):
        args = ("ab", "bc", "ac", [2, 2], [2, 2], [2, 2])
        self.assertEqual(bench_data.intensity(*args, is_complex=True), 4 * bench_data.intensity(*args))


@unittest.skipUnless(
    tapp_bindings.library_available(),
    "TAPP library not found (not built, or TAPP_LIBRARY_PATH points to a missing file)",
)
class TestBenchBySpecs(unittest.TestCase):
    def make_matmul(self, dtype):
        return {
            "name": f"matmul_{dtype}",
            "indices": "ab-bc-ac",
            "extents": {"a": 2, "b": 2, "c": 2},
            **{f"datatype_{t}": dtype for t in ("a", "b", "c", "d")},
            **{f"op_{t}": "identity" for t in ("a", "b", "c", "d")},
            "precision": "default",
            "alpha": 1.0,
            "beta": 0.0,
            "repeats": 1,
        }

    def run_quietly(self, specs):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            results = bench.bench_by_specs(specs)
        return results, out.getvalue()

    def test_real_and_complex_run_without_error(self):
        results, _ = self.run_quietly([self.make_matmul("f32"), self.make_matmul("c64")])
        for min_time, avg_time, gflops, error in results:
            self.assertIsNone(error)
            self.assertGreater(gflops, 0)

    def test_bad_spec_is_skipped_and_others_still_run(self):
        bad = {**self.make_matmul("f32"), "name": "bad", "datatype_a": "f99"}
        results, output = self.run_quietly([bad, self.make_matmul("f32")])
        self.assertIsInstance(results[0][3], ValueError)
        self.assertIsNone(results[1][3])
        self.assertIn("bad: SKIPPED", output)

    def test_zero_repeats_is_skipped_and_others_still_run(self):
        bad = {**self.make_matmul("f32"), "name": "no_repeats", "repeats": 0}
        results, output = self.run_quietly([bad, self.make_matmul("f32")])
        self.assertIsInstance(results[0][3], ValueError)
        self.assertIn("repeats", str(results[0][3]))
        self.assertIsNone(results[1][3])
        self.assertIn("no_repeats: SKIPPED", output)


if __name__ == "__main__":
    unittest.main()
