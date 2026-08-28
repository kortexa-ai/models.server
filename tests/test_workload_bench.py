import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "bench" / "workload_bench.py"
SPEC = importlib.util.spec_from_file_location("workload_bench", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
workload_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workload_bench)


class WorkloadBenchTest(unittest.TestCase):
    def test_mixed_chat_profiles_have_distinct_history_shapes(self):
        histories = [
            workload_bench.mixed_chat_messages(label, words, turns)
            for label, words, turns in workload_bench.CHAT_PROFILES
        ]

        self.assertEqual([len(messages) for messages in histories], [6, 10, 18, 34])
        self.assertEqual(
            [profile[1] for profile in workload_bench.CHAT_PROFILES],
            [512, 8192, 32768, 131072],
        )
        self.assertLess(len(str(histories[0])), len(str(histories[-1])))

    def test_pipeline_images_are_distinct_but_same_dimensions(self):
        first = workload_bench.pipeline_png(64, 0)
        second = workload_bench.pipeline_png(64, 1)

        self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(second.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertNotEqual(first, second)

    def test_aggregate_reports_latency_and_throughput(self):
        results = [
            {
                "ok": True,
                "wall_seconds": 1.0,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
            {
                "ok": True,
                "wall_seconds": 3.0,
                "usage": {"prompt_tokens": 200, "completion_tokens": 20},
            },
            {"ok": False, "error": "bounded failure"},
        ]

        summary = workload_bench.aggregate(results, 4.0)

        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["requests_per_second"], 0.5)
        self.assertEqual(summary["prompt_tokens_per_second"], 75.0)
        self.assertEqual(summary["latency_seconds"]["p50"], 1.0)
        self.assertEqual(summary["latency_seconds"]["p95"], 3.0)


if __name__ == "__main__":
    unittest.main()
