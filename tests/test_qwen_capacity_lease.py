from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts/qwen_capacity_lease.py"
SPEC = importlib.util.spec_from_file_location("qwen_capacity_lease", MODULE_PATH)
assert SPEC and SPEC.loader
lease = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lease
SPEC.loader.exec_module(lease)


def owner(name: str = "worker-a") -> dict[str, str]:
    return {
        "actorId": name,
        "harness": "Prime Agent",
        "model": "Qwen",
        "rootSessionId": f"root-{name}",
        "delegatedWorkerId": f"delegated-{name}",
        "sessionRef": f"session-{name}",
    }


def healthy_snapshot() -> dict:
    return {
        "host": "smarty",
        "roster": {
            "models/qwen-3.8-27b": {
                "installed": True,
                "enabled": True,
                "running": True,
            }
        },
        "health": {"models/qwen-3.8-27b": True},
        "gpu": {
            "index": 0,
            "memoryTotalMiB": 97887,
            "memoryUsedMiB": 65000,
            "memoryFreeMiB": 32887,
            "utilizationPercent": 5,
            "powerDrawWatts": 100,
            "powerLimitWatts": 450,
        },
        "gpuProcesses": [
            {
                "pid": 101,
                "unit": "kortexa-ai-llm-qwen-3.8-27b.service",
                "usedMemoryMiB": 30396,
            }
        ],
        "unknownGpuProcesses": [],
        "inconsistentManagedGpuProcesses": [],
        "legolmGpuOwners": [],
        "qwen": {
            "endpoint": "http://127.0.0.1:2053",
            "modelPresent": True,
            "totalSlots": 2,
            "observedSlots": 2,
            "busySlots": 0,
            "portPids": [101],
            "gpuPids": [101],
        },
    }


class FakeClock:
    def __init__(self) -> None:
        self.value = dt.datetime(2026, 8, 23, 20, 0, tzinfo=dt.timezone.utc)
        self.lock = threading.Lock()

    def __call__(self) -> dt.datetime:
        with self.lock:
            return self.value

    def advance(self, seconds: int) -> None:
        with self.lock:
            self.value += dt.timedelta(seconds=seconds)


class FakeCollector:
    def __init__(self, snapshots: list[dict] | None = None) -> None:
        self.snapshots = snapshots or [healthy_snapshot()]
        self.index = 0
        self.canary_error: Exception | None = None
        self.canary_count = 0
        self.lock = threading.Lock()

    def collect(self) -> dict:
        with self.lock:
            result = self.snapshots[min(self.index, len(self.snapshots) - 1)]
            self.index += 1
            return copy.deepcopy(result)

    def canary(self) -> dict:
        with self.lock:
            self.canary_count += 1
        if self.canary_error:
            raise self.canary_error
        return {
            "elapsedMs": 125,
            "promptTokens": 12,
            "completionTokens": 2,
            "responseStored": False,
        }


class LeaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.state_file = Path(self.temp.name) / "leases.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def manager(self, collector: FakeCollector | None = None) -> lease.LeaseManager:
        registry = lease.LeaseRegistry(self.state_file, clock=self.clock)
        return lease.LeaseManager(registry, collector or FakeCollector())

    def read_state(self) -> dict:
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def test_healthy_canary_admits_one_bounded_lease(self) -> None:
        collector = FakeCollector()
        result = self.manager(collector).acquire(owner(), 900)

        self.assertEqual("admit", result["decision"])
        self.assertEqual(1, collector.canary_count)
        self.assertEqual(
            {
                "maxConcurrency": 1,
                "maxContextTokens": 65536,
                "maxOutputTokens": 4096,
            },
            result["lease"]["budget"],
        )
        self.assertFalse(result["lease"]["admission"]["canary"]["responseStored"])
        self.assertEqual("http://127.0.0.1:2053", result["lease"]["endpoint"])

    def test_concurrent_agents_get_one_lease_and_one_queue(self) -> None:
        collector = FakeCollector()
        manager_a = self.manager(collector)
        manager_b = self.manager(collector)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda pair: pair[0].acquire(pair[1], 900),
                    ((manager_a, owner("a")), (manager_b, owner("b"))),
                )
            )

        self.assertEqual(["admit", "queue"], sorted(item["decision"] for item in results))
        queued = next(item for item in results if item["decision"] == "queue")
        self.assertIn("agent-capacity-already-leased", queued["reasonCodes"])
        self.assertEqual(1, len(self.read_state()["active"]))

    def test_stale_lease_from_crashed_agent_expires_before_reclaim(self) -> None:
        manager = self.manager()
        first = manager.acquire(owner("crashed"), 60)
        self.clock.advance(61)
        second = manager.acquire(owner("replacement"), 60)

        self.assertEqual("admit", first["decision"])
        self.assertEqual("admit", second["decision"])
        state = self.read_state()
        self.assertEqual(1, len(state["active"]))
        self.assertEqual("expired", state["history"][0]["status"])
        self.assertEqual("heartbeat-expired", state["history"][0]["release"]["reasonCodes"][0])

    def test_legolm_gpu_owner_queues_without_canary(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["legolmGpuOwners"] = [
            {"pid": 333, "cwd": "/home/francip/src/legolm", "unit": ""}
        ]
        collector = FakeCollector([snapshot])
        result = self.manager(collector).acquire(owner(), 900)

        self.assertEqual("queue", result["decision"])
        self.assertIn("legolm-active", result["reasonCodes"])
        self.assertEqual(0, collector.canary_count)

    def test_unknown_cuda_process_queues(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["unknownGpuProcesses"] = [
            {"pid": 444, "unit": "", "usedMemoryMiB": 1024}
        ]
        result = self.manager(FakeCollector([snapshot])).acquire(owner(), 900)
        self.assertIn("unknown-cuda-process", result["reasonCodes"])

    def test_managed_gpu_process_with_stopped_roster_entry_queues(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["inconsistentManagedGpuProcesses"] = [
            {"pid": 202, "unit": "kortexa-ai-vision-server.service"}
        ]
        result = self.manager(FakeCollector([snapshot])).acquire(owner(), 900)
        self.assertIn("managed-cuda-roster-mismatch", result["reasonCodes"])

    def test_insufficient_headroom_queues(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["gpu"]["memoryFreeMiB"] = 16000
        result = self.manager(FakeCollector([snapshot])).acquire(owner(), 900)
        self.assertIn("insufficient-vram-headroom", result["reasonCodes"])

    def test_busy_reserved_slot_queues(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["qwen"]["busySlots"] = 1
        result = self.manager(FakeCollector([snapshot])).acquire(owner(), 900)
        self.assertIn("qwen-request-capacity-busy", result["reasonCodes"])

    def test_mismatched_slot_telemetry_queues(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["qwen"]["observedSlots"] = 1
        result = self.manager(FakeCollector([snapshot])).acquire(owner(), 900)
        self.assertIn("qwen-slot-state-unknown", result["reasonCodes"])

    def test_unhealthy_protected_service_queues(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["health"]["vision.server"] = False
        result = self.manager(FakeCollector([snapshot])).acquire(owner(), 900)
        self.assertIn("protected-service-unhealthy", result["reasonCodes"])

    def test_wrong_or_duplicate_qwen_port_owner_queues(self) -> None:
        snapshot = healthy_snapshot()
        snapshot["qwen"]["portPids"] = [101, 202]
        result = self.manager(FakeCollector([snapshot])).acquire(owner(), 900)
        self.assertIn("qwen-port-owner-mismatch", result["reasonCodes"])

    def test_canary_failure_creates_no_lease(self) -> None:
        collector = FakeCollector()
        collector.canary_error = lease.CapacityError("qwen-canary-failed")
        result = self.manager(collector).acquire(owner(), 900)

        self.assertEqual("queue", result["decision"])
        self.assertEqual(["qwen-canary-failed"], result["reasonCodes"])
        self.assertEqual({}, self.read_state()["active"])

    def test_qwen_pid_change_during_canary_queues(self) -> None:
        after = healthy_snapshot()
        after["qwen"]["portPids"] = [202]
        after["qwen"]["gpuPids"] = [202]
        result = self.manager(FakeCollector([healthy_snapshot(), after])).acquire(
            owner(), 900
        )
        self.assertIn("qwen-process-changed", result["reasonCodes"])

    def test_unsafe_heartbeat_revokes_active_lease(self) -> None:
        unsafe = healthy_snapshot()
        unsafe["gpu"]["utilizationPercent"] = 99
        collector = FakeCollector([healthy_snapshot(), healthy_snapshot(), unsafe])
        manager = self.manager(collector)
        acquired = manager.acquire(owner(), 900)
        result = manager.heartbeat(acquired["lease"]["leaseId"], owner(), 900)

        self.assertEqual("blocked", result["decision"])
        self.assertIn("production-gpu-load-high", result["reasonCodes"])
        state = self.read_state()
        self.assertEqual({}, state["active"])
        self.assertEqual("blocked", state["history"][0]["release"]["outcome"])

    def test_release_is_idempotent_for_same_owner(self) -> None:
        manager = self.manager()
        acquired = manager.acquire(owner(), 900)
        lease_id = acquired["lease"]["leaseId"]
        first = manager.release(lease_id, owner(), "completed")
        second = manager.release(lease_id, owner(), "completed")

        self.assertEqual("released", first["decision"])
        self.assertEqual("released", second["decision"])
        self.assertEqual(1, len(self.read_state()["history"]))

    def test_owner_mismatch_cannot_heartbeat_or_release(self) -> None:
        manager = self.manager()
        acquired = manager.acquire(owner(), 900)
        lease_id = acquired["lease"]["leaseId"]
        heartbeat = manager.heartbeat(lease_id, owner("other"), 900)
        released = manager.release(lease_id, owner("other"), "cancelled")

        self.assertEqual(["lease-owner-mismatch"], heartbeat["reasonCodes"])
        self.assertEqual(["lease-owner-mismatch"], released["reasonCodes"])
        self.assertIn(lease_id, self.read_state()["active"])

    def test_secret_shaped_owner_is_rejected_before_persistence(self) -> None:
        args = type(
            "Args",
            (),
            {
                "actor_id": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                "harness": "Prime",
                "owner_model": "Qwen",
                "root_session_id": "root",
                "delegated_worker_id": "worker",
                "session_ref": "session",
            },
        )()
        with self.assertRaisesRegex(lease.CapacityError, "secret-shaped-actor-id"):
            lease.owner_from_args(args)
        self.assertFalse(self.state_file.exists())

    def test_state_files_are_owner_only_and_store_no_prompt_or_output(self) -> None:
        self.manager().acquire(owner(), 900)
        mode = stat.S_IMODE(os.stat(self.state_file).st_mode)
        payload = self.state_file.read_text(encoding="utf-8")

        self.assertEqual(0o600, mode)
        self.assertEqual(0o700, stat.S_IMODE(os.stat(self.state_file.parent).st_mode))
        self.assertNotIn("Reply with exactly", payload)
        self.assertNotIn('"content"', payload)
        self.assertNotIn('"response"', payload)

    def test_roster_has_explicit_success_decision(self) -> None:
        result = self.manager().roster()
        self.assertEqual("listed", result["decision"])


class ParsingAndCommandTests(unittest.TestCase):
    @staticmethod
    def canary_result(
        content: object = "OK",
        reasoning_content: object = "",
        prompt_tokens: object = 10,
        completion_tokens: object = 2,
    ) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning_content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    prompt_tokens + completion_tokens
                    if isinstance(prompt_tokens, int)
                    and not isinstance(prompt_tokens, bool)
                    and isinstance(completion_tokens, int)
                    and not isinstance(completion_tokens, bool)
                    else None
                ),
            },
        }

    def test_canary_accepts_content_without_retaining_it(self) -> None:
        parsed = lease.parse_canary_response(self.canary_result(), 25)
        self.assertEqual(["content"], parsed["outputChannels"])
        self.assertFalse(parsed["responseStored"])
        self.assertNotIn("OK", json.dumps(parsed))

    def test_canary_accepts_observed_reasoning_only_shape(self) -> None:
        result = self.canary_result(content="", reasoning_content="fixture-reasoning")
        parsed = lease.parse_canary_response(result, 25)
        self.assertEqual(["reasoning-content"], parsed["outputChannels"])
        self.assertNotIn("fixture-reasoning", json.dumps(parsed))

    def test_canary_accepts_openai_typed_text_parts(self) -> None:
        result = self.canary_result(content=[{"type": "text", "text": "OK"}])
        parsed = lease.parse_canary_response(result, 25)
        self.assertEqual(["content-parts"], parsed["outputChannels"])

    def test_canary_accepts_empty_parts_when_reasoning_is_present(self) -> None:
        result = self.canary_result(content=[], reasoning_content="fixture-reasoning")
        parsed = lease.parse_canary_response(result, 25)
        self.assertEqual(["reasoning-content"], parsed["outputChannels"])

    def test_canary_rejects_empty_output_channels(self) -> None:
        with self.assertRaisesRegex(lease.CapacityError, "qwen-canary-empty"):
            lease.parse_canary_response(self.canary_result(content=""), 25)

    def test_canary_rejects_error_and_malformed_shapes(self) -> None:
        malformed = (
            {"error": {"message": "fixture-error"}},
            {"choices": []},
            {"choices": ["not-an-object"]},
            {"choices": [{"message": {"role": "user", "content": "OK"}}]},
            {**self.canary_result(), "error": {"message": "fixture-error"}},
            self.canary_result(content=42),
            self.canary_result(content="", reasoning_content=["not-a-string"]),
            self.canary_result(content=[{"type": "image", "text": "not-text"}]),
        )
        for result in malformed:
            with self.subTest(result=result), self.assertRaisesRegex(
                lease.CapacityError, "qwen-canary-invalid"
            ):
                lease.parse_canary_response(result, 25)

    def test_canary_rejects_missing_or_out_of_budget_usage(self) -> None:
        missing = self.canary_result()
        del missing["usage"]
        missing_total = self.canary_result()
        del missing_total["usage"]["total_tokens"]
        mismatched_total = self.canary_result()
        mismatched_total["usage"]["total_tokens"] = 99
        invalid = (
            missing,
            missing_total,
            mismatched_total,
            self.canary_result(prompt_tokens=True),
            self.canary_result(prompt_tokens=-1),
            self.canary_result(prompt_tokens=257),
            self.canary_result(completion_tokens=0),
            self.canary_result(completion_tokens=9),
        )
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(lease.CapacityError):
                lease.parse_canary_response(result, 25)

    def test_ktxsvc_roster_preserves_grouped_service_names(self) -> None:
        output = """\
PROJECT                        INSTALLED  ENABLED    RUNNING
-------                        ---------  -------    -------
models.server                  -          -          -
  qwen-3.8-27b                 yes        yes        yes
alt-image-gen.server           -          -          -
  base                         yes        yes        yes
vision.server                  yes        yes        yes
"""
        roster = lease.parse_ktxsvc_list(output)
        self.assertTrue(roster["models/qwen-3.8-27b"]["running"])
        self.assertTrue(roster["alt-image-gen.server/base"]["running"])
        self.assertTrue(roster["vision.server"]["running"])

    def test_read_only_runner_rejects_service_and_process_mutation(self) -> None:
        runner = lease.ReadOnlyRunner()
        forbidden = (
            ("ktxsvc", "stop", "models/qwen-3.8-27b"),
            ("ktxsvc", "start", "models/qwen-3.8-27b"),
            ("systemctl", "restart", "anything"),
            ("kill", "101"),
            ("pkill", "python"),
            ("nvidia-smi", "--gpu-reset"),
            ("nvidia-smi", "--power-limit=600"),
        )
        for command in forbidden:
            with self.subTest(command=command), self.assertRaisesRegex(
                lease.CapacityError, "non-read-only-command-rejected"
            ):
                runner.run(command)

    def test_gpu_parsers_fail_closed_on_unknown_shape(self) -> None:
        with self.assertRaises(lease.CapacityError):
            lease.parse_gpu_summary("broken")
        with self.assertRaises(lease.CapacityError):
            lease.parse_gpu_processes("1, missing-column")


if __name__ == "__main__":
    unittest.main()
