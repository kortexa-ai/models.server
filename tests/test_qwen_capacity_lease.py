from __future__ import annotations

import contextlib
import copy
import datetime as dt
import importlib.util
import io
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
        "model": "qwen-3.8-27b",
        "rootSessionId": "prime:0198ff96-2e31-7c30-9eca-4d4f22265e90",
        "delegatedWorkerId": f"/root/{name}",
        "sessionRef": f"/root/{name}",
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
            "outputChannels": ["content"],
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
                "harness": "Prime Agent",
                "owner_model": "qwen-3.8-27b",
                "root_session_id": "prime:0198ff96-2e31-7c30-9eca-4d4f22265e90",
                "delegated_worker_id": "/root/worker",
                "session_ref": "/root/worker",
            },
        )()
        with self.assertRaisesRegex(lease.CapacityError, "secret-shaped-actor-id"):
            lease.owner_from_args(args)
        self.assertFalse(self.state_file.exists())

    def test_formatted_credentials_are_rejected_by_cli_owner_parser(self) -> None:
        secret_values = (
            '{"password":"fixture-secret-value"}',
            '{"authorization":"Bearer fixture-token"}',
            "AWS_SECRET_ACCESS_KEY=fixture-secret-value",
            "awsSecretAccessKey=fixture-secret-value",
            '"client_secret"=fixture-secret-value',
            '"private-key" = "fixture-key-value"',
            '{" password ":"fixture-secret-value"}',
            "' authorizationHeader ': 'Bearer fixture-token'",
            '{"\\u0070assword":"fixture-secret-value"}',
            "apiKeyValue=fixture-key-value",
            "authorizationHeaderValue=Bearer fixture-token",
            "privateKeyPem=fixture-key-value",
            '{"header":"Bearer fixture-token"}',
            "https://fixture-user:fixture-password@example.invalid",
            "api_key_backup=fixture-key",
            "private_key_backup=fixture-key",
            "access_token_copy=fixture-token",
            "client_secret_previous=fixture-secret",
            "password_archive=fixture-password",
            "credentials_backup=fixture-credentials",
            "dbTokenOld=fixture-token",
            "secretKeyReplica=fixture-key",
            "signingKeyDuplicate=fixture-key",
            "refresh_token_snapshot=fixture-token",
            "passphraseFormer=fixture-passphrase",
            "credentialCopy=fixture-credential",
            "api\\u002fkey_backup=fixture-key",
            "api/key_backup=fixture-key",
            '["api_key_backup"]=fixture-key',
            "`api_key_backup`=fixture-key",
            '{"api/key_backup":"fixture-key"}',
            "['private/key_backup']: 'fixture-key'",
            "export API/KEY_BACKUP=fixture-key",
            "[authorization/header_previous] = Bearer fixture-token",
            "`client/secret_archive`: fixture-secret",
            "auth.api/key_copy = fixture-key",
            "APIKEY=fixture-value",
            "CLIENTSECRET: fixture-value",
            "ACCESSTOKEN = fixture-value",
            "PRIVATEKEY:=fixture-value",
            "SECRETKEY => fixture-value",
            "SIGNINGKEY: fixture-value",
            "AUTHORIZATIONHEADER = Basic fixture-value",
            "ACCESSKEYID=fixture-value",
            '{"APIKEY":"fixture-value"}',
            '["CLIENTSECRET"]=fixture-value',
            "`ACCESSTOKEN`: fixture-value",
            "PRIVATEKEY # deployment backup = fixture-value",
            '"SECRETKEY" /* rotated */: "fixture-value"',
            "`SIGNINGKEY` # old: fixture-value",
            "AUTHORIZATIONHEADER /* transport */ = Basic fixture-value",
            "ACCESSKEYID # comment=fixture-value",
            "api/*note*/key=fixture-value",
            "access/*note*/key=fixture-value",
            "private/*note*/key=fixture-value",
            "signing/*note*/key=fixture-value",
            "pass/*note*/word=fixture-value",
            "api<!--note-->key=fixture-value",
            "client<!--note-->secret=fixture-value",
            '{"api":{"key":"fixture-value"}}',
            '{"private":{"key":"fixture-value"}}',
            '{"access":{"token":"fixture-value"}}',
            '{"client":{"secret":"fixture-value"}}',
            '{"authorization":{"header":"Basic fixture-value"}}',
            "PrIvAtE/KeY=fixture-value",
            "aPi/kEy=fixture-value",
            "AcCeSs/ToKeN=fixture-value",
            "ClIeNt/SeCrEt=fixture-value",
            "AuThOrIzAtIoN/HeAdEr=Basic fixture-value",
            "PaSs/WoRd=fixture-value",
            "api#note#key=fixture-value",
            "api # note\n key = fixture-value",
            "private# note\n key: fixture-value",
        )
        for value in secret_values:
            args = type(
                "Args",
                (),
                {
                    "actor_id": "worker",
                    "harness": "Prime Agent",
                    "owner_model": "qwen-3.8-27b",
                    "root_session_id": "prime:0198ff96-2e31-7c30-9eca-4d4f22265e90",
                    "delegated_worker_id": "/root/worker",
                    "session_ref": value,
                },
            )()
            with self.subTest(value=value), self.assertRaisesRegex(
                lease.CapacityError, "secret-shaped-session-ref"
            ):
                lease.owner_from_args(args)
            self.assertFalse(self.state_file.exists())

    def test_compound_and_header_credentials_are_rejected(self) -> None:
        secret_values = (
            "passwordHash=fixture-secret",
            "PASSWORD_HASH = fixture-secret",
            "Authorization: Bearer fixture-token",
            "authorization = bearer   fixture-token",
            "AUTHORIZATION: Basic fixture-credential",
            "  BeArEr fixture-token  ",
            "refreshToken=fixture-token",
            "client_secret: fixture-secret",
            "privateKey=fixture-key",
            "github_token: fixture-token",
        )
        for value in secret_values:
            with self.subTest(value=value), self.assertRaisesRegex(
                lease.CapacityError, "secret-shaped-session-ref"
            ):
                lease.validate_identifier("session-ref", value)

    def test_secret_owner_cannot_bypass_cli_validation_or_reach_state(self) -> None:
        secret_values = (
            "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "ghu_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "ghr_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "AKIAABCDEFGHIJKLMNOP",
            "ASIAABCDEFGHIJKLMNOP",
            "-----BEGIN PRIVATE KEY-----",
            "-----BEGIN ENCRYPTED PRIVATE KEY-----",
            "Authorization: Bearer fixture-token",
            "authorizationHeader = 'Bearer fixture-token'",
            "label: authorizationHeader=Bearer fixture-token",
            "passwordHash=fixture-secret",
            "my password = \"fixture-secret\"",
            "build_passwd: fixture-secret",
            "deploy-passphrase = fixture-secret",
            "credentials='fixture-credential'",
            "refresh Token = fixture-token",
            "api Key = fixture-key",
            "client-secret=fixture-secret",
            "privateKey=fixture-key",
            "secret_key = fixture-key",
            "signing-key: fixture-key",
            '{"password":"fixture-secret-value"}',
            '{"authorization":"Bearer fixture-token"}',
            "AWS_SECRET_ACCESS_KEY=fixture-secret-value",
            "awsSecretAccessKey=fixture-secret-value",
            '"client_secret"=fixture-secret-value',
            '"private-key" = "fixture-key-value"',
            '{" password ":"fixture-secret-value"}',
            "' authorizationHeader ': 'Bearer fixture-token'",
            '{"\\u0070assword":"fixture-secret-value"}',
            "apiKeyValue=fixture-key-value",
            "authorizationHeaderValue=Bearer fixture-token",
            "privateKeyPem=fixture-key-value",
            '{"header":"Bearer fixture-token"}',
            "https://fixture-user:fixture-password@example.invalid",
            "api_key_backup=fixture-key",
            "private_key_backup=fixture-key",
            "access_token_copy=fixture-token",
            "client_secret_previous=fixture-secret",
            "password_archive=fixture-password",
            "credentials_backup=fixture-credentials",
            "dbTokenOld=fixture-token",
            "secretKeyReplica=fixture-key",
            "signingKeyDuplicate=fixture-key",
            "refresh_token_snapshot=fixture-token",
            "passphraseFormer=fixture-passphrase",
            "credentialCopy=fixture-credential",
            "api\\u002fkey_backup=fixture-key",
            "api/key_backup=fixture-key",
            '["api_key_backup"]=fixture-key',
            "`api_key_backup`=fixture-key",
            '{"api/key_backup":"fixture-key"}',
            "['private/key_backup']: 'fixture-key'",
            "export API/KEY_BACKUP=fixture-key",
            "[authorization/header_previous] = Bearer fixture-token",
            "`client/secret_archive`: fixture-secret",
            "auth.api/key_copy = fixture-key",
            "APIKEY=fixture-value",
            "CLIENTSECRET: fixture-value",
            "ACCESSTOKEN = fixture-value",
            "PRIVATEKEY:=fixture-value",
            "SECRETKEY => fixture-value",
            "SIGNINGKEY: fixture-value",
            "AUTHORIZATIONHEADER = Basic fixture-value",
            "ACCESSKEYID=fixture-value",
            '{"APIKEY":"fixture-value"}',
            '["CLIENTSECRET"]=fixture-value',
            "`ACCESSTOKEN`: fixture-value",
            "PRIVATEKEY # deployment backup = fixture-value",
            '"SECRETKEY" /* rotated */: "fixture-value"',
            "`SIGNINGKEY` # old: fixture-value",
            "AUTHORIZATIONHEADER /* transport */ = Basic fixture-value",
            "ACCESSKEYID # comment=fixture-value",
            "api/*note*/key=fixture-value",
            "access/*note*/key=fixture-value",
            "private/*note*/key=fixture-value",
            "signing/*note*/key=fixture-value",
            "pass/*note*/word=fixture-value",
            "api<!--note-->key=fixture-value",
            "client<!--note-->secret=fixture-value",
            '{"api":{"key":"fixture-value"}}',
            '{"private":{"key":"fixture-value"}}',
            '{"access":{"token":"fixture-value"}}',
            '{"client":{"secret":"fixture-value"}}',
            '{"authorization":{"header":"Basic fixture-value"}}',
            "PrIvAtE/KeY=fixture-value",
            "aPi/kEy=fixture-value",
            "AcCeSs/ToKeN=fixture-value",
            "ClIeNt/SeCrEt=fixture-value",
            "AuThOrIzAtIoN/HeAdEr=Basic fixture-value",
            "PaSs/WoRd=fixture-value",
            "api#note#key=fixture-value",
            "api # note\n key = fixture-value",
            "private# note\n key: fixture-value",
        )
        for value in secret_values:
            with self.subTest(value=value):
                secret_owner = owner()
                secret_owner["sessionRef"] = value
                manager = self.manager()

                with self.assertRaisesRegex(
                    lease.CapacityError, "secret-shaped-session-ref"
                ):
                    manager.acquire(secret_owner, 900)
                self.assertFalse(self.state_file.exists())
                self.assertFalse(manager.registry.lock_file.exists())

    def test_invalid_release_outcome_cannot_mutate_state_or_history(self) -> None:
        manager = self.manager()
        acquired = manager.acquire(owner(), 900)
        lease_id = acquired["lease"]["leaseId"]
        state_before = self.state_file.read_bytes()
        lock_before = manager.registry.lock_file.stat()
        invalid_outcomes = (
            None,
            "",
            "expired",
            " completed ",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        )

        for outcome in invalid_outcomes:
            with self.subTest(outcome=outcome), self.assertRaisesRegex(
                lease.CapacityError, "invalid-release-outcome"
            ):
                manager.release(lease_id, owner(), outcome)  # type: ignore[arg-type]
            self.assertEqual(state_before, self.state_file.read_bytes())
            lock_after = manager.registry.lock_file.stat()
            self.assertEqual(lock_before.st_mtime_ns, lock_after.st_mtime_ns)
            self.assertEqual(lock_before.st_size, lock_after.st_size)

        state = self.read_state()
        self.assertIn(lease_id, state["active"])
        self.assertEqual([], state["history"])

    def test_cli_rejects_invalid_release_outcome_before_state_access(self) -> None:
        argv = [
            "release",
            "--actor-id",
            "worker",
            "--harness",
            "Prime",
            "--owner-model",
            "Qwen",
            "--root-session-id",
            "root",
            "--delegated-worker-id",
            "delegated",
            "--session-ref",
            "session",
            "--lease-id",
            "qwen-fixture",
            "--outcome",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            lease.parse_args(argv)
        self.assertFalse(self.state_file.exists())

    def test_final_persistence_boundary_rejects_secret_strings(self) -> None:
        registry = lease.LeaseRegistry(self.state_file, clock=self.clock)
        secret_values = (
            '{"\\u0070assword":"fixture-secret-value"}',
            "api_key_backup=fixture-key",
            "private_key_backup=fixture-key",
            "access_token_copy=fixture-token",
            "client_secret_previous=fixture-secret",
            "password_archive=fixture-password",
            "credentials_backup=fixture-credentials",
            "dbTokenOld=fixture-token",
            "secretKeyReplica=fixture-key",
            "signingKeyDuplicate=fixture-key",
            "refresh_token_snapshot=fixture-token",
            "passphraseFormer=fixture-passphrase",
            "credentialCopy=fixture-credential",
            "api\\u002fkey_backup=fixture-key",
            "api/key_backup=fixture-key",
            '["api_key_backup"]=fixture-key',
            "`api_key_backup`=fixture-key",
            '{"api/key_backup":"fixture-key"}',
            "['private/key_backup']: 'fixture-key'",
            "export API/KEY_BACKUP=fixture-key",
            "[authorization/header_previous] = Bearer fixture-token",
            "`client/secret_archive`: fixture-secret",
            "auth.api/key_copy = fixture-key",
            "APIKEY=fixture-value",
            "CLIENTSECRET: fixture-value",
            "ACCESSTOKEN = fixture-value",
            "PRIVATEKEY:=fixture-value",
            "SECRETKEY => fixture-value",
            "SIGNINGKEY: fixture-value",
            "AUTHORIZATIONHEADER = Basic fixture-value",
            "ACCESSKEYID=fixture-value",
            '{"APIKEY":"fixture-value"}',
            '["CLIENTSECRET"]=fixture-value',
            "`ACCESSTOKEN`: fixture-value",
            "PRIVATEKEY # deployment backup = fixture-value",
            '"SECRETKEY" /* rotated */: "fixture-value"',
            "`SIGNINGKEY` # old: fixture-value",
            "AUTHORIZATIONHEADER /* transport */ = Basic fixture-value",
            "ACCESSKEYID # comment=fixture-value",
            "api/*note*/key=fixture-value",
            "access/*note*/key=fixture-value",
            "private/*note*/key=fixture-value",
            "signing/*note*/key=fixture-value",
            "pass/*note*/word=fixture-value",
            "api<!--note-->key=fixture-value",
            "client<!--note-->secret=fixture-value",
            '{"api":{"key":"fixture-value"}}',
            '{"private":{"key":"fixture-value"}}',
            '{"access":{"token":"fixture-value"}}',
            '{"client":{"secret":"fixture-value"}}',
            '{"authorization":{"header":"Basic fixture-value"}}',
            "PrIvAtE/KeY=fixture-value",
            "aPi/kEy=fixture-value",
            "AcCeSs/ToKeN=fixture-value",
            "ClIeNt/SeCrEt=fixture-value",
            "AuThOrIzAtIoN/HeAdEr=Basic fixture-value",
            "PaSs/WoRd=fixture-value",
            "api#note#key=fixture-value",
            "api # note\n key = fixture-value",
            "private# note\n key: fixture-value",
        )
        for value in secret_values:
            state = lease.empty_state()
            state["history"].append(
                {
                    "leaseId": "fixture",
                    "release": {"outcome": "completed"},
                    "callerControlled": ("safe-metadata", value),
                }
            )
            with self.subTest(value=value), self.assertRaisesRegex(
                lease.CapacityError, "secret-shaped-persistent-state"
            ):
                registry._write(state)
            self.assertFalse(self.state_file.exists())
            self.assertFalse(registry.lock_file.exists())

    def test_final_persistence_boundary_rejects_nested_credential_paths(self) -> None:
        registry = lease.LeaseRegistry(self.state_file, clock=self.clock)
        secret_structures = (
            {"api": {"key": "fixture-value"}},
            {"private": {"key": "fixture-value"}},
            {"access": {"token": "fixture-value"}},
            {"access": {"key": {"id": "fixture-value"}}},
            {"client": {"secret": "fixture-value"}},
            {"signing": {"key": "fixture-value"}},
            {"authorization": {"header": "Basic fixture-value"}},
            {"pass": {"word": "fixture-value"}},
            {"outer": [{"api": {"key": "fixture-value"}}]},
        )
        for value in secret_structures:
            state = lease.empty_state()
            state["history"].append(
                {
                    "leaseId": "fixture",
                    "release": {"outcome": "completed"},
                    "callerControlled": value,
                }
            )
            with self.subTest(value=value), self.assertRaisesRegex(
                lease.CapacityError, "secret-shaped-persistent-state"
            ):
                registry._write(state)
            self.assertFalse(self.state_file.exists())
            self.assertFalse(registry.lock_file.exists())

    def test_closed_state_rejects_benign_unknown_metadata(self) -> None:
        registry = lease.LeaseRegistry(self.state_file, clock=self.clock)
        state = lease.empty_state()
        state["history"].append(
            {
                "leaseId": "fixture",
                "release": {"outcome": "completed"},
                "callerControlled": {
                    "private": {"key": {"path": "fixture.pem"}},
                    "api": {"key": {"documentation": "approved"}},
                    "authorization": {"header": {"tests": "enabled"}},
                },
            }
        )

        with self.assertRaisesRegex(lease.CapacityError, "lease-state-invalid"):
            registry._write(state)

        self.assertFalse(self.state_file.exists())
        self.assertFalse(registry.lock_file.exists())

    def test_structured_secret_scan_limits_fail_closed_without_recursion(self) -> None:
        oversized_json = json.dumps(
            {"safe": "x" * lease.MAX_STRUCTURED_STRING_CHARS}
        )
        deeply_nested_json = "[" * 1_000 + "0" + "]" * 1_000
        too_many_nodes = ["safe"] * (lease.MAX_SECRET_SCAN_NODES + 1)

        self.assertTrue(lease.contains_secret(oversized_json))
        self.assertTrue(lease.contains_secret(deeply_nested_json))
        self.assertTrue(lease.structured_contains_secret(too_many_nodes))

    def test_benign_native_and_serialized_structures_have_matching_policy(
        self,
    ) -> None:
        safe_structures = (
            {"authorization": {"header": {"tests": "enabled"}}},
            {"private": {"key": {"path": "fixture.pem"}}},
            {"api": {"key": {"documentation": "approved"}}},
        )
        serialized = tuple(json.dumps(value) for value in safe_structures)
        for native, text in zip(safe_structures, serialized, strict=True):
            with self.subTest(native=native):
                self.assertFalse(lease.structured_contains_secret(native))
                self.assertFalse(lease.contains_secret(text))

    def test_positive_identifier_grammar_accepts_canonical_shapes(self) -> None:
        benign_values = (
            "password-policy-review",
            "api-key-documentation",
            "access-token-rotation",
            "authorization-header-tests",
            "bearer-capacity-worker",
            "token-budget-4096",
            "secret-scanner-critic",
            "/root/factory_4_builder",
            "prime:0198ff96-2e31-7c30-9eca-4d4f22265e90",
            "openai/gpt-5.6-sol:xhigh",
        )
        for value in benign_values:
            with self.subTest(value=value):
                self.assertEqual(
                    value, lease.validate_identifier("session-ref", value)
                )

    def test_positive_identifier_grammar_rejects_ambiguous_encodings(self) -> None:
        invalid_values = (
            "name=value",
            "name /* comment */ value",
            "name<!--comment-->value",
            "name # comment",
            '{"id":"worker"}',
            '["worker"]',
            " leading",
            "trailing ",
            "two words",
            "worker\nnext",
            "worker\x00next",
            "wörker",
            "a" * 301,
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(
                lease.CapacityError, "invalid-session-ref"
            ):
                lease.validate_identifier("session-ref", value)

    def test_identifier_native_and_serialized_forms_share_closed_policy(self) -> None:
        canonical = "/root/factory_4_builder"
        self.assertEqual(
            canonical, lease.validate_identifier("session-ref", canonical)
        )
        for value in (
            json.dumps(canonical),
            json.dumps({"sessionRef": canonical}),
            {"sessionRef": canonical},
            [canonical],
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                lease.CapacityError, "invalid-session-ref"
            ):
                lease.validate_identifier("session-ref", value)  # type: ignore[arg-type]

    def test_owner_accepts_codex_omp_prime_and_agent_deck_session_shapes(self) -> None:
        variants = (
            ("Codex", "codex:01a02f9b-04d8-7be2-bd52-db5f5bbb570d"),
            ("Codex", "01a02f9b-04d8-7be2-bd52-db5f5bbb570d"),
            ("OMP", "0198ff96-2e31-7c30-9eca-4d4f22265e90"),
            ("Prime Agent", "prime:0198ff96-2e31-7c30-9eca-4d4f22265e90"),
        )
        for harness, root_session_id in variants:
            candidate = owner()
            candidate["harness"] = harness
            candidate["rootSessionId"] = root_session_id
            with self.subTest(harness=harness, root_session_id=root_session_id):
                self.assertEqual(candidate, lease.validate_owner(candidate))

    def test_owner_rejects_unknown_harness_and_noncanonical_session_ids(self) -> None:
        mutations = (
            ("harness", "Prime"),
            ("harness", "codex"),
            ("rootSessionId", "root-session"),
            ("rootSessionId", "omp:0198ff96-2e31-7c30-9eca-4d4f22265e90"),
            ("rootSessionId", "0198ff96-2e31-4c30-9eca-4d4f22265e90"),
        )
        for field, value in mutations:
            candidate = owner()
            candidate[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(
                lease.CapacityError
            ):
                lease.validate_owner(candidate)

    def test_state_files_are_owner_only_and_store_no_prompt_or_output(self) -> None:
        self.manager().acquire(owner(), 900)
        mode = stat.S_IMODE(os.stat(self.state_file).st_mode)
        payload = self.state_file.read_text(encoding="utf-8")

        self.assertEqual(0o600, mode)
        self.assertEqual(0o700, stat.S_IMODE(os.stat(self.state_file.parent).st_mode))
        self.assertNotIn("Reply with exactly", payload)
        self.assertNotIn('"content":', payload)
        self.assertNotIn('"response"', payload)

    def test_roster_has_explicit_success_decision(self) -> None:
        result = self.manager().roster()
        self.assertEqual("listed", result["decision"])

    def test_closed_state_schema_rejects_unknown_fields_and_wrong_types(self) -> None:
        self.manager().acquire(owner(), 900)
        baseline = self.read_state()
        lease_id = next(iter(baseline["active"]))
        mutations = (
            lambda state: state.update({"extra": True}),
            lambda state: state.update({"version": True}),
            lambda state: state["active"][lease_id].update({"extra": True}),
            lambda state: state["active"][lease_id].update({"status": []}),
            lambda state: state["active"][lease_id].update({"generation": 2}),
            lambda state: state["active"][lease_id]["owner"].update(
                {"extra": "worker"}
            ),
            lambda state: state["active"][lease_id]["budget"].update(
                {"maxConcurrency": True}
            ),
            lambda state: state["active"][lease_id]["admission"]["canary"].update(
                {"responseStored": "false"}
            ),
            lambda state: state["active"][lease_id]["admission"]["canary"].update(
                {"outputChannels": [{}]}
            ),
            lambda state: state["active"][lease_id]["admission"].update(
                {"reservedSlots": 0}
            ),
            lambda state: state["active"][lease_id].update(
                {"heartbeats": "2026-08-23T20:00:00Z"}
            ),
            lambda state: state["active"][lease_id].update(
                {"expiresAt": "2026-08-23T20:00:30Z"}
            ),
            lambda state: state["active"][lease_id].update(
                {"acquiredAt": "0001-01-01T00:00:00+14:00"}
            ),
            lambda state: state.update({"history": {}}),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaisesRegex(
                lease.CapacityError, "lease-state-invalid"
            ):
                lease.validate_persisted_state(candidate)

    def test_closed_state_schema_rejects_invalid_terminal_lifecycle(self) -> None:
        manager = self.manager()
        acquired = manager.acquire(owner(), 900)
        manager.release(acquired["lease"]["leaseId"], owner(), "completed")
        baseline = self.read_state()
        mutations = (
            lambda state: state["history"][0].update({"status": "active"}),
            lambda state: state["history"][0]["release"].update(
                {"outcome": "expected-success"}
            ),
            lambda state: state["history"][0]["release"].update(
                {"reasonCodes": ["caller-defined"]}
            ),
            lambda state: state["history"][0]["release"].update(
                {"reasonCodes": [{}]}
            ),
            lambda state: state["history"][0]["release"].update({"extra": True}),
            lambda state: state["history"][0]["release"].update(
                {"releasedAt": "2026-08-23T19:59:59Z"}
            ),
            lambda state: state["history"][0]["release"].update(
                {"releasedAt": "2026-08-23T20:15:00Z"}
            ),
            lambda state: state["history"][0]["release"].update(
                {
                    "outcome": "completed",
                    "reasonCodes": ["production-gpu-load-high"],
                }
            ),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaisesRegex(
                lease.CapacityError, "lease-state-invalid"
            ):
                lease.validate_persisted_state(candidate)

    def test_registry_validates_after_load_and_before_write(self) -> None:
        manager = self.manager()
        manager.acquire(owner(), 900)
        valid_bytes = self.state_file.read_bytes()
        invalid = self.read_state()
        invalid["active"][next(iter(invalid["active"]))]["unknown"] = True
        self.state_file.write_text(json.dumps(invalid), encoding="utf-8")

        with self.assertRaisesRegex(lease.CapacityError, "lease-state-invalid"):
            manager.registry._load()

        self.state_file.write_bytes(valid_bytes)
        with (
            self.assertRaisesRegex(lease.CapacityError, "lease-state-invalid"),
            manager.registry.transaction() as state,
        ):
            state["unknown"] = True
        self.assertEqual(valid_bytes, self.state_file.read_bytes())

    def test_exact_version_one_state_migrates_and_incompatible_state_fails_closed(
        self,
    ) -> None:
        manager = self.manager()
        manager.acquire(owner(), 900)
        legacy = self.read_state()
        legacy["version"] = lease.LEGACY_STATE_VERSION
        self.state_file.write_text(
            json.dumps(legacy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        self.assertEqual("listed", manager.roster()["decision"])
        self.assertEqual(lease.STATE_VERSION, self.read_state()["version"])

        incompatible = copy.deepcopy(legacy)
        incompatible["unknown"] = "fixture"
        payload = json.dumps(incompatible, sort_keys=True).encode()
        self.state_file.write_bytes(payload)
        with self.assertRaisesRegex(lease.CapacityError, "lease-state-invalid"):
            manager.registry._load()
        self.assertEqual(payload, self.state_file.read_bytes())


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
