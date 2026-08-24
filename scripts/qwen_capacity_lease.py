#!/usr/bin/env python3
"""Fail-closed capacity leases for coding agents using Smarty's managed Qwen.

This program observes services and GPU state. It never starts, stops, restarts,
or signals a process. Lease consumers must heartbeat before every model request
and honor a denied or expired decision.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import json
import os
import re
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UTC = dt.timezone.utc
STATE_VERSION = 1
MAX_HISTORY = 256
MAX_HEARTBEATS = 1_000
MAX_HTTP_BYTES = 1_048_576
MAX_STATE_BYTES = 4_194_304
MAX_CANARY_PROMPT_TOKENS = 256
DEFAULT_STATE_FILE = Path.home() / ".local/state/kortexa-qwen-capacity/leases.json"
RELEASE_OUTCOMES = ("completed", "blocked", "failed", "cancelled")
PERSISTED_RELEASE_OUTCOMES = (*RELEASE_OUTCOMES, "expired")
GPU_SUMMARY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw,power.limit",
    "--format=csv,noheader,nounits",
)
GPU_PROCESS_COMMAND = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
)


class CapacityError(RuntimeError):
    """An expected fail-closed admission or lease error."""


@dataclass(frozen=True)
class Policy:
    endpoint: str = "http://127.0.0.1:2053"
    model: str = "qwen-3.8-27b"
    managed_service: str = "models/qwen-3.8-27b"
    managed_unit: str = "kortexa-ai-llm-qwen-3.8-27b.service"
    min_free_vram_mib: int = 16_384
    max_gpu_utilization_percent: int = 80
    reserved_slots: int = 1
    max_concurrency: int = 1
    max_context_tokens: int = 65_536
    max_output_tokens: int = 4_096
    min_ttl_seconds: int = 60
    max_ttl_seconds: int = 3_600


POLICY = Policy()

OWNER_FIELDS = {
    "actorId": "actor-id",
    "harness": "harness",
    "model": "model",
    "rootSessionId": "root-session-id",
    "delegatedWorkerId": "delegated-worker-id",
    "sessionRef": "session-ref",
}

HEALTH_ENDPOINTS = {
    "alt-image-gen.server/base": "http://127.0.0.1:4004/health",
    "vision.server": "http://127.0.0.1:4001/health",
    "models/gemma-4-e2b": "http://127.0.0.1:2039/health",
    "tts.server": "http://127.0.0.1:4003/health",
    "asr.server": "http://127.0.0.1:4002/health",
    "models/lfm2.5-vl-3b": "http://127.0.0.1:2055/health",
    "models/qwen-3.8-27b": "http://127.0.0.1:2053/health",
    "comfyui.server": "http://127.0.0.1:8050/system_stats",
}

GPU_UNIT_TO_SERVICE = {
    "kortexa-ai-comfyui-server.service": "comfyui.server",
    "kortexa-ai-llm-lfm2.5-vl-3b.service": "models/lfm2.5-vl-3b",
    "kortexa-ai-llm-gemma-4-e2b.service": "models/gemma-4-e2b",
    "kortexa-ai-llm-qwen-3.8-27b.service": "models/qwen-3.8-27b",
    "kortexa-ai-tts-server.service": "tts.server",
    "kortexa-ai-asr-server.service": "asr.server",
    "kortexa-ai-vision-server.service": "vision.server",
}

SECRET_VALUE_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@"),
)

JSON_UNICODE_ESCAPE_PATTERN = re.compile(r"\\u([0-9a-fA-F]{4})")
INTER_WORD_COMMENT_PATTERN = re.compile(r"/\*.*?\*/|<!--.*?-->", re.DOTALL)
PAIRED_HASH_COMMENT_PATTERN = re.compile(r"#[^#\r\n]*#")
HASH_LINE_COMMENT_PATTERN = re.compile(r"#[^\r\n]*(?:\r?\n|$)")
ASSIGNMENT_LEFT_BOUNDARIES = frozenset(",;{}\n\r")
MAX_ASSIGNMENT_KEY_CHARS = 300
MAX_STRUCTURED_STRING_CHARS = 65_536
MAX_SECRET_SCAN_CHARS = MAX_STATE_BYTES
MAX_SECRET_SCAN_NODES = 10_000
MAX_SECRET_SCAN_DEPTH = 64
MAX_SECRET_PATH_COMPONENTS = 4
SECRET_KEY_PHRASES = (
    ("access", "key", "id"),
    ("authorization", "header"),
    ("password", "hash"),
    ("passwd", "hash"),
    ("passphrase", "hash"),
    ("pass", "word"),
    ("api", "key"),
    ("access", "token"),
    ("access", "key"),
    ("client", "secret"),
    ("private", "key"),
    ("secret", "key"),
    ("signing", "key"),
    ("password",),
    ("passwordhash",),
    ("passwd",),
    ("passphrase",),
    ("credential",),
    ("credentials",),
    ("token",),
    ("secret",),
    ("authorization",),
)
SINGLE_SECRET_KEY_WORDS = {phrase[0] for phrase in SECRET_KEY_PHRASES if len(phrase) == 1}
COLLAPSED_SECRET_KEY_PHRASES = {
    "".join(phrase): phrase
    for phrase in SECRET_KEY_PHRASES
    if len(phrase) > 1 and "".join(phrase) not in SINGLE_SECRET_KEY_WORDS
}
COLLAPSED_SECRET_KEY_PATTERN = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in sorted(
            COLLAPSED_SECRET_KEY_PHRASES, key=len, reverse=True
        )
    )
)
BENIGN_CREDENTIAL_QUALIFIERS = {
    "budget",
    "documentation",
    "docs",
    "file",
    "id",
    "identifier",
    "name",
    "path",
    "policy",
    "ref",
    "reference",
    "rotation",
    "scanner",
    "status",
    "test",
    "tests",
    "type",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def timestamp(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def contains_secret(value: str) -> bool:
    return structured_contains_secret(value)


def flat_string_contains_secret(value: str) -> bool:
    value = decode_json_unicode_escapes(value)
    if any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS):
        return True
    representations = (value, normalize_inter_word_comments(value))
    for representation in representations:
        for key in assignment_key_candidates(representation):
            if is_secret_assignment_key(key):
                return True
    return False


def parsed_object_like(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    if len(stripped) > MAX_STRUCTURED_STRING_CHARS:
        raise CapacityError("structured-secret-scan-limit")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    except (MemoryError, RecursionError) as exc:
        raise CapacityError("structured-secret-scan-limit") from exc
    return parsed if isinstance(parsed, (dict, list)) else None


def structured_contains_secret(root: Any) -> bool:
    """Boundedly screen strings and semantic key paths in nested structures."""
    stack: list[tuple[Any, tuple[str, ...], int]] = [(root, (), 0)]
    nodes = 0
    scanned_chars = 0
    while stack:
        item, path, depth = stack.pop()
        nodes += 1
        if nodes > MAX_SECRET_SCAN_NODES or depth > MAX_SECRET_SCAN_DEPTH:
            return True
        if isinstance(item, dict):
            if len(stack) + len(item) > MAX_SECRET_SCAN_NODES:
                return True
            for key, child in item.items():
                if not isinstance(key, str):
                    key = str(key)
                scanned_chars += len(key)
                if scanned_chars > MAX_SECRET_SCAN_CHARS or flat_string_contains_secret(key):
                    return True
                child_path = (*path[-(MAX_SECRET_PATH_COMPONENTS - 1) :], key)
                stack.append((child, child_path, depth + 1))
            continue
        if isinstance(item, (list, tuple)):
            if len(stack) + len(item) > MAX_SECRET_SCAN_NODES:
                return True
            stack.extend((child, path, depth + 1) for child in item)
            continue

        if path and is_secret_assignment_key("/".join(path)):
            return True
        if not isinstance(item, str):
            continue
        scanned_chars += len(item)
        if scanned_chars > MAX_SECRET_SCAN_CHARS:
            return True
        try:
            parsed = parsed_object_like(item)
        except CapacityError:
            return True
        if parsed is not None:
            stack.append((parsed, path, depth))
        elif flat_string_contains_secret(item):
            return True
    return False


def decode_json_unicode_escapes(value: str) -> str:
    """Decode bounded JSON-style Unicode escapes for credential-key screening."""
    for _ in range(64):
        decoded = JSON_UNICODE_ESCAPE_PATTERN.sub(
            lambda match: chr(int(match.group(1), 16)), value
        )
        if decoded == value:
            break
        value = decoded
    return value


def assignment_key_candidates(value: str) -> Iterator[str]:
    """Extract bounded assignment keys without depending on wrapper syntax."""
    value = decode_json_unicode_escapes(value)
    last_value = len(value) - 1
    while last_value >= 0 and value[last_value].isspace():
        last_value -= 1
    for separator, character in enumerate(value):
        if character not in {":", "="} or separator >= last_value:
            continue
        lower_bound = max(-1, separator - MAX_ASSIGNMENT_KEY_CHARS - 1)
        start = separator
        while start - 1 > lower_bound:
            if value[start - 1] in ASSIGNMENT_LEFT_BOUNDARIES:
                break
            start -= 1
        candidate = value[start:separator].strip()
        if candidate:
            yield candidate


def normalize_inter_word_comments(key: str) -> str:
    key = INTER_WORD_COMMENT_PATTERN.sub(" ", key)
    key = PAIRED_HASH_COMMENT_PATTERN.sub(" ", key)
    return HASH_LINE_COMMENT_PATTERN.sub(" ", key)


def expand_collapsed_key_parts(parts: Iterator[str]) -> tuple[str, ...]:
    words: list[str] = []
    for part in parts:
        cursor = 0
        for match in COLLAPSED_SECRET_KEY_PATTERN.finditer(part):
            if match.start() > cursor:
                words.append(part[cursor : match.start()])
            words.extend(COLLAPSED_SECRET_KEY_PHRASES[match.group()])
            cursor = match.end()
        if cursor < len(part):
            words.append(part[cursor:])
    return tuple(words)


def assignment_key_word_variants(key: str) -> tuple[tuple[str, ...], ...]:
    """Return case-normalized and conventional camel-case key tokenizations."""
    key = normalize_inter_word_comments(decode_json_unicode_escapes(key))
    normalized = expand_collapsed_key_parts(
        iter(re.findall(r"[a-z0-9]+", key.lower()))
    )
    camel_key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", key)
    camel_key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", camel_key)
    camel = expand_collapsed_key_parts(
        iter(re.findall(r"[a-z0-9]+", camel_key.lower()))
    )
    return (normalized,) if camel == normalized else (normalized, camel)


def assignment_key_words(key: str) -> tuple[str, ...]:
    """Return the case-normalized semantic words for an assignment key."""
    return assignment_key_word_variants(key)[0]


def words_contain_secret_phrase(words: tuple[str, ...]) -> bool:
    index = 0
    while index < len(words):
        phrase = next(
            (
                candidate
                for candidate in SECRET_KEY_PHRASES
                if words[index : index + len(candidate)] == candidate
            ),
            None,
        )
        if phrase is None:
            index += 1
            continue
        trailing = words[index + len(phrase) :]
        if not trailing or trailing[0] not in BENIGN_CREDENTIAL_QUALIFIERS:
            return True
        index += len(phrase)
    return False


def is_secret_assignment_key(key: str) -> bool:
    return any(
        words_contain_secret_phrase(words)
        for words in assignment_key_word_variants(key)
    )


def validate_release_outcome(value: Any) -> str:
    if not isinstance(value, str) or value not in RELEASE_OUTCOMES:
        raise CapacityError("invalid-release-outcome")
    return value


def validate_persisted_state(state: dict[str, Any]) -> None:
    """Reject secret-shaped strings and invalid outcomes before writing state."""
    if structured_contains_secret(state):
        raise CapacityError("secret-shaped-persistent-state")

    for lease in [*state.get("active", {}).values(), *state.get("history", [])]:
        release = lease.get("release") if isinstance(lease, dict) else None
        if release is not None and (
            not isinstance(release, dict)
            or release.get("outcome") not in PERSISTED_RELEASE_OUTCOMES
        ):
            raise CapacityError("invalid-persisted-release-outcome")


def validate_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 300:
        raise CapacityError(f"invalid-{name}")
    if contains_secret(value):
        raise CapacityError(f"secret-shaped-{name}")
    return value


def owner_from_args(args: argparse.Namespace) -> dict[str, str]:
    return validate_owner(
        {
            "actorId": args.actor_id,
            "harness": args.harness,
            "model": args.owner_model,
            "rootSessionId": args.root_session_id,
            "delegatedWorkerId": args.delegated_worker_id,
            "sessionRef": args.session_ref,
        }
    )


def validate_owner(owner: dict[str, str]) -> dict[str, str]:
    if not isinstance(owner, dict) or set(owner) != set(OWNER_FIELDS):
        raise CapacityError("invalid-owner")
    return {
        field: validate_identifier(name, owner[field])
        for field, name in OWNER_FIELDS.items()
    }


def empty_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "active": {}, "history": []}


class LeaseRegistry:
    """A local JSON registry guarded by a POSIX advisory lock."""

    def __init__(
        self,
        state_file: Path = DEFAULT_STATE_FILE,
        clock: Callable[[], dt.datetime] = utc_now,
    ) -> None:
        self.state_file = state_file
        self.lock_file = state_file.with_suffix(state_file.suffix + ".lock")
        self.clock = clock

    def _load(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return empty_state()
        try:
            if self.state_file.stat().st_size > MAX_STATE_BYTES:
                raise CapacityError("lease-state-too-large")
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CapacityError("lease-state-unreadable") from exc
        if (
            not isinstance(state, dict)
            or state.get("version") != STATE_VERSION
            or not isinstance(state.get("active"), dict)
            or not isinstance(state.get("history"), list)
        ):
            raise CapacityError("lease-state-invalid")
        return state

    def _write(self, state: dict[str, Any]) -> None:
        validate_persisted_state(state)
        if len(state["history"]) > MAX_HISTORY:
            raise CapacityError("lease-history-cap-reached")
        self.state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state_file.parent, 0o700)
        payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
        fd, temp_name = tempfile.mkstemp(
            dir=self.state_file.parent,
            prefix=f".{self.state_file.name}.",
            text=True,
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.state_file)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[dict[str, Any]]:
        self.lock_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as lock:
            os.chmod(self.lock_file, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = self._load()
            try:
                yield state
                self._write(state)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def expire_stale(self, state: dict[str, Any]) -> None:
        now = self.clock()
        for lease_id, lease in list(state["active"].items()):
            if parse_timestamp(lease["expiresAt"]) > now:
                continue
            lease["status"] = "expired"
            lease["release"] = {
                "releasedAt": timestamp(now),
                "outcome": "expired",
                "reasonCodes": ["heartbeat-expired"],
            }
            state["history"].append(lease)
            del state["active"][lease_id]


class ReadOnlyRunner:
    """Run only the three commands needed for observation."""

    def run(self, args: Sequence[str], timeout: int = 10) -> str:
        command = tuple(args)
        allowed = (
            command == ("ktxsvc", "list")
            or command in {GPU_SUMMARY_COMMAND, GPU_PROCESS_COMMAND}
            or command == ("ss", "-H", "-ltnp", "sport = :2053")
        )
        if not allowed:
            raise CapacityError("non-read-only-command-rejected")
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise CapacityError(f"{command[0]}-observation-failed")
        return result.stdout


def parse_ktxsvc_list(output: str) -> dict[str, dict[str, bool]]:
    roster: dict[str, dict[str, bool]] = {}
    parent: str | None = None
    for raw in output.splitlines():
        parts = raw.split()
        if len(parts) != 4 or parts[-1] not in {"yes", "no", "-"}:
            continue
        name, installed, enabled, running = parts
        child = raw[:1].isspace()
        if child and parent:
            key = f"models/{name}" if parent == "models.server" else f"{parent}/{name}"
        else:
            key = name
            parent = name if {installed, enabled, running} == {"-"} else None
        roster[key] = {
            "installed": installed == "yes",
            "enabled": enabled == "yes",
            "running": running == "yes",
        }
    return roster


def parse_gpu_summary(output: str) -> dict[str, int]:
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 8:
        raise CapacityError("gpu-summary-invalid")
    row = [item.strip() for item in rows[0]]
    try:
        return {
            "index": int(row[0]),
            "memoryTotalMiB": int(row[2]),
            "memoryUsedMiB": int(row[3]),
            "memoryFreeMiB": int(row[4]),
            "utilizationPercent": int(row[5]),
            "powerDrawWatts": round(float(row[6])),
            "powerLimitWatts": round(float(row[7])),
        }
    except ValueError as exc:
        raise CapacityError("gpu-summary-invalid") from exc


def parse_gpu_processes(output: str) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) != 3:
            raise CapacityError("gpu-process-list-invalid")
        try:
            processes.append(
                {
                    "pid": int(row[0].strip()),
                    "processName": Path(row[1].strip()).name,
                    "usedMemoryMiB": int(row[2].strip()),
                }
            )
        except ValueError as exc:
            raise CapacityError("gpu-process-list-invalid") from exc
    return processes


def parse_canary_response(result: Any, elapsed_ms: int) -> dict[str, Any]:
    """Validate liveness without returning or retaining generated text."""
    if not isinstance(result, dict) or "error" in result:
        raise CapacityError("qwen-canary-invalid")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise CapacityError("qwen-canary-invalid")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise CapacityError("qwen-canary-invalid")
    message = choice["message"]
    if message.get("role") != "assistant":
        raise CapacityError("qwen-canary-invalid")

    channels: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        if content.strip():
            channels.append("content")
    elif isinstance(content, list):
        has_text = False
        for part in content:
            if (
                not isinstance(part, dict)
                or part.get("type") != "text"
                or not isinstance(part.get("text"), str)
            ):
                raise CapacityError("qwen-canary-invalid")
            has_text = has_text or bool(part["text"].strip())
        if has_text:
            channels.append("content-parts")
    else:
        raise CapacityError("qwen-canary-invalid")

    reasoning = message.get("reasoning_content", "")
    if not isinstance(reasoning, str):
        raise CapacityError("qwen-canary-invalid")
    if reasoning.strip():
        channels.append("reasoning-content")
    if not channels:
        raise CapacityError("qwen-canary-empty")

    usage = result.get("usage")
    if not isinstance(usage, dict):
        raise CapacityError("qwen-canary-invalid")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or not 0 <= prompt_tokens <= MAX_CANARY_PROMPT_TOKENS
        or not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or not 1 <= completion_tokens <= 8
    ):
        raise CapacityError("qwen-canary-budget-invalid")
    total_tokens = usage.get("total_tokens")
    if (
        not isinstance(total_tokens, int)
        or isinstance(total_tokens, bool)
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise CapacityError("qwen-canary-budget-invalid")

    return {
        "elapsedMs": elapsed_ms,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "outputChannels": channels,
        "responseStored": False,
    }


class LiveCollector:
    """Collect a bounded, secret-free Smarty admission snapshot."""

    def __init__(self, runner: ReadOnlyRunner | None = None, policy: Policy = POLICY) -> None:
        self.runner = runner or ReadOnlyRunner()
        self.policy = policy

    def _http_status(self, url: str) -> int:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return response.status
        except (OSError, urllib.error.URLError):
            return 0

    def _fetch_json(self, url: str) -> Any:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = response.read(MAX_HTTP_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise CapacityError("qwen-observation-failed") from exc
        if len(payload) > MAX_HTTP_BYTES:
            raise CapacityError("qwen-observation-too-large")
        try:
            return json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CapacityError("qwen-observation-invalid") from exc

    @staticmethod
    def _process_details(pid: int) -> tuple[str, str]:
        base = Path("/proc") / str(pid)
        try:
            cwd = os.readlink(base / "cwd")
        except OSError:
            cwd = ""
        unit = ""
        try:
            cgroup = (base / "cgroup").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            cgroup = ""
        match = re.search(r"system\.slice/([^/]+\.service)", cgroup)
        if match:
            unit = match.group(1)
        return cwd, unit

    def collect(self) -> dict[str, Any]:
        if socket.gethostname().split(".", 1)[0] != "smarty" or os.uname().sysname != "Linux":
            raise CapacityError("not-smarty-linux")

        roster = parse_ktxsvc_list(self.runner.run(("ktxsvc", "list")))
        gpu = parse_gpu_summary(self.runner.run(GPU_SUMMARY_COMMAND))
        gpu_processes = parse_gpu_processes(self.runner.run(GPU_PROCESS_COMMAND))
        for process in gpu_processes:
            cwd, unit = self._process_details(process["pid"])
            process["cwd"] = cwd
            process["unit"] = unit

        health: dict[str, bool] = {}
        for service, url in HEALTH_ENDPOINTS.items():
            if roster.get(service, {}).get("running"):
                health[service] = self._http_status(url) == 200

        models = self._fetch_json(f"{self.policy.endpoint}/v1/models")
        props = self._fetch_json(f"{self.policy.endpoint}/props")
        slots = self._fetch_json(f"{self.policy.endpoint}/slots")
        if not isinstance(slots, list):
            raise CapacityError("qwen-slots-invalid")
        if any(
            not isinstance(slot, dict)
            or not isinstance(slot.get("is_processing"), bool)
            for slot in slots
        ):
            raise CapacityError("qwen-slots-invalid")
        model_ids = {
            entry.get("id")
            for entry in models.get("data", [])
            if isinstance(entry, dict)
        } if isinstance(models, dict) else set()

        port_output = self.runner.run(("ss", "-H", "-ltnp", "sport = :2053"))
        port_pids = sorted({int(value) for value in re.findall(r"pid=(\d+)", port_output)})
        legolm = [
            {"pid": item["pid"], "cwd": item["cwd"], "unit": item["unit"]}
            for item in gpu_processes
            if "/legolm" in item["cwd"].lower() or "legolm" in item["unit"].lower()
        ]
        if any(
            details.get("running") and service.lower().startswith("legolm")
            for service, details in roster.items()
        ):
            legolm.append({"pid": None, "cwd": "", "unit": "managed-legolm"})

        unknown_gpu = [
            {"pid": item["pid"], "unit": item["unit"], "usedMemoryMiB": item["usedMemoryMiB"]}
            for item in gpu_processes
            if item["unit"] not in GPU_UNIT_TO_SERVICE
        ]
        inconsistent_managed_gpu = [
            {"pid": item["pid"], "unit": item["unit"]}
            for item in gpu_processes
            if item["unit"] in GPU_UNIT_TO_SERVICE
            and not roster.get(GPU_UNIT_TO_SERVICE[item["unit"]], {}).get("running")
        ]
        qwen_gpu_pids = [
            item["pid"] for item in gpu_processes if item["unit"] == self.policy.managed_unit
        ]
        return {
            "host": "smarty",
            "roster": roster,
            "health": health,
            "gpu": gpu,
            "gpuProcesses": [
                {
                    "pid": item["pid"],
                    "unit": item["unit"],
                    "usedMemoryMiB": item["usedMemoryMiB"],
                }
                for item in gpu_processes
            ],
            "unknownGpuProcesses": unknown_gpu,
            "inconsistentManagedGpuProcesses": inconsistent_managed_gpu,
            "legolmGpuOwners": legolm,
            "qwen": {
                "endpoint": self.policy.endpoint,
                "modelPresent": self.policy.model in model_ids,
                "totalSlots": props.get("total_slots") if isinstance(props, dict) else None,
                "observedSlots": len(slots),
                "busySlots": sum(slot["is_processing"] for slot in slots),
                "portPids": port_pids,
                "gpuPids": qwen_gpu_pids,
            },
        }

    def canary(self) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.policy.model,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 8,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.policy.endpoint}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read(MAX_HTTP_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise CapacityError("qwen-canary-failed") from exc
        elapsed_ms = round((time.monotonic() - started) * 1_000)
        if len(payload) > MAX_HTTP_BYTES:
            raise CapacityError("qwen-canary-too-large")
        try:
            result = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CapacityError("qwen-canary-invalid") from exc
        return parse_canary_response(result, elapsed_ms)


def evaluate_snapshot(
    snapshot: dict[str, Any],
    active_lease_count: int,
    policy: Policy = POLICY,
) -> list[str]:
    reasons: list[str] = []
    if snapshot.get("host") != "smarty":
        reasons.append("wrong-host")
    roster = snapshot.get("roster", {})
    if not roster.get(policy.managed_service, {}).get("running"):
        reasons.append("managed-qwen-not-running")
    unhealthy = sorted(service for service, healthy in snapshot.get("health", {}).items() if not healthy)
    if unhealthy:
        reasons.append("protected-service-unhealthy")
    if policy.managed_service not in snapshot.get("health", {}):
        reasons.append("qwen-health-unavailable")

    gpu = snapshot.get("gpu", {})
    if gpu.get("memoryFreeMiB", -1) < policy.min_free_vram_mib:
        reasons.append("insufficient-vram-headroom")
    if gpu.get("utilizationPercent", 101) > policy.max_gpu_utilization_percent:
        reasons.append("production-gpu-load-high")
    if snapshot.get("unknownGpuProcesses"):
        reasons.append("unknown-cuda-process")
    if snapshot.get("inconsistentManagedGpuProcesses"):
        reasons.append("managed-cuda-roster-mismatch")
    if snapshot.get("legolmGpuOwners"):
        reasons.append("legolm-active")

    qwen = snapshot.get("qwen", {})
    if not qwen.get("modelPresent"):
        reasons.append("wrong-qwen-endpoint")
    total_slots = qwen.get("totalSlots")
    observed_slots = qwen.get("observedSlots")
    busy_slots = qwen.get("busySlots")
    if (
        not isinstance(total_slots, int)
        or not isinstance(observed_slots, int)
        or not isinstance(busy_slots, int)
        or observed_slots != total_slots
    ):
        reasons.append("qwen-slot-state-unknown")
    elif busy_slots + policy.max_concurrency + policy.reserved_slots > total_slots:
        reasons.append("qwen-request-capacity-busy")
    port_pids = qwen.get("portPids", [])
    gpu_pids = qwen.get("gpuPids", [])
    if len(port_pids) != 1 or port_pids != gpu_pids:
        reasons.append("qwen-port-owner-mismatch")
    if active_lease_count:
        reasons.append("agent-capacity-already-leased")
    return sorted(set(reasons))


class LeaseManager:
    def __init__(
        self,
        registry: LeaseRegistry,
        collector: LiveCollector,
        policy: Policy = POLICY,
    ) -> None:
        self.registry = registry
        self.collector = collector
        self.policy = policy

    def _ttl(self, ttl_seconds: int) -> int:
        if not self.policy.min_ttl_seconds <= ttl_seconds <= self.policy.max_ttl_seconds:
            raise CapacityError("ttl-out-of-range")
        return ttl_seconds

    def probe(self) -> dict[str, Any]:
        with self.registry.transaction() as state:
            self.registry.expire_stale(state)
            snapshot = self.collector.collect()
            reasons = evaluate_snapshot(snapshot, len(state["active"]), self.policy)
            return {
                "decision": "admit" if not reasons else "queue",
                "reasonCodes": reasons,
                "snapshot": public_snapshot(snapshot),
                "activeLeaseCount": len(state["active"]),
            }

    def acquire(self, owner: dict[str, str], ttl_seconds: int) -> dict[str, Any]:
        owner = validate_owner(owner)
        ttl_seconds = self._ttl(ttl_seconds)
        with self.registry.transaction() as state:
            self.registry.expire_stale(state)
            pre = self.collector.collect()
            reasons = evaluate_snapshot(pre, len(state["active"]), self.policy)
            if reasons:
                return {"decision": "queue", "reasonCodes": reasons}
            try:
                canary = self.collector.canary()
                post = self.collector.collect()
            except CapacityError as exc:
                return {"decision": "queue", "reasonCodes": [str(exc)]}
            reasons = evaluate_snapshot(post, len(state["active"]), self.policy)
            if pre["qwen"]["portPids"] != post["qwen"]["portPids"]:
                reasons.append("qwen-process-changed")
            if reasons:
                return {"decision": "queue", "reasonCodes": sorted(set(reasons))}

            now = self.registry.clock()
            lease_id = f"qwen-{secrets.token_hex(12)}"
            lease = {
                "leaseId": lease_id,
                "status": "active",
                "generation": 1,
                "owner": owner,
                "endpoint": self.policy.endpoint,
                "model": self.policy.model,
                "budget": {
                    "maxConcurrency": self.policy.max_concurrency,
                    "maxContextTokens": self.policy.max_context_tokens,
                    "maxOutputTokens": self.policy.max_output_tokens,
                },
                "acquiredAt": timestamp(now),
                "heartbeatAt": timestamp(now),
                "expiresAt": timestamp(now + dt.timedelta(seconds=ttl_seconds)),
                "heartbeats": [timestamp(now)],
                "release": None,
                "admission": {
                    "qwenPid": post["qwen"]["portPids"][0],
                    "freeVramMiB": post["gpu"]["memoryFreeMiB"],
                    "totalSlots": post["qwen"]["totalSlots"],
                    "reservedSlots": self.policy.reserved_slots,
                    "canary": canary,
                },
            }
            state["active"][lease_id] = lease
            return {"decision": "admit", "lease": lease}

    def heartbeat(
        self, lease_id: str, owner: dict[str, str], ttl_seconds: int
    ) -> dict[str, Any]:
        owner = validate_owner(owner)
        ttl_seconds = self._ttl(ttl_seconds)
        with self.registry.transaction() as state:
            self.registry.expire_stale(state)
            lease = state["active"].get(lease_id)
            if lease is None:
                return {"decision": "blocked", "reasonCodes": ["lease-not-active"]}
            if lease["owner"] != owner:
                return {"decision": "blocked", "reasonCodes": ["lease-owner-mismatch"]}
            snapshot = self.collector.collect()
            reasons = evaluate_snapshot(snapshot, len(state["active"]) - 1, self.policy)
            if reasons:
                self._release_locked(state, lease, "blocked", reasons)
                return {"decision": "blocked", "reasonCodes": reasons}
            if len(lease["heartbeats"]) >= MAX_HEARTBEATS:
                reasons = ["heartbeat-history-cap-reached"]
                self._release_locked(state, lease, "blocked", reasons)
                return {"decision": "blocked", "reasonCodes": reasons}
            now = self.registry.clock()
            lease["heartbeatAt"] = timestamp(now)
            lease["expiresAt"] = timestamp(now + dt.timedelta(seconds=ttl_seconds))
            lease["heartbeats"].append(timestamp(now))
            return {"decision": "admit", "lease": lease}

    def release(
        self, lease_id: str, owner: dict[str, str], outcome: str
    ) -> dict[str, Any]:
        owner = validate_owner(owner)
        outcome = validate_release_outcome(outcome)
        with self.registry.transaction() as state:
            self.registry.expire_stale(state)
            lease = state["active"].get(lease_id)
            if lease is None:
                prior = next(
                    (item for item in reversed(state["history"]) if item["leaseId"] == lease_id),
                    None,
                )
                if prior and prior["owner"] == owner:
                    return {"decision": "released", "lease": prior}
                return {"decision": "blocked", "reasonCodes": ["lease-not-found"]}
            if lease["owner"] != owner:
                return {"decision": "blocked", "reasonCodes": ["lease-owner-mismatch"]}
            self._release_locked(state, lease, outcome, [])
            return {"decision": "released", "lease": lease}

    def _release_locked(
        self,
        state: dict[str, Any],
        lease: dict[str, Any],
        outcome: str,
        reasons: list[str],
    ) -> None:
        lease["status"] = "released"
        lease["release"] = {
            "releasedAt": timestamp(self.registry.clock()),
            "outcome": outcome,
            "reasonCodes": reasons,
        }
        state["history"].append(lease)
        del state["active"][lease["leaseId"]]

    def roster(self) -> dict[str, Any]:
        with self.registry.transaction() as state:
            self.registry.expire_stale(state)
            return {
                "decision": "listed",
                "active": list(state["active"].values()),
                "history": state["history"],
            }


def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "host": snapshot["host"],
        "runningServices": sorted(
            name for name, state in snapshot["roster"].items() if state["running"]
        ),
        "serviceHealth": snapshot["health"],
        "gpu": snapshot["gpu"],
        "gpuProcesses": snapshot["gpuProcesses"],
        "unknownGpuProcessCount": len(snapshot["unknownGpuProcesses"]),
        "managedGpuRosterMismatchCount": len(
            snapshot["inconsistentManagedGpuProcesses"]
        ),
        "legolmGpuOwnerCount": len(snapshot["legolmGpuOwners"]),
        "qwen": snapshot["qwen"],
    }


def add_owner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--owner-model", required=True)
    parser.add_argument("--root-session-id", required=True)
    parser.add_argument("--delegated-worker-id", required=True)
    parser.add_argument("--session-ref", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe", help="show current fail-closed admission state")
    subparsers.add_parser("list", help="show the bounded lease roster and history")

    acquire = subparsers.add_parser("acquire", help="measure and acquire one lease")
    add_owner_arguments(acquire)
    acquire.add_argument("--ttl-seconds", type=int, default=900)

    heartbeat = subparsers.add_parser("heartbeat", help="recheck safety and extend a lease")
    add_owner_arguments(heartbeat)
    heartbeat.add_argument("--lease-id", required=True)
    heartbeat.add_argument("--ttl-seconds", type=int, default=900)

    release = subparsers.add_parser("release", help="idempotently release a lease")
    add_owner_arguments(release)
    release.add_argument("--lease-id", required=True)
    release.add_argument(
        "--outcome",
        choices=RELEASE_OUTCOMES,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manager = LeaseManager(LeaseRegistry(), LiveCollector())
    try:
        if args.command == "probe":
            result = manager.probe()
        elif args.command == "list":
            result = manager.roster()
        elif args.command == "acquire":
            result = manager.acquire(owner_from_args(args), args.ttl_seconds)
        elif args.command == "heartbeat":
            result = manager.heartbeat(
                args.lease_id, owner_from_args(args), args.ttl_seconds
            )
        else:
            result = manager.release(args.lease_id, owner_from_args(args), args.outcome)
    except CapacityError as exc:
        result = {"decision": "blocked", "reasonCodes": [str(exc)]}
    except (KeyError, IndexError, OSError, subprocess.SubprocessError, TypeError, ValueError):
        result = {"decision": "blocked", "reasonCodes": ["internal-observation-error"]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("decision") in {"admit", "released", "listed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
