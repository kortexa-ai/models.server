#!/usr/bin/env python3
"""Bounded streaming/tool probes. Generated tools are never executed."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def stream(base, payload):
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    result = {"content": "", "reasoning_content": "", "tool_calls": [],
              "done": False, "usage": {}, "finish_reason": None}
    calls = {}
    first = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                result["done"] = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if any(delta.get(key) for key in ("content", "reasoning", "reasoning_content", "tool_calls")):
                    first = first or time.monotonic()
                result["content"] += delta.get("content") or ""
                result["reasoning_content"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
                for part in delta.get("tool_calls", []):
                    call = calls.setdefault(part["index"], {"id": "", "type": "function",
                                                          "function": {"name": "", "arguments": ""}})
                    call["id"] += part.get("id") or ""
                    for key in ("name", "arguments"):
                        call["function"][key] += part.get("function", {}).get(key) or ""
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
    result["tool_calls"] = [calls[index] for index in sorted(calls)]
    result["elapsed_s"] = round(time.monotonic() - start, 3)
    result["ttft_s"] = round(first - start, 3) if first else None
    if not result["done"]:
        raise RuntimeError("Missing SSE terminator")
    if result["tool_calls"] and result["finish_reason"] != "tool_calls":
        raise RuntimeError("Tool calls have the wrong finish reason")
    for call in result["tool_calls"]:
        if not call["id"] or not call["function"]["name"]:
            raise RuntimeError("Incomplete streamed tool identity")
        json.loads(call["function"]["arguments"])
    return result


def assistant(result):
    return {"role": "assistant", "content": result["content"] or None,
            "reasoning_content": result["reasoning_content"],
            "tool_calls": result["tool_calls"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-file", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output file; evidence must not be overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    def request(label, messages, **extra):
        body = {"model": args.model, "messages": messages, "temperature": 0,
                "max_tokens": 1024, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"reasoning_effort": "low"}, **extra}
        response = stream(args.base, body)
        response["label"] = label
        response["request_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        results.append(response)
        args.output.write_text(json.dumps({"base": args.base, "model": args.model,
                                          "results": results}, indent=2))
        print(json.dumps({"label": label, "elapsed_s": response["elapsed_s"],
                          "finish_reason": response["finish_reason"],
                          "tool_calls": len(response["tool_calls"])}), flush=True)
        return response

    tool = {"type": "function", "function": {"name": "read_inventory",
            "description": "Read the inventory for a named bin. Empty output means the bin has no entries.",
            "parameters": {"type": "object", "properties": {"bin": {"type": "string"}},
                           "required": ["bin"]}}}
    messages = [{"role": "system", "content": "Use tools when requested. An empty tool result is valid. Do not repeat an unchanged lookup. "
                 "If a lookup returns an error with a corrected bin, make exactly one corrected call."},
                {"role": "user", "content": "Read inventory for bin violet-17 using read_inventory."}]
    result = request("streamed_auto_tool", messages, tools=[tool], tool_choice="auto")
    assert len(result["tool_calls"]) == 1, "Expected one parsed call"
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "read_inventory"
    assert json.loads(call["function"]["arguments"])["bin"] == "violet-17"
    messages += [assistant(result), {"role": "tool", "tool_call_id": call["id"], "content": ""},
                 {"role": "user", "content": "The empty result is valid. Reply EMPTY_OK and do not retry."}]
    result = request("empty_result_no_repeat", messages, tools=[tool], tool_choice="auto")
    assert not result["tool_calls"] and "EMPTY_OK" in result["content"]
    messages += [assistant(result), {"role": "user", "content": "Now read bin copper-42 using the tool."}]
    result = request("next_tool_after_empty", messages, tools=[tool], tool_choice="auto")
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert json.loads(call["function"]["arguments"])["bin"] == "copper-42"
    messages += [assistant(result), {"role": "tool", "tool_call_id": call["id"],
                 "content": '{"error":"renamed","correct_bin":"copper-43"}'}]
    result = request("tool_error_correction", messages, tools=[tool], tool_choice="auto")
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert json.loads(call["function"]["arguments"])["bin"] == "copper-43"
    messages += [assistant(result), {"role": "tool", "tool_call_id": call["id"],
                 "content": '{"count":23,"verification":"COPPER_VERIFIED_43"}'},
                 {"role": "user", "content": "Report the count and verification string, then stop."}]
    result = request("nonempty_result_continuation", messages, tools=[tool], tool_choice="auto")
    assert not result["tool_calls"]
    assert "23" in result["content"] and "COPPER_VERIFIED_43" in result["content"]

    if args.replay_file:
        body = json.loads(args.replay_file.read_text())
        body.update(model=args.model, temperature=0, max_tokens=2048, stream=True,
                    stream_options={"include_usage": True})
        result = stream(args.base, body)
        result.update(label="reconstructed_omp_boundary", reconstructed=True,
                      executed_generated_tools=False,
                      source_sha256=hashlib.sha256(args.replay_file.read_bytes()).hexdigest())
        results.append(result)
        assert result["finish_reason"] in ("tool_calls", "stop")
        assert result["tool_calls"] or result["content"], "No usable continuation"
        # This is one bounded continuation, not proof against future tool loops.

    args.output.write_text(json.dumps({"base": args.base, "model": args.model,
                                      "passed": True, "results": results}, indent=2))
    print(json.dumps({"passed": True, "probes": len(results)}), flush=True)


if __name__ == "__main__":
    main()
