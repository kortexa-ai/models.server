#!/usr/bin/env python3
"""Check SM120 packed-FP8 attention and two-shard LSE merging on one GPU.

Run inside the pinned experimental image with the model stopped. This tests
the kernel arithmetic, including empty local shards. It is not a distributed
model test and does not establish model-level DCP compatibility.
"""
import json
import math
from types import SimpleNamespace

import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
import vllm._custom_ops as ops


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    device = "cuda"
    tokens, rows, heads, dimension = 8, 4096, 64, 512
    lengths = [0, 1, 3, 4, 5, 63, 257, 2048]
    query = torch.randn(tokens, heads, dimension, device=device,
                        dtype=torch.bfloat16) * 0.5
    latent = torch.randn(rows, dimension, device=device,
                         dtype=torch.bfloat16) * 0.5
    packed = torch.zeros(rows // 64, 64, 656, dtype=torch.uint8, device=device)
    ops.concat_and_cache_mla(
        latent, torch.empty(rows, 0, dtype=torch.bfloat16, device=device),
        packed, torch.arange(rows, dtype=torch.int64, device=device),
        "fp8_ds_mla", torch.ones(1, dtype=torch.float32, device=device))
    flat = packed.view(rows, 656)
    scales = flat[:, 512:528].contiguous().view(torch.float32)
    values = (flat[:, :512].contiguous().view(torch.float8_e4m3fn).float()
              * scales.repeat_interleave(128, dim=1))
    indices = torch.full((tokens, 2048), -1, dtype=torch.int32, device=device)
    for token, length in enumerate(lengths):
        # Short rows exercise an empty rank; long rows span both shards.
        indices[token, :length] = torch.arange(length, device=device)
    workspace = torch.zeros(128 * 1024 ** 2, dtype=torch.int8, device=device)
    scale = dimension ** -0.5

    def run(cache, slots):
        result, lse = trtllm_batch_decode_with_kv_cache_mla(
            query=query.unsqueeze(1), kv_cache=cache.unsqueeze(1),
            workspace_buffer=workspace, qk_nope_head_dim=128,
            kv_lora_rank=512, qk_rope_head_dim=0,
            block_tables=slots.unsqueeze(1), seq_lens=None,
            max_seq_len=2048, sparse_mla_top_k=2048,
            bmm1_scale=scale, bmm2_scale=1.0,
            kv_scale_format="arbitrary_fp32", return_lse=True)
        torch.cuda.synchronize()
        result, lse = result.squeeze(1).float(), lse.reshape(tokens, heads)
        # FlashInfer leaves all-masked rows unspecified. An adapter must
        # supply the neutral attention result before the cross-rank merge.
        empty = (slots < 0).all(dim=-1)
        result = torch.where(empty[:, None, None], 0.0, result)
        lse = torch.where(empty[:, None], -torch.inf, lse)
        return result, lse

    full, full_lse = run(packed, indices)
    reference, reference_lse = torch.zeros_like(full), torch.full_like(full_lse, -torch.inf)
    for token, length in enumerate(lengths):
        if length:
            scores = query[token].float() @ values[:length].T * scale
            reference[token] = torch.softmax(scores, -1) @ values[:length]
            reference_lse[token] = torch.logsumexp(scores, -1)
    active = torch.isfinite(reference_lse)
    natural_error = (full_lse[active] - reference_lse[active]).abs().max().item()
    binary_error = (full_lse[active] * math.log(2) - reference_lse[active]).abs().max().item()
    base_e = natural_error < binary_error
    print(json.dumps({"natural_lse_error": natural_error,
                      "binary_lse_error": binary_error,
                      "reference_max_abs_error": (full - reference).abs().max().item(),
                      "reference_row_errors": (full - reference).abs().flatten(1).amax(1).tolist()}), flush=True)
    # SM120 quantizes queries internally; comparison with the FP32 reference
    # includes that rounding. Shard merging below has its own tighter check.
    assert min(natural_error, binary_error) < 0.04, (natural_error, binary_error)
    assert torch.isfinite(full).all() and torch.equal(full[0], torch.zeros_like(full[0]))
    assert torch.isneginf(full_lse[0]).all()
    torch.testing.assert_close(full, reference, atol=0.02, rtol=0.05)
    results = []
    row_ids = torch.arange(rows, device=device)
    for interleave in (1, 4, 8):
        outputs, log_sums = [], []
        for rank in (0, 1):
            local_cache = flat[(row_ids // interleave) % 2 == rank].contiguous().view(-1, 64, 656)
            owned = (indices >= 0) & ((indices // interleave) % 2 == rank)
            local_slots = torch.where(
                owned, indices // (2 * interleave) * interleave + indices % interleave, -1)
            output, lse = run(local_cache, local_slots)
            assert torch.isfinite(output).all(), "Nonfinite output from empty local shard"
            outputs.append(output)
            log_sums.append(lse if base_e else lse * math.log(2))
        logs = torch.stack(log_sums)
        weights = torch.nan_to_num(torch.softmax(logs, dim=0), nan=0.0)
        merged = (torch.stack(outputs) * weights.unsqueeze(-1)).sum(0)
        print(json.dumps({"interleave": interleave,
                          "merge_max_abs_error": (merged - full).abs().max().item()}), flush=True)
        torch.testing.assert_close(merged, full, atol=0.003, rtol=0.02)
        results.append({"interleave": interleave,
                        "max_abs_merge_error": (merged - full).abs().max().item()})
    import sm120_dcp
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as backend

    sm120_dcp.install(backend)
    adapter_results = []
    for interleave in (1, 4, 8):
        outputs, log_sums = [], []
        for rank in (0, 1):
            cache = flat[(row_ids // interleave) % 2 == rank].contiguous().view(-1, 64, 656)
            impl = SimpleNamespace(
                dcp_world_size=2, dcp_rank=rank, pcp_world_size=1, index_group=None,
                topk_indices_buffer=indices, _workspace_buffer=workspace,
                kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=0,
                scale=scale, kv_scale_format="arbitrary_fp32")
            metadata = SimpleNamespace(
                block_size=64, cp_kv_cache_interleave_size=interleave,
                req_id_per_token=torch.zeros(tokens, device=device, dtype=torch.int32),
                block_table=torch.arange(rows // 128, device=device,
                                         dtype=torch.int32).unsqueeze(0))
            output, lse = backend.FlashInferMLASparseSM120Impl.forward_mqa(
                impl, query, cache, metadata, None)
            outputs.append(output.float())
            log_sums.append(lse * math.log(2))
        weights = torch.nan_to_num(torch.softmax(torch.stack(log_sums), dim=0), nan=0.0)
        merged = (torch.stack(outputs) * weights.unsqueeze(-1)).sum(0)
        torch.testing.assert_close(merged, full, atol=0.003, rtol=0.02)
        adapter_results.append({"interleave": interleave,
                                "max_abs_merge_error": (merged - full).abs().max().item()})
    print(json.dumps({"passed": True, "device": torch.cuda.get_device_name(),
                      "lse_base_e": base_e, "natural_lse_error": natural_error,
                      "binary_lse_error": binary_error,
                      "max_abs_reference_error": (full - reference).abs().max().item(),
                      "shard_checks": results,
                      "adapter_checks": adapter_results,
                      "scope": "single-GPU arithmetic; distributed model remains untested"}), flush=True)


if __name__ == "__main__":
    main()
