"""Optional SM120 DCP adapter for the pinned vLLM/FlashInfer experiment.

Reuse upstream's DCP index conversion and cross-rank reduction. FlashInfer
0.7.0.post1 supplies base-2 LSE on SM120, but upstream's SM120 backend does
not expose it. Keep DCP=1 on the original implementation. No PCP or HiSparse.
"""
import functools


def install(module):
    cls = module.FlashInferMLASparseSM120Impl
    original = cls.forward_mqa

    @functools.wraps(original)
    def forward(self, q, kv_cache, metadata, layer):
        if self.dcp_world_size == 1:
            return original(self, q, kv_cache, metadata, layer)
        import torch
        from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
        from vllm.v1.attention.backends.mla.sparse_utils import (
            flat_kv_row_view, triton_filter_and_convert_dcp_index,
        )
        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        if self.pcp_world_size != 1 or isinstance(self.index_group, HiSparseMLAIndexGroup):
            raise NotImplementedError("Experimental SM120 DCP excludes PCP and HiSparse")
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        tokens, heads = q.shape[:2]
        indices = self.topk_indices_buffer[:tokens]
        rows, stride = flat_kv_row_view(kv_cache, metadata.block_size)
        slots, counts = triton_filter_and_convert_dcp_index(
            metadata.req_id_per_token[:tokens], metadata.block_table, indices,
            dcp_size=self.dcp_world_size, dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=metadata.cp_kv_cache_interleave_size,
            BLOCK_SIZE=metadata.block_size, BLOCK_STRIDE_ROWS=stride,
            NUM_TOPK_TOKENS=indices.shape[1], return_valid_counts=True)
        if self._workspace_buffer is None:
            self._workspace_buffer = module._get_workspace_buffer(q.device)
        output = q.new_empty((tokens, heads, self.kv_lora_rank))
        output, lse = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=rows.view(-1, 64, rows.shape[-1]).view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank, qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=slots.unsqueeze(1), seq_lens=None,
            max_seq_len=indices.shape[1], out=output.unsqueeze(1),
            bmm1_scale=self.scale, bmm2_scale=1.0,
            sparse_mla_top_k=indices.shape[1], kv_scale_format=self.kv_scale_format,
            return_lse=True)
        # The native kernel does not initialize all-masked rows. Neutralize
        # empty local shards before the upstream DCP softmax merge.
        empty = counts == 0
        output = torch.where(empty[:, None, None], 0.0, output.squeeze(1))
        lse = torch.where(empty[:, None], -torch.inf, lse.reshape(tokens, heads))
        return output, lse

    cls.forward_mqa = forward
    cls.supports_dcp = True
    cls.can_return_lse_for_decode = True
    cls.lse_base_on_e = False
    cls.supports_mtp_with_cp_non_trivial_interleave_size = True
    print("glm53-main: experimental SM120 DCP adapter enabled", flush=True)
