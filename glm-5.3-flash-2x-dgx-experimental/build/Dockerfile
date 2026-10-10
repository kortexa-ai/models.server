# Source and native binaries are pinned separately. The native-source check
# rejects using the older wheel when compiled inputs differ.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:f853a6833fea509edf1298cad31b630003c242a8b2b70c353e831fa716c8da0b
FROM local/glm53-tp2:770d115-20261009 AS transport
FROM ${BASE_IMAGE}

ARG VLLM_REVISION=276fbcff2717bd934cfa37c8a2e4c391f3e7237b
ARG NATIVE_REVISION=187a0eb98aa42341d703f83421d693fa7585581b
ENV VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=${NATIVE_REVISION} \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    MAX_JOBS=2

RUN apt-get update \
 && apt-get install --yes --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git init /opt/vllm-main \
 && git -C /opt/vllm-main remote add origin https://github.com/vllm-project/vllm.git \
 && git -C /opt/vllm-main fetch --depth=2 origin ${VLLM_REVISION} \
 && git -C /opt/vllm-main checkout --detach ${VLLM_REVISION}
COPY prepare_source.py /opt/glm53-build/prepare_source.py
RUN python3 /opt/glm53-build/prepare_source.py /opt/vllm-main ${VLLM_REVISION} ${NATIVE_REVISION} \
 && cd /opt/vllm-main \
 && uv pip install --system -r requirements/build/cuda.txt \
 && uv pip install --system --no-deps --no-build-isolation .

# Keep the existing collective implementation available as a separate test
# arm. It is disabled until the new vLLM interfaces pass its compatibility test.
COPY --from=transport /opt/glm-roce /opt/glm-roce
COPY --from=transport /opt/glm-roce-b12x /opt/glm-roce-b12x
COPY qmix.py /opt/glm53-main/qmix.py
COPY glm_fast_load.py /opt/glm53-main/glm_fast_load.py
COPY LICENSE.knapcio NOTICE.knapcio /opt/glm53-main/
COPY cache_audit.py /opt/glm53-main/cache_audit.py
COPY sm120_dcp.py /opt/glm53-main/sm120_dcp.py
COPY draft_cache.py /opt/glm53-main/draft_cache.py
COPY sitecustomize.py /opt/glm53-main/sitecustomize.py
ENV PYTHONPATH=/opt/glm53-main \
    GLM_ROCE_ALLREDUCE=0 \
    GLM_FAST_LOAD=0 \
    GLM_SM120_DCP=0 \
    B12X_ROCE_CACHE_DIR=/opt/glm-roce-b12x/proxy-cache
RUN python3 -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)'
LABEL ai.kortexa.glm53.vllm-revision=${VLLM_REVISION} \
      ai.kortexa.glm53.native-revision=${NATIVE_REVISION}
