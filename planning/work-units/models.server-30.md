# Fast Qwen listener host

Work item: https://github.com/kortexa-ai/models.server/issues/30

Use Smarty LAN address 192.168.2.3 in both fast-model manifests. The old stock
manifest carries Snappy metadata, which must not be copied to a Linux-only
listener. Verify each configured address with a socket bind on Smarty without
loading the models or disturbing the Shingi GPU handoff.
