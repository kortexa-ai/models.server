# Bonsai 2 27B GPU placement

Work unit: https://github.com/kortexa-ai/models.server/issues/24

Bind the Bonsai systemd service on smarty to RTX PRO 6000 UUID
`GPU-a71210ca-e14a-755a-88bb-77f53a2102f6`. Check that UUID before starting
the isolated Prism runtime on port 2062.

Validate the service definition and existing Bonsai runtime tests. Deliver
through Git and `ktxsvc`, then verify endpoint health, a completion, and the
running process's GPU UUID. Keep deployment evidence in the linked issue.

If installation fails, stop and uninstall only Bonsai through `ktxsvc` to
restore its prior stopped and uninstalled state.
