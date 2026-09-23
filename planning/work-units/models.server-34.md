# Unattended model archive

Work order: https://github.com/kortexa-ai/models.server/issues/34

Prepare the approved original/NVFP4 file inventory for the external WD Elements
disk on Smarty. Pin revisions, reject inventory drift, and merge duplicate
repository selections. Keep LFM within 2.5 and source SAM 3 from ModelScope.

Use a separate managed archival service with sequential, resumable transfers,
checksums, atomic progress, retries, mount identity and free-space guards.
Keep serving caches and production services unchanged. Test failure/resume and
checksum rejection, deploy through Git, then verify progress across a service
restart and a fresh SSH connection. The source inventory and runtime progress
live on the external disk; delivery evidence belongs in the work order.
