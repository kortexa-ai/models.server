# One GLM identity across three paired recipes

Owning issue: https://github.com/kortexa-ai/models.server/issues/62
Public catalog integration: https://github.com/kortexa-ai/api.server/issues/106

Preserve all three inference configurations and image pins. Keep distinct
recipe IDs for services, containers and setup receipts, with one served model
name (`glm-5.3-flash-2x-dgx`) and Static API port 2070. Select the 1M recipe
as the only enabled service on Static and Shock. Existing harness defaults
and provider addresses remain valid.

Before deployment, preserve the old experimental receipts and verify the
stable recovery recipe. Test name/port rendering and ownership isolation.
After Git synchronization and explicit setup, use managed stop/start order
and check each recipe with a canonical-ID streamed completion. Finish with
the 1M recipe installed, and verify direct/public catalog and tool calls.
Keep validation and deployment receipts in the issues and external evidence.
