# Runtime snapshot

Minimal files used by the reference experiment, originating from:

<https://github.com/ZhanqiuHu/flash-dlm-experimental>

`UPSTREAM_FILES.json` records the original experiment file hashes before the portability patch. There was no Git metadata in the experimental copy, so no commit ID is asserted.

The only packaging patch is in `src/model/dream_flash/generation_utils.py`: its global tokenizer loader reads `REP_RESOLVED_DREAM_MODEL` and `REP_OFFLINE`, so a local model directory can be used without a second hard-coded Hub lookup. Class definitions, tensor math, KV caching and native token commitment are retained.

The research scripts wrap the sampler in memory, verify the expected source anchors and write the generated wrapper to the run output directory. This directory itself is never edited during training.
