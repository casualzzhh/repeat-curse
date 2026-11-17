# Third-party sources

- FlashDLM experimental implementation: <https://github.com/ZhanqiuHu/flash-dlm-experimental>, MIT, Copyright (c) 2025 Zhanqiu Hu. The minimal runtime snapshot used for this experiment is included in `third_party/flash_dlm/`; original file hashes are recorded in `UPSTREAM_FILES.json`. Its MIT notice is retained in `third_party/flash_dlm/LICENSE`.
- `generation_utils.py` states it was adapted from <https://huggingface.co/Dream-org/Dream-v0-Instruct-7B>. Original attribution is retained. The only release patch changes global tokenizer loading to respect the configured cached Dream path and offline setting; it does not change model or decoding math.
- GSM8K: <https://github.com/openai/grade-school-math>, MIT, Copyright (c) 2021 OpenAI. The bundled question split/token IDs and evaluation questions are derived from that dataset. See `data/LICENSE.gsm8k`.
- Dream and Qwen base model weights/tokenizers are separately obtained from their Hugging Face repositories and governed by their own terms. No base weights or trained adapters are bundled in this source release.

The vendored snapshot has no recorded upstream Git commit; content hashes, not an invented commit, identify the version used. This project does not claim to be the official FlashDLM implementation.
