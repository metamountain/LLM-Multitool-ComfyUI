# LLM_Multitool for ComfyUI

One node for vision and text LLMs on any OpenAI-compatible server
(BeeLlama / llama.cpp, Unsloth Studio, OpenAI). It can also start a small
local vision model by itself, so no loader node or extra wires are needed.

> Replaces the earlier *Unsloth API PRO* + *Unsloth Model Picker* nodes.
> The node id is still `UnslothProAPI`, so saved workflows keep working.

## Features

- **Modes**: `auto` (vision if an image is connected, otherwise text), `vision`, `text`
- **`text` input socket**: chain text-to-text (output of another LLM, prompt nodes ...).
  Use `{text}` in `user_prompt` to place it, otherwise it is appended.
- **`local_model` dropdown**: lists every local GGUF that has an `mmproj*.gguf`
  next to it (Hugging Face cache and `ComfyUI/models/LLM`). Picking one starts
  its own `llama-server` on port **8081**; switching models restarts it.
  `server (api_url)` uses whatever the server at `api_url` serves.
- **Active model shown on the node** after each run (read-only `active` field).
- **Hover help** on every input.
- Thinking / reasoning output, seed, retries with back-off, `extra_body` JSON overrides.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/metamountain/Comfyui-Unsloth-Open-Api- llm_multitool
```

Needs only `numpy`, `Pillow`, `requests` (already in ComfyUI).

For `local_model` you need a `llama-server` binary (llama.cpp or BeeLlama).
It is found via the env var `BEELLAMA_BIN`, then `PATH`.

## Recommended small vision model (< 4 GB)

[unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF):
`Qwen3.5-4B-UD-Q4_K_XL.gguf` (2.9 GB) + `mmproj-F16.gguf` (0.7 GB).

```
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf mmproj-F16.gguf
```

Press **R** in ComfyUI and pick it in `local_model`. On an RTX 4090 an image
description takes about 0.5 s (~180 tok/s); the first run adds a few seconds
for loading. The server unloads the model after 5 min idle.

## Inputs

| Input | Notes |
|---|---|
| `api_url`, `api_key` | Server for `server (api_url)` (default `http://127.0.0.1:8080`) |
| `mode` | `auto` / `vision` / `text` |
| `system_prompt`, `user_prompt` | Prompts; `{text}` placeholder for the `text` socket |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty` | Sampling |
| `max_tokens`, `seed`, `timeout_s`, `max_retries` | Output / connection |
| `enable_thinking`, `reasoning_effort` | Reasoning |
| `image` (optional) | IMAGE for vision |
| `model_id` (optional) | `auto` = ask the server |
| `backend` (optional) | `beellama` / `unsloth` / `openai` parameter dialect |
| `extra_body` (optional) | JSON merged into the request last |
| `local_model` (optional) | Local GGUF to run on port 8081 |
| `text` (optional socket) | Text input for text-to-text chains |

## Outputs

`response`, `thinking`, `active_model`, `active_mode`
