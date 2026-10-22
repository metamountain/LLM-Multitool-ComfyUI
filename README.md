# LLM_Multitool for ComfyUI

One node for vision and text LLMs on any OpenAI-compatible server
(llama.cpp, Ollama, LM Studio, vLLM, Unsloth Studio, OpenAI ...). It can also start a small
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
- **`unload local model` button**: stops the port-8081 server and frees its VRAM.
  There is one such server shared by all LLM_Multitool nodes. It also unloads by
  itself after 5 min idle and when ComfyUI exits.
- **Hover help** on every input.
- Thinking / reasoning output, seed, retries with back-off, `extra_body` JSON overrides.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/metamountain/LLM-Multitool-ComfyUI llm_multitool
```

Needs only `numpy`, `Pillow`, `requests` (already in ComfyUI).

For `local_model` you need a `llama-server` binary from
[llama.cpp](https://github.com/ggml-org/llama.cpp/releases) (or a fork) with vision
support. It is found via `PATH`, or set the env var `LLAMA_SERVER_BIN` to its full
path, e.g. in your ComfyUI start script:

```
set LLAMA_SERVER_BIN=C:\llama.cpp\llama-server.exe
```

Use a recent llama.cpp build: the server is started with `--fit off`,
`--flash-attn on`, `--image-max-tokens` and `--sleep-idle-seconds`. If it exits
right away, the node's error message shows the last lines of its log.

Optional environment variables:

| Variable | Default | |
|---|---|---|
| `LLAMA_SERVER_BIN` | `llama-server` on PATH | server binary |
| `LLM_MULTITOOL_PORT` | `8081` | port of the local server |
| `LLM_MULTITOOL_CTX` | `16384` | context size of the local server |
| `LLM_MULTITOOL_LOAD_S` | `300` | max seconds to wait for a model to load |

Tested on Windows; Linux and macOS should work but are untested so far.
A saved workflow that names a `local_model` missing on your machine still loads;
the node only reports it when it runs.

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
| `image` (optional) | IMAGE for vision (first image of a batch) |
| `model_id` (optional) | `auto` = ask the server |
| `backend` (optional) | `llama.cpp` / `unsloth` / `openai` parameter dialect (`beellama` = old name of `llama.cpp`) |
| `extra_body` (optional) | JSON merged into the request last |
| `local_model` (optional) | Local GGUF to run on port 8081 (projector picked by matching name, F16 preferred) |
| `text` (optional socket) | Text input for text-to-text chains |

## Outputs

`response`, `thinking`, `active_model`, `active_mode`
