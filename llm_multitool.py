"""
LLM_Multitool - ComfyUI Custom Node
Node: UnslothProAPI

Modes:
  auto      — vision if an image is connected, otherwise text
  vision    — Image + text prompt -> text response
  text      — Text only -> text response

local_model: starts its own BeeLlama server (port 8081) for a local GGUF + mmproj.

Defaults tuned for Gemma 4 2B (gemma-4-2b-it / gemma-4-2b-it-GGUF)
"""

import numpy as np
import atexit
import base64
import json
import os
import re
import shutil
import subprocess
import time
import requests
from io import BytesIO
from PIL import Image


# =============================================================================
#  HELP TEXTS
#  Edit these strings to update all tooltips throughout all nodes at once.
# =============================================================================

HELP = {
    # Connection
    "api_url":
        "Base URL of the BeeLlama / llama.cpp server (OpenAI-compatible).\n"
        "Default: http://127.0.0.1:8080\n"
        "Change only if the server runs on a different port or machine.",

    "api_key":
        "API key (server started with --api-key).\n"
        "Leave blank if the server runs without a key.",

    # Mode
    "mode":
        "Select operating mode:\n"
        "  auto     -- vision if an image is connected, otherwise text\n"
        "  vision   -- Send image + text prompt, receive text reply\n"
        "  text     -- Text-only prompt, no image needed",

    # Prompts
    "system_prompt":
        "Defines the model role and behaviour.\n"
        "Sent as the system message before your prompt.",

    "user_prompt":
        "Your message to the model.\n"
        "In vision mode the image is attached automatically.",

    # Sampling
    "temperature":
        "Controls output randomness.\n"
        "0.0 = fully deterministic\n"
        "1.0 = Gemma 4 recommended default\n"
        "2.0 = very random / creative",

    "top_p":
        "Nucleus sampling threshold.\n"
        "Only tokens with cumulative prob <= top_p are sampled.\n"
        "Gemma 4 recommended: 0.95",

    "top_k":
        "Limit sampling to top-k most likely tokens.\n"
        "Gemma 4 recommended: 64\n"
        "-1 = disabled (no limit)",

    "min_p":
        "Minimum probability filter.\n"
        "Excludes tokens below min_p x prob(top token).\n"
        "0.0 = disabled (Gemma 4 default)",

    "repetition_penalty":
        "Penalise tokens that already appeared in the output.\n"
        "1.0 = no effect (neutral)\n"
        ">1.0 = reduce repetition  (recommended: 1.1)\n"
        "<1.0 = encourage repetition (unusual)\n"
        "Range: 0.0 - 2.0",

    "presence_penalty":
        "OpenAI-style penalty for tokens seen at least once.\n"
        "0.0 = no effect\n"
        "Positive values push the model toward new topics.",

    # Output controls
    "max_tokens":
        "Maximum tokens in the model reply.\n"
        "Gemma 4 2B context window: 8 192 tokens.",

    "seed":
        "Random seed for reproducibility.\n"
        "-1 = random (server picks a seed each run)\n"
        "Positive integer = fixed seed -> identical output for same input.",

    "timeout_s":
        "HTTP request timeout in seconds.\n"
        "Increase for slow hardware or very large models.\n"
        "Range: 30 - 600 s",

    "max_retries":
        "Auto-retries on timeout or connection drop.\n"
        "Uses exponential back-off: 1 s, 2 s, 4 s ...\n"
        "0 = no retries (fail immediately)",

    "enable_thinking":
        "Enable chain-of-thought reasoning.\n"
        "Internal reasoning appears in the 'thinking' output socket.\n"
        "Increases latency. Best for complex reasoning tasks.",

    "reasoning_effort":
        "Depth of chain-of-thought reasoning:\n"
        "  none   -- disabled (fastest)\n"
        "  low    -- brief reasoning\n"
        "  medium -- balanced\n"
        "  high   -- deepest reasoning (slowest)\n"
        "Has no effect when enable_thinking=False.",

    "model_id_optional":
        "Model ID to use. 'auto' = use whatever model the server\n"
        "currently serves (detected via GET /v1/models).",

    "backend":
        "Which server API dialect to speak:\n"
        "  beellama -- BeeLlama / llama.cpp server (top_k, min_p,\n"
        "              repeat_penalty, thinking via chat_template_kwargs)\n"
        "  unsloth  -- Unsloth Studio (model from /api/inference/status)\n"
        "  openai   -- plain OpenAI API, only standard fields are sent\n"
        "              (top_k, min_p, repetition_penalty are ignored)",

    "extra_body":
        "Optional JSON object merged into the request last,\n"
        "overriding any field above. Example:\n"
        "  {\"top_k\": 20, \"cache_prompt\": true}\n"
        "Leave empty for none.",

    "local_model":
        "Model to run:\n"
        "  server (api_url) -- use whatever the server at api_url serves\n"
        "  <gguf>           -- start a BeeLlama server on port 8081 with this\n"
        "                      GGUF + its mmproj (vision); api_url is ignored.\n"
        "Lists every GGUF that has an mmproj*.gguf next to it (HF cache and\n"
        "ComfyUI models/LLM). Switching models restarts that server.",

    "text":
        "Optional text input (e.g. from another LLM or a prompt node).\n"
        "If user_prompt contains {text} it is inserted there,\n"
        "otherwise it is appended below user_prompt.",
}


# =============================================================================
#  GEMMA 4 2B RECOMMENDED SAMPLING DEFAULTS
# =============================================================================

_GEMMA4_TEMP       = 1.0
_GEMMA4_TOP_P      = 0.95
_GEMMA4_TOP_K      = 64
_GEMMA4_MIN_P      = 0.0
_GEMMA4_REP_PEN    = 1.1
_GEMMA4_PRES_PEN   = 0.0
_GEMMA4_MAX_TOKENS = 2048


# =============================================================================
#  SHARED HELPERS
# =============================================================================

def _build_headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key.strip():
        h["Authorization"] = f"Bearer {api_key.strip()}"
    return h


BACKENDS = ["beellama", "unsloth", "openai"]


def _get_active_model(base_url: str, api_key: str, backend: str):
    """Ask the server which model it serves -> (model_id, extra_payload, error_or_None)

    unsloth  : GET /api/inference/status (also yields gguf_variant)
    otherwise: GET /v1/models, first entry
    """
    endpoint = "/api/inference/status" if backend == "unsloth" else "/v1/models"
    try:
        r = requests.get(
            f"{base_url}{endpoint}",
            headers=_build_headers(api_key),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if backend == "unsloth":
            if not data.get("loaded", True):
                return None, {}, "No model loaded in Unsloth Studio -- load one first."
            model_id = (data.get("model_path") or data.get("model")
                        or data.get("model_name") or "unknown")
            extra = {"gguf_variant": data["gguf_variant"]} if data.get("gguf_variant") else {}
            return model_id, extra, None
        models = data.get("data") or []
        if not models:
            return None, {}, "Server lists no model -- is it started with a model?"
        return models[0].get("id") or "unknown", {}, None
    except requests.exceptions.ConnectionError:
        return None, {}, f"Cannot reach {base_url} -- is the {backend} server running?"
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        if code == 401:
            return None, {}, "401 Unauthorized -- check your API key."
        if code == 503:
            return None, {}, "503 -- server is still loading the model, try again."
        return None, {}, f"HTTP {code} from {endpoint}"
    except Exception as exc:
        return None, {}, f"Model query failed: {exc}"


def _parse_extra_body(text: str):
    """JSON object string -> (dict, error_or_None). Empty string = no extras."""
    if not text or not text.strip():
        return {}, None
    try:
        extra = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"extra_body is not valid JSON -- {exc}"
    if not isinstance(extra, dict):
        return None, "extra_body must be a JSON object, e.g. {\"top_k\": 20}"
    return extra, None


# =============================================================================
#  LOCAL MODEL LOADER -- own BeeLlama server for a local GGUF + mmproj
# =============================================================================

SERVER_CHOICE  = "server (api_url)"
LOCAL_PORT     = 8081
LOCAL_URL      = f"http://127.0.0.1:{LOCAL_PORT}"
LOCAL_CTX      = 8192
LOCAL_LOAD_S   = 180      # max wait for the model to load
BEELLAMA_BIN   = (os.environ.get("BEELLAMA_BIN") or shutil.which("llama-server")
                  or r"D:\BeeLlama\bin\llama-server.exe")
_NODE_DIR      = os.path.dirname(os.path.abspath(__file__))
_PID_FILE      = os.path.join(_NODE_DIR, "local_server.pid")
_LOG_FILE      = os.path.join(_NODE_DIR, "local_server.log")
_MMPROJ_PREF   = ("f16", "bf16", "f32")

_local_models  = {}       # label -> (model_path, mmproj_path)
_local_proc    = None     # Popen of the server started by this ComfyUI process


def _model_dirs():
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    dirs = [os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub")]
    try:
        import folder_paths
        dirs.append(os.path.join(folder_paths.models_dir, "LLM"))
    except ImportError:
        pass
    return [d for d in dirs if os.path.isdir(d)]


def _scan_local_models():
    """Every GGUF with an mmproj*.gguf in the same folder -> {label: (model, mmproj)}"""
    found = {}
    for root_dir in _model_dirs():
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in ("blobs", ".cache", ".locks")]
            ggufs  = [f for f in filenames if f.lower().endswith(".gguf")
                      and os.path.exists(os.path.join(dirpath, f))]   # skip dangling links
            mmproj = [f for f in ggufs if f.lower().startswith("mmproj")]
            if not mmproj:
                continue
            mmproj.sort(key=lambda f: next((i for i, p in enumerate(_MMPROJ_PREF)
                                            if f.lower()[:-5].endswith(p)), 99))
            for f in ggufs:
                part = re.search(r"-(\d{5})-of-\d{5}\.gguf$", f)
                if f in mmproj or f.lower().startswith("mtp-") or (part and part.group(1) != "00001"):
                    continue    # skip projectors, MTP draft heads, non-first split parts
                path  = os.path.join(dirpath, f)
                label = f[:-5]
                # Same file in several snapshots / folders: keep the newest.
                if label in found and os.path.getmtime(found[label][0]) >= os.path.getmtime(path):
                    continue
                found[label] = (path, os.path.join(dirpath, mmproj[0]))
    sizes = {k: os.path.getsize(v[0]) for k, v in found.items()}
    return dict(sorted(found.items(), key=lambda kv: sizes[kv[0]]))


def _local_alias(label: str) -> str:
    return "local-" + "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def _local_served():
    """Alias served on LOCAL_PORT, or None if nothing answers."""
    try:
        r = requests.get(f"{LOCAL_URL}/v1/models", timeout=3)
        r.raise_for_status()
        models = r.json().get("data") or []
        return models[0].get("id") if models else ""
    except Exception:
        return None


def _stop_local_server():
    global _local_proc
    if _local_proc is not None and _local_proc.poll() is None:
        _local_proc.terminate()
        try:
            _local_proc.wait(10)
        except subprocess.TimeoutExpired:
            _local_proc.kill()
    _local_proc = None
    # Server left over from an earlier ComfyUI run: only kill it if it is ours.
    if os.path.exists(_PID_FILE):
        try:
            pid = int(open(_PID_FILE).read().strip())
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True).stdout
            if b"llama-server" in out.lower():
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        except (ValueError, OSError):
            pass
        try:
            os.remove(_PID_FILE)
        except OSError:
            pass
    for _ in range(20):    # wait until the port is free
        if _local_served() is None:
            return
        time.sleep(0.5)


atexit.register(_stop_local_server)


def _ensure_local_server(label: str):
    """Start (or reuse) the BeeLlama server for `label` -> (base_url, model_id, error_or_None)"""
    global _local_proc
    paths = _local_models.get(label) or _scan_local_models().get(label)
    if not paths:
        return None, None, f"local model '{label}' not found -- refresh ComfyUI (R)."
    alias = _local_alias(label)
    served = _local_served()
    if served == alias:
        return LOCAL_URL, alias, None
    if served is not None and _local_proc is None and not os.path.exists(_PID_FILE):
        return None, None, (f"Port {LOCAL_PORT} is used by another server ('{served}') "
                            "-- stop it or pick 'server (api_url)'.")
    if not os.path.exists(BEELLAMA_BIN):
        return None, None, f"llama-server not found: {BEELLAMA_BIN} (set env BEELLAMA_BIN)"

    _stop_local_server()
    model_path, mmproj_path = paths
    cmd = [
        BEELLAMA_BIN,
        "--model", model_path, "--mmproj", mmproj_path,
        "--alias", alias,
        "--host", "127.0.0.1", "--port", str(LOCAL_PORT),
        "--ctx-size", str(LOCAL_CTX), "--parallel", "1",
        "--n-gpu-layers", "99", "--fit", "off", "--flash-attn", "on",
        "--image-max-tokens", "1024",
        "--sleep-idle-seconds", "300",   # unload after 5 min idle, frees VRAM
    ]
    print(f"[LLM_Multitool] Starting local server: {label} on port {LOCAL_PORT}")
    log = open(_LOG_FILE, "w", encoding="utf-8", errors="replace")
    _local_proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    with open(_PID_FILE, "w") as f:
        f.write(str(_local_proc.pid))

    deadline = time.time() + LOCAL_LOAD_S
    while time.time() < deadline:
        if _local_proc.poll() is not None:
            return None, None, f"local server exited (code {_local_proc.returncode}) -- see {_LOG_FILE}"
        try:
            if requests.get(f"{LOCAL_URL}/health", timeout=2).status_code == 200:
                return LOCAL_URL, alias, None
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return None, None, f"local server did not get ready in {LOCAL_LOAD_S} s -- see {_LOG_FILE}"


# =============================================================================
#  NODE -- LLM_Multitool
# =============================================================================

class UnslothProAPI:
    """
    BeeLlama (llama.cpp) inference node -- Vision / Text.

    Outputs
    -------
    response      Model reply text (or error message)
    thinking      Chain-of-thought (only when enable_thinking=True)
    active_model  Model ID that was actually used
    active_mode   Current mode string
    """

    @classmethod
    def INPUT_TYPES(cls):
        global _local_models
        local = _local_models = _scan_local_models()
        return {
            "required": {

                # CONNECTION
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:8080",
                    "tooltip": HELP["api_url"],
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": HELP["api_key"],
                }),

                # MODE
                "mode": (["auto", "vision", "text"], {
                    "default": "auto",
                    "tooltip": HELP["mode"],
                }),

                # PROMPTS
                "_header_system_prompt": (["SYSTEM PROMPT"], {
                    "default": "SYSTEM PROMPT",
                }),
                "system_prompt": ("STRING", {
                    "multiline":      True,
                    "default":        "You are a helpful assistant.",
                    "tooltip":        HELP["system_prompt"],
                    "dynamicPrompts": False,
                }),
                "_header_user_prompt": (["USER PROMPT"], {
                    "default": "USER PROMPT",
                }),
                "user_prompt": ("STRING", {
                    "multiline":      True,
                    "default":        "Describe this image in detail.",
                    "tooltip":        HELP["user_prompt"],
                    "dynamicPrompts": False,
                }),

                # SAMPLING
                "temperature": ("FLOAT", {
                    "default": _GEMMA4_TEMP,
                    "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": HELP["temperature"],
                }),
                "top_p": ("FLOAT", {
                    "default": _GEMMA4_TOP_P,
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": HELP["top_p"],
                }),
                "top_k": ("INT", {
                    "default": _GEMMA4_TOP_K,
                    "min": -1, "max": 200, "step": 1,
                    "tooltip": HELP["top_k"],
                }),
                "min_p": ("FLOAT", {
                    "default": _GEMMA4_MIN_P,
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": HELP["min_p"],
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": _GEMMA4_REP_PEN,
                    "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": HELP["repetition_penalty"],
                }),
                "presence_penalty": ("FLOAT", {
                    "default": _GEMMA4_PRES_PEN,
                    "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": HELP["presence_penalty"],
                }),

                # OUTPUT CONTROLS
                "max_tokens": ("INT", {
                    "default": _GEMMA4_MAX_TOKENS,
                    "min": 1, "max": 32768, "step": 64,
                    "tooltip": HELP["max_tokens"],
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1, "max": 2**32 - 1, "step": 1,
                    "tooltip": HELP["seed"],
                }),
                "timeout_s": ("INT", {
                    "default": 120,
                    "min": 30, "max": 600, "step": 10,
                    "tooltip": HELP["timeout_s"],
                }),
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 0, "max": 5, "step": 1,
                    "tooltip": HELP["max_retries"],
                }),

                # THINKING
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": HELP["enable_thinking"],
                }),
                "reasoning_effort": (["none", "low", "medium", "high"], {
                    "default": "none",
                    "tooltip": HELP["reasoning_effort"],
                }),
            },

            "optional": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Input image -- required for vision mode.\n"
                        "ComfyUI IMAGE tensor shape: [B,H,W,C], C=3, float32 0-1."
                    ),
                }),
                "model_id": ("STRING", {
                    "default": "auto",
                    "tooltip": HELP["model_id_optional"],
                }),
                # Kept optional and last so saved workflows keep their widget order.
                "backend": (BACKENDS, {
                    "default": "beellama",
                    "tooltip": HELP["backend"],
                }),
                "extra_body": ("STRING", {
                    "multiline":      True,
                    "default":        "",
                    "tooltip":        HELP["extra_body"],
                    "dynamicPrompts": False,
                }),
                # New widgets go last, again for saved-workflow widget order.
                "local_model": ([SERVER_CHOICE] + list(local), {
                    "default": SERVER_CHOICE,
                    "tooltip": HELP["local_model"],
                }),
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip":    HELP["text"],
                }),
            },
        }

    RETURN_TYPES    = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES    = ("response","thinking","active_model","active_mode")
    OUTPUT_TOOLTIPS = (
        "The model's reply text",
        "Chain-of-thought / reasoning (only when enable_thinking=True)",
        "Model ID that was actually used for this inference",
        "Active mode string (vision / text)",
    )
    FUNCTION = "chat"
    CATEGORY = "LLM"

    # -- Image / Tensor helpers ------------------------------------------------

    @staticmethod
    def _tensor_to_b64_jpeg(image_tensor) -> str:
        arr = np.clip(255.0 * image_tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # -- Main execute ---------------------------------------------------------

    def chat(self, **kwargs):
        result = self._chat(**kwargs)
        # Shown on the node itself (web/llm_multitool.js), no wire needed.
        return {"ui": {"active_model": [result[2]]}, "result": result}

    def _chat(
        self,
        api_url, api_key, mode,
        _header_system_prompt, system_prompt, user_prompt, _header_user_prompt,
        temperature, top_p, top_k, min_p,
        repetition_penalty, presence_penalty,
        max_tokens, seed, timeout_s, max_retries,
        enable_thinking, reasoning_effort,
        image=None, model_id="auto",
        backend="beellama", extra_body="",
        local_model=SERVER_CHOICE, text=None,
    ):
        base_url   = api_url.rstrip("/")
        if mode == "auto":
            mode = "vision" if image is not None else "text"
        is_vision  = mode == "vision"

        if text and text.strip():
            if "{text}" in user_prompt:
                user_prompt = user_prompt.replace("{text}", text)
            else:
                user_prompt = f"{user_prompt}\n\n{text}" if user_prompt.strip() else text

        extra, err = _parse_extra_body(extra_body)
        if err:
            return (f"ERROR: {err}", "", "none", mode)

        # 1 -- Resolve model
        model_extra = {}
        if local_model and local_model != SERVER_CHOICE:
            base_url, model_id, err = _ensure_local_server(local_model)
            if err:
                return (f"ERROR: {err}", "", "none", mode)
            backend, api_key = "beellama", ""
        elif not model_id or model_id.strip().lower() in ("auto", ""):
            resolved, model_extra, err = _get_active_model(base_url, api_key, backend)
            if err:
                return (f"ERROR: {err}", "", "none", mode)
            model_id = resolved
        model_label = model_id

        # 2 -- Vision guard
        if is_vision and image is None:
            return (
                f"ERROR: mode='{mode}' requires an IMAGE input -- "
                "connect an image node or switch to 'text' mode.",
                "", model_label, mode,
            )

        # 3 -- Build user content
        img_b64 = None
        if is_vision:
            img_b64 = self._tensor_to_b64_jpeg(image)
            user_content = [
                {"type": "text",      "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                }},
            ]
        else:
            user_content = user_prompt

        # 4 -- Build payload
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "temperature":        temperature,
            "top_p":              top_p,
            "presence_penalty":   presence_penalty,
            "max_tokens":         max_tokens,
            "stream":             False,
        }

        # Backend-specific parameter names
        if backend == "beellama":
            payload["top_k"]          = top_k
            payload["min_p"]          = min_p
            payload["repeat_penalty"] = repetition_penalty
            # llama.cpp hands this to the chat template (Qwen3 / Gemma thinking switch)
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        elif backend == "unsloth":
            payload["top_k"]              = top_k
            payload["min_p"]              = min_p
            payload["repetition_penalty"] = repetition_penalty
            payload["enable_thinking"]    = enable_thinking
            payload.update(model_extra)

        if enable_thinking and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort
        if seed != -1:               payload["seed"]             = seed

        # extra_body wins over everything above
        payload.update(extra)

        # 5 -- Send with exponential back-off retry
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers=_build_headers(api_key),
                    timeout=timeout_s,
                )
                r.raise_for_status()
                msg      = r.json()["choices"][0]["message"]
                content  = msg.get("content") or ""
                thinking = msg.get("reasoning_content") or msg.get("thinking") or ""

                return (content, thinking, model_label, mode)

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as exc:
                last_err = str(exc)
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[LLM_Multitool] Retry {attempt+1}/{max_retries} in {wait}s ...")
                    time.sleep(wait)
                    continue
                return (
                    f"ERROR: Connection failed after {max_retries+1} attempts -- {last_err}",
                    "", model_label, mode,
                )

            except requests.exceptions.HTTPError:
                code   = r.status_code
                detail = {
                    401: "401 Unauthorized -- check your API key.",
                    400: f"400 Bad request -- {r.text[:400]}",
                    503: "503 -- server is still loading the model, try again.",
                    422: f"422 Validation error -- {r.text[:400]}",
                    500: f"500 Server error -- {r.text[:200]}",
                }
                return (
                    f"ERROR: {detail.get(code, f'HTTP {code} -- {r.text[:200]}')}",
                    "", model_label, mode,
                )

            except KeyError:
                return (
                    f"ERROR: Unexpected response format -- {r.text[:200]}",
                    "", model_label, mode,
                )
            except Exception as exc:
                return (
                    f"ERROR: {exc}",
                    "", model_label, mode,
                )

        return (
            f"ERROR: All retries exhausted -- {last_err}",
            "", model_label, mode,
        )


# =============================================================================
#  Registry
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "UnslothProAPI":  UnslothProAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnslothProAPI":  "LLM_Multitool",
}
