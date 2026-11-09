"""
LLM_Multitool - ComfyUI Custom Node
Node id: LLM_Multitool

Modes:
  auto      — vision if an image is connected, otherwise text
  vision    — Image + text prompt -> text response
  text      — Text only -> text response

local_model: starts its own llama-server (port 8081) for a local GGUF (+ mmproj for vision).
"""

import numpy as np
import atexit
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
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
        "Base URL of the server (llama.cpp, Ollama, LM Studio, vLLM, OpenAI ...).\n"
        "Default: http://127.0.0.1:8080 (Ollama: http://127.0.0.1:11434).\n"
        "Ignored when local_model is set.",

    "api_key_env":
        "Name of the environment variable that holds the API key\n"
        "(e.g. OPENAI_API_KEY, GEMINI_API_KEY).\n"
        "The key is read via os.environ.get() at run time -- it is NEVER\n"
        "stored in the workflow file.\n"
        "Leave empty if the server needs no key (local servers).",

    "unload_after":
        "Unload the model after the request finishes.\n"
        "  llama.cpp / router -- POST /models/unload (frees VRAM)\n"
        "  ollama             -- keep_alive 0 in the request\n"
        "  openai / unsloth   -- no effect (cloud / managed)\n"
        "Recommended: True when sharing the GPU with ComfyUI rendering.",

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
        "1.0 = typical default\n"
        "2.0 = very random / creative",

    "top_p":
        "Nucleus sampling threshold.\n"
        "Only tokens with cumulative prob <= top_p are sampled.\n"
        "Typical: 0.95",

    "top_k":
        "Limit sampling to top-k most likely tokens.\n"
        "Typical: 20-64\n"
        "-1 = disabled (no limit)",

    "min_p":
        "Minimum probability filter.\n"
        "Excludes tokens below min_p x prob(top token).\n"
        "0.0 = disabled (default)",

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
        "Maximum tokens in the model reply (thinking counts too).\n"
        "Empty response? Raise this or turn enable_thinking off.",

    "on_error":
        "What happens on an error (server down, bad key, empty reply ...):\n"
        "  stop        -- stop the queue with a red error message (safe)\n"
        "  return text -- continue; 'response' holds a line starting with ERROR:\n"
        "                 (careful: that text could end up as your image prompt)",

    "seed":
        "Random seed for reproducibility.\n"
        "-1 = random: the node runs again on every queue (never cached)\n"
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
        "  llama.cpp -- llama-server and forks (top_k, min_p,\n"
        "              repeat_penalty, thinking via chat_template_kwargs)\n"
        "  unsloth  -- Unsloth Studio (model from /api/inference/status)\n"
        "  openai   -- plain OpenAI API, only standard fields are sent\n"
        "              (top_k, min_p, repetition_penalty are ignored)\n"
        "  ollama   -- Ollama native API (/api/chat, images as base64 array,\n"
        "              keep_alive for VRAM management)",

    "extra_body":
        "Optional JSON object merged into the request last,\n"
        "overriding any field above. Example:\n"
        "  {\"top_k\": 20, \"cache_prompt\": true}\n"
        "Leave empty for none.",

    "local_model":
        "Model to run:\n"
        "  server (api_url) -- use whatever the server at api_url serves\n"
        "  <gguf>           -- start a llama-server on port 8081 with this\n"
        "                      GGUF (+ its mmproj for vision); api_url is ignored.\n"
        "  <gguf> (text)    -- no mmproj next to it: text only, no images.\n"
        "Scans the HF cache and ComfyUI models/LLM, text_encoders, clip\n"
        "(and extra_model_paths.yaml) for GGUF files. Switching restarts the server.",

    "keep_alive":
        "Ollama only: seconds the model stays loaded after the reply.\n"
        "0 = unload right away (frees VRAM for rendering).",

    "clean_output":
        "Clean the reply: remove think blocks, Markdown and surrounding quotes.\n"
        "Useful when the reply goes straight into a render prompt.",

    "image_mode":
        "first -- send only the first image of a batch\n"
        "all   -- send every image of the batch (up to 8) in one request",

    "image":
        "Optional image. With mode 'auto' the node switches to vision\n"
        "when an image is connected.",

    "text":
        "Optional text input (e.g. from another LLM or a prompt node).\n"
        "If user_prompt contains {text} it is inserted there,\n"
        "otherwise it is appended below user_prompt.",
}


# =============================================================================
#  SAMPLING DEFAULTS
# =============================================================================

_DEFAULT_TEMP       = 1.0
_DEFAULT_TOP_P      = 0.95
_DEFAULT_TOP_K      = 64
_DEFAULT_MIN_P      = 0.0
_DEFAULT_REP_PEN    = 1.1
_DEFAULT_PRES_PEN   = 0.0
_DEFAULT_MAX_TOKENS = 2048


# =============================================================================
#  SHARED HELPERS
# =============================================================================

def _clean_response(text: str, max_words: int = 0) -> str:
    """Remove think blocks, Markdown, surrounding quotes. Optional word limit.
    """
    if not text:
        return text
    # Remove think/reasoning blocks (also a leftover block whose opening tag is missing)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if re.search(r"</think>", text, flags=re.I):
        text = re.split(r"</think>", text, flags=re.I)[-1]
    # Remove Markdown formatting
    text = re.sub(r"[*_#`>]", "", text)
    # Remove surrounding quotes
    text = text.strip().strip('"').strip("'").strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Word limit
    if max_words > 0:
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
    return text


def _validate_api_key_env(name: str) -> str | None:
    """Return an error string if *name* is not a valid env-var name, else None.

    Rules:
    - must match ^[A-Za-z_][A-Za-z0-9_]*$
    - must NOT look like an actual key (sk-, AIza, > 64 chars)
    """
    if not name or not name.strip():
        return None  # empty = no key
    n = name.strip()
    # Key check first, and never echo the value: it may be a real key.
    if re.match(r"^(sk-|sk_|AIza|ghp_|hf_|xai-|gsk_)", n) or len(n) > 64:
        return ("api_key_env looks like an API key value, not a variable name. "
                "Put the key in an environment variable and enter only its NAME "
                "(e.g. OPENAI_API_KEY), not the key itself.")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n):
        return ("api_key_env is not a valid variable name. "
                "It must match [A-Za-z_][A-Za-z0-9_]* (letters, digits, underscores).")
    return None


def _resolve_api_key(api_key_env: str) -> tuple[str, str | None]:
    """API key from the environment variable named in api_key_env -> (key, error_or_None)."""
    err = _validate_api_key_env(api_key_env)
    if err:
        return "", err
    if api_key_env and api_key_env.strip():
        key = os.environ.get(api_key_env.strip(), "")
        if not key:
            return "", (f"Environment variable '{api_key_env.strip()}' is not set. "
                        f"Set it in your ComfyUI start script before launching.")
        return key, None
    return "", None


def _normalize_url(url: str) -> str:
    """Forgive common api_url mistakes: missing http://, trailing slash or /v1."""
    u = (url or "").strip() or "http://127.0.0.1:8080"
    if not re.match(r"^https?://", u, flags=re.I):
        u = "http://" + u
    u = u.rstrip("/")
    if u.lower().endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


def _pick_router_model(models: list):
    """First model a llama.cpp router has in VRAM ('loaded' / 'sleeping').

    Plain llama-server entries have no 'status' -> first entry.
    Returns None if this is a router with no model loaded (asking for an
    unloaded one would make the router load it).
    """
    if not models:
        return None
    if not any(isinstance(m.get("status"), dict) for m in models):
        return models[0].get("id")
    for m in models:
        if (m.get("status") or {}).get("value") in ("loaded", "sleeping", "loading"):
            return m.get("id")
    return None


def _build_headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key.strip():
        h["Authorization"] = f"Bearer {api_key.strip()}"
    return h


BACKENDS = ["llama.cpp", "ollama", "openai", "unsloth"]


def _get_active_model(base_url: str, api_key: str, backend: str):
    """Ask the server which model it serves -> (model_id, extra_payload, error_or_None)

    unsloth  : GET /api/inference/status (also yields gguf_variant)
    ollama   : GET /api/tags (first model)
    otherwise: GET /v1/models, first entry
    """
    if backend == "ollama":
        endpoint = "/api/tags"
    elif backend == "unsloth":
        endpoint = "/api/inference/status"
    else:
        endpoint = "/v1/models"
    try:
        r = requests.get(
            f"{base_url}{endpoint}",
            headers=_build_headers(api_key),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if backend == "ollama":
            models = data.get("models") or []
            if not models:
                return None, {}, "Ollama lists no model -- pull one first (ollama pull <name>)."
            try:   # prefer a model that is already in VRAM
                ps = requests.get(f"{base_url}/api/ps", headers=_build_headers(api_key),
                                  timeout=15).json().get("models") or []
                if ps:
                    return ps[0].get("name") or "unknown", {}, None
            except Exception:
                pass
            return models[0].get("name") or "unknown", {}, None
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
        picked = _pick_router_model(models)    # router: the loaded model, else the first
        if not picked:
            return None, {}, ("Router has no model loaded -- load one, or enter its id "
                              "in model_id (the router then loads it).")
        return picked, {}, None
    except requests.exceptions.ConnectionError:
        return None, {}, f"Cannot reach {base_url} -- is the {backend} server running?"
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        if code == 401:
            return None, {}, "401 Unauthorized -- check your API key."
        if code == 503:
            return None, {}, "503 -- server is still loading the model, try again."
        if code == 404:
            return None, {}, (f"{base_url} has no {endpoint} -- wrong 'backend'? "
                              f"('{backend}' selected; llama.cpp / LM Studio / vLLM = llama.cpp or "
                              f"openai, Ollama = ollama, Unsloth Studio = unsloth)")
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
#  LOCAL MODEL LOADER -- own llama-server for a local GGUF + mmproj
# =============================================================================

SERVER_CHOICE  = "server (api_url)"
# Overridable via environment, e.g. in the ComfyUI start script.
LOCAL_PORT     = int(os.environ.get("LLM_MULTITOOL_PORT", "8081"))
LOCAL_URL      = f"http://127.0.0.1:{LOCAL_PORT}"
LOCAL_CTX      = int(os.environ.get("LLM_MULTITOOL_CTX", "16384"))
LOCAL_LOAD_S   = int(os.environ.get("LLM_MULTITOOL_LOAD_S", "300"))   # max wait for loading
# llama.cpp (or a fork such as BeeLlama): env LLAMA_SERVER_BIN, old name BEELLAMA_BIN, then PATH
LLAMA_SERVER_BIN = (os.environ.get("LLAMA_SERVER_BIN") or os.environ.get("BEELLAMA_BIN")
                    or shutil.which("llama-server") or "")
_STATE_DIR     = os.path.join(tempfile.gettempdir(), "llm_multitool")
_PID_FILE      = os.path.join(_STATE_DIR, f"local_server_{LOCAL_PORT}.pid")
_LOG_FILE      = os.path.join(_STATE_DIR, f"local_server_{LOCAL_PORT}.log")
_MMPROJ_PREF   = ("f16", "bf16", "f32")
_SCAN_TTL_S    = 30       # INPUT_TYPES runs on every queue -- don't walk the disk each time

_local_models  = {}       # label -> (model_path, mmproj_path)
_local_scan_t  = 0.0
_local_proc    = None     # Popen of the server started by this ComfyUI process
_local_label   = None     # label that _local_proc serves (or is still loading)
_local_lock    = threading.RLock()


def _model_dirs():
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "huggingface")
    dirs = [os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.path.join(hf_home, "hub")]
    try:    # ComfyUI folders, including extra_model_paths.yaml
        import folder_paths
        dirs.append(os.path.join(folder_paths.models_dir, "LLM"))
        for kind in ("LLM", "text_encoders", "clip"):
            try:
                dirs += folder_paths.get_folder_paths(kind)
            except Exception:
                pass
    except ImportError:
        pass
    seen, out = set(), []
    for d in dirs:
        key = os.path.normcase(os.path.abspath(d)) if d else ""
        if d and key not in seen and os.path.isdir(d):
            seen.add(key)
            out.append(d)
    return out


# GGUFs that cannot chat on their own: draft / MTP heads (speculative decoding),
# encoder-only text encoders (T5 ...).
_NOT_CHAT = re.compile(r"^mtp-|dflash|draft|encoder", re.I)


def _pick_mmproj(model_file, mmprojs):
    """Projector for `model_file`: longest shared name prefix, then F16 > BF16 > F32."""
    base = model_file.lower()

    def score(f):
        name = f.lower()[len("mmproj"):].lstrip("-_.")
        common = len(os.path.commonprefix([base, name]))
        tok = re.split(r"[-_.]", f.lower()[:-5])[-1]    # exact, "bf16" must not match "f16"
        pref = _MMPROJ_PREF.index(tok) if tok in _MMPROJ_PREF else 99
        return (-common, pref)

    return min(mmprojs, key=score)


def _scan_local_models():
    """Every GGUF -> {label: (model, mmproj or None)}.

    With an mmproj*.gguf in the same folder it can see images; without one it is
    listed as "<name> (text)" and runs text only.
    """
    found, mtimes = {}, {}
    for root_dir in _model_dirs():
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in ("blobs", ".cache", ".locks")]
            ggufs  = [f for f in filenames if f.lower().endswith(".gguf")
                      and os.path.exists(os.path.join(dirpath, f))]   # skip dangling links
            mmproj = [f for f in ggufs if f.lower().startswith("mmproj")]
            for f in ggufs:
                part = re.search(r"-(\d{5})-of-\d{5}\.gguf$", f)
                if (f in mmproj or _NOT_CHAT.search(f) or (part and part.group(1) != "00001")):
                    continue    # skip projectors, draft heads, encoders, non-first split parts
                path  = os.path.join(dirpath, f)
                label = f[:-5] if mmproj else f"{f[:-5]} (text)"
                try:
                    mtime, size = os.path.getmtime(path), os.path.getsize(path)
                except OSError:
                    continue
                # Same file in several snapshots / folders: keep the newest.
                if label in found and mtimes[label][0] >= mtime:
                    continue
                found[label] = (path, os.path.join(dirpath, _pick_mmproj(f, mmproj))
                                if mmproj else None)
                mtimes[label] = (mtime, size)
    return dict(sorted(found.items(), key=lambda kv: mtimes[kv[0]][1]))


def _local_models_cached(force=False):
    global _local_models, _local_scan_t
    if force or time.time() - _local_scan_t > _SCAN_TTL_S:
        try:
            _local_models = _scan_local_models()
        except Exception as exc:    # never break node registration over a scan
            print(f"[LLM_Multitool] model scan failed: {exc}")
        _local_scan_t = time.time()
    return _local_models


def _local_alias(label: str) -> str:
    label = label.removesuffix(" (text)")
    return "local-" + "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def _local_served():
    """Alias served on LOCAL_PORT, None if nothing answers, "" while it is still loading."""
    try:
        r = requests.get(f"{LOCAL_URL}/v1/models", timeout=3)
        if r.status_code == 503:
            return ""
        r.raise_for_status()
        models = r.json().get("data") or []
        return models[0].get("id") if models else ""
    except Exception:
        return None


def _is_llama_server(pid: int) -> bool:
    """True if `pid` is still our server binary (PIDs get reused)."""
    names = {b"llama-server"}
    if LLAMA_SERVER_BIN:
        names.add(os.path.splitext(os.path.basename(LLAMA_SERVER_BIN))[0].lower().encode())
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True).stdout
    else:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True).stdout
    return any(n in out.lower() for n in names)


def _log_tail(n=8) -> str:
    try:
        with open(_LOG_FILE, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except OSError:
        return ""


def _stop_local_server():
    global _local_proc, _local_label
    with _local_lock:
        if _local_proc is not None and _local_proc.poll() is None:
            _local_proc.terminate()
            try:
                _local_proc.wait(10)
            except subprocess.TimeoutExpired:
                _local_proc.kill()
        _local_proc, _local_label = None, None
        # Server left over from an earlier ComfyUI run: only kill it if it is ours.
        if os.path.exists(_PID_FILE):
            try:
                pid = int(open(_PID_FILE).read().strip())
                if _is_llama_server(pid):
                    os.kill(pid, signal.SIGTERM)    # Windows: TerminateProcess
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


def _unload_remote(api_url: str, backend: str, api_key: str = ""):
    """Free the VRAM of the server at api_url -> (unloaded_names, note).

    llama.cpp router : POST /models/unload for every loaded / sleeping / loading model
    plain llama-server: no unload API -> note only
    ollama           : keep_alive 0 for every model in /api/ps
    openai / unsloth : nothing to unload
    """
    base, h, done = _normalize_url(api_url), _build_headers(api_key), []
    try:
        if backend == "ollama":
            for m in requests.get(f"{base}/api/ps", headers=h, timeout=10).json().get("models") or []:
                name = m.get("name") or m.get("model")
                requests.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0},
                              headers=h, timeout=30)
                done.append(name)
            return done, "" if done else "Ollama has no model loaded."
        if backend == "llama.cpp":
            models = requests.get(f"{base}/v1/models", headers=h, timeout=10).json().get("data") or []
            if not any(isinstance(m.get("status"), dict) for m in models):
                return [], "plain llama-server has no unload API -- stop that server yourself."
            for m in models:
                if (m.get("status") or {}).get("value") in ("loaded", "sleeping", "loading"):
                    r = requests.post(f"{base}/models/unload", json={"model": m["id"]},
                                      headers=h, timeout=30)
                    if r.ok:
                        done.append(m["id"])
            return done, "" if done else "router has no model loaded."
        return [], f"{backend}: nothing to unload (cloud / managed server)."
    except requests.exceptions.ConnectionError:
        return done, f"cannot reach {base}."
    except Exception as exc:
        return done, f"unload failed: {exc}"


def _unload_all(api_url="", backend="llama.cpp", api_key_env=""):
    """'unload model' button: own local server + model at api_url -> (any_unloaded, message)."""
    parts, any_done = [], False
    if _local_served() is not None:
        _stop_local_server()
        parts.append(f"local server (port {LOCAL_PORT}) stopped")
        any_done = True
    if api_url:
        key, err = _resolve_api_key(api_key_env)
        if err:
            parts.append(err)
        else:
            names, note = _unload_remote(api_url, backend, key)
            if names:
                parts.append("unloaded " + ", ".join(names))
                any_done = True
            elif note and not any_done:
                parts.append(note)
    return any_done, "; ".join(parts) or "nothing loaded."


# "unload model" button on the node (web/llm_multitool.js) -> free VRAM.
try:
    import asyncio
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/llm_multitool/unload")
    async def _unload_route(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        args = {k: str(body.get(k) or "") for k in ("api_url", "backend", "api_key_env")}
        args["backend"] = args["backend"] or "llama.cpp"
        done, msg = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _unload_all(**args))
        return web.json_response({"unloaded": done, "message": msg})
except Exception:    # outside ComfyUI (tests) there is no PromptServer
    pass


def _wait_ready(proc, alias: str):
    """Wait (without holding the lock) for the server we started -> (base_url, model_id, error)"""
    deadline = time.time() + LOCAL_LOAD_S
    while time.time() < deadline:
        if _local_proc is not proc:
            return None, None, "local server was unloaded or replaced while loading."
        if proc.poll() is not None:
            return None, None, (f"local server exited (code {proc.returncode}) -- "
                                f"log {_LOG_FILE}:\n{_log_tail()}")
        try:
            if requests.get(f"{LOCAL_URL}/health", timeout=2).status_code == 200:
                return LOCAL_URL, alias, None
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return None, None, (f"local server still loading after {LOCAL_LOAD_S} s -- run again to keep "
                        f"waiting, or raise env LLM_MULTITOOL_LOAD_S. Log: {_LOG_FILE}")


def _ensure_local_server(label: str):
    """Start (or reuse) the llama-server for `label` -> (base_url, model_id, error_or_None)"""
    with _local_lock:
        paths = _local_models_cached().get(label) or _local_models_cached(force=True).get(label)
        if not paths:
            return None, None, f"local model '{label}' not found on this machine -- refresh ComfyUI (R)."
        alias = _local_alias(label)
        # Ours and still loading (e.g. after a timeout): keep waiting instead of restarting it.
        if _local_label == label and _local_proc is not None and _local_proc.poll() is None:
            proc = _local_proc
            if _local_served() == alias:
                return LOCAL_URL, alias, None
        else:
            proc = _start_local_server(label, paths, alias)
            if isinstance(proc, tuple):    # finished: ready or error
                return proc
    return _wait_ready(proc, alias)


def _start_local_server(label, paths, alias):
    """Caller holds _local_lock. -> Popen to wait for, or a final result tuple"""
    global _local_proc, _local_label
    served = _local_served()
    if served == alias:
        return LOCAL_URL, alias, None
    if served is not None and _local_proc is None and not os.path.exists(_PID_FILE):
        return None, None, (f"Port {LOCAL_PORT} is used by another server ('{served}') -- "
                            "stop it, set env LLM_MULTITOOL_PORT, or pick 'server (api_url)'.")
    if not LLAMA_SERVER_BIN or not os.path.exists(LLAMA_SERVER_BIN):
        return None, None, ("llama-server not found -- put it on PATH or set env "
                            "LLAMA_SERVER_BIN to its full path.")

    _stop_local_server()
    model_path, mmproj_path = paths
    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        *(["--mmproj", mmproj_path] if mmproj_path else []),
        "--alias", alias,
        "--host", "127.0.0.1", "--port", str(LOCAL_PORT),
        "--ctx-size", str(LOCAL_CTX), "--parallel", "1",
        "--n-gpu-layers", "99", "--fit", "off", "--flash-attn", "on",
        "--image-max-tokens", "1024",
        "--sleep-idle-seconds", "300",   # unload after 5 min idle, frees VRAM
    ]
    print(f"[LLM_Multitool] Starting local server: {label} on port {LOCAL_PORT}")
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(_LOG_FILE, "w", encoding="utf-8", errors="replace") as log:
        _local_proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    _local_label = label
    with open(_PID_FILE, "w") as f:
        f.write(str(_local_proc.pid))
    return _local_proc


# =============================================================================
#  NODE -- LLM_Multitool
# =============================================================================

class LLMMultitool:
    """
    Vision / text LLM node for llama.cpp, Ollama, Unsloth Studio and OpenAI-compatible servers.

    Outputs
    -------
    response      Model reply text (or error message)
    thinking      Chain-of-thought (only when enable_thinking=True)
    active_model  Model ID that was actually used
    active_mode   Current mode string
    """

    @classmethod
    def INPUT_TYPES(cls):
        local = _local_models_cached()
        multiline = {"multiline": True, "dynamicPrompts": False}
        return {
            "required": {
                # MODEL / CONNECTION
                "local_model": ([SERVER_CHOICE] + list(local), {
                    "default": SERVER_CHOICE, "tooltip": HELP["local_model"]}),
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:8080", "tooltip": HELP["api_url"]}),
                "backend": (BACKENDS, {
                    "default": "llama.cpp", "tooltip": HELP["backend"]}),
                "model_id": ("STRING", {
                    "default": "auto", "tooltip": HELP["model_id_optional"]}),
                "api_key_env": ("STRING", {
                    "default": "", "tooltip": HELP["api_key_env"]}),
                # PROMPT
                "mode": (["auto", "vision", "text"], {
                    "default": "auto", "tooltip": HELP["mode"]}),
                "system_prompt": ("STRING", {**multiline,
                    "default": "You are a helpful assistant.", "tooltip": HELP["system_prompt"]}),
                "user_prompt": ("STRING", {**multiline,
                    "default": "Describe this image in detail.", "tooltip": HELP["user_prompt"]}),
                # SAMPLING
                "temperature": ("FLOAT", {"default": _DEFAULT_TEMP, "min": 0.0, "max": 2.0,
                                          "step": 0.01, "tooltip": HELP["temperature"]}),
                "top_p": ("FLOAT", {"default": _DEFAULT_TOP_P, "min": 0.0, "max": 1.0,
                                    "step": 0.01, "tooltip": HELP["top_p"]}),
                "top_k": ("INT", {"default": _DEFAULT_TOP_K, "min": -1, "max": 200,
                                  "tooltip": HELP["top_k"]}),
                "min_p": ("FLOAT", {"default": _DEFAULT_MIN_P, "min": 0.0, "max": 1.0,
                                    "step": 0.01, "tooltip": HELP["min_p"]}),
                "repetition_penalty": ("FLOAT", {"default": _DEFAULT_REP_PEN, "min": 0.0, "max": 2.0,
                                                 "step": 0.01, "tooltip": HELP["repetition_penalty"]}),
                "presence_penalty": ("FLOAT", {"default": _DEFAULT_PRES_PEN, "min": -2.0, "max": 2.0,
                                               "step": 0.01, "tooltip": HELP["presence_penalty"]}),
                "max_tokens": ("INT", {"default": _DEFAULT_MAX_TOKENS, "min": 1, "max": 32768,
                                       "step": 64, "tooltip": HELP["max_tokens"]}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1,
                                 "tooltip": HELP["seed"]}),
                # THINKING
                "enable_thinking": ("BOOLEAN", {"default": False, "tooltip": HELP["enable_thinking"]}),
                "reasoning_effort": (["none", "low", "medium", "high"], {
                    "default": "none", "tooltip": HELP["reasoning_effort"]}),
                # OUTPUT / VRAM / SAFETY
                "clean_output": ("BOOLEAN", {"default": False, "tooltip": HELP["clean_output"]}),
                "image_mode": (["first", "all"], {"default": "first", "tooltip": HELP["image_mode"]}),
                "unload_after": ("BOOLEAN", {"default": False, "tooltip": HELP["unload_after"]}),
                "keep_alive": ("INT", {"default": 0, "min": 0, "max": 86400, "step": 60,
                                       "tooltip": HELP["keep_alive"]}),
                "on_error": (["stop", "return text"], {"default": "stop", "tooltip": HELP["on_error"]}),
                "timeout_s": ("INT", {"default": 120, "min": 30, "max": 600, "step": 10,
                                      "tooltip": HELP["timeout_s"]}),
                "max_retries": ("INT", {"default": 2, "min": 0, "max": 5,
                                        "tooltip": HELP["max_retries"]}),
                "extra_body": ("STRING", {**multiline, "default": "", "tooltip": HELP["extra_body"]}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": HELP["image"]}),
                "text": ("STRING", {"forceInput": True, "tooltip": HELP["text"]}),
            },
        }

    OUTPUT_NODE     = True
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

    @classmethod
    def VALIDATE_INPUTS(cls, local_model=SERVER_CHOICE):
        # Taking local_model here skips ComfyUI's "value not in list" check for it:
        # a workflow from another machine must not fail just because that GGUF is
        # missing here -- _chat() reports it only if this node actually runs.
        return True

    # -- Main execute ---------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        # seed -1 = random -> NaN never equals itself, so ComfyUI never serves a cached reply
        return float("nan") if seed == -1 else ""

    def chat(self, **kwargs):
        on_error = kwargs.pop("on_error", "stop")
        result = self._chat(**kwargs)
        if on_error == "stop" and str(result[0]).startswith("ERROR:"):
            raise RuntimeError(f"LLM_Multitool: {result[0][6:].strip()}")
        # Shown on the node itself (web/llm_multitool.js), no wire needed.
        return {"ui": {"active_model": [result[2]]}, "result": result}

    def _chat(
        self, local_model=SERVER_CHOICE, api_url="http://127.0.0.1:8080", backend="llama.cpp",
        model_id="auto", api_key_env="", mode="auto",
        system_prompt="", user_prompt="",
        temperature=_DEFAULT_TEMP, top_p=_DEFAULT_TOP_P, top_k=_DEFAULT_TOP_K,
        min_p=_DEFAULT_MIN_P, repetition_penalty=_DEFAULT_REP_PEN,
        presence_penalty=_DEFAULT_PRES_PEN, max_tokens=_DEFAULT_MAX_TOKENS, seed=-1,
        enable_thinking=False, reasoning_effort="none",
        clean_output=False, image_mode="first", unload_after=False, keep_alive=0,
        timeout_s=120, max_retries=2, extra_body="",
        image=None, text=None,
    ):
        api_key, key_err = _resolve_api_key(api_key_env)
        if key_err:
            return (f"ERROR: {key_err}", "", "none", mode)

        base_url   = _normalize_url(api_url)
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
        is_local = bool(local_model and local_model != SERVER_CHOICE)
        if is_local:
            if is_vision and local_model.endswith(" (text)"):
                return (f"ERROR: '{local_model}' is text only (no mmproj next to it) -- "
                        "pick a model without '(text)' for images, or set mode to 'text'.",
                        "", "none", mode)
            base_url, model_id, err = _ensure_local_server(local_model)
            if err:
                return (f"ERROR: {err}", "", "none", mode)
            backend, api_key = "llama.cpp", ""
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
        img_b64_list = []
        if is_vision:
            n_imgs = min(image.shape[0], 8) if image_mode == "all" else 1
            if image_mode == "all" and image.shape[0] > 8:
                print(f"[LLM_Multitool] Batch has {image.shape[0]} images, only the first 8 are sent.")
            for i in range(n_imgs):
                arr = np.clip(255.0 * image[i].cpu().numpy(), 0, 255).astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=92)
                img_b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            img_b64 = img_b64_list[0] if img_b64_list else None
            if backend == "ollama":
                # Ollama: images as separate base64 array
                user_content = user_prompt
            else:
                user_content = [{"type": "text", "text": user_prompt}]
                for b64 in img_b64_list:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
        else:
            user_content = user_prompt
            img_b64 = None

        # 4 -- Build payload
        if backend == "ollama":
            # Ollama native: POST /api/chat, images as base64 array, keep_alive
            user_msg = {"role": "user", "content": user_content}
            if is_vision and img_b64_list:
                user_msg["images"] = img_b64_list      # /api/chat: images belong to the message
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    user_msg,
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repeat_penalty": repetition_penalty,
                    "presence_penalty": presence_penalty,
                    "num_predict": max_tokens,
                },
            }
            if seed != -1:
                payload["options"]["seed"] = seed
            if enable_thinking:
                payload["think"] = reasoning_effort if reasoning_effort != "none" else True
            # keep_alive: 0 = unload immediately (frees VRAM)
            payload["keep_alive"] = keep_alive if not unload_after else 0
            payload.update(extra)
        else:
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
            if backend == "llama.cpp":
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
        endpoint = "/api/chat" if backend == "ollama" else "/v1/chat/completions"
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(
                    f"{base_url}{endpoint}",
                    json=payload,
                    headers=_build_headers(api_key),
                    timeout=timeout_s,
                )
                r.raise_for_status()
                if backend == "ollama":
                    data     = r.json()
                    msg      = data.get("message") or {}
                    content  = msg.get("content") or ""
                    thinking = msg.get("thinking") or ""
                    finish   = data.get("done_reason") or ""
                else:
                    choice   = r.json()["choices"][0]
                    msg      = choice["message"]
                    content  = msg.get("content") or ""
                    thinking = msg.get("reasoning_content") or msg.get("thinking") or ""
                    finish   = choice.get("finish_reason") or ""

                if not content.strip():
                    why = ("max_tokens was used up" + (" by thinking" if thinking else "")
                           + " -- raise max_tokens or turn enable_thinking off."
                           if finish == "length" else "the model returned no text.")
                    return (f"ERROR: Empty response -- {why}", thinking, model_label, mode)

                # Optional clean-up, e.g. when the reply goes straight into a render prompt
                if clean_output:
                    content = _clean_response(content)

                # Unload after run
                if unload_after and is_local:
                    _stop_local_server()       # own server: no router API, just stop it
                elif unload_after and backend == "llama.cpp":
                    try:
                        requests.post(
                            f"{base_url}/models/unload",
                            json={"model": model_id},
                            headers=_build_headers(api_key),
                            timeout=10,
                        )
                        print(f"[LLM_Multitool] Unloaded model '{model_id}' (unload_after=True)")
                    except Exception as ue:
                        print(f"[LLM_Multitool] Unload failed (non-fatal): {ue}")

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
                if code == 503 and attempt < max_retries:    # model still (re)loading
                    wait = 2 ** (attempt + 1)
                    print(f"[LLM_Multitool] Server loading, retry {attempt+1}/{max_retries} in {wait}s ...")
                    time.sleep(wait)
                    continue
                detail = {
                    401: "401 Unauthorized -- check your API key.",
                    400: f"400 Bad request -- {r.text[:400]}",
                    503: "503 -- server is still loading the model, try again.",
                    422: f"422 Validation error -- {r.text[:400]}",
                    500: f"500 Server error -- {r.text[:200]}",
                }
                msg = detail.get(code, f'HTTP {code} -- {r.text[:200]}')
                if is_vision and code in (400, 500):
                    msg += " (Sending an image? The model may have no vision projector / mmproj.)"
                return (f"ERROR: {msg}", "", model_label, mode)

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
    "LLM_Multitool": LLMMultitool,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Multitool": "LLM_Multitool",
}
