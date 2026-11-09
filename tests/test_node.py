"""Offline tests for LLM_Multitool (no GPU, no server): python -m pytest tests -q"""
import importlib.util
import math
import os

import numpy as np
import pytest
import requests

_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_multitool.py")
_spec = importlib.util.spec_from_file_location("llm_multitool_under_test", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# -- fakes --------------------------------------------------------------------

class FakeTensor:
    """Stands in for a torch IMAGE batch [B,H,W,C]."""
    def __init__(self, b=1, h=8, w=8):
        self.arr = np.random.rand(b, h, w, 3).astype(np.float32)
        self.shape = self.arr.shape

    def __getitem__(self, i):
        a = self.arr[i]

        class _T:
            def cpu(self):
                return self

            def numpy(self):
                return a
        return _T()


class FakeResp:
    def __init__(self, data=None, code=200, text=""):
        self._data, self.status_code, self.text = data, code, text or str(data)

    def json(self):
        return self._data

    @property
    def ok(self):
        return self.status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


@pytest.fixture
def net(monkeypatch):
    """Records POSTs; replies are configured via net['post'] / net['get']."""
    state = {"posts": [], "post": FakeResp({"choices": [{"message": {"content": "ok"}}]}),
             "get": {}}

    def fake_post(url, json=None, **kw):
        state["posts"].append((url, json))
        return state["post"]

    def fake_get(url, **kw):
        for suffix, resp in state["get"].items():
            if url.endswith(suffix):
                return resp
        raise requests.exceptions.ConnectionError(url)

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod.requests, "get", fake_get)
    return state


def kw(**over):
    base = dict(
        api_url="http://127.0.0.1:8080", mode="text",
        system_prompt="sys", user_prompt="hi",
        temperature=1.0, top_p=0.95, top_k=64, min_p=0.0,
        repetition_penalty=1.1, presence_penalty=0.0,
        max_tokens=64, seed=-1, timeout_s=30, max_retries=0,
        enable_thinking=False, reasoning_effort="none",
        model_id="m", backend="llama.cpp", api_key_env="", local_model=mod.SERVER_CHOICE,
        clean_output=False, image_mode="first", unload_after=False, keep_alive=0,
        extra_body="",
    )
    base.update(over)
    return base


def run(**over):
    return mod.LLMMultitool().chat(**kw(**over))["result"]


# -- helpers ------------------------------------------------------------------

def test_clean_keeps_normal_words():
    assert mod._clean_response("I think the weekend is fine") == "I think the weekend is fine"


def test_clean_removes_think_and_markdown():
    assert mod._clean_response("<think>abc</think>Hello **world**") == "Hello world"


def test_clean_drops_orphan_closing_tag():
    assert mod._clean_response("reasoning here</think> Final") == "Final"


def test_normalize_url():
    assert mod._normalize_url("127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    assert mod._normalize_url("") == "http://127.0.0.1:8080"
    assert mod._normalize_url("https://api.openai.com/v1") == "https://api.openai.com"


def test_pick_router_model():
    pick = mod._pick_router_model
    assert pick([{"id": "a", "status": {"value": "unloaded"}},
                 {"id": "b", "status": {"value": "sleeping"}}]) == "b"
    assert pick([{"id": "a", "status": {"value": "unloaded"}}]) is None
    assert pick([{"id": "x"}]) == "x"
    assert pick([]) is None


def test_is_changed():
    v = mod.LLMMultitool.IS_CHANGED(seed=-1)
    assert isinstance(v, float) and math.isnan(v)
    assert mod.LLMMultitool.IS_CHANGED(seed=3) == ""


# -- requests -----------------------------------------------------------------

def test_ollama_vision_payload(net):
    net["post"] = FakeResp({"message": {"content": "hi", "thinking": "t"}})
    res = run(backend="ollama", mode="vision", image=FakeTensor(), seed=5,
              enable_thinking=True, api_url="http://127.0.0.1:11434")
    url, payload = net["posts"][-1]
    assert url.endswith("/api/chat")
    assert "images" not in payload
    assert len(payload["messages"][1]["images"]) == 1
    assert payload["options"]["seed"] == 5
    assert payload["think"] is True
    assert res == ("hi", "t", "m", "vision")


def test_router_auto_picks_loaded_model(net):
    net["get"]["/v1/models"] = FakeResp({"data": [
        {"id": "a", "status": {"value": "unloaded"}},
        {"id": "b", "status": {"value": "sleeping"}}]})
    run(model_id="auto")
    assert net["posts"][-1][1]["model"] == "b"


def test_empty_response_return_text(net):
    net["post"] = FakeResp({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    res = run(on_error="return text")
    assert res[0].startswith("ERROR: Empty response") and "max_tokens" in res[0]


def test_empty_response_stops_by_default(net):
    net["post"] = FakeResp({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    with pytest.raises(RuntimeError):
        run()
    with pytest.raises(RuntimeError):
        run(on_error="stop")


def test_vision_400_mentions_mmproj(net):
    net["post"] = FakeResp({"error": "x"}, code=400, text="image input is not supported")
    res = run(mode="vision", image=FakeTensor(), on_error="return text")
    assert "mmproj" in res[0]


def test_url_with_v1_is_fixed(net):
    run(api_url="127.0.0.1:8080/v1")
    assert net["posts"][-1][0] == "http://127.0.0.1:8080/v1/chat/completions"


# -- unload button --------------------------------------------------------------

def test_unload_router_only_loaded_models(net, monkeypatch):
    monkeypatch.setattr(mod, "_local_served", lambda: None)
    net["post"] = FakeResp({"success": True})
    net["get"]["/v1/models"] = FakeResp({"data": [
        {"id": "a", "status": {"value": "unloaded"}},
        {"id": "b", "status": {"value": "loaded"}}]})
    done, msg = mod._unload_all(api_url="127.0.0.1:8080", backend="llama.cpp")
    assert done and "b" in msg
    assert net["posts"] == [("http://127.0.0.1:8080/models/unload", {"model": "b"})]


def test_unload_plain_llama_server_only_notes(net, monkeypatch):
    monkeypatch.setattr(mod, "_local_served", lambda: None)
    net["get"]["/v1/models"] = FakeResp({"data": [{"id": "x"}]})
    done, msg = mod._unload_all(api_url="http://h:8080", backend="llama.cpp")
    assert not done and "no unload API" in msg and net["posts"] == []


def test_unload_ollama_keep_alive_0(net, monkeypatch):
    monkeypatch.setattr(mod, "_local_served", lambda: None)
    net["post"] = FakeResp({})
    net["get"]["/api/ps"] = FakeResp({"models": [{"name": "llava"}]})
    done, msg = mod._unload_all(api_url="http://h:11434", backend="ollama")
    assert done
    assert net["posts"] == [("http://h:11434/api/generate", {"model": "llava", "keep_alive": 0})]


def test_unload_openai_does_nothing(net, monkeypatch):
    monkeypatch.setattr(mod, "_local_served", lambda: None)
    done, msg = mod._unload_all(api_url="https://api.openai.com/v1", backend="openai")
    assert not done and "nothing to unload" in msg and net["posts"] == []


# -- model scan -----------------------------------------------------------------

def test_scan_lists_text_only_and_skips_non_chat(tmp_path, monkeypatch):
    vis, txt = tmp_path / "vis", tmp_path / "txt"
    vis.mkdir(); txt.mkdir()
    for f in ("Model-A-Q4.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf"):
        (vis / f).write_bytes(b"x")
    for f in ("Chat-B-Q4.gguf", "t5-xxl-encoder-Q8.gguf", "X-DFlash2-Q4.gguf", "mtp-head.gguf"):
        (txt / f).write_bytes(b"x")
    monkeypatch.setattr(mod, "_model_dirs", lambda: [str(tmp_path)])
    found = mod._scan_local_models()
    assert set(found) == {"Model-A-Q4", "Chat-B-Q4 (text)"}
    assert found["Model-A-Q4"][1].endswith("mmproj-F16.gguf")
    assert found["Chat-B-Q4 (text)"][1] is None


def test_text_only_model_rejects_image(net):
    res = run(local_model="Chat-B-Q4 (text)", mode="vision", image=FakeTensor(), on_error="return text")
    assert "text only" in res[0] and net["posts"] == []


def test_inputs_have_no_legacy_fields():
    it = mod.LLMMultitool.INPUT_TYPES()
    names = set(it["required"]) | set(it["optional"])
    assert not {"api_key", "_header_system_prompt", "_header_user_prompt"} & names
    assert "beellama" not in mod.BACKENDS
    assert list(it["required"])[0] == "local_model"
    assert set(it["optional"]) == {"image", "text"}
