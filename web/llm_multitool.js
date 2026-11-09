// Shows the active model on the LLM_Multitool node itself (no output wire needed)
// and adds an "unload model" button that frees the VRAM of the active model.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SERVER_CHOICE = "server (api_url)";

// Prompt boxes grow with their text (min/max in lines), the node grows with them.
const GROW = { system_prompt: [4, 24], user_prompt: [4, 24], extra_body: [2, 12] };
const LINE_PX = 16;
const CHAR_PX = 7.2;

function setupAutoGrow(node) {
    const boxes = node.widgets.filter((w) => GROW[w.name]);
    const lines = (w) => {
        const el = w.element ?? w.inputEl;
        const text = String(el?.value ?? w.value ?? "");
        const cols = Math.max(20, Math.floor(((el?.clientWidth || node.size[0] - 30) - 12) / CHAR_PX));
        const n = text.split("\n").reduce((sum, l) => sum + Math.max(1, Math.ceil(l.length / cols)), 0);
        const [lo, hi] = GROW[w.name];
        return Math.min(hi, Math.max(lo, n));
    };
    for (const w of boxes) {
        w.options = w.options ?? {};
        w.options.getMinHeight = () => lines(w) * LINE_PX + 12;
        w.options.getMaxHeight = () => lines(w) * LINE_PX + 12;
    }
    let pending = false;
    const fit = () => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => {
            pending = false;
            node.setSize([node.size[0], node.computeSize()[1]]);
            node.setDirtyCanvas(true, true);
        });
    };
    // The textareas may not exist yet when the node is created: listen on the document.
    const onInput = (e) => {
        if (boxes.some((w) => (w.element ?? w.inputEl) === e.target)) fit();
    };
    document.addEventListener("input", onInput, true);
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        document.removeEventListener("input", onInput, true);
        return onRemoved?.apply(this, arguments);
    };
    setTimeout(fit, 0);
    return fit;
}

app.registerExtension({
    name: "llm_multitool.ActiveModel",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LLM_Multitool") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const status = this.addWidget("text", "active", "-", () => {}, { serialize: false });
            status.serialize = false;
            status.disabled = true;
            status.tooltip = "Model that answered the last run (read-only).\n" +
                "local-<name> = own llama-server on port 8081 from 'local_model'.";
            this._llmStatus = status;
            // The current frontend draws a disabled widget by its label only,
            // so the text goes into the label as well as the value.
            const show = (text) => {
                status.value = text;
                status.label = `active: ${text}`;
                this.setDirtyCanvas(true, true);
            };
            this._llmShow = show;

            const val = (name) => this.widgets.find((w) => w.name === name)?.value ?? "";
            const unload = this.addWidget("button", "unload model", null, async () => {
                show("unloading ...");
                try {
                    const res = await api.fetchApi("/llm_multitool/unload", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            // with a local_model picked, only the own server is stopped
                            api_url: val("local_model") === SERVER_CHOICE || !val("local_model")
                                ? val("api_url") : "",
                            backend: val("backend") || "llama.cpp",
                            api_key_env: val("api_key_env"),
                        }),
                    });
                    const data = await res.json();
                    show(data.unloaded ? `VRAM free: ${data.message}` : data.message);
                } catch (e) {
                    show(`unload failed: ${e}`);
                }
            }, { serialize: false });
            unload.serialize = false;
            unload.tooltip = "Free VRAM now:\n" +
                "- stops the node's own llama-server (local_model, port 8081, shared by all nodes)\n" +
                "- llama.cpp router at api_url: unloads its loaded model\n" +
                "- ollama: unloads its loaded models (keep_alive 0)\n" +
                "- openai / unsloth / plain llama-server: nothing to unload";
            this.setSize(this.computeSize());
            this._llmFit = setupAutoGrow(this);

            // Show the pick right away, the real model id after the next run.
            const pick = this.widgets.find((w) => w.name === "local_model");
            if (pick) {
                const cb = pick.callback;
                pick.callback = (v, ...rest) => {
                    show(v === SERVER_CHOICE ? "server (after next run)" : `${v} (loads on run)`);
                    return cb?.(v, ...rest);
                };
            }
            return r;
        };

        // Saved workflows set the text after creation: fit again once loaded.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => this._llmFit?.(), 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const model = message?.active_model?.[0];
            if (model && this._llmShow) this._llmShow(model);
        };
    },
});
