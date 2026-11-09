// Shows the active model on the LLM_Multitool node itself (no output wire needed)
// and adds an "unload model" button that frees the VRAM of the active model.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SERVER_CHOICE = "server (api_url)";

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

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const model = message?.active_model?.[0];
            if (model && this._llmShow) this._llmShow(model);
        };
    },
});
