// Shows the active model on the LLM_Multitool node itself (no output wire needed)
// and adds an "unload" button for the node's own llama-server (port 8081).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SERVER_CHOICE = "server (api_url)";

app.registerExtension({
    name: "llm_multitool.ActiveModel",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "UnslothProAPI") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const status = this.addWidget("text", "active", "-", () => {}, { serialize: false });
            status.serialize = false;
            status.disabled = true;
            status.tooltip = "Model that answered the last run (read-only).\n" +
                "local-<name> = own llama-server on port 8081 from 'local_model'.";
            this._llmStatus = status;

            const unload = this.addWidget("button", "unload local model", null, async () => {
                try {
                    const res = await api.fetchApi("/llm_multitool/unload", { method: "POST" });
                    const data = await res.json();
                    status.value = data.unloaded ? "unloaded (VRAM free)" : "nothing loaded";
                } catch (e) {
                    status.value = `unload failed: ${e}`;
                }
                this.setDirtyCanvas(true, true);
            }, { serialize: false });
            unload.serialize = false;
            unload.tooltip = "Stop the llama-server that 'local_model' started (port 8081, shared\n" +
                "by all LLM_Multitool nodes) and free its VRAM. A server at api_url is not touched.";
            this.setSize(this.computeSize());

            // Show the pick right away, the real model id after the next run.
            const pick = this.widgets.find((w) => w.name === "local_model");
            if (pick) {
                const cb = pick.callback;
                pick.callback = (v, ...rest) => {
                    status.value = v === SERVER_CHOICE ? "server (after next run)" : `${v} (loads on run)`;
                    this.setDirtyCanvas(true, true);
                    return cb?.(v, ...rest);
                };
            }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const model = message?.active_model?.[0];
            if (model && this._llmStatus) {
                this._llmStatus.value = model;
                this.setDirtyCanvas(true, true);
            }
        };
    },
});
