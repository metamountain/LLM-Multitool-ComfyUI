// Shows the active model on the LLM_Multitool node itself (no output wire needed).
import { app } from "../../scripts/app.js";

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
